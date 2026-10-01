from bisect import bisect_right
from copy import copy
import math

import torch

import comfy.hooks
import comfy.lora_convert
import comfy.model_patcher
import comfy.patcher_extension
import comfy.utils
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter
import folder_paths
from comfy_api.latest import io


BASE_BASE = "Base composition / Base style"
BASE_TURBO = "Base composition / Turbo style"
TURBO_BASE = "Turbo composition / Base style"
TURBO_TURBO = "Turbo composition / Turbo style"
MIXER_KEY = "turbo_base_trajectory_mixer"


def trajectory_steps(steps, scheme):
    mode = scheme["scheme"]
    composition, style = {
        BASE_BASE: (False, False), BASE_TURBO: (False, True),
        TURBO_BASE: (True, False), TURBO_TURBO: (True, True),
    }[mode]
    if composition != style:
        prefix = math.floor(steps * scheme["composition_ratio"] + 0.5)
        return [composition] * prefix + [style] * (steps - prefix)
    first_half = (steps + 1) // 2
    second_half = steps // 2
    prefix = math.floor(first_half * scheme["prefix_ratio"] + 0.5)
    suffix = math.floor(second_half * scheme["suffix_ratio"] + 0.5)
    return [composition] * prefix + [not composition] * (steps - prefix - suffix) + [style] * suffix


def lora_weight(progress, cuts, segment_roles, smooth_steps, total_steps):
    for i, cut in enumerate(cuts):
        previous = cuts[i - 1] if i else 0
        following = cuts[i + 1] if i + 1 < len(cuts) else total_steps
        half_width = min(smooth_steps, cut - previous, following - cut) / 2
        if half_width > 0 and cut - half_width < progress < cut + half_width:
            blend = (progress - cut + half_width) / (2 * half_width)
            blend = blend * blend * (3 - 2 * blend)
            return float(segment_roles[i]) + (float(segment_roles[i + 1]) - float(segment_roles[i])) * blend
    return float(segment_roles[bisect_right(cuts, progress)])


class StrengthKeyframes(comfy.hooks.HookKeyframeGroup):
    def __init__(self):
        super().__init__()
        self.value = 1.0
        self.previous = 1.0

    @property
    def strength(self):
        return self.value

    def prepare_current_keyframe(self, curr_t, transformer_options):
        changed = self.value != self.previous
        self.previous = self.value
        return changed


class ScheduledLoRABypass(BypassForwardHook):
    def __init__(self, module, adapter, multiplier, strength, device):
        super().__init__(module, adapter, multiplier)
        self.strength = strength
        self.device = device
        self.source_weights = adapter.weights

    def _move_adapter_weights_to_device(self, device, dtype=None):
        super()._move_adapter_weights_to_device(self.device, dtype)

    def _bypass_forward(self, x, *args, **kwargs):
        multiplier = self.multiplier * self.strength.value
        if multiplier == 0:
            return self.original_forward(x, *args, **kwargs)
        self.adapter.multiplier = multiplier
        return super()._bypass_forward(x, *args, **kwargs)

    def release(self):
        self.eject()
        self.adapter.weights = self.source_weights


def make_bypass_injection(model, hooks, strength):
    # Native injection ejection does not unwind stacked forward wrappers in reverse order.
    if model.injections:
        return None, {}
    patches = []
    for hook in hooks.hooks:
        for key, entries in model.hook_patches[hook.hook_ref].items():
            if not key.endswith(".weight"):
                return None, {}
            module = comfy.utils.get_attr(model.model, key[:-7])
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return None, {}
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) and (module.groups != 1 or module.padding_mode != "zeros"):
                return None, {}
            for scale, adapter, model_scale, offset, function in entries:
                if (not isinstance(adapter, LoRAAdapter) or any(value is not None for value in adapter.weights[3:])
                        or model_scale != 1 or offset is not None or function is not None):
                    return None, {}
                patches.append((key[:-7], adapter, scale))
    if not patches:
        return None, {}
    active = {}

    def inject(patcher):
        if patcher not in active:
            active[patcher] = [ScheduledLoRABypass(comfy.utils.get_attr(patcher.model, key), copy(adapter), scale,
                                                strength, patcher.load_device) for key, adapter, scale in patches]
        for bypass in active[patcher]:
            bypass.inject()

    def eject(patcher):
        for bypass in reversed(active.get(patcher, [])):
            bypass.eject()

    return comfy.patcher_extension.PatcherInjection(inject, eject), active


class TrajectoryHook:
    def __init__(self, hooks, scheme, turbo_cfg, lora_smooth_steps=0.0):
        self.hooks = hooks
        self.scheme = scheme.copy()
        self.turbo_cfg = turbo_cfg
        self.lora_smooth_steps = lora_smooth_steps

    def register(self, patcher, hooks, target_dict, model_options, registered):
        for hook in self.hooks.hooks:
            registered.add(hook)

    def sample(self, executor, noise, latent_image, sampler, sigmas, *args, **kwargs):
        guider = executor.class_obj
        roles = trajectory_steps(len(sigmas) - 1, self.scheme)
        sigma_values = sigmas.detach().cpu().tolist()
        negative_sigmas = [-sigma for sigma in sigma_values]
        cuts = [i for i in range(1, len(roles)) if roles[i] != roles[i - 1]]
        thresholds = [-sigma_values[i] for i in cuts]
        segment_roles = [roles[0]] + [roles[i] for i in cuts] if roles else [False]
        turbo_conds = None
        combined = {}
        strength = StrengthKeyframes()
        bypass, bypass_hooks = make_bypass_injection(guider.model_patcher, self.hooks, strength) if any(roles) else (None, {})
        sample_hooks = self.hooks.clone()
        for hook in sample_hooks.hooks:
            hook.hook_keyframe = strength

        def predict(pred_executor, x, timestep, model_options, seed=None):
            nonlocal turbo_conds
            sigma = float(timestep[0])
            turbo = segment_roles[bisect_right(thresholds, -sigma)]
            weight = float(turbo)
            if self.lora_smooth_steps > 0 and roles:
                interval = bisect_right(negative_sigmas, -sigma) - 1
                if interval < 0:
                    progress = 0.0
                elif interval >= len(roles):
                    progress = float(len(roles))
                else:
                    progress = interval + (sigma_values[interval] - sigma) / (sigma_values[interval] - sigma_values[interval + 1])
                weight = lora_weight(progress, cuts, segment_roles, self.lora_smooth_steps, len(roles))
            strength.value = weight
            current_guider = pred_executor.class_obj
            original_conds = current_guider.conds
            original_cfg = current_guider.cfg
            if weight > 0 and bypass is None:
                if turbo_conds is None:
                    turbo_conds = {}
                    for name, conds in original_conds.items():
                        turbo_conds[name] = []
                        for cond in conds:
                            existing = cond.get("hooks")
                            if existing not in combined:
                                combined[existing] = sample_hooks if existing is None else existing.clone_and_combine(sample_hooks)
                            turbo_conds[name].append(cond | {"hooks": combined[existing]})
                current_guider.conds = turbo_conds
            if turbo:
                current_guider.cfg = self.turbo_cfg
            try:
                return pred_executor(x, timestep, model_options, seed)
            finally:
                current_guider.conds = original_conds
                current_guider.cfg = original_cfg

        original_options = guider.model_options
        original_hook_mode = guider.model_patcher.hook_mode
        guider.model_options = comfy.model_patcher.create_model_options_clone(original_options)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE, MIXER_KEY, predict,
            guider.model_options, is_model_options=True,
        )
        # Avoid retaining a second model's worth of patched weights between segments.
        guider.model_patcher.set_hook_mode(comfy.hooks.EnumHookMode.MinVram)
        patchers = [guider.model_patcher] + guider.model_patcher.get_additional_models_with_key("multigpu")
        original_injected = []
        try:
            if bypass is not None:
                for patcher in patchers:
                    original_injected.append((patcher, patcher.is_injected))
                    patcher.eject_model()
                    patcher.set_injections(MIXER_KEY, [bypass])
                guider.model_patcher.inject_model()
            return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)
        finally:
            for patcher, was_injected in reversed(original_injected):
                bypass.eject(patcher)
                patcher.eject_model()
                patcher.remove_injections(MIXER_KEY)
                if was_injected:
                    patcher.inject_model()
            for active in bypass_hooks.values():
                for hook in active:
                    hook.release()
            bypass_hooks.clear()
            guider.model_options = original_options
            guider.model_patcher.set_hook_mode(original_hook_mode)


def attach_mixer(model, hooks, scheme, turbo_cfg, lora_smooth_steps=0.0):
    mixer = TrajectoryHook(hooks, scheme, turbo_cfg, lora_smooth_steps)
    previous = model.get_attachment(MIXER_KEY)
    if previous is not None:
        for hook in previous.hooks:
            model.hook_patches.pop(hook.hook_ref)
    model.set_attachments(MIXER_KEY, hooks)
    model.remove_callbacks_with_key(comfy.patcher_extension.CallbacksMP.ON_REGISTER_ALL_HOOK_PATCHES, MIXER_KEY)
    model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, MIXER_KEY)
    model.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_REGISTER_ALL_HOOK_PATCHES, MIXER_KEY, mixer.register)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, MIXER_KEY, mixer.sample)
    return model


class TurboBaseTrajectoryMixer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        def same_inputs():
            return [
                io.Float.Input("prefix_ratio", default=0.5, min=0.0, max=1.0, step=0.01,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="Fraction of the first half assigned to the selected composition source."),
                io.Float.Input("suffix_ratio", default=0.5, min=0.0, max=1.0, step=0.01,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="Fraction of the second half assigned to the selected style source. The middle uses the other source."),
            ]

        def different_inputs():
            return [io.Float.Input("composition_ratio", default=0.5, min=0.0, max=1.0, step=0.01,
                                   display_mode=io.NumberDisplay.slider,
                                   tooltip="Fraction of all active steps assigned to composition; the rest use the style source.")]

        return io.Schema(
            node_id="TurboBaseTrajectoryMixer",
            display_name="Turbo / Base Trajectory Mixer",
            category="model/sampling",
            description="Switch a Turbo LoRA on/off along the existing sampler schedule. Composition/style labels describe a sampling hypothesis, not guaranteed visual separation.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("turbo_lora", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("lora_strength", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("turbo_cfg", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="CFG during Turbo segments with standard CFG guiders. Base segments use the sampler's CFG. Custom guiders may implement their own guidance."),
                io.DynamicCombo.Input("scheme", options=[
                    io.DynamicCombo.Option(BASE_BASE, same_inputs()),
                    io.DynamicCombo.Option(BASE_TURBO, different_inputs()),
                    io.DynamicCombo.Option(TURBO_BASE, different_inputs()),
                    io.DynamicCombo.Option(TURBO_TURBO, same_inputs()),
                ]),
                io.Float.Input("lora_smooth_steps", optional=True, default=0.0, min=0.0, max=32.0, step=0.5,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="LoRA strength transition width in active sampling steps, centered on each switch. 0 keeps hard switching. Neighboring segments limit the width; CFG keeps its segment settings."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, turbo_lora, lora_strength, turbo_cfg, scheme, lora_smooth_steps=0.0):
        path = folder_paths.get_full_path_or_raise("loras", turbo_lora)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        lora = comfy.lora_convert.convert_lora(lora)
        # ComfyUI uses the same delegate for conditioning-based weight hooks.
        if model.is_dynamic():
            model = model.get_non_dynamic_delegate()
        mixed, _, hooks = comfy.hooks.load_hook_lora_for_models(model, None, lora, lora_strength, 0.0)
        if not any(mixed.hook_patches[hook.hook_ref] for hook in hooks.hooks):
            raise ValueError("The Turbo LoRA contains no compatible diffusion-model weights for this model.")
        return io.NodeOutput(attach_mixer(mixed, hooks, scheme, turbo_cfg, lora_smooth_steps))
