from bisect import bisect_right
import math

import torch

import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.model_patcher
import comfy.patcher_extension
import comfy.utils
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


def cache_lora_deltas(model, hooks):
    cached = {}
    for hook in hooks.hooks:
        cached[hook.hook_ref] = {}
        for key, patches in model.hook_patches[hook.hook_ref].items():
            cached[hook.hook_ref][key] = []
            for patch in patches:
                adapter = patch[1]
                if isinstance(adapter, LoRAAdapter) and adapter.weights[4] is None and adapter.weights[5] is None:
                    weight, _, convert_func = comfy.model_patcher.get_key_weight(model.model, key)
                    if convert_func is None:
                        shape = list(weight.shape)
                        if patch[3] is not None:
                            shape[patch[3][0]] = patch[3][2]
                        delta = torch.zeros(shape, device="cpu", dtype=torch.float32)
                        delta = comfy.lora.calculate_weight([(1.0, adapter, 1.0, None, None)], delta, key)
                        patch = (patch[0], ("diff", (delta,)), *patch[2:])
                cached[hook.hook_ref][key].append(patch)
    return cached


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
            if weight > 0:
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
        original_hook_patches = guider.model_patcher.hook_patches
        cache_enabled = self.lora_smooth_steps > 0 and any(roles)
        guider.model_options = comfy.model_patcher.create_model_options_clone(original_options)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE, MIXER_KEY, predict,
            guider.model_options, is_model_options=True,
        )
        # Avoid retaining a second model's worth of patched weights between segments.
        guider.model_patcher.set_hook_mode(comfy.hooks.EnumHookMode.MinVram)
        try:
            if cache_enabled:
                guider.model_patcher.hook_patches = original_hook_patches | cache_lora_deltas(guider.model_patcher, self.hooks)
            return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)
        finally:
            # Multigpu clones copy hook patches during sampling preparation.
            if cache_enabled:
                for clone in guider.model_patcher.get_additional_models_with_key("multigpu"):
                    for hook in self.hooks.hooks:
                        clone.hook_patches[hook.hook_ref] = original_hook_patches[hook.hook_ref]
            guider.model_patcher.hook_patches = original_hook_patches
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
