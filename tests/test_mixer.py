import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
import weakref
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT.parents[1]
sys.path.insert(0, str(HOST))
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.hooks
import comfy.model_base
import comfy.model_patcher
import comfy.samplers
import comfy.supported_models
from comfy_api.latest import _io as internal_io
import nodes as host_nodes
sys.argv = original_argv

spec = importlib.util.spec_from_file_location("trajectory_test_plugin", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
mixer = sys.modules["trajectory_test_plugin.nodes"]

POSITIVE = [[torch.ones(1, 1, 2), {}]]
NEGATIVE = [[torch.zeros(1, 1, 2), {}]]
SIGMAS = torch.linspace(1.0, 0.0, 13)
LORA = {"diffusion_model.proj.lora_up.weight": torch.ones(2, 1),
        "diffusion_model.proj.lora_down.weight": torch.ones(1, 2)}


class TinyDiffusion(torch.nn.Module):
    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False, device=device)
        self.proj.weight.data.fill_(1.0)
        self.dtype = torch.float32
        self.seen = []

    def forward(self, x, timestep, context=None, **kwargs):
        effective = self.proj(torch.ones(x.shape[0], 2, device=x.device, dtype=x.dtype) / 2).mean()
        self.seen.append((float(effective), x.shape[0]))
        guidance = context.mean(dim=(1, 2)).view(-1, *([1] * (x.ndim - 1)))
        return torch.ones_like(x) * (effective + guidance)


def tiny_model(model_type=comfy.model_base.ModelType.FLOW):
    config = comfy.supported_models.Anima({"image_model": "anima"})
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.BaseModel(config, model_type, device=torch.device("cpu"), unet_model=TinyDiffusion)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def scheme(mode=mixer.BASE_BASE, **ratios):
    return {"scheme": mode, "prefix_ratio": 0.5, "suffix_ratio": 0.5, "composition_ratio": 0.5} | ratios


def mixed_model(base, selection=None, lora_smooth_steps=0.0):
    mixed, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, LORA, 1.0, 0.0)
    return mixer.attach_mixer(mixed, hooks, selection or scheme(), 1.0, lora_smooth_steps)


def sample(model, sigmas=SIGMAS, sampler_name="euler", positive=POSITIVE, negative=NEGATIVE, callback=None, mask=None, sampler=None):
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(4.0)
    noise = torch.full((1, 16, 1, 2, 2), 3.0)
    output = guider.sample(noise, torch.zeros_like(noise), sampler or comfy.samplers.ksampler(sampler_name), sigmas,
                           denoise_mask=mask, callback=callback, disable_pbar=True, seed=0)
    return output, guider


class MixerTests(unittest.TestCase):
    def test_bypass_skips_zero_strength_and_never_merges_turbo_weights(self):
        base = tiny_model()
        mixed = mixed_model(base, lora_smooth_steps=4.0)
        forward = base.model.diffusion_model.proj.forward
        original_h = mixer.LoRAAdapter.h
        calls = []
        copies = []
        def apply(adapter, x, base_out):
            calls.append(adapter.multiplier)
            copies.append(weakref.ref(adapter))
            return original_h(adapter, x, base_out)
        def check_weights(step, denoised, x, total):
            torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2), rtol=0, atol=0)
        with patch.object(mixer.LoRAAdapter, "calculate_weight") as merge:
            with patch.object(mixer.LoRAAdapter, "h", new=apply):
                sample(mixed, callback=check_weights)
                self.assertEqual(len(calls), 9)
                calls.clear()
                sample(mixed_model(base))
                self.assertEqual(calls, [1.0] * 6)
                calls.clear()
                sample(mixed_model(base, scheme(prefix_ratio=1, suffix_ratio=1)))
                self.assertFalse(calls)
            merge.assert_not_called()
        self.assertTrue(all(ref() is None for ref in copies))
        self.assertEqual(base.model.diffusion_model.proj.forward, forward)
        self.assertFalse(mixed.injections)

    def test_bypass_matches_weight_hooks_with_alpha_negative_strength_and_heun(self):
        lora = {"diffusion_model.proj.lora_up.weight": torch.tensor([[0.7], [-0.4]]),
                "diffusion_model.proj.lora_down.weight": torch.tensor([[0.3, -0.2]]),
                "diffusion_model.proj.alpha": torch.tensor(2.0)}
        for scale in (-0.75, 0.0, 1.25):
            for solver in ("euler", "heun", "dpmpp_2m"):
                with self.subTest(scale=scale, solver=solver):
                    base = tiny_model()
                    mixed, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, lora, scale, 0.0)
                    mixed = mixer.attach_mixer(mixed, hooks, scheme(), 1.0, 4.0)
                    bypass, _ = sample(mixed, sampler_name=solver)
                    with patch.object(mixer, "make_bypass_injection", return_value=(None, {})):
                        merged, _ = sample(mixed, sampler_name=solver)
                    torch.testing.assert_close(bypass, merged)
                    self.assertFalse(mixed.injections)

    def test_dora_and_existing_injections_use_weight_hooks(self):
        base = tiny_model()
        dora = LORA | {"diffusion_model.proj.dora_scale": torch.ones(2, 1)}
        mixed, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, dora, 1.0, 0.0)
        mixed = mixer.attach_mixer(mixed, hooks, scheme(), 1.0, 2.0)
        injection, _ = mixer.make_bypass_injection(mixed, hooks, mixer.StrengthKeyframes())
        self.assertIsNone(injection)
        with patch.object(mixer.LoRAAdapter, "h") as bypass:
            output, _ = sample(mixed)
            self.assertTrue(torch.isfinite(output).all())
            bypass.assert_not_called()
        torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))
        existing = comfy.patcher_extension.PatcherInjection(lambda patcher: None, lambda patcher: None)
        base.set_injections("other", [existing])
        mixed = mixed_model(base, lora_smooth_steps=2.0)
        sample(mixed)
        self.assertEqual(mixed.get_injections("other"), [existing])
        self.assertIsNone(mixed.get_injections(mixer.MIXER_KEY))

    def test_bypass_interruption_releases_adapters_and_restores_forward(self):
        base = tiny_model()
        mixed = mixed_model(base, lora_smooth_steps=4.0)
        forward = base.model.diffusion_model.proj.forward
        released = []
        original_release = mixer.ScheduledLoRABypass.release
        def release(bypass):
            original_release(bypass)
            self.assertIs(bypass.adapter.weights, bypass.source_weights)
            released.append(weakref.ref(bypass.adapter))
        def interrupt(step, denoised, x, total):
            if step == 3:
                raise RuntimeError("bypass interruption")
        with patch.object(mixer.ScheduledLoRABypass, "release", new=release):
            with self.assertRaisesRegex(RuntimeError, "bypass interruption"):
                sample(mixed, callback=interrupt)
        self.assertTrue(released)
        self.assertTrue(all(ref() is None for ref in released))
        self.assertEqual(base.model.diffusion_model.proj.forward, forward)
        self.assertFalse(mixed.injections)
        self.assertFalse(mixed.is_injected)
        again, _ = sample(mixed)
        self.assertTrue(torch.isfinite(again).all())

    def test_conv_bypass_matches_native_merged_weight_with_stride_and_dilation(self):
        model = torch.nn.Module()
        model.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=2, dilation=2)
        patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        adapter = mixer.LoRAAdapter(set(), (torch.randn(4, 2, 1, 1), torch.randn(2, 3, 3, 3), 1.5, None, None, None))
        hook = comfy.hooks.WeightHook()
        hooks = comfy.hooks.HookGroup()
        hooks.add(hook)
        patcher.hook_patches[hook.hook_ref] = {"conv.weight": [(-0.75, adapter, 1.0, None, None)]}
        strength = mixer.StrengthKeyframes()
        strength.value = 0.37
        injection, active = mixer.make_bypass_injection(patcher, hooks, strength)
        forward = model.conv.forward
        patcher.set_injections(mixer.MIXER_KEY, [injection])
        x = torch.randn(2, 3, 17, 19)
        try:
            patcher.inject_model()
            result = model.conv(x)
            weight = comfy.lora.calculate_weight([(-0.75 * strength.value, adapter, 1.0, None, None)], model.conv.weight.detach().clone(), "conv.weight")
            expected = torch.nn.functional.conv2d(x, weight, model.conv.bias, stride=2, padding=2, dilation=2)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        finally:
            patcher.eject_model()
            patcher.remove_injections(mixer.MIXER_KEY)
            for bypass in active[patcher]:
                bypass.release()
        self.assertEqual(model.conv.forward, forward)
        model.conv.padding_mode = "reflect"
        injection, _ = mixer.make_bypass_injection(patcher, hooks, strength)
        self.assertIsNone(injection)

    def test_partial_bypass_injection_failure_restores_forward(self):
        base = tiny_model()
        mixed = mixed_model(base)
        forward = base.model.diffusion_model.proj.forward
        original_inject = mixer.ScheduledLoRABypass.inject
        def fail(bypass):
            original_inject(bypass)
            raise RuntimeError("partial injection")
        with patch.object(mixer.ScheduledLoRABypass, "inject", new=fail):
            with self.assertRaisesRegex(RuntimeError, "partial injection"):
                sample(mixed)
        self.assertEqual(base.model.diffusion_model.proj.forward, forward)
        self.assertFalse(mixed.injections)
        self.assertFalse(mixed.is_injected)
        sample(mixed)

    def test_bypass_injection_uses_each_patchers_model_and_device(self):
        main = mixed_model(tiny_model())
        clone = tiny_model()
        main.set_additional_models("multigpu", [clone])
        modules = [patcher.model.diffusion_model.proj for patcher in (main, clone)]
        forwards = [module.forward for module in modules]
        def run(noise, latent, sampler, sigmas):
            clone.inject_model()
            for patcher, module in zip((main, clone), modules):
                bypass = module.forward.__self__
                self.assertIsInstance(bypass, mixer.ScheduledLoRABypass)
                self.assertEqual(bypass.device, patcher.load_device)
                torch.testing.assert_close(module(torch.ones(1, 2)), torch.full((1, 2), 4.0))
            raise RuntimeError("multigpu interruption")
        guider = comfy.samplers.CFGGuider(main)
        wrapper = main.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, mixer.MIXER_KEY)[0]
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(run, guider, [wrapper])
        with self.assertRaisesRegex(RuntimeError, "multigpu interruption"):
            executor.execute(None, None, None, SIGMAS)
        for patcher, module, forward in zip((main, clone), modules, forwards):
            self.assertEqual(module.forward, forward)
            self.assertFalse(patcher.injections)
            self.assertFalse(patcher.is_injected)

    def test_four_default_trajectories(self):
        expected = {
            mixer.BASE_BASE: [False] * 3 + [True] * 6 + [False] * 3,
            mixer.BASE_TURBO: [False] * 6 + [True] * 6,
            mixer.TURBO_BASE: [True] * 6 + [False] * 6,
            mixer.TURBO_TURBO: [True] * 3 + [False] * 6 + [True] * 3,
        }
        for mode, roles in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(mixer.trajectory_steps(12, scheme(mode)), roles)
                base = tiny_model()
                mixed = mixed_model(base, scheme(mode))
                result, guider = sample(mixed)
                self.assertEqual(base.model.diffusion_model.seen, [(2.0, 1) if role else (1.0, 2) for role in roles])
                expected_raw = torch.full_like(result, 3.0 - sum(3 if role else 5 for role in roles) / 12)
                torch.testing.assert_close(result, base.model.process_latent_out(expected_raw))
                self.assertEqual(guider.cfg, 4.0)
                self.assertFalse(base.patches)
                self.assertFalse(mixed.hook_backup)
                self.assertFalse(mixed.cached_hook_patches)
                torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))

    def test_endpoints_and_odd_halves(self):
        self.assertEqual(mixer.trajectory_steps(5, scheme(prefix_ratio=1, suffix_ratio=0)), [False] * 3 + [True] * 2)
        self.assertEqual(mixer.trajectory_steps(5, scheme(prefix_ratio=0, suffix_ratio=1)), [True] * 3 + [False] * 2)
        self.assertEqual(mixer.trajectory_steps(12, scheme(prefix_ratio=1, suffix_ratio=1)), [False] * 12)
        self.assertEqual(mixer.trajectory_steps(12, scheme(prefix_ratio=0, suffix_ratio=0)), [True] * 12)
        self.assertEqual(mixer.trajectory_steps(12, scheme(mixer.BASE_TURBO, composition_ratio=0)), [True] * 12)
        self.assertEqual(mixer.trajectory_steps(12, scheme(mixer.BASE_TURBO, composition_ratio=1)), [False] * 12)

    def test_short_partial_schedule_and_repeat_have_no_state_leak(self):
        base = tiny_model()
        mixed = mixed_model(base)
        partial = torch.tensor([0.6, 0.5, 0.3, 0.1, 0.0])
        first, _ = sample(mixed, sigmas=partial)
        self.assertEqual(base.model.diffusion_model.seen, [(1.0, 2), (2.0, 1), (2.0, 1), (1.0, 2)])
        second, _ = sample(mixed, sigmas=partial)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertNotIn("wrappers", mixed.model_options.get("transformer_options", {}))

    def test_eps_v_prediction_flow_and_multistep_samplers(self):
        for kind in (comfy.model_base.ModelType.EPS, comfy.model_base.ModelType.V_PREDICTION, comfy.model_base.ModelType.FLOW):
            for solver in ("euler", "heun", "dpmpp_2m"):
                with self.subTest(kind=kind, solver=solver):
                    base = tiny_model(kind)
                    result, _ = sample(mixed_model(base), sampler_name=solver)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertEqual(base.model.diffusion_model.seen[0][0], 1.0)
                    self.assertIn((2.0, 1), base.model.diffusion_model.seen)
                    self.assertEqual(base.model.diffusion_model.seen[-1][0], 1.0)
                    torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))

    def test_existing_static_and_conditioning_loras_are_preserved(self):
        base = tiny_model()
        base.add_patches({"diffusion_model.proj.weight": ("diff", (torch.full((2, 2), 0.25),))})
        other, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, LORA, 0.5, 0.0)
        positive = comfy.hooks.set_hooks_for_conditioning(POSITIVE, hooks)
        negative = comfy.hooks.set_hooks_for_conditioning(NEGATIVE, hooks)
        mixed = mixed_model(other)
        result, _ = sample(mixed, positive=positive, negative=negative)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(base.model.diffusion_model.seen, [(1.75, 2)] * 3 + [(2.75, 1)] * 6 + [(1.75, 2)] * 3)
        self.assertFalse(mixed.hook_backup)

    def test_interruption_restores_weights_and_next_run(self):
        base = tiny_model()
        mixed = mixed_model(base)
        def interrupt(step, denoised, x, total):
            if step == 4:
                raise RuntimeError("test interruption")
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            sample(mixed, callback=interrupt)
        self.assertFalse(mixed.hook_backup)
        torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))
        result, guider = sample(mixed)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(guider.cfg, 4.0)

    def test_native_mask_and_pure_base_equivalence(self):
        base = tiny_model()
        pure, _ = sample(base)
        mixed = mixed_model(base, scheme(prefix_ratio=1, suffix_ratio=1))
        same, _ = sample(mixed)
        torch.testing.assert_close(pure, same, rtol=0, atol=0)
        masked, _ = sample(mixed_model(base), mask=torch.zeros(1, 1, 1, 2, 2))
        torch.testing.assert_close(masked, base.model.process_latent_out(torch.zeros_like(masked)))

    def test_safe_loader_real_lora_mapping_and_dynamic_delegate(self):
        base = tiny_model()
        with patch.object(mixer.folder_paths, "get_full_path_or_raise", return_value="local.safetensors") as resolve:
            with patch.object(mixer.comfy.utils, "load_torch_file", return_value=LORA) as load:
                with patch.object(base, "is_dynamic", return_value=True):
                    with patch.object(base, "get_non_dynamic_delegate", return_value=tiny_model(), create=True) as delegate:
                        output = mixer.TurboBaseTrajectoryMixer.execute(base, "selected.safetensors", 1.0, 1.0, scheme())
                        mixed = output.result[0]
                        delegate.assert_called_once()
                resolve.assert_called_once_with("loras", "selected.safetensors")
                load.assert_called_once_with("local.safetensors", safe_load=True)
        sample(mixed)
        self.assertFalse(mixed.patches)
        self.assertEqual(len(mixed.hook_patches), 1)

    def test_incompatible_lora_fails_clearly(self):
        with patch.object(mixer.folder_paths, "get_full_path_or_raise", return_value="wrong.safetensors"):
            with patch.object(mixer.comfy.utils, "load_torch_file", return_value={}):
                with self.assertRaisesRegex(ValueError, "no compatible"):
                    mixer.TurboBaseTrajectoryMixer.execute(tiny_model(), "wrong.safetensors", 1.0, 1.0, scheme())

    def test_replacing_mixer_removes_old_adapter_without_changing_input(self):
        first = mixed_model(tiny_model())
        second = mixed_model(first, scheme(mixer.TURBO_BASE))
        self.assertEqual(len(first.hook_patches), 1)
        self.assertEqual(len(second.hook_patches), 1)
        sample(second)
        self.assertEqual(second.model.diffusion_model.seen, [(2.0, 1)] * 6 + [(1.0, 2)] * 6)

    def test_native_v3_registration_and_conditional_sliders(self):
        self.assertTrue(asyncio.run(host_nodes.load_custom_node(str(ROOT))))
        node = host_nodes.NODE_CLASS_MAPPINGS["TurboBaseTrajectoryMixer"]
        schema = node.define_schema()
        selector = schema.inputs[-2]
        self.assertEqual([len(option.inputs) for option in selector.options], [2, 1, 1, 2])
        for option in selector.options:
            self.assertTrue(all(widget.display_mode == mixer.io.NumberDisplay.slider for widget in option.inputs))
            live_inputs = {"scheme": option.key} | {"scheme." + widget.id: 0.5 for widget in option.inputs}
            finalized, _, v3_data = internal_io.get_finalized_class_inputs(node.INPUT_TYPES(), live_inputs)
            self.assertTrue(all("scheme." + widget.id in finalized["required"] for widget in option.inputs))
            nested = internal_io.build_nested_inputs(live_inputs, v3_data)
            self.assertEqual(nested["scheme"]["scheme"], option.key)
            self.assertTrue(all(nested["scheme"][widget.id] == 0.5 for widget in option.inputs))

    def test_smoothing_changes_real_lora_weights_and_preserves_cfg(self):
        base = tiny_model()
        mixed = mixed_model(base, lora_smooth_steps=2.0)
        result, _ = sample(mixed)
        expected = [(1.0, 2)] * 3 + [(1.5, 1)] + [(2.0, 1)] * 5 + [(1.5, 2)] + [(1.0, 2)] * 2
        self.assertEqual(base.model.diffusion_model.seen, expected)
        expected_raw = torch.full_like(result, 3.0 - sum(weight + (4 if branches == 2 else 1) for weight, branches in expected) / 12)
        torch.testing.assert_close(result, base.model.process_latent_out(expected_raw))
        again, _ = sample(mixed)
        torch.testing.assert_close(result, again, rtol=0, atol=0)
        self.assertFalse(mixed.hook_backup)
        self.assertFalse(mixed.cached_hook_patches)
        torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))

    def test_smooth_curve_continuity_short_segments_and_pure_modes(self):
        for before, after in ((False, True), (True, False)):
            cuts, roles = [6], [before, after]
            self.assertEqual(mixer.lora_weight(4, cuts, roles, 4, 12), float(before))
            self.assertEqual(mixer.lora_weight(8, cuts, roles, 4, 12), float(after))
            self.assertEqual(mixer.lora_weight(6, cuts, roles, 4, 12), 0.5)
            self.assertAlmostEqual(mixer.lora_weight(5, cuts, roles, 4, 12), 0.15625 if after else 0.84375)
        self.assertEqual(mixer.lora_weight(1.5, [1, 2], [False, True, False], 32, 4), 1.0)
        for role in (False, True):
            for progress in (0, 0.5, 4, 12):
                self.assertEqual(mixer.lora_weight(progress, [], [role], 32, 12), float(role))

    def test_smoothing_preserves_other_hooks_and_restores_after_interruption(self):
        base = tiny_model()
        other, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, LORA, 0.5, 0.0)
        positive = comfy.hooks.set_hooks_for_conditioning(POSITIVE, hooks)
        negative = comfy.hooks.set_hooks_for_conditioning(NEGATIVE, hooks)
        mixed = mixed_model(other, lora_smooth_steps=2.0)
        sample(mixed, positive=positive, negative=negative)
        self.assertEqual(base.model.diffusion_model.seen[3], (2.0, 1))
        self.assertEqual(base.model.diffusion_model.seen[4], (2.5, 1))
        def interrupt(step, denoised, x, total):
            if step == 3:
                raise RuntimeError("smooth interruption")
        with self.assertRaisesRegex(RuntimeError, "smooth interruption"):
            sample(mixed, callback=interrupt)
        self.assertFalse(mixed.hook_backup)
        torch.testing.assert_close(base.model.diffusion_model.proj.weight, torch.ones(2, 2))

    def test_smooth_strength_at_intermediate_and_revisited_sigmas(self):
        base = tiny_model()
        mixed = mixed_model(base, lora_smooth_steps=2.0)
        midpoint = (float(SIGMAS[2]) + float(SIGMAS[3])) / 2
        def probe(model, x, sigmas, extra_args=None, callback=None, disable=None):
            for sigma in (midpoint, float(SIGMAS[3]), float(SIGMAS[4]), midpoint):
                model(x, x.new_full((x.shape[0],), sigma), **extra_args)
            return x
        sample(mixed, sampler=comfy.samplers.KSAMPLER(probe))
        weights = [weight for weight, _ in base.model.diffusion_model.seen]
        for actual, expected in zip(weights, (1.15625, 1.5, 2.0, 1.15625)):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(len(weights), 4)
        self.assertFalse(mixed.hook_backup)


if __name__ == "__main__":
    unittest.main()
