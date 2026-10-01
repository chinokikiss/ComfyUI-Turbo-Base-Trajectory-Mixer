import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
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
        self.seen.append((float(self.proj.weight.mean()), x.shape[0]))
        guidance = context.mean(dim=(1, 2)).view(-1, *([1] * (x.ndim - 1)))
        return torch.ones_like(x) * (self.proj.weight.mean() + guidance)


def tiny_model(model_type=comfy.model_base.ModelType.FLOW):
    config = comfy.supported_models.Anima({"image_model": "anima"})
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.BaseModel(config, model_type, device=torch.device("cpu"), unet_model=TinyDiffusion)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def scheme(mode=mixer.BASE_BASE, **ratios):
    return {"scheme": mode, "prefix_ratio": 0.5, "suffix_ratio": 0.5, "composition_ratio": 0.5} | ratios


def mixed_model(base, selection=None):
    mixed, _, hooks = comfy.hooks.load_hook_lora_for_models(base, None, LORA, 1.0, 0.0)
    return mixer.attach_mixer(mixed, hooks, selection or scheme(), 1.0)


def sample(model, sigmas=SIGMAS, sampler_name="euler", positive=POSITIVE, negative=NEGATIVE, callback=None, mask=None):
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(4.0)
    noise = torch.full((1, 16, 1, 2, 2), 3.0)
    output = guider.sample(noise, torch.zeros_like(noise), comfy.samplers.ksampler(sampler_name), sigmas,
                           denoise_mask=mask, callback=callback, disable_pbar=True, seed=0)
    return output, guider


class MixerTests(unittest.TestCase):
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
        selector = schema.inputs[-1]
        self.assertEqual([len(option.inputs) for option in selector.options], [2, 1, 1, 2])
        for option in selector.options:
            self.assertTrue(all(widget.display_mode == mixer.io.NumberDisplay.slider for widget in option.inputs))
            live_inputs = {"scheme": option.key} | {"scheme." + widget.id: 0.5 for widget in option.inputs}
            finalized, _, v3_data = internal_io.get_finalized_class_inputs(node.INPUT_TYPES(), live_inputs)
            self.assertTrue(all("scheme." + widget.id in finalized["required"] for widget in option.inputs))
            nested = internal_io.build_nested_inputs(live_inputs, v3_data)
            self.assertEqual(nested["scheme"]["scheme"], option.key)
            self.assertTrue(all(nested["scheme"][widget.id] == 0.5 for widget in option.inputs))


if __name__ == "__main__":
    unittest.main()
