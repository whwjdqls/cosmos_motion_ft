"""Focused CPU tests for the BONES native-schedule POC."""
from __future__ import annotations

import unittest

import numpy as np
import torch

import bs_native_flow


class NativeScheduleTest(unittest.TestCase):
    def test_shift_matches_cosmos_formula(self):
        raw = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        shifted = bs_native_flow.shift_sigma(raw, 3.0)
        expected = 3.0 * raw / (1.0 + 2.0 * raw)
        torch.testing.assert_close(shifted, expected)
        torch.testing.assert_close(bs_native_flow.shift_sigma(raw, 1.0), raw)

    def test_training_sampler_is_shifted_logitnormal(self):
        seed = 123
        expected_generator = torch.Generator().manual_seed(seed)
        raw = torch.sigmoid(torch.randn((16,), generator=expected_generator))
        expected = bs_native_flow.shift_sigma(raw, 3.0)

        actual_generator = torch.Generator().manual_seed(seed)
        actual = bs_native_flow.sample_train_sigma(
            16, "cpu", shift=3.0, generator=actual_generator
        )
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool(((actual > 0.0) & (actual < 1.0)).all()))

    def test_diagnostic_inference_ladder_is_single_shift(self):
        sigmas, timesteps = bs_native_flow.inference_schedule(
            4, shift=3.0, num_train_timesteps=1000
        )
        base = np.linspace(0.999, 0.0, 5).copy()[:-1]
        expected_sigmas = 3.0 * base / (1.0 + 2.0 * base)
        expected_timesteps = (expected_sigmas * 1000).astype(np.int64)

        np.testing.assert_allclose(sigmas[:-1].numpy(), expected_sigmas, rtol=1e-7)
        np.testing.assert_array_equal(timesteps.numpy(), expected_timesteps)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertTrue(bool((sigmas[:-1] > sigmas[1:]).all()))

    def test_unipc_factory_is_official_cosmos_scheduler(self):
        scheduler = bs_native_flow.create_unipc_scheduler(
            4,
            shift=3.0,
            num_train_timesteps=1000,
            device="cpu",
        )
        self.assertEqual(
            scheduler.__class__.__module__,
            "cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc",
        )
        self.assertEqual(scheduler.__class__.__name__, "FlowUniPCMultistepScheduler")
        self.assertEqual(scheduler.config.solver_order, 2)
        self.assertEqual(scheduler.config.solver_type, "bh2")
        self.assertEqual(scheduler.config.prediction_type, "flow_prediction")
        self.assertTrue(scheduler.config.predict_x0)
        self.assertTrue(scheduler.config.lower_order_final)

        train_sigma_max = 3.0 * 0.999 / (1.0 + 2.0 * 0.999)
        base = np.linspace(train_sigma_max, 0.0, 5).copy()[:-1]
        shifted = 3.0 * base / (1.0 + 2.0 * base)
        expected_timesteps = (shifted * 1000).astype(np.int64)
        expected_sigmas = np.concatenate([shifted, [0.0]]).astype(np.float32)
        np.testing.assert_allclose(scheduler.sigmas.numpy(), expected_sigmas, rtol=1e-7)
        np.testing.assert_array_equal(scheduler.timesteps.cpu().numpy(), expected_timesteps)

        diagnostic_sigmas, _ = bs_native_flow.inference_schedule(4, shift=3.0)
        self.assertFalse(torch.equal(scheduler.sigmas, diagnostic_sigmas))

    def test_perfect_x0_model_reaches_clean_endpoint(self):
        target = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]], dtype=torch.float32)

        class PerfectModel:
            def __init__(self):
                self.model_sigmas = []

            def __call__(self, x, sigma, text, text_pad, neutral):
                del x, text, text_pad, neutral
                self.model_sigmas.append(sigma.detach().clone())
                return target

        model = PerfectModel()
        text = torch.zeros(1, 1, 4)
        neutral = torch.zeros(1, 30, 3)
        result = bs_native_flow.sample_x0(
            model,
            text,
            None,
            neutral,
            T=2,
            motion_dim=2,
            steps=5,
            guidance=1.0,
            device="cpu",
            generator=torch.Generator().manual_seed(0),
            native_shift=3.0,
        )
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)
        self.assertEqual(len(model.model_sigmas), 5)
        for sigma in model.model_sigmas:
            scaled = sigma * bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS
            torch.testing.assert_close(scaled, scaled.round())

    def test_heun_perfect_x0_model_reaches_clean_endpoint(self):
        target = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]], dtype=torch.float32)

        class PerfectModel:
            def __init__(self):
                self.model_sigmas = []

            def __call__(self, x, sigma, text, text_pad, neutral):
                del x, text, text_pad, neutral
                self.model_sigmas.append(sigma.detach().clone())
                return target

        model = PerfectModel()
        result = bs_native_flow.sample_x0_heun(
            model,
            torch.zeros(1, 1, 4),
            None,
            torch.zeros(1, 30, 3),
            T=2,
            motion_dim=2,
            steps=5,
            guidance=1.0,
            device="cpu",
            generator=torch.Generator().manual_seed(0),
            native_shift=3.0,
        )
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)
        self.assertEqual(len(model.model_sigmas), 2 * 5 - 1)
        for sigma in model.model_sigmas:
            scaled = sigma * bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS
            torch.testing.assert_close(scaled, scaled.round())

    def test_official_unipc_perfect_x0_model_reaches_clean_endpoint(self):
        target = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]], dtype=torch.float32)

        class PerfectModel:
            def __init__(self):
                self.model_sigmas = []

            def __call__(self, x, sigma, text, text_pad, neutral):
                del x, text, text_pad, neutral
                self.model_sigmas.append(sigma.detach().clone())
                return target

        model = PerfectModel()
        result = bs_native_flow.sample_x0_unipc(
            model,
            torch.zeros(1, 1, 4),
            None,
            torch.zeros(1, 30, 3),
            T=2,
            motion_dim=2,
            steps=5,
            guidance=1.0,
            device="cpu",
            generator=torch.Generator().manual_seed(0),
            native_shift=3.0,
        )
        torch.testing.assert_close(result, target, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(model.model_sigmas), 5)
        for sigma in model.model_sigmas:
            scaled = sigma * bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS
            torch.testing.assert_close(scaled, scaled.round())


if __name__ == "__main__":
    unittest.main()
