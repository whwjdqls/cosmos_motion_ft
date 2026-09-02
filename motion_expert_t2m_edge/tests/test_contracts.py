from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import config
from checkpoint import capture_rng_state, latest_checkpoint, restore_rng_state
from data import normalize_task_weights
from dataset import humanize_caption
from flow import add_noise_x0_masked, native_inference_schedule
from layer import EdgeMotionMLP
from losses import masked_mse
from train_visualization import _sample_id, _sample_seed


class ArchitectureContractTest(unittest.TestCase):
    def test_seven_small_nano_style_motion_layers(self):
        self.assertEqual(config.HIDDEN_SIZE, 2048)
        self.assertEqual(config.MOTION_LAYER_INDICES, (3, 7, 11, 15, 19, 23, 27))
        self.assertEqual(config.MOTION_INTERMEDIATE_SIZE, 3072)
        self.assertEqual(config.MOTION_MLP_TYPE, "motion_swiglu")
        self.assertLess(config.MOTION_INTERMEDIATE_SIZE, 9216)

    def test_motion_ffn_is_three_linear_swiglu(self):
        module = EdgeMotionMLP(8, 12, bias=False)
        self.assertEqual(tuple(module.gate_proj.weight.shape), (12, 8))
        self.assertEqual(tuple(module.up_proj.weight.shape), (12, 8))
        self.assertEqual(tuple(module.down_proj.weight.shape), (8, 12))
        self.assertEqual(tuple(module(torch.randn(3, 8)).shape), (3, 8))

    def test_phase2_default_is_nymeria_t2m_and_ti2m(self):
        self.assertEqual(
            normalize_task_weights(),
            {"text2motion": 0.75, "textimg2motion": 0.25},
        )
        self.assertEqual(config.BONES_TEXT2M_FRAC, 0.0)
        self.assertEqual(config.DEFAULT_BATCH_SIZE, 128)
        self.assertEqual(config.DEFAULT_GRAD_ACCUM, 1)
        self.assertEqual(config.TI2M_FRAMES, 97)
        self.assertFalse(config.BONES_CAMERA_HEAD_EQUIVALENT)
        self.assertEqual(config.CONTRACT_SCHEMA_VERSION, 3)
        self.assertEqual(
            config.CAPTION_SUBJECT_POLICY,
            "standalone_C_to_sentence_aware_camera_wearer",
        )

    def test_generator_tokens_remain_out_of_scope(self):
        self.assertEqual(config.REASONER_IMAGE_SIZE, 256)
        self.assertEqual(config.MOTION_LAYER_INDICES, (3, 7, 11, 15, 19, 23, 27))

    def test_nymeria_subject_is_camera_wearer_and_only_standalone_uppercase_c(self):
        caption = "C carries a cup. ABC, c, coffee, C2, _C, and (C) stay distinct. C turns."
        self.assertEqual(
            humanize_caption(caption),
            "The camera wearer carries a cup. ABC, c, coffee, C2, _C, and "
            "(the camera wearer) stay distinct. The camera wearer turns.",
        )


class FlowAndLossContractTest(unittest.TestCase):
    def test_padding_is_never_noised(self):
        torch.manual_seed(0)
        x0 = torch.randn(2, 5, 3)
        pad = torch.tensor([[False, False, True, True, True], [False] * 5])
        x_sigma, _, target, noised = add_noise_x0_masked(
            x0, pad, torch.tensor([0.5, 0.8])
        )
        torch.testing.assert_close(x_sigma[0, 2:], x0[0, 2:])
        torch.testing.assert_close(target, x0)
        self.assertTrue(torch.equal(noised, ~pad))

    def test_shifted_schedule_ends_at_zero(self):
        sigmas, timesteps = native_inference_schedule(4)
        self.assertEqual(tuple(sigmas.shape), (5,))
        self.assertEqual(tuple(timesteps.shape), (4,))
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertTrue(bool(torch.all(sigmas[:-1] > sigmas[1:])))

    def test_masked_mse_ignores_padding(self):
        left = torch.tensor([[[1.0], [100.0]]])
        right = torch.zeros_like(left)
        valid = torch.tensor([[True, False]])
        self.assertEqual(float(masked_mse(left, right, valid)), 1.0)


class VisualizationContractTest(unittest.TestCase):
    def test_sample_identity_and_noise_seed_are_stable_and_task_specific(self):
        self.assertEqual(
            _sample_id("text2motion", 17, "walk forward"),
            _sample_id("text2motion", 17, "walk forward"),
        )
        self.assertNotEqual(
            _sample_id("text2motion", 17, "walk forward"),
            _sample_id("textimg2motion", 17, "walk forward"),
        )
        self.assertEqual(
            _sample_seed(20260901, "text2motion", 3),
            _sample_seed(20260901, "text2motion", 3),
        )
        self.assertNotEqual(
            _sample_seed(20260901, "text2motion", 3),
            _sample_seed(20260901, "textimg2motion", 3),
        )


class RecoveryCheckpointContractTest(unittest.TestCase):
    def test_latest_checkpoint_selects_newest_complete_regular_or_recovery_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(latest_checkpoint(root))
            regular = root / "step_000000250.pt"
            recovery = root / "recovery_latest.pt"
            ignored_temporary = root / "recovery_latest.pt.tmp.123"
            regular.touch()
            recovery.touch()
            ignored_temporary.touch()
            os.utime(regular, ns=(1_000, 1_000))
            os.utime(recovery, ns=(2_000, 2_000))
            os.utime(ignored_temporary, ns=(3_000, 3_000))
            self.assertEqual(latest_checkpoint(root), recovery)
            os.utime(regular, ns=(4_000, 4_000))
            self.assertEqual(latest_checkpoint(root), regular)

    def test_cpu_rng_bundle_restores_python_numpy_and_torch(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        state = capture_rng_state(include_cuda=False)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertTrue(restore_rng_state(state))
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
