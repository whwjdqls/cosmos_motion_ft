"""Exercise Edge's actual local head constructor, not the Nano namesake."""
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn

import config
import motion_heads
from motion_heads import MotionHeads
from model import EdgePhase2MotionExpert


class EdgeMotionHeadsTest(unittest.TestCase):
    def test_283_default_and_explicit_constructor_match_original_bitwise(self):
        self.assertEqual(Path(motion_heads.__file__).resolve().parent,
                         Path(__file__).resolve().parents[1])
        source = subprocess.check_output(
            ["git", "-C", str(config.REPO_ROOT), "show", "f3a7a89:motion_expert_t2m_edge/motion_heads.py"],
            text=True,
        )
        legacy = ModuleType("_original_edge_motion_heads")
        exec(compile(source, motion_heads.__file__, "exec"), legacy.__dict__)
        with mock.patch.object(config, "MOTION_DIM", 283):
            torch.manual_seed(19)
            original = legacy.MotionHeads(hidden_size=32)
            original_rng = torch.random.get_rng_state()
            torch.manual_seed(19)
            default = MotionHeads(hidden_size=32)
            default_rng = torch.random.get_rng_state()
            torch.manual_seed(19)
            explicit = MotionHeads(hidden_size=32, motion_dim=283)
        self.assertEqual(set(original.state_dict()), set(default.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, default.state_dict()[key]), key)
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        self.assertTrue(torch.equal(original_rng, default_rng))
        self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
        # Compare non-persistent timestep frequencies and the real forward path too.
        self.assertTrue(torch.equal(original.time_embedder._timestep_frequencies,
                                    explicit.time_embedder._timestep_frequencies))
        x = torch.randn(2, 7, 283)
        sigma = torch.tensor([.2, .8])
        mask = torch.ones(2, 7, dtype=torch.bool)
        shape = torch.randn(2, 30, 3)
        old_hidden = original.encode_motion(x, sigma, mask)
        new_hidden = explicit.encode_motion(x, sigma, mask)
        self.assertTrue(torch.equal(old_hidden, new_hidden))
        self.assertTrue(torch.equal(original.encode_shape(shape), explicit.encode_shape(shape)))
        self.assertTrue(torch.equal(original.decode(old_hidden), explicit.decode(new_hidden)))
        explicit.load_state_dict(original.state_dict(), strict=True)

    def test_369_head_forward_backward_and_strict_reload(self):
        # Explicit width must take precedence over the currently selected default.
        with mock.patch.object(config, "MOTION_DIM", 283):
            heads = MotionHeads(hidden_size=32, motion_dim=369)
        self.assertEqual(tuple(heads.motion2llm.weight.shape), (32, 369))
        self.assertEqual(tuple(heads.llm2motion.weight.shape), (369, 32))
        x = torch.randn(2, 9, 369)
        sigma = torch.tensor([.3, .7])
        mask = torch.ones(2, 9, dtype=torch.bool)
        hidden = heads.encode_motion(x, sigma, mask)
        prediction = heads.decode(hidden)
        self.assertEqual(tuple(prediction.shape), (2, 9, 369))
        target = torch.randn_like(prediction)
        loss = (prediction - target).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(heads.llm2motion.weight.grad).all())
        self.assertGreater(float(heads.llm2motion.weight.grad.abs().sum()), 0)
        copy = MotionHeads(hidden_size=32, motion_dim=369)
        copy.load_state_dict(heads.state_dict(), strict=True)
        self.assertTrue(torch.equal(copy.decode(hidden.detach()), prediction.detach()))
        with mock.patch.object(config, "MOTION_DIM", 369):
            self.assertEqual(MotionHeads(hidden_size=32).motion_dim, 369)

    def test_model_construction_passes_width_to_actual_edge_heads(self):
        # Inject only a tiny frozen backbone, keeping the production constructor
        # and the actual framework timestep/head modules. No base weights/GPU needed.
        for dimension in (283, 369):
            backbone = nn.Module()
            backbone.layers = nn.ModuleList()
            backbone.norm = nn.LayerNorm(32)
            edge = SimpleNamespace(backbone=backbone,
                geometry=SimpleNamespace(hidden_size=32, num_attention_heads=4,
                                         num_key_value_heads=2, head_dim=8))
            with mock.patch.object(config, "MOTION_DIM", dimension):
                model = EdgePhase2MotionExpert(device="cpu", dtype=torch.float32,
                                                verbose=False, frozen_edge=edge)
            self.assertIsInstance(model.heads, MotionHeads)
            self.assertEqual(model.heads.motion_dim, dimension)
            self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
            self.assertFalse(any(name.startswith("backbone.") for name in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
