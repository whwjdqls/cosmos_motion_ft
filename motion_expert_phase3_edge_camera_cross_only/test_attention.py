"""Independent cross-only equations, gradients, empty rows and policy guards."""
import copy
import unittest
from dataclasses import replace
import torch
import modality_bridge as bridge
from scripts.phase3_performance import execution_policy
from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_camera_directional_nojoint.policy import configure as baseline
from .attention import cross_attention_mask, cross_only_attention, grouped_cross_attention
from .policy import configure, configure_checkpoint


def metadata(mode):
    return bridge.BridgeMeta(mode, torch.tensor([0, 1, 1, 2]),
        torch.ones(4, dtype=torch.bool) if mode == 'video2motion' else torch.tensor([1, 0, 0, 0], dtype=torch.bool),
        torch.tensor([-1, 0, 1, 4, 5, 8]),
        torch.ones(6, dtype=torch.bool) if mode == 'motimg2video' else torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool),
        torch.tensor([0, 1, 1, 5]), torch.tensor([0, 4, 4, 8]))


class CrossOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.recipe = copy.deepcopy(RECIPE)

    def tearDown(self):
        RECIPE.clear(); RECIPE.update(self.recipe)

    def test_only_cross_edges_with_wan_alignment_and_clean_queries(self):
        for mode in ['video2motion', 'motimg2video', 'joint']:
            meta = metadata(mode)
            for direction in ['directional', 'bidirectional']:
                mask = cross_attention_mask(meta, direction)
                self.assertFalse(mask[:4, :4].any()); self.assertFalse(mask[4:, 4:].any())
                self.assertFalse(mask[4].any()); self.assertFalse(mask[:, 4].any())
                self.assertFalse(mask[1, 8]); self.assertFalse(mask[8, 1])
                if direction == 'directional':
                    self.assertFalse(mask[:4][meta.gen_clean].any())
                    self.assertFalse(mask[4:][meta.motion_clean].any())

    def test_grouped_outputs_and_gradients_match_independent_rectangular_attention(self):
        for mode in ['video2motion', 'motimg2video', 'joint']:
            meta = metadata(mode)
            q, k, v = [torch.randn(2, 10, 4, requires_grad=True) for _ in range(3)]
            # Build each target row independently from physical frame alignment.
            results = []
            frames = list(meta.gen_frame) + list(meta.motion_frame)
            clean = list(meta.gen_clean) + list(meta.motion_clean)
            for row, frame in enumerate(frames):
                group = int(frame) if row < 4 else (int(frame) + 3) // 4
                keys = [j for j, f in enumerate(frames) if int(frame) >= 0 and int(f) >= 0
                        and (j < 4) != (row < 4) and not clean[row]
                        and (int(f) if j < 4 else (int(f) + 3) // 4) == group]
                if keys:
                    weights = (q[:, row:row+1] @ k[:, keys].transpose(-2, -1) * .5).softmax(-1)
                    results.append((weights @ v[:, keys]).squeeze(1))
                else:
                    results.append(q[:, row] * 0 + k.sum((1,2))[:, None] * 0 + v.sum((1,2))[:, None] * 0)
            expected = torch.stack(results, dim=1)
            actual = grouped_cross_attention(q, k, v, meta, 'directional')
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            ga = torch.autograd.grad(actual.square().sum(), [q, k, v], retain_graph=True)
            gr = torch.autograd.grad(expected.square().sum(), [q, k, v])
            for a, b in zip(ga, gr):torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_identity_clean_stream_shape_and_dense_grouped_backward_parity(self):
        for mode in ['video2motion', 'motimg2video', 'joint']:
            a = bridge.LocalModalityBridge(8, 2, 4, 'directional', 'mmdit')
            g, m, meta = torch.randn(4, 8), torch.randn(6, 8), metadata(mode)
            before = {k: v.shape for k, v in a.state_dict().items()}
            with cross_only_attention():
                for x, y in zip(a(g, m, meta), (g, m)):torch.testing.assert_close(x, y, rtol=0, atol=0)
                with torch.no_grad():a.gate_g.fill_(.2); a.gate_m.fill_(.3)
                b = copy.deepcopy(a)
                expected = a(g, m, meta)
                with execution_policy(grouped_attention=True):actual = b(g, m, meta)
                for x, y in zip(actual, expected):torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(actual[0][meta.gen_clean], g[meta.gen_clean], rtol=0, atol=0)
                torch.testing.assert_close(actual[1][meta.motion_clean], m[meta.motion_clean], rtol=0, atol=0)
                sum(x.square().sum() for x in expected).backward()
                sum(x.square().sum() for x in actual).backward()
                for x, y in zip(a.parameters(), b.parameters()):
                    self.assertTrue(torch.isfinite(y.grad).all())
                    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
            self.assertEqual(before, {k: v.shape for k, v in a.state_dict().items()})

    def test_bf16_empty_rows_and_checkpoint_recompute_are_finite(self):
        from torch.utils.checkpoint import checkpoint
        a = bridge.LocalModalityBridge(8, 2, 4, 'directional', 'mmdit')
        with torch.no_grad():a.gate_g.fill_(.2);a.gate_m.fill_(.3)
        for mode in ['video2motion', 'motimg2video']:
            g, m, meta = torch.randn(4, 8), torch.randn(6, 8), metadata(mode)
            with cross_only_attention(), execution_policy(grouped_attention=True, precision='bf16_bridge'):
                out = checkpoint(lambda x, y: a(x, y, meta), g, m, use_reentrant=False)
                sum(x.square().sum() for x in out).backward()
                for value in out:self.assertTrue(torch.isfinite(value).all())
                for p in a.parameters():self.assertTrue(torch.isfinite(p.grad).all())
                torch.testing.assert_close(out[0][meta.gen_clean], g[meta.gen_clean], rtol=0, atol=0)
            a.zero_grad()

    def test_exact_matched_recipe_and_checkpoint_rejection(self):
        base = copy.deepcopy(baseline('temporal_local'))
        cross = copy.deepcopy(configure())
        self.assertEqual({k for k in base.keys() | cross.keys() if base.get(k) != cross.get(k)},
                         {'experiment', 'bridge_attention_type'})
        configure_checkpoint({'scheduler': {'recipe': cross}})
        with self.assertRaises(ValueError):configure_checkpoint({'scheduler': {'recipe': base}})
        with self.assertRaises(ValueError):configure('global')

    def test_nested_context_restores_original_masks_and_kernels(self):
        originals = (bridge.temporally_local_attention_mask, bridge.grouped_joint_attention, bridge.LocalModalityBridge._joint_attention)
        with cross_only_attention():
            with cross_only_attention():pass
            self.assertIs(bridge.temporally_local_attention_mask, cross_attention_mask)
        self.assertEqual(originals, (bridge.temporally_local_attention_mask, bridge.grouped_joint_attention, bridge.LocalModalityBridge._joint_attention))


if __name__ == '__main__':unittest.main()
