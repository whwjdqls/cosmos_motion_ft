"""Exercise native packing and information boundaries, including temporal alignment."""
import unittest
from types import SimpleNamespace
import torch
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from native_motion_phase1.data import TASKS, make_sample


class MotionTaskTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(9)
        self.motion = torch.randn(97, 283, generator=g)
        self.neutral = torch.randn(30, 3, generator=g)
        self.latent = torch.randn(48, 25, 16, 16, generator=g)

    def test_native_packed_masks(self):
        for task in TASKS:
            with self.subTest(task=task):
                s = make_sample(self.motion, self.neutral, self.latent, task, "Walking.")
                clean = GenerationDataClean(batch_size=1, is_image_batch=False,
                    x0_tokens_vision=[s["video_latents"][None]], fps_vision=torch.tensor([20.]),
                    x0_tokens_action=[s["action"]], fps_action=torch.tensor([20.]),
                    action_domain_id=[s["domain_id"]], raw_action_dim=[s["raw_action_dim"]])
                packed = pack_input_sequence(sequence_plans=[s["sequence_plan"]], input_text_indexes=[[10, 11]],
                    gen_data_clean=clean, input_timesteps=torch.tensor([.5]),
                    special_tokens={"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
                    latent_patch_size=2, action_dim=288, enable_fps_modulation=True, base_fps=24.,
                    temporal_compression_factor=4, unified_3d_mrope_temporal_modality_margin=15000)
                vm, am = packed.vision.condition_mask[0], packed.action.condition_mask[0]
                self.assertEqual(int((vm == 0).sum()), 24 if task == "motion2video" else 0)
                self.assertEqual(int((am == 0).sum()), 0 if task == "motion2video" else 97)
                self.assertTrue(bool(am[0].all()))
                torch.testing.assert_close(s["action"][1:, :283], self.motion, rtol=0, atol=0)
                self.assertEqual(int(s["action"][:, 283:].count_nonzero()), 0)

    def test_text_only_is_independent_of_video(self):
        a = make_sample(self.motion, self.neutral, None, "text2motion", "Walking.")
        b = make_sample(self.motion, self.neutral, self.latent * 100, "text2motion", "Walking.")
        torch.testing.assert_close(a["video_latents"], b["video_latents"], rtol=0, atol=0)
        self.assertEqual(int(a["video_latents"].count_nonzero()), 0)

    def test_ti2m_cannot_see_future_video(self):
        changed = self.latent.clone()
        changed[:, 1:] += 100
        a = make_sample(self.motion, self.neutral, self.latent, "textimg2motion", "Walking.")
        b = make_sample(self.motion, self.neutral, changed, "textimg2motion", "Walking.")
        torch.testing.assert_close(a["video_latents"], b["video_latents"], rtol=0, atol=0)
        self.assertEqual(a["video_latents"].shape[1], 1)

    def test_nontext_tasks_drop_caption(self):
        for task in TASKS[2:]:
            s = make_sample(self.motion, self.neutral, self.latent, task, "Hidden target description.")
            self.assertEqual(s["ai_caption"], "")

    def test_native_loss_excludes_shape_prefix_and_padding(self):
        from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
        pred = torch.ones(98, 288, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros(98, 1)
        mask[0] = 1
        rf = SimpleNamespace(train_time_weight=lambda timestep, kwargs: torch.ones_like(timestep))
        loss, _ = compute_flow_matching_loss(
            [pred], [target], [mask], torch.tensor([[.5]]), True, rf,
            {"device": "cpu", "dtype": torch.float32}, raw_action_dim=[torch.tensor(283)])
        loss.backward()
        self.assertEqual(int(pred.grad[0].count_nonzero()), 0)
        self.assertEqual(int(pred.grad[:, 283:].count_nonzero()), 0)
        self.assertTrue(bool((pred.grad[1:, :283] > 0).all()))
        self.assertAlmostEqual(float(loss.detach()), 97 / 98, places=6)


if __name__ == "__main__":
    unittest.main()
