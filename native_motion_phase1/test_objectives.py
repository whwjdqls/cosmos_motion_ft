"""Numerical parity with specialist loss, native video loss, and UniPC states."""
from types import SimpleNamespace as NS
import unittest
import torch

from native_motion_phase1.objectives import reconstruction_loss, t2m_reconstruction_loss, MOTION_WEIGHTS, x0_segment_to_velocity
from native_motion_phase1.model import MotionX0Model, sampling_sigmas
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSamplerConfig
from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss


def fixture():
    torch.manual_seed(8)
    mean, std = torch.zeros(283), torch.ones(283)
    mean[-4:], std[-4:] = .5, .5
    raw = torch.zeros(2, 97, 283)
    pose = raw[..., :270].view(2, 97, 30, 9)
    pose[..., 0], pose[..., 4] = 1., 1.
    pose[..., 6:9] = torch.randn(2, 1, 30, 3) * .1
    raw[..., 270], raw[..., 274], raw[..., -4:] = 1., 1., 1.
    targets = (raw - mean) / std
    packed = torch.zeros(2, 98, 288)
    packed[:, 1:, :283] = targets
    pred = (packed + torch.randn_like(packed) * .01).requires_grad_()
    masks = torch.zeros(2, 98, 1); masks[:, 0] = 1
    return pred, packed, masks, mean, std


class ObjectiveTests(unittest.TestCase):
    def test_exact_specialist_loss_and_gradient_parity(self):
        pred, target, masks, mean, std = fixture()
        actual, terms = reconstruction_loss(list(pred), list(target), list(masks), mean, std)
        expected, expected_terms = t2m_reconstruction_loss(pred[:, 1:, :283], target[:, 1:, :283],
            torch.ones(2, 97, dtype=torch.bool), mean, std, **MOTION_WEIGHTS,
            joint_loss_mode="centered", fps=20., modes=["text2motion"] * 2,
            reduction="per_task", decoded_nonfinite="skip")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in expected_terms:
            torch.testing.assert_close(terms[key], expected_terms[key], rtol=0, atol=0)
        grad = torch.autograd.grad(actual, pred, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, pred)[0]
        torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertEqual(int(grad[:, 0].count_nonzero()), 0)
        self.assertEqual(int(grad[:, :, 283:].count_nonzero()), 0)

    def test_motion_to_video_has_no_motion_loss(self):
        pred, target, masks, mean, std = fixture()
        masks[:] = 1
        loss, _ = reconstruction_loss(list(pred), list(target), list(masks), mean, std)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(int(pred.grad.count_nonzero()), 0)

    def test_native_video_loss_and_gradient_unchanged(self):
        pred, target, masks, mean, std = fixture()
        masks[:] = 1
        model = MotionX0Model.__new__(MotionX0Model)
        torch.nn.Module.__init__(model)
        model.motion_mean, model.motion_std, model._motion_targets = mean, std, tuple(target)
        model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
        model.config = NS(vision_gen=True, action_gen=True, sound_gen=False, lidar_state_ch=None,
            rectified_flow_training_config=NS(normalize_loss_by_active=False, image_loss_scale=None,
                loss_scale=10., action_loss_weight=0., sample_level_loss_averaging=False))
        model.rectified_flow_video = NS(train_time_weight=lambda t, kwargs: torch.ones_like(t))
        model.rectified_flow_action = model.rectified_flow_video
        model._get_load_balancing_loss_meshes = lambda: (None, None)
        vp = [torch.randn(3, 4, 2, 2, requires_grad=True) for _ in range(2)]
        vt = [torch.randn_like(p) for p in vp]
        vm = [torch.zeros(4, 1, 1) for _ in vp]
        for m in vm: m[0] = 1
        packed = NS(vision=NS(tokens=vp, condition_mask=vm, mse_loss_indexes=torch.arange(3)),
                    action=NS(tokens=list(pred), condition_mask=list(masks), mse_loss_indexes=torch.empty(0, dtype=torch.long),
                              raw_action_dim=[torch.tensor(283)]*2, action_valid_mask=None))
        noised = GenerationDataNoised(batch_size=2, epsilon_vision=vt, xt_tokens_vision=vp,
                                     vt_target_vision=vt, vt_target_action=list(target))
        ts = torch.tensor([[100.], [750.]])
        actual, _ = model._compute_losses(dict(preds_action=list(pred), preds_vision=vp), packed, noised, ts, False)
        expected = 10. * compute_flow_matching_loss(vp, vt, vm, ts, True,
            model.rectified_flow_video, model.tensor_kwargs_fp32)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grads = torch.autograd.grad(actual, vp, retain_graph=True)
        refs = torch.autograd.grad(expected, vp)
        for g, r in zip(grads, refs): torch.testing.assert_close(g, r, rtol=0, atol=0)

    def test_x0_conversion_uses_actual_sigma_and_preserves_video(self):
        cfg = UniPCSamplerConfig()
        mapping = sampling_sigmas(cfg, 30, 10.)
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1., use_dynamic_shifting=False)
        scheduler.set_timesteps(30, device="cpu", shift=10.)
        timestep, sigma = int(scheduler.timesteps[12]), float(scheduler.sigmas[12])
        self.assertEqual(mapping[timestep], sigma)
        self.assertNotEqual(mapping[timestep], timestep / 1000.)
        x0 = torch.randn(98, 288)
        epsilon = torch.randn_like(x0)
        state = (1-sigma)*x0 + sigma*epsilon
        video = torch.randn(19)
        out = x0_segment_to_velocity(torch.cat((video, x0.flatten())),
            torch.cat((torch.zeros(19), state.flatten())), 19, (98, 288), [0], sigma)
        torch.testing.assert_close(out[:19], video, rtol=0, atol=0)
        velocity = out[19:].view(98, 288)
        torch.testing.assert_close(velocity[1:, :283], (epsilon-x0)[1:, :283])
        self.assertEqual(int(velocity[0].count_nonzero()), 0)
        self.assertEqual(int(velocity[:, 283:].count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main()
