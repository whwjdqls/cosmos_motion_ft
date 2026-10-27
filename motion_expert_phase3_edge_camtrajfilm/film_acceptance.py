"""Approved compute-node tests of the direct trajectory and timestep paths."""
import numpy as np
import torch
from modality_bridge import LocalModalityBridge
from .backbone import bridge_meta
from .camera_features import trajectory18
from .film import CameraFiLMBridge


def check_film(device):
    checks = []
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        torch.manual_seed(19)
        # Upright Z-up input: retain height after conversion and subtract the
        # same TOTAL offset used by calibrated motion, not initial height.
        positions = np.zeros((97, 3)); positions[:, 2] = 1.7
        rotations = np.tile(np.eye(3), (97, 1, 1))
        first = trajectory18(positions, rotations, 0.2)
        higher = positions.copy(); higher[:, 2] += 0.4
        second = trajectory18(higher, rotations, 0.2)
        np.testing.assert_allclose(first[:, 7], 1.5, atol=1e-6)
        np.testing.assert_allclose(second[:, 7] - first[:, 7], 0.4, atol=1e-6)
        np.testing.assert_allclose(second[:, :7], first[:, :7], atol=1e-6)
        np.testing.assert_allclose(second[:, 8:], first[:, 8:], atol=1e-6)
        shifted = positions.copy(); shifted[:, :2] += [5, -8]
        np.testing.assert_allclose(trajectory18(shifted, rotations, 0.2), first, atol=1e-6)
        checks.append("trajectory18_height_retained_total_floor_offset_and_horizontal_origin_invariance")
        bridge = CameraFiLMBridge(LocalModalityBridge(8, 2, 4).to(device), 4, 8)
        camera = torch.randn(97, 18, device=device)
        noise = torch.randn(8, device=device)
        with torch.no_grad():
            bridge.attention.gate_g.fill_(0.2)
            bridge.attention.gate_m.fill_(0.2)
            for mode in ("motimg2video", "video2motion", "camera2motion", "cameravideo2motion"):
                meta = bridge_meta(mode, device)
                g = torch.randn(meta.gen_frame.numel(), 8, device=device)
                m = torch.randn(98, 8, device=device)
                expected_g, expected_m = bridge.attention(g, m, meta)
                actual_g, actual_m = bridge(g, m, meta, camera, noise)
                torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
                torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
            checks.append("trajectory_FiLM_zero_initialization_bit_exact_with_nonzero_attention_gates")
            bridge.film.mlp[-1].weight.normal_(std=0.05)
            bridge.film.mlp[-1].bias.normal_(std=0.05)
            for mode in ("motimg2video", "video2motion"):
                meta = bridge_meta(mode, device)
                g = torch.randn(meta.gen_frame.numel(), 8, device=device)
                m = torch.randn(98, 8, device=device)
                expected_g, expected_m = bridge.attention(g, m, meta)
                actual_g, actual_m = bridge(g, m, meta, camera * 2, noise * 3)
                torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
                torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
            meta = bridge_meta("cameravideo2motion", device)
            g = torch.randn(meta.gen_frame.numel(), 8, device=device)
            m = torch.randn(98, 8, device=device)
            expected_g, expected_m = bridge.attention(g, m, meta)
            actual_g, actual_m = bridge(g, m, meta, camera, noise)
            torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
            torch.testing.assert_close(actual_m[:1], expected_m[:1], rtol=0, atol=0)
            assert not torch.equal(actual_m[1:], expected_m[1:])
            motion = torch.randn(97, 8, device=device)
            delta = bridge.film.delta(motion, camera, noise)
            changed = camera.clone(); changed[9, 7] += 3
            affected = (bridge.film.delta(motion, changed, noise) - delta).abs().amax(-1).nonzero().flatten().tolist()
            assert affected == [9], affected
            changed = motion.clone(); changed[10, 0] += 3
            affected = (bridge.film.delta(changed, camera, noise) - delta).abs().amax(-1).nonzero().flatten().tolist()
            assert affected == [10], affected
            changed = noise.clone(); changed[0] += 3
            assert (bridge.film.delta(motion, camera, changed) - delta).abs().max().item() > 0
            assert not any("gate" in name for name, _ in bridge.film.named_parameters())
            checks.append("trajectory_FiLM_frame_local_camera_motion_and_noise_dependence_shape_and_generator_unchanged")
    return checks
