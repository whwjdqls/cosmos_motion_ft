"""Focused FiLM tests, called only by the approved compute-node preflight."""
import torch
from modality_bridge import LocalModalityBridge
from .film import CameraFiLMBridge, CameraMotionFiLM
from .backbone import bridge_meta


def check_film(device):
    checks = []
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        torch.manual_seed(19)
        camera = torch.arange(96 * 8, device=device, dtype=torch.float32).reshape(96, 8)
        incoming, outgoing, valid = CameraMotionFiLM.align_transitions(camera)
        assert torch.equal(incoming[0], torch.zeros_like(camera[0]))
        assert torch.equal(outgoing[0], camera[0])
        assert torch.equal(incoming[10], camera[9]) and torch.equal(outgoing[10], camera[10])
        assert torch.equal(incoming[96], camera[95]) and torch.count_nonzero(outgoing[96]) == 0
        assert valid[0].tolist() == [0, 1] and valid[-1].tolist() == [1, 0]
        checks.append("FiLM_incoming_outgoing_transition_alignment_and_boundary_masks")
        bridge = CameraFiLMBridge(LocalModalityBridge(8, 2, 4).to(device), 4, 8)
        with torch.no_grad():
            bridge.attention.gate_g.fill_(0.2)
            bridge.attention.gate_m.fill_(0.2)
            for mode in ("motimg2video", "video2motion", "camera2motion", "cameravideo2motion"):
                meta = bridge_meta(mode, device)
                g = torch.randn(meta.gen_frame.numel(), 8, device=device)
                m = torch.randn(98, 8, device=device)
                expected_g, expected_m = bridge.attention(g, m, meta)
                actual_g, actual_m = bridge(g, m, meta)
                torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
                torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
            checks.append("FiLM_zero_initialization_bit_exact_with_nonzero_attention_gates")
            bridge.film.mlp[-1].weight.normal_(std=0.05)
            bridge.film.mlp[-1].bias.normal_(std=0.05)
            for mode in ("motimg2video", "video2motion"):
                meta = bridge_meta(mode, device)
                g = torch.randn(meta.gen_frame.numel(), 8, device=device)
                m = torch.randn(98, 8, device=device)
                expected_g, expected_m = bridge.attention(g, m, meta)
                actual_g, actual_m = bridge(g, m, meta)
                torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
                torch.testing.assert_close(actual_m, expected_m, rtol=0, atol=0)
            meta = bridge_meta("cameravideo2motion", device)
            g = torch.randn(meta.gen_frame.numel(), 8, device=device)
            m = torch.randn(98, 8, device=device)
            expected_g, expected_m = bridge.attention(g, m, meta)
            actual_g, actual_m = bridge(g, m, meta)
            torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
            torch.testing.assert_close(actual_m[:1], expected_m[:1], rtol=0, atol=0)
            assert not torch.equal(actual_m[1:], expected_m[1:])
            checks.append("FiLM_changes_only_target_motion_frames_not_shape_or_generator_or_non_camera_tasks")
            motion = torch.randn(97, 8, device=device)
            camera = torch.randn(96, 8, device=device)
            delta = bridge.film.delta(motion, camera)
            changed = camera.clone(); changed[9, 0] += 3
            delta_camera = bridge.film.delta(motion, changed)
            affected = (delta_camera - delta).abs().amax(-1).nonzero().flatten().tolist()
            assert affected == [9, 10], affected
            changed = motion.clone(); changed[10, 0] += 3
            delta_motion = bridge.film.delta(changed, camera)
            affected = (delta_motion - delta).abs().amax(-1).nonzero().flatten().tolist()
            assert affected == [10], affected
            checks.append("FiLM_uses_camera_and_motion_context_with_frame_local_modulation")
    return checks
