"""Fast CPU contracts for native-schedule Phase-3 bridge training."""
from __future__ import annotations

import torch

import flow
from modality_bridge import LocalModalityBridge


def verify_waver() -> None:
    g1 = torch.Generator().manual_seed(17)
    got = flow.sample_sigma_native_waver(32, "cpu", shift=3.0, generator=g1)

    g2 = torch.Generator().manual_seed(17)
    u = torch.rand((32,), dtype=torch.float32, generator=g2)
    t_raw = 1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1.0 + u)
    expected = flow.shift_sigma(1.0 - t_raw, 3.0)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    assert bool(((got >= 0.0) & (got <= 1.0)).all())


def verify_causal_locality() -> None:
    bridge = LocalModalityBridge(hidden=8, num_heads=2, head_dim=4)
    gen = torch.arange(25)
    motion = torch.arange(97)
    mask = bridge._local_pair_mask(gen, motion)
    expected_groups = {
        0: [0],
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8],
        24: [93, 94, 95, 96],
    }
    for latent_frame, source_frames in expected_groups.items():
        assert torch.nonzero(mask[latent_frame]).view(-1).tolist() == source_frames
    assert mask.sum().item() == 97
    assert mask.sum(dim=0).eq(1).all(), "every source frame needs exactly one local latent"


def main() -> None:
    verify_waver()
    verify_causal_locality()
    print("native bridge CPU contracts: PASS")


if __name__ == "__main__":
    main()
