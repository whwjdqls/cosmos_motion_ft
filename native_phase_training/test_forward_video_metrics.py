from __future__ import annotations

import numpy as np

from native_phase_training.forward_video_metric_utils import (
    CDFVD_FRAME_INDICES,
    HORIZON_SUFFIX_SLICES,
    SUFFIX_RGB_INDICES,
    aggregate_scalars,
)


def test_suffix_and_horizon_contract() -> None:
    assert SUFFIX_RGB_INDICES.tolist() == list(range(1, 97))
    covered: list[int] = []
    for frame_slice in HORIZON_SUFFIX_SLICES.values():
        covered.extend(SUFFIX_RGB_INDICES[frame_slice].tolist())
    assert covered == list(range(1, 97))


def test_cdfvd_uses_every_suffix_and_horizon_frame() -> None:
    expected_bounds = {
        "full_suffix_frames_1_96": list(range(1, 97)),
        "early_frames_1_32": list(range(1, 33)),
        "middle_frames_33_64": list(range(33, 65)),
        "late_frames_65_96": list(range(65, 97)),
    }
    for name, indices in CDFVD_FRAME_INDICES.items():
        assert indices.tolist() == expected_bounds[name]
        assert 0 not in indices


def test_aggregate_validation() -> None:
    summary = aggregate_scalars([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
