"""Focused CPU checks for the Phase-2 C45 evaluation data/sampling helpers."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from bs_shape_metrics import farthest_shape_indices
from shape_tmr_eval_common import (
    EvalCase,
    chunked_farthest_indices,
    read_jsonl,
    seeded_initial_noise,
    stable_seed,
    write_jsonl,
)


def _case(case_id: str, frames: int) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        cohort="test",
        text="a person walks",
        num_frames=frames,
        seed=stable_seed(case_id),
        motion_path="/tmp/not-read.npz",
        gt_start=0,
        gt_end=frames,
        source_kind="bones",
    )


def main() -> None:
    rng = np.random.default_rng(7)
    bones = rng.normal(size=(37, 29))
    expected = farthest_shape_indices(bones)
    actual = chunked_farthest_indices(bones, chunk_size=5)
    np.testing.assert_array_equal(actual, expected)

    cases = [_case("case-a", 7), _case("case-b", 11), _case("case-c", 5)]
    together = seeded_initial_noise(cases, 11, "cpu")
    for index, case in enumerate(cases):
        alone = seeded_initial_noise([case], case.num_frames, "cpu")
        torch.testing.assert_close(
            together[index, : case.num_frames], alone[0], rtol=0, atol=0
        )
        assert bool((together[index, case.num_frames :] == 0).all())

    assert stable_seed("case-a") == stable_seed("case-a")
    assert stable_seed("case-a") != stable_seed("case-b")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cases.jsonl"
        write_jsonl(path, cases)
        assert read_jsonl(path) == cases

    print("shape-TMR eval contracts PASS: exact farthest shape, per-case noise, JSONL roundtrip")


if __name__ == "__main__":
    main()
