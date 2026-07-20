"""Validate the production Phase-2 shape-TMR GPU smoke artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _require_finite_numbers(value, path: str = "summary") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases-per-cohort", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    summary = json.loads((root / "summary.json").read_text())
    expected = {
        "bones_content_overview": {"t2m_cfg2"},
        "nymeria_t2m": {"t2m_cfg2"},
        "nymeria_ti2m": {"ti2m_cfg2", "ti2m_no_cfg"},
    }
    if set(summary["cohorts"]) != set(expected):
        raise ValueError(
            f"smoke cohorts mismatch: {sorted(summary['cohorts'])} vs {sorted(expected)}"
        )
    for cohort, variants in expected.items():
        rows = summary["cohorts"][cohort]
        if set(rows) != variants:
            raise ValueError(f"{cohort}: variants {sorted(rows)} vs {sorted(variants)}")
        for variant, row in rows.items():
            if int(row["n"]) != args.cases_per_cohort:
                raise ValueError(f"{cohort}/{variant}: expected n={args.cases_per_cohort}")
    _require_finite_numbers(summary)

    videos = sorted((root / "viz").glob("**/*.mp4"))
    conditions = sorted((root / "viz" / "nymeria_ti2m").glob("**/*_condition.png"))
    if len(videos) < 4:
        raise ValueError(f"expected at least four smoke MP4s, found {len(videos)}")
    if len(conditions) < 2:
        raise ValueError(f"expected both TI2M condition images, found {len(conditions)}")
    print(
        f"shape-TMR GPU smoke PASS: cohorts=3 variants=4 videos={len(videos)} "
        f"condition_images={len(conditions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
