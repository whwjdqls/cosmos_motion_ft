"""Merge independently computed Phase-1 resolution-matrix model reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from native_phase_training.analyze_phase1_resolution_matrix import CELLS, MODELS, _write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = None
    for model in MODELS:
        path = args.partials_root / model / "resolution_matrix_metrics.json"
        partial = json.loads(path.read_text())
        if set(partial["models"]) != {model}:
            raise ValueError(f"{path} does not contain exactly model {model!r}")
        if payload is None:
            payload = {**partial, "models": {}}
        else:
            for key in (
                "schema_version",
                "matrix_root",
                "eval_root",
                "conditioned_frame_excluded",
                "common_comparison_size",
            ):
                if partial[key] != payload[key]:
                    raise ValueError(f"partial report mismatch for {key}: {path}")
        payload["models"][model] = partial["models"][model]

    assert payload is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "resolution_matrix_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    summary_path = args.output_dir / "SUMMARY.md"
    _write_summary(summary_path, payload)
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(summary_path),
                "models": list(MODELS),
                "cells": list(CELLS),
                "partials_root": str(args.partials_root),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[resolution-analysis] merged {len(MODELS)} model reports into {args.output_dir}")


if __name__ == "__main__":
    main()
