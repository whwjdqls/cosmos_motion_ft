"""Merge six Kimodo text-to-motion evaluation JSONs into one compact report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SUITES = (
    ("content", "overview"),
    ("content", "timeline_single"),
    ("content", "timeline_multi"),
    ("repetition", "overview"),
    ("repetition", "timeline_single"),
    ("repetition", "timeline_multi"),
)


def _metric_row(payload: dict, path: Path) -> dict:
    protocol = payload["protocol"]
    generators = payload["generators"]
    if len(generators) != 1:
        raise ValueError(f"expected one generator in {path}, found {sorted(generators)}")
    label, result = next(iter(generators.items()))
    physical = result["physical_20fps"]
    shape = result["shape"]
    tracking = shape["population_tracking"]
    row = {
        "split": protocol["split"],
        "group": protocol["group"],
        "result": str(path),
        "generator_label": label,
        "case_audit": protocol["case_audit"],
        "protocol_R01": result["tmr"]["TMR/t2m_R/R01"],
        "protocol_R02": result["tmr"]["TMR/t2m_R/R02"],
        "protocol_R03": result["tmr"]["TMR/t2m_R/R03"],
        "protocol_R05": result["tmr"]["TMR/t2m_R/R05"],
        "protocol_R10": result["tmr"]["TMR/t2m_R/R10"],
        "plain_R01": result["plain_t2m_gen"]["R01"],
        "plain_R02": result["plain_t2m_gen"]["R02"],
        "plain_R03": result["plain_t2m_gen"]["R03"],
        "plain_R05": result["plain_t2m_gen"]["R05"],
        "plain_R10": result["plain_t2m_gen"]["R10"],
        "fid_gen_gt": result["tmr"]["TMR/FID/gen_gt"],
        "contact_skate_cm_s": physical["foot_skate_from_pred_contacts"] * 100.0,
        "height_skate_cm_s": physical["foot_skate_from_height"] * 100.0,
        "max_contact_velocity_cm_s": physical["foot_skate_max_vel"] * 100.0,
        "contact_consistency": physical["foot_contact_consistency"],
        "skate_ratio": physical["foot_skate_ratio"],
        "bone_mae_cm": shape["bone_length_mae_cm_mean"],
        "shape_centered_correlation": tracking["actor_centered_correlation"],
        "shape_response_slope": tracking["actor_centered_response_slope"],
        "shape_variance_ratio": tracking["actor_centered_variance_ratio"],
    }
    farthest = shape["counterfactuals"].get("farthest_natural")
    if farthest is not None:
        row.update(
            {
                "shape_cf_delta_cosine": farthest["delta_cosine"],
                "shape_cf_response_slope": farthest["delta_response_slope"],
                "shape_cf_magnitude_ratio": farthest["delta_magnitude_ratio"],
                "shape_cf_target_advantage_cm": farthest[
                    "counterfactual_target_advantage_cm"
                ],
                "shape_cf_plain_R03": farthest["plain_t2m"]["R03"],
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    common = None
    for split, group in SUITES:
        path = input_dir / f"{split}_{group}.json"
        payload = json.loads(path.read_text())
        if (payload["protocol"]["split"], payload["protocol"]["group"]) != (split, group):
            raise ValueError(f"suite identity mismatch in {path}")
        rows.append(_metric_row(payload, path))
        current = {
            "evaluator": payload["evaluator"],
            "sampling_steps": payload["protocol"]["sampling_steps"],
            "native_solver": payload["protocol"]["requested_native_solver"],
            "guidance": payload["protocol"]["guidance"],
            "shape_counterfactual_strategy": payload["protocol"]["shape_counterfactual"][
                "strategy"
            ],
        }
        if common is None:
            common = current
        elif current != common:
            raise ValueError(f"evaluation protocol differs in {path}")

    metric_names = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    total_used = sum(row["case_audit"]["used"] for row in rows)
    weighted = {
        key: sum(row[key] * row["case_audit"]["used"] for row in rows) / total_used
        for key in metric_names
        if all(key in row and math.isfinite(float(row[key])) for row in rows)
    }
    report = {
        "scope": {
            "included": [f"{split}/text2motion/{group}" for split, group in SUITES],
            "excluded_as_not_applicable": [
                {
                    "categories": ["constraints_withtext", "constraints_notext"],
                    "reason": (
                        "The BONES MotionExpert accepts text and skeleton shape only; it has no "
                        "trajectory, keyframe, or end-effector constraint input."
                    ),
                }
            ],
        },
        "protocol": common,
        "total_discovered_cases": sum(row["case_audit"]["discovered"] for row in rows),
        "total_used_cases": total_used,
        "suites": rows,
        "case_weighted_mean_of_suite_metrics": weighted,
        "weighted_mean_note": (
            "These are case-weighted means of independently computed suite metrics, not metrics "
            "recomputed over one merged retrieval pool."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[all-benchmarks] wrote {out} ({total_used} usable cases)", flush=True)


if __name__ == "__main__":
    main()
