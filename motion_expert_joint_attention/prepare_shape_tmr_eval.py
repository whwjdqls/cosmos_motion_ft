"""Prepare immutable BONES-SEED and floor-valid Nymeria C45 evaluation manifests."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import config
from dataset import SHARED_MEAN_PATH, SHARED_STD_PATH, humanize_caption
from nymeria_joint_dataset import NymeriaJointDataset
from shape_tmr_eval_common import (
    DEFAULT_BONES_UNIEGO_ROOT,
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_TESTSUITE,
    SUITES,
    EvalCase,
    load_case_features,
    sha256_file,
    stable_seed,
    write_jsonl,
)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _bones_motion_path(root: Path, seed_motion: dict) -> Path:
    relative = str(seed_motion["bvh_path"])
    relative = relative[4:] if relative.startswith("BVH/") else relative
    relative = relative[:-4] if relative.endswith(".bvh") else relative
    return root / f"{relative}.npz"


def build_bones_cases(
    testsuite: Path,
    uniego_root: Path,
    split: str,
    group: str,
    *,
    fps: float = 20.0,
    min_frames: int = 10,
    max_cases: int = 0,
) -> tuple[list[EvalCase], np.ndarray, dict]:
    case_dirs = sorted(glob.glob(str(testsuite / split / "text2motion" / group / "*")))
    cases: list[EvalCase] = []
    neutrals: list[np.ndarray] = []
    skipped: Counter[str] = Counter()
    for case_dir in case_dirs:
        if max_cases > 0 and len(cases) >= max_cases:
            break
        try:
            meta = json.load(open(os.path.join(case_dir, "meta.json")))
            seed_motion = json.load(open(os.path.join(case_dir, "seed_motion.json")))
            text = str(meta["text"])
            duration = float(meta["duration"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            skipped["bad_metadata"] += 1
            continue
        motion_path = _bones_motion_path(uniego_root, seed_motion)
        if not motion_path.is_file():
            skipped["missing_motion"] += 1
            continue
        try:
            with np.load(motion_path, mmap_mode="r") as data:
                if "features" not in data or "neutral_joints" not in data:
                    skipped["wrong_motion_schema"] += 1
                    continue
                n_total = int(data["features"].shape[0])
                start = int(round(int(seed_motion["crop_start_frame_index"]) * fps / 30.0))
                end = int(round(int(seed_motion["crop_end_frame_index"]) * fps / 30.0))
                start = max(0, min(start, n_total))
                end = max(start, min(end, n_total))
                gt = np.asarray(data["features"][start:end])
                neutral = np.asarray(data["neutral_joints"])
        except (OSError, KeyError, TypeError, ValueError, EOFError):
            skipped["bad_motion"] += 1
            continue
        if end - start < min_frames:
            skipped["too_short_gt"] += 1
            continue
        if not np.isfinite(gt).all():
            skipped["nonfinite_gt"] += 1
            continue
        if neutral.shape != (30, 3) or not np.isfinite(neutral).all():
            skipped["bad_neutral_joints"] += 1
            continue
        num_frames = int(duration * fps)
        if num_frames < min_frames:
            skipped["too_short_request"] += 1
            continue
        if num_frames > 200:
            skipped["request_over_phase2_limit"] += 1
            continue
        basename = os.path.basename(case_dir)
        case_id = f"bones_{split}_{group}_{basename}"
        cases.append(
            EvalCase(
                case_id=case_id,
                cohort=f"bones_{split}_{group}",
                text=text,
                num_frames=num_frames,
                seed=int(meta.get("seed", stable_seed(case_id))),
                motion_path=str(motion_path),
                gt_start=start,
                gt_end=end,
                source_kind="bones",
            )
        )
        neutrals.append((neutral - neutral.mean(axis=0, keepdims=True)).astype(np.float32))
    order = sorted(range(len(cases)), key=lambda index: (cases[index].num_frames, cases[index].case_id))
    cases = [cases[index] for index in order]
    neutral_array = np.stack([neutrals[index] for index in order]) if order else np.empty((0, 30, 3), np.float32)
    return cases, neutral_array, {
        "discovered": len(case_dirs),
        "used": len(cases),
        "skipped": dict(sorted(skipped.items())),
        "min_generated_frames": min((case.num_frames for case in cases), default=0),
        "max_generated_frames": max((case.num_frames for case in cases), default=0),
    }


def _nymeria_case(
    entry: dict,
    row_index: int,
    cohort: str,
    num_frames: int,
    *,
    with_image: bool,
) -> EvalCase:
    uuid = str(entry["uuid"])
    start = int(entry["s"])
    stem = uuid.replace("/", "__")
    text = humanize_caption(str(entry["cap"]))
    case_id = f"{cohort}_{row_index:05d}_{stem}_{start}"
    return EvalCase(
        case_id=case_id,
        cohort=cohort,
        text=text,
        num_frames=num_frames,
        seed=stable_seed(case_id),
        motion_path=str(entry["uni"]),
        gt_start=start,
        gt_end=start + num_frames,
        source_kind="nymeria",
        floor_offset=None if entry["off"] is None else float(entry["off"]),
        image_path=str(entry["vis"]) if with_image else None,
        image_start=start if with_image else None,
        uuid=uuid,
    )


def build_nymeria_cases(dataset: NymeriaJointDataset, mean: np.ndarray, std: np.ndarray):
    cohorts = {}
    cohort_neutrals = {}
    audits = {}
    definitions = (
        ("nymeria_t2m", dataset._t2m_index, False),
        ("nymeria_ti2m", dataset._index, True),
    )
    for cohort, entries, with_image in definitions:
        cases = []
        neutrals = []
        skipped: Counter[str] = Counter()
        for row_index, entry in enumerate(entries):
            num_frames = 97 if with_image else min(200, int(entry["avail"]))
            case = _nymeria_case(
                entry,
                row_index,
                cohort,
                num_frames,
                with_image=with_image,
            )
            try:
                features, neutral = load_case_features(case)
            except (OSError, KeyError, ValueError, EOFError):
                skipped["invalid_motion"] += 1
                continue
            if len(features) != num_frames:
                skipped["short_motion"] += 1
                continue
            zmax = float(np.max(np.abs((features - mean) / std)))
            if not np.isfinite(zmax) or zmax > 20.0:
                skipped["runtime_feature_guard"] += 1
                continue
            cases.append(case)
            neutrals.append(neutral)
        cohorts[cohort] = cases
        cohort_neutrals[cohort] = (
            np.stack(neutrals).astype(np.float32)
            if neutrals else np.empty((0, 30, 3), dtype=np.float32)
        )
        audits[cohort] = {
            "floor_filtered_annotation_rows": len(entries),
            "used": len(cases),
            "skipped": dict(sorted(skipped.items())),
            "held_out_sequences": len({case.uuid for case in cases}),
            "physical_windows": len({(case.uuid, case.gt_start) for case in cases}),
            "unique_captions": len({case.text for case in cases}),
            "min_generated_frames": min((case.num_frames for case in cases), default=0),
            "max_generated_frames": max((case.num_frames for case in cases), default=0),
        }
    return cohorts, cohort_neutrals, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--testsuite", default=str(DEFAULT_TESTSUITE))
    parser.add_argument("--bones-uniego-root", default=str(DEFAULT_BONES_UNIEGO_ROOT))
    parser.add_argument("--max-bones-cases", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    bundle_root = Path(args.bundle_root).resolve()
    testsuite = Path(args.testsuite).resolve()
    bones_root = Path(args.bones_uniego_root).resolve()
    base_cache_path = bundle_root / "artifacts" / "text" / "benchmark_llm2vec.pt"
    if not base_cache_path.is_file():
        raise FileNotFoundError(f"benchmark text cache is missing: {base_cache_path}")

    mean = np.load(SHARED_MEAN_PATH).astype(np.float32)
    std = np.load(SHARED_STD_PATH).astype(np.float32)
    dataset = NymeriaJointDataset(
        split="test",
        num_frames=200,
        aligned_num_frames=97,
        task_weights={"text2motion": 0.75, "textimg2motion": 0.25},
        bones_text2motion_frac=0.0,
        cfg_dropout=0.0,
        prefer_latents=False,
        force_on_the_fly=True,
        reasoner_image_for_textimg=True,
        reasoner_image_size=256,
        train=False,
    )
    nymeria, nymeria_neutrals, nymeria_audit = build_nymeria_cases(dataset, mean, std)
    for cohort, cases in nymeria.items():
        write_jsonl(out_dir / f"{cohort}.jsonl", cases)
        np.save(out_dir / f"{cohort}_neutral.npy", nymeria_neutrals[cohort])
        print(f"[prepare] {cohort}: {json.dumps(nymeria_audit[cohort], sort_keys=True)}")

    bones_audit = {}
    bones_cases = {}
    for split, group in SUITES:
        cases, neutrals, audit = build_bones_cases(
            testsuite,
            bones_root,
            split,
            group,
            max_cases=args.max_bones_cases,
        )
        cohort = f"bones_{split}_{group}"
        bones_cases[cohort] = cases
        write_jsonl(out_dir / f"{cohort}.jsonl", cases)
        np.save(out_dir / f"{cohort}_neutral.npy", neutrals)
        bones_audit[cohort] = audit
        print(f"[prepare] {cohort}: {json.dumps(audit, sort_keys=True)}")

    nymeria_captions = {
        case.text for cases in nymeria.values() for case in cases
    }
    bones_captions = {
        case.text for cases in bones_cases.values() for case in cases
    }
    evaluator_captions = sorted(
        {""} | nymeria_captions | bones_captions
    )
    _atomic_json(out_dir / "evaluator_captions.json", evaluator_captions)
    protocol = {
        "version": 2,
        "phase2_checkpoint_contract": {
            "T_text2motion_max": 200,
            "T_textimg2motion": 97,
            "fps": 20,
            "motion_representation": "283-D proportional UniEgo",
        },
        "bones": bones_audit,
        "nymeria": nymeria_audit,
        "nymeria_floor_policy": (
            "NymeriaJointDataset split=test with active floor_calibration.json; all annotation "
            "windows retained after wrong_floor/residual_penetration/extreme_y filtering, then "
            "directly filtered by the training-time old-stat |z|max<=20 guard without substitution"
        ),
        "nymeria_annotation_duplicates": (
            "preserved intentionally: different captions over the same physical start are distinct "
            "text-conditioned evaluation cases"
        ),
        "paths": {
            "testsuite": str(testsuite),
            "bones_uniego_root": str(bones_root),
            "nymeria_manifest": config.NYMERIA_MANIFEST,
            "nymeria_split": config.NYMERIA_SPLIT_FILE,
            "floor_calibration": config.FLOOR_CALIBRATION_JSON,
            "generator_mean": SHARED_MEAN_PATH,
            "generator_std": SHARED_STD_PATH,
            "bundle_root": str(bundle_root),
            "benchmark_text_cache": str(base_cache_path),
        },
        "sha256": {
            "floor_calibration": sha256_file(config.FLOOR_CALIBRATION_JSON),
            "generator_mean": sha256_file(SHARED_MEAN_PATH),
            "generator_std": sha256_file(SHARED_STD_PATH),
            "benchmark_text_cache": sha256_file(base_cache_path),
        },
        "unique_evaluator_captions": {
            "all_including_empty": len(evaluator_captions),
            "bones": len(bones_captions),
            "nymeria": len(nymeria_captions),
        },
    }
    _atomic_json(out_dir / "protocol.json", protocol)
    print(f"[prepare] wrote manifests and protocol to {out_dir}")


if __name__ == "__main__":
    main()
