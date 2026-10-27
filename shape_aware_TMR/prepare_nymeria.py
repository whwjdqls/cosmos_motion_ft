"""Freeze Nymeria splits/captions and small raw-window NPY caches on a compute node."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from dataset import humanize_caption
from kimodo.sanitize import sanitize_text
from motion_array_cache import load_motion_window
from nymeria_data import atomic_json, require_compute, sha256_file


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--uniego-root", required=True)
    p.add_argument("--floor-calibration", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--representation", default="camera_head_recanonicalization_v1")
    p.add_argument("--data-fps", type=float, default=20.0)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    require_compute()
    if not 0 < args.validation_fraction < 1 or not 0 < args.min_frames <= args.max_frames:
        p.error("invalid validation fraction or frame limits")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = {
"schema": "nymeria_c45_windows_v2",
        "input_sha256": {name: sha256_file(getattr(args, name))
                         for name in ("manifest", "split_file", "floor_calibration")},
        "uniego_root": str(Path(args.uniego_root).resolve()),
        "representation": args.representation, "data_fps": args.data_fps,
        "validation_fraction": args.validation_fraction,
        "min_frames": args.min_frames, "max_frames": args.max_frames, "seed": args.seed,
        "caption_policy": "humanize_caption + kimodo.sanitize_text",
        "deduplication_policy": "same window and sanitized caption; reject conflicting floor metadata",
        "length_policy": "native annotated windows; reject outside min/max, no new windows",
    }
    marker = root / "prepared.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["request"] != request:
            raise RuntimeError("prepared data has different provenance; select a new --out")
        print(f"Reuse immutable prepared data: {root}", flush=True)
        return

    split = json.loads(Path(args.split_file).read_text())
    train = set(split["train"])
    test = set(split["test"])
    validation = set(split.get("validation", split.get("val", [])))
    if "val" in split and "validation" in split and set(split["val"]) != set(split["validation"]):
        raise ValueError("ambiguous validation and val recording lists")
    if train & test or train & validation or test & validation:
        raise ValueError("source split contains overlapping recordings")
    derived_validation = not validation
    if derived_validation:
        if len(train) < 2:
            raise ValueError("need at least two training recordings for held-out validation")
        ordered = sorted(train, key=lambda u: hashlib.sha256(
            f"{args.seed}:{u}".encode()).hexdigest())
        count = min(len(train) - 1, max(1, round(len(train) * args.validation_fraction)))
        validation = set(ordered[:count])
        train -= validation
    if not train or not validation or not test:
        raise ValueError("train, validation, and test must all have recordings")
    partitions = {"train": train, "validation": validation, "test": test}
    owner = {uuid: name for name, ids in partitions.items() for uuid in ids}

    floor = json.loads(Path(args.floor_calibration).read_text())
    deltas = floor.get("deltas", {})
    fallback = float(floor.get("global_delta", 0.0))
    drops = {u: {(int(e[0]), int(e[1])): str(e[2]) if len(e) > 2 else "quality_drop"
                 for e in entries}
             for u, entries in floor.get("dropped_windows", {}).items()}
    rows = {name: [] for name in partitions}
    rejected = []
    seen_uuid, seen_case = set(), {}
    cached_windows = {}
    duplicate_counts = Counter()
    sequence_sources = {}
    captions = {""}

    def reject(uuid, reason, window=None):
        row = {"uuid": uuid, "reason": reason}
        if window:
            row.update(start_frame=window.get("start_frame"), end_frame=window.get("end_frame"))
        rejected.append(row)

    with open(args.manifest) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid not in owner:
                continue
            if uuid in seen_uuid:
                raise ValueError(f"duplicate recording in source manifest: {uuid}")
            seen_uuid.add(uuid)
            relative = Path(uuid)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe recording ID: {uuid}")
            motion_path = Path(args.uniego_root) / f"{uuid}.npz"
            if not motion_path.is_file():
                reject(uuid, "missing_motion")
                continue
            total = int(record.get("nb_frames", 0))
            if total < args.min_frames:
                reject(uuid, "short_recording")
                continue
            features, neutral = load_motion_window(motion_path, 0, total)
            if features.ndim != 2 or features.shape[1] != 283:
                reject(uuid, "invalid_feature_shape")
                continue
            if neutral.shape != (30, 3) or not np.isfinite(neutral).all():
                reject(uuid, "invalid_neutral_joints")
                continue
            sequence_sources[uuid] = {
                "path": str(motion_path), "bytes": motion_path.stat().st_size,
                "mtime_ns": motion_path.stat().st_mtime_ns, "loaded_frames": len(features),
            }
            skeleton_name = hashlib.sha256(uuid.encode()).hexdigest()
            neutral_file = f"neutral/{skeleton_name}.npy"
            save_array(root / neutral_file, neutral)
            for window in record.get("t2w_windows", []):
                if not window.get("usable", False):
                    reject(uuid, "not_usable", window)
                    continue
                caption = window.get("caption")
                if not isinstance(caption, str) or not caption.strip():
                    reject(uuid, "missing_caption", window)
                    continue
                text = sanitize_text(humanize_caption(caption))
                if not text:
                    reject(uuid, "empty_sanitized_caption", window)
                    continue
                start, end = int(window["start_frame"]), int(window["end_frame"])
                reason = drops.get(uuid, {}).get((start, end))
                if reason is not None:
                    reject(uuid, f"floor_quality:{reason}", window)
                    continue
                offset = window.get("ground_offset_y")
                if offset is None:
                    reject(uuid, "missing_floor_offset", window)
                    continue
                delta = float(deltas.get(uuid, fallback))
                offset = float(offset) + delta
                if not math.isfinite(offset):
                    reject(uuid, "invalid_floor_offset", window)
                    continue
                if start < 0 or end > min(total, len(features)) or end <= start:
                    reject(uuid, "out_of_bounds_window", window)
                    continue
                if not args.min_frames <= end - start <= args.max_frames:
                    reject(uuid, "window_length_outside_limits", window)
                    continue
                raw = features[start:end]
                if not np.isfinite(raw).all():
                    reject(uuid, "nonfinite_window", window)
                    continue
                window_id = f"{uuid}:{start}:{end}"
                caption_key = (window_id, text)
                floor_signature = json.dumps({
                    "floor_offset": offset, "floor_gt": float(window["ground_offset_y"]),
                    "floor_delta": delta,
                    **{name: window.get(name) for name in (
                        "floor_source", "floor_status", "ambiguous", "est_ambiguous",
                        "n_floors_in_slice", "foot_skating_cms")},
                }, sort_keys=True, allow_nan=False)
                if caption_key in seen_case:
                    if seen_case[caption_key] != floor_signature:
                        raise ValueError(f"conflicting floor metadata for duplicate caption: {window_id}")
                    reject(uuid, "duplicate_window_caption", window)
                    duplicate_counts[owner[uuid]] += 1
                    continue
                seen_case[caption_key] = floor_signature
                case_id = f"{window_id}:{hashlib.sha256(text.encode()).hexdigest()}"
                if window_id not in cached_windows:
                    digest = hashlib.sha256(window_id.encode()).hexdigest()
                    features_file = f"windows/{digest[:2]}/{digest}.npy"
                    save_array(root / features_file, raw)
                    cached_windows[window_id] = (
                        features_file, sha256_file(root / features_file))
                features_file, features_sha256 = cached_windows[window_id]
                rows[owner[uuid]].append({
                    "case_id": case_id, "window_id": window_id,
                    "uuid": uuid, "text": text, "raw_caption": caption,
                    "start_frame": start, "end_frame": end, "num_frames": end - start,
                    "features_file": features_file, "neutral_file": neutral_file,
                    "features_sha256": features_sha256,
                    "neutral_sha256": sha256_file(root / neutral_file),
                    "floor_offset": offset, "floor_gt": float(window["ground_offset_y"]),
                    "floor_delta": delta, "floor_delta_source": "sequence" if uuid in deltas else "global",
                })
                captions.add(text)
            print(f"[prepare] {uuid}: " + " ".join(
                f"{name}={len(values)}" for name, values in rows.items()), flush=True)
    for name, values in rows.items():
        if not values:
            raise ValueError(f"no eligible windows in {name}; inspect source paths and exclusions")
        values.sort(key=lambda r: (r["uuid"], r["start_frame"], r["end_frame"], r["case_id"]))
        write_jsonl(root / f"{name}.jsonl", values)
    write_jsonl(root / "rejections.jsonl", rejected)
    atomic_json(root / "captions.json", sorted(captions))
    atomic_json(root / "recording_splits.json", {name: sorted(ids) for name, ids in partitions.items()})
    atomic_json(marker, {
        "request": request,
        "derived_validation_from_train": derived_validation,
        "counts": {name: {"windows": len(values), "caption_cases": len(values),
                          "unique_motion_windows": len({r["window_id"] for r in values}),
                          "duplicate_window_captions_removed": duplicate_counts[name],
                          "recordings": len({r["uuid"] for r in values}),
                          "unique_captions": len({r["text"] for r in values})}
                   for name, values in rows.items()},
        "rejections": dict(Counter(r["reason"] for r in rejected)),
        "missing_manifest_recordings": sorted(set(owner) - seen_uuid),
        "sequence_sources": sequence_sources,
        "cache_semantics": "raw unnormalized UniEgo283 and uncentered neutral joints; floor applied on read",
        "status": "prepared_not_trained",
    })
    print(json.dumps({"out": str(root), "windows": {k: len(v) for k, v in rows.items()}}), flush=True)


if __name__ == "__main__":
    main()
