"""Offline (kimodo env) builder: emit `bones_pairs_{train,val}.jsonl` over the 283-D
BONES-SEED UniEgoMotion tree so the joint-attention `train.py` (cosmos env) can load BONES
**without importing kimodo**.

Why this exists
---------------
The MoT joint-attention trainer runs in the `cosmos` env (it instantiates the real
`Cosmos3VFMNetwork`). The BONES-SEED (text, segment) index, however, lives in kimodo
(`SOMABonesSeedDataset`: the natural/single/multi pools, the split restriction, and the
`_resolve_segment` windowing). We cannot import kimodo from the cosmos env (Py 3.13 vs 3.10,
torch 2.10 vs 2.4 — they cannot share a process). So we run the kimodo index build ONCE here,
flatten it to a plain JSONL of `(uuid, uniego_path, start, end, caption, source)` rows pointing
straight at the 283-D uniego npz, and the cosmos-env `train.py` then reads each window with a
plain `np.load(uniego_path)["features"][start:end]` — zero kimodo dependency at train time.

This mirrors `motion_expert/build_pairs.py` (the Nymeria pair builder) but over the BONES-SEED
uniego tree (`…/soma_proportional_uniegomotion_20fps`). The emitted row schema is intentionally
identical to Nymeria's so the joint `dataset.py` can `ConcatDataset` them with one code path:

    {"uuid": "<reldir>/<filename>", "uniego_path": "<abs>.npz",
     "start": int, "end": int, "caption": str,
     "ground_offset_y": null,      # BONES-SEED uniego is already floor-grounded; no offset
     "source": "bones"}

Determinism / idempotency
-------------------------
We reuse `BonesSeedUniegoDataset` (PoC `bs_dataset.py`) to build the kimodo index, then enumerate
its three pools directly (`self._pools[src]`). By default, the natural/overview pool is filtered to
`content_natural_desc_4` only; single-timeline and multi-timeline entries are kept as-is. Each kept
`_BSEntry` becomes exactly ONE row. Unlike the
runtime dataset — whose `_resolve_segment` adds a *random* per-fetch offset — we emit the entry's
**stored segment bounds** capped to `max_frames` (= a fixed front-cropped window). That makes the
output a pure function of the index, so re-running reproduces the same JSONL (idempotent), and a
crashed run is `--resume`-able (rows already present, keyed by `(uniego_path,start,end,caption)`,
are skipped). Windows shorter than `--min_frames` (32) or NaN/Inf-tainted are dropped.

Run (kimodo env)
----------------
    source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh && conda activate kimodo
    export PYTHONPATH=/home/jungbin_cho/kimodo_open:/home/jungbin_cho/cosmos_motion_ft/motion_expert:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
    python /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/build_bones_pairs.py

  (PYTHONPATH needs kimodo_open for `SOMABonesSeedDataset`, and `motion_expert` for
   `bs_dataset.py` + `uniego_layout.py` which `BonesSeedUniegoDataset` imports.)

Outputs (under RUNS_ROOT/joint_attention/, default /weka/jungbin/cosmos_motion_ft_runs/):
    joint_attention/bones_pairs_train.jsonl
    joint_attention/bones_pairs_val.jsonl
    joint_attention/bones_index_{train,val}.json   # cached kimodo index (fast re-runs)

Optional --export_windows mode
------------------------------
If a row's npz slice should be materialized as a standalone per-window npz (e.g. the proportional
tree is unavailable at train time, or you want a flat per-clip cache), pass `--export_windows
<dir>`: each row then writes `<dir>/<hash>.npz` with `features[start:end]` + `neutral_joints`, and
`uniego_path` points at that small npz with `start=0,end=n`. Default (no flag) keeps `uniego_path`
pointing at the source tree (no copy) — the train loader slices on the fly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os

import numpy as np

# `BonesSeedUniegoDataset` lives in the PoC; it subclasses kimodo's `SOMABonesSeedDataset` and
# carries the uniego-specific `_build_natural_pool`. Importing it (kimodo env, PYTHONPATH has both
# kimodo_open and motion_expert) gives us the exact same index the runtime POC builds.
from bs_dataset import (
    BonesSeedUniegoDataset,
    DATA_ROOT,
    NATURAL_CSV,
    TEMPORAL_JSONL,
    MULTI_JSONL,
)
from uniego_layout import FEAT_DIM

log = logging.getLogger("build_bones_pairs")

# ---- defaults for the cluster (/weka) box; bs_dataset.py defaults to the A100 /mnt paths ----
WEKA_DATA_ROOT = "/weka/jungbin/seed/soma_proportional_uniegomotion_20fps"
RUNS_ROOT = "/weka/jungbin/cosmos_motion_ft_runs"
TRAIN_SPLIT = "/weka/jungbin/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt"
# Held-out: the content-generalization test split (disjoint motions from train).
VAL_SPLIT = "/weka/jungbin/Kimodo-Motion-Gen-Benchmark/splits/test_content_split_paths.txt"
WEKA_METADATA = "/weka/jungbin/seed/metadata"
WEKA_NATURAL_CSV = os.path.join(WEKA_METADATA, "seed_metadata_v004.csv")
WEKA_TEMPORAL = os.path.join(WEKA_METADATA, "seed_metadata_v002_temporal_labels.jsonl")
WEKA_MULTI = "/weka/jungbin/seed/multi_timeline.jsonl"


def _row_key(uniego_path: str, start: int, end: int, caption: str) -> str:
    """Stable identity for a pair row (used for --resume dedup)."""
    return f"{uniego_path}|{start}|{end}|{caption}"


def _window_hash(uniego_path: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{uniego_path}|{start}|{end}".encode("utf-8")).hexdigest()
    return h[:16]


def _existing_keys(path: str) -> set:
    """Read an existing JSONL and return the set of row keys already present (for --resume)."""
    keys: set = set()
    if not os.path.isfile(path):
        return keys
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add(_row_key(r["uniego_path"], int(r["start"]), int(r["end"]), r["caption"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return keys


class _Entry:
    """Lightweight (text, segment) entry, matching kimodo's `_BSEntry` fields we use."""
    __slots__ = ("source", "motion_path", "filename", "seg_start_sec", "seg_end_sec", "text")

    def __init__(self, source, motion_path, filename, seg_start_sec, seg_end_sec, text, **_):
        self.source = source
        self.motion_path = motion_path
        self.filename = filename
        self.seg_start_sec = float(seg_start_sec)
        self.seg_end_sec = float(seg_end_sec)
        self.text = text


# Canonical source order, fixed so enumeration (and the output JSONL) is deterministic.
SOURCES = ("natural", "single", "multi")


def _natural_desc4_by_file(natural_csv_path: str, include_mirrored: bool) -> dict[str, str]:
    """Return filename -> cleaned content_natural_desc_4 for overview/natural rows."""
    import csv

    # Keep the mirror filtering aligned with BonesSeedUniegoDataset._build_natural_pool.
    from bs_dataset import _truthy

    out: dict[str, str] = {}
    with open(natural_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if not include_mirrored and (_truthy(row.get("is_mirror")) or fname.endswith("_M")):
                continue
            desc4 = row.get("content_natural_desc_4", "")
            if isinstance(desc4, str):
                desc4 = desc4.strip()
            if desc4:
                out[fname] = desc4
    return out


def _apply_natural_caption_policy(pools: dict, args) -> dict:
    """Select which overview/natural captions survive before JSONL materialization."""
    policy = getattr(args, "natural_caption_policy", "desc4")
    if policy == "all":
        return pools
    if policy != "desc4":
        raise ValueError(f"unknown natural_caption_policy={policy!r}")

    desc4 = _natural_desc4_by_file(args.natural_csv, args.include_mirrored)
    before = len(pools.get("natural", []))
    pools = dict(pools)
    pools["natural"] = [
        e for e in pools.get("natural", [])
        if desc4.get(e.filename) == (e.text or "").strip()
    ]
    log.info("natural caption policy desc4: kept %d/%d overview entries",
             len(pools["natural"]), before)
    return pools


def load_pools(split_name: str, train_split_path: str, cache_index: str, args) -> dict:
    """Return {source: [_Entry,...]} for a split, decoupled from the kimodo runtime.

    Builds the kimodo index via `BonesSeedUniegoDataset` (which writes the JSON cache before its
    own empty-pool guard fires). For the held-out val split the `multi` pool is legitimately
    empty (the multi-timeline file references train-split motions), which trips kimodo's
    `RuntimeError("Empty pool …")` — so we catch it and enumerate from the just-written cache
    JSON instead. Either way we only consume the (text, segment) index, never kimodo motion I/O.
    """
    try:
        ds = BonesSeedUniegoDataset(
            train_split_path=train_split_path,
            data_root=args.data_root,
            natural_csv_path=args.natural_csv,
            temporal_labels_path=args.temporal_labels,
            multi_timeline_path=args.multi_timeline,
            mean_path=os.path.join(args.data_root, "Mean_uniego.npy"),
            std_path=os.path.join(args.data_root, "Std_uniego.npy"),
            fps=args.fps,
            max_clip_sec=args.max_clip_sec,
            min_frames=args.min_frames,
            include_mirrored=args.include_mirrored,
            cache_index=cache_index,
            train=(split_name == "train"),
            seed=0,
        )
        return {src: list(ds._pools[src]) for src in SOURCES}
    except RuntimeError as e:
        # Empty-pool guard (common for val/multi). The index cache was already written; use it.
        if "Empty pool" not in str(e) or not os.path.isfile(cache_index):
            raise
        log.warning("[%s] %s — enumerating from index cache %s", split_name, e, cache_index)
        raw = json.load(open(cache_index))
        return {src: [_Entry(**d) for d in raw.get(src, [])] for src in SOURCES}


def build_split(
    split_name: str,
    train_split_path: str,
    args,
) -> int:
    """Build one split's JSONL by enumerating the kimodo index pools deterministically."""
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bones_pairs_{split_name}.jsonl")
    cache_index = os.path.join(out_dir, f"bones_index_{split_name}.json")

    # Build the kimodo index (cached → near-instant on re-run). normalize=False inside the
    # subclass, so no stats are touched; we only need the (text, segment) pools, not motion I/O.
    log.info("[%s] building kimodo index over %s (split=%s)…",
             split_name, args.data_root, os.path.basename(train_split_path))
    pools = load_pools(split_name, train_split_path, cache_index, args)
    pools = _apply_natural_caption_policy(pools, args)
    log.info("[%s] pools: %s", split_name,
             " ".join(f"{s}={len(pools[s])}" for s in SOURCES))
    max_frames = int(round(args.max_clip_sec * args.fps))

    # --resume: skip rows already written; append. Otherwise start fresh.
    done_keys = _existing_keys(out_path) if args.resume else set()
    mode = "a" if (args.resume and done_keys) else "w"
    if args.export_windows:
        os.makedirs(args.export_windows, exist_ok=True)

    # Cache npz length + finiteness checks so duplicated motion_paths (a clip backs many
    # captions/segments) only hit the NFS once.
    len_cache: dict[str, int] = {}
    nj_finite_cache: dict[str, bool] = {}

    n_written = n_skip_resume = n_short = n_nonfinite = n_oob = n_missing = 0
    tmp_path = out_path + f".tmp.{os.getpid()}"
    # Stream-append into the live file under --resume; else write into a tmp then atomic-rename.
    sink_path = out_path if mode == "a" else tmp_path
    with open(sink_path, mode) as fout:
        for src in SOURCES:                          # ("natural", "single", "multi")
            for entry in pools[src]:
                npz = entry.motion_path
                # Deterministic window: stored segment bounds → frames, front-capped to max_frames.
                sf = int(round(entry.seg_start_sec * args.fps))
                ef = int(round(entry.seg_end_sec * args.fps))
                if npz not in len_cache:
                    if not os.path.isfile(npz):
                        len_cache[npz] = -1
                        nj_finite_cache[npz] = False
                    else:
                        try:
                            with np.load(npz, mmap_mode="r") as d:
                                len_cache[npz] = int(d["features"].shape[0])
                                nj = np.asarray(d["neutral_joints"]).astype(np.float32)
                                nj_finite_cache[npz] = bool(np.isfinite(nj).all())
                        except (OSError, KeyError, ValueError, EOFError) as e:
                            log.warning("[%s] bad npz %s: %s", split_name, npz, e)
                            len_cache[npz] = -1
                            nj_finite_cache[npz] = False
                ulen = len_cache[npz]
                if ulen < 0:
                    n_missing += 1
                    continue

                sf = max(0, min(sf, ulen))
                ef = max(sf, min(ef, ulen))
                ef = min(ef, sf + max_frames)
                if ef - sf < args.min_frames:
                    n_short += 1
                    continue
                if sf < 0 or ef > ulen:
                    n_oob += 1
                    continue
                if not nj_finite_cache[npz]:
                    n_nonfinite += 1
                    continue

                cap = (entry.text or "").strip()
                if not cap:
                    continue

                # Verify the features window itself is finite (some proportional npz are NaN-tainted;
                # kimodo's shape-aware audit drops ~679). A NaN GT window would poison the loss.
                try:
                    with np.load(npz, mmap_mode="r") as d:
                        feats = np.asarray(d["features"][sf:ef]).astype(np.float32)
                        if args.export_windows:
                            nj_full = np.asarray(d["neutral_joints"]).astype(np.float32)
                except (OSError, KeyError, ValueError, EOFError) as e:
                    log.warning("[%s] window read fail %s[%d:%d]: %s", split_name, npz, sf, ef, e)
                    n_missing += 1
                    continue
                if feats.shape[-1] != FEAT_DIM or not np.isfinite(feats).all():
                    n_nonfinite += 1
                    continue

                # rel uuid: "<reldir>/<filename>" (drop data_root prefix + .npz suffix)
                rel = os.path.relpath(npz, args.data_root)
                uuid = rel[:-4] if rel.endswith(".npz") else rel

                row_uniego_path = npz
                row_start, row_end = sf, ef
                if args.export_windows:
                    # Materialize a standalone per-window npz; row then slices [0:n].
                    wpath = os.path.join(args.export_windows, _window_hash(npz, sf, ef) + ".npz")
                    if not os.path.isfile(wpath):
                        np.savez(wpath, features=feats, neutral_joints=nj_full)
                    row_uniego_path = wpath
                    row_start, row_end = 0, int(feats.shape[0])

                key = _row_key(row_uniego_path, row_start, row_end, cap)
                if key in done_keys:
                    n_skip_resume += 1
                    continue
                done_keys.add(key)

                fout.write(json.dumps({
                    "uuid": uuid,
                    "uniego_path": row_uniego_path,
                    "start": int(row_start),
                    "end": int(row_end),
                    "caption": cap,
                    "ground_offset_y": None,   # BONES-SEED uniego is already floor-grounded
                    "source": "bones",
                }) + "\n")
                n_written += 1

    if mode != "a":
        os.replace(tmp_path, out_path)

    log.info(
        "[%s] %d rows -> %s | skipped: resume=%d short(<%d)=%d nonfinite=%d oob=%d missing=%d",
        split_name, n_written if mode != "a" else len(done_keys), out_path,
        n_skip_resume, args.min_frames, n_short, n_nonfinite, n_oob, n_missing,
    )
    return n_written


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=WEKA_DATA_ROOT,
                    help=f"283-D uniego tree (default weka; bs_dataset default={DATA_ROOT})")
    ap.add_argument("--train_split", default=TRAIN_SPLIT)
    ap.add_argument("--val_split", default=VAL_SPLIT)
    ap.add_argument("--natural_csv", default=WEKA_NATURAL_CSV)
    ap.add_argument("--temporal_labels", default=WEKA_TEMPORAL)
    ap.add_argument("--multi_timeline", default=WEKA_MULTI)
    ap.add_argument("--out_dir", default=os.path.join(RUNS_ROOT, "joint_attention"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_clip_sec", type=float, default=10.0,
                    help="front-cap each window to this many seconds (= max_frames/fps)")
    ap.add_argument("--min_frames", type=int, default=32,
                    help="drop windows shorter than this (matches Nymeria build_pairs)")
    ap.add_argument("--include_mirrored", action="store_true", default=True)
    ap.add_argument("--no_mirrored", dest="include_mirrored", action="store_false")
    ap.add_argument("--natural_caption_policy", choices=("desc4", "all"), default="desc4",
                    help="overview/natural caption policy: desc4 keeps only "
                         "content_natural_desc_4; all keeps desc_1..4")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping rows already present (keyed by uniego_path,start,end,caption)")
    ap.add_argument("--export_windows", default=None,
                    help="if set, materialize each window as a standalone npz under this dir "
                         "and point uniego_path at it (start=0,end=n)")
    ap.add_argument("--splits", default="train,val",
                    help="comma list of which splits to build")
    args = ap.parse_args()

    # Fall back to bs_dataset defaults if the weka metadata is absent (A100 box).
    for attr, fallback in [("natural_csv", NATURAL_CSV),
                           ("temporal_labels", TEMPORAL_JSONL),
                           ("multi_timeline", MULTI_JSONL)]:
        if not os.path.isfile(getattr(args, attr)) and os.path.isfile(fallback):
            log.warning("%s not found; falling back to bs_dataset default %s", attr, fallback)
            setattr(args, attr, fallback)
    if not os.path.isdir(args.data_root) and os.path.isdir(DATA_ROOT):
        log.warning("data_root %s absent; falling back to bs_dataset default %s",
                    args.data_root, DATA_ROOT)
        args.data_root = DATA_ROOT

    split_paths = {"train": args.train_split, "val": args.val_split}
    totals = {}
    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if name not in split_paths:
            raise ValueError(f"unknown split '{name}' (expected train/val)")
        totals[name] = build_split(name, split_paths[name], args)

    log.info("[done] %s", " ".join(f"{k}={v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
