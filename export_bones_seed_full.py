#!/usr/bin/env python
"""Scalable, sharded FULL export of the kimodo BONES-SEED (raw_text, motion[T,369])
dataset for Cosmos3 training.

This is the streaming, memory-safe big brother of
``export_bones_seed_text_motion.py``. It produces the SAME on-disk contract the
trainer reads:

  features.npy  -- float32 [total_frames, 369], all clips concatenated along time
  index.json    -- {offsets:int64[N+1], texts, lengths, filenames, sources, meta}

Sample i = features[offsets[i]:offsets[i+1]] -> [T_i, 369].

Two differences vs the subset script, both required for the full set:

1. UNIQUE ENTRIES, NOT THE VIRTUAL ROUND-ROBIN.
   ``SOMABonesSeedDataset.__len__`` is a VIRTUAL 1:1:1 round-robin =
   ``3 * max(pool)`` (~1.45M) which DUPLICATES the shorter pools. For a static
   export we want each UNIQUE (source, caption, segment) once. We read the three
   internal pools ``ds._pools["natural"|"single"|"multi"]`` (lists of ``_BSEntry``),
   take their UNION, and dedup by ``(source, filename, seg_start_sec, seg_end_sec,
   text)``. Each unique entry is fetched ONCE via the same deterministic load
   pipeline ``__getitem__`` uses (canonicalize + normalize; no random heading /
   no random offset, exactly as the subset export configured the dataset).

2. STREAM TO DISK (memory-safe).
   We do NOT hold all motions in RAM. We append each clip's float32 bytes to a
   growing temp file ``features.raw`` while recording offsets/texts/lengths/...
   in Python lists. When done we know ``total_frames``; we then write a valid
   .npy v1.0 header for shape (total_frames, 369) and stream-copy the raw body
   after it, yielding a real ``features.npy`` that np.load(mmap_mode="r") opens.
   Peak RAM is one clip + the (small) Python metadata lists.

SHARDING for an sbatch array: ``--shard k --num-shards N`` makes this task export
unique entries ``[k::N]`` (strided) to ``<out>/shard_k/{features.npy,index.json}``.
``--max-samples`` caps per-shard count for testing. The index build is cached to
``--cache-index`` so array tasks after the first reuse it (the build is ~10-14 min).

Run in the kimodo env on a compute node (CPU-only is fine; no GPU needed):
  /home/jungbin_cho/miniforge3/envs/kimodo/bin/python export_bones_seed_full.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

KIMODO_ROOT = "/home/jungbin_cho/kimodo_open"
if KIMODO_ROOT not in sys.path:
    sys.path.insert(0, KIMODO_ROOT)

import torch  # noqa: E402
from kimodo.data import SOMABonesSeedDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Paths -- identical to the verified subset exporter (the /weka fixes).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/jungbin/seed/soma_uniform_motions_20fps"
NATURAL_CSV_PATH = "/weka/jungbin/seed/metadata/seed_metadata_v004.csv"
TEMPORAL_LABELS_PATH = "/weka/jungbin/seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
MULTI_TIMELINE_PATH = "/weka/jungbin/seed/multi_timeline.jsonl"
TRAIN_SPLIT_PATH = "/weka/jungbin/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt"
STATS_PATH = "/weka/jungbin/seed/stats/soma_uniform_motions_20fps/"

FPS = 20
MIN_FRAMES = 10
MAX_CLIP_SEC = 10.0       # -> max_frames = 200
MAX_SEGMENT_SEC = 15.0
INCLUDE_MIRRORED = True

DEFAULT_OUT = "/weka/jungbin/seed/cosmos_text_motion_full"
DEFAULT_CACHE = "/weka/jungbin/seed/cosmos_text_motion_full/bones_seed_index_cache.json"

LAYOUT = "smooth_root3|heading2|jpos90|rot6d180|vel90|footc4"
DIM = 369


def build_dataset(seed: int, cache_index: str | None) -> SOMABonesSeedDataset:
    """Build the BONES-SEED dataset, deterministic (no aug / no random offset).

    Same args as the subset exporter. ``cache_index`` lets array tasks after the
    first reuse the (~10-14 min) pool build.
    """
    return SOMABonesSeedDataset(
        data_root=DATA_ROOT,
        natural_csv_path=NATURAL_CSV_PATH,
        temporal_labels_path=TEMPORAL_LABELS_PATH,
        multi_timeline_path=MULTI_TIMELINE_PATH,
        train_split_path=TRAIN_SPLIT_PATH,
        fps=FPS,
        max_clip_sec=MAX_CLIP_SEC,
        max_segment_sec=MAX_SEGMENT_SEC,
        rand_offset_max_sec=0.0,        # deterministic window
        min_frames=MIN_FRAMES,
        include_mirrored=INCLUDE_MIRRORED,
        stats_path=STATS_PATH,
        normalize=True,
        random_heading_aug=False,       # deterministic features
        cache_index=cache_index,
        seed=seed,
    )


def enumerate_unique_entries(ds: SOMABonesSeedDataset) -> list:
    """Union of the 3 pools, each unique (source,file,seg,text) entry ONCE.

    Order is natural ++ single ++ multi (stable within each pool). Dedup key
    includes ``source`` so the same window described identically under two
    sources is still kept once per source (they are genuinely distinct training
    captions/segments; identical only if every field matches).
    """
    seen: set = set()
    uniq: list = []
    for src in ("natural", "single", "multi"):
        for e in ds._pools[src]:
            key = (e.source, e.filename, round(float(e.seg_start_sec), 6),
                   round(float(e.seg_end_sec), 6), e.text)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
    return uniq


def fetch_entry(ds: SOMABonesSeedDataset, entry) -> np.ndarray | None:
    """Load ONE _BSEntry through the same deterministic pipeline __getitem__ uses.

    canonicalize + (no random heading) + normalize. Returns float32 [T,369], or
    None if the segment is too short (mirrors __getitem__'s skip-and-advance).
    """
    local_rot, root_pos, T = ds._load_segment(entry)
    if T < ds.min_frames:
        return None
    features = ds._features_with_canonicalize(local_rot, root_pos, T)
    features, _ = ds._apply_random_heading(features)  # no-op (aug off)
    if ds.normalize:
        features = ds.motion_rep.normalize(features)
    m = features.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(m, dtype=np.float32)


def finalize_npy(raw_path: Path, npy_path: Path, total_frames: int) -> None:
    """Concatenate (valid .npy v1.0 header for shape (total_frames,DIM)) || raw.

    Memory-safe: the header is tiny; the body is stream-copied with shutil.
    """
    from numpy.lib import format as npformat

    header_meta = {
        "descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
        "fortran_order": False,
        "shape": (int(total_frames), DIM),
    }
    tmp_npy = npy_path.with_suffix(".npy.tmp")
    with open(tmp_npy, "wb") as fout:
        npformat.write_array_header_1_0(fout, header_meta)
        with open(raw_path, "rb") as fin:
            shutil.copyfileobj(fin, fout, length=64 * 1024 * 1024)
    os.replace(tmp_npy, npy_path)
    os.remove(raw_path)


def export_shard(ds, uniq, shard: int, num_shards: int, out_dir: Path,
                 max_samples: int) -> dict:
    """Stream-export this shard's strided slice [shard::num_shards] to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "features.raw"
    npy_path = out_dir / "features.npy"
    index_path = out_dir / "index.json"

    my_entries = uniq[shard::num_shards]
    if max_samples is not None and max_samples >= 0:
        my_entries = my_entries[:max_samples]
    n_planned = len(my_entries)
    print(f"[shard {shard}/{num_shards}] {n_planned} unique entries assigned "
          f"(of {len(uniq)} total)", flush=True)

    offsets = [0]
    texts: list[str] = []
    lengths: list[int] = []
    filenames: list[str] = []
    sources: list[str] = []

    total_frames = 0
    n_written = 0
    n_skipped = 0
    t0 = time.time()
    with open(raw_path, "wb") as raw_f:
        for i, entry in enumerate(my_entries):
            m = fetch_entry(ds, entry)
            if m is None:
                n_skipped += 1
                continue
            assert m.shape[1] == DIM, f"dim {m.shape[1]} != {DIM}"
            T = int(m.shape[0])
            raw_f.write(m.tobytes(order="C"))
            total_frames += T
            offsets.append(total_frames)
            texts.append(str(entry.text))
            lengths.append(T)
            filenames.append(str(entry.filename))
            sources.append(str(entry.source))
            n_written += 1

            if (i + 1) % 1000 == 0:
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-9)
                eta = (n_planned - (i + 1)) / max(rate, 1e-9)
                print(f"[shard {shard}] {i+1}/{n_planned} processed "
                      f"({n_written} written, {n_skipped} skipped, "
                      f"{total_frames} frames) {rate:.1f}/s ETA {eta/60:.1f}m",
                      flush=True)

    print(f"[shard {shard}] streaming done: {n_written} written, "
          f"{n_skipped} skipped, {total_frames} frames. Finalizing npy...",
          flush=True)
    finalize_npy(raw_path, npy_path, total_frames)

    index = {
        "offsets": offsets,                       # list[int], len N+1
        "texts": texts,
        "lengths": lengths,
        "filenames": filenames,
        "sources": sources,
        "meta": {
            "fps": FPS,
            "dim": DIM,
            "normalized": True,
            "layout": LAYOUT,
            "source_dataset": "bones_seed soma_uniform_motions_20fps",
            "shard": shard,
            "num_shards": num_shards,
            "unique_entries_total": len(uniq),
        },
    }
    with open(index_path, "w") as f:
        json.dump(index, f)

    size_mb = npy_path.stat().st_size / 1e6
    print(f"[shard {shard}] DONE  N={n_written}  total_frames={total_frames}  "
          f"features.npy={size_mb:.1f}MB  -> {out_dir}", flush=True)
    return {
        "shard": shard, "n": n_written, "skipped": n_skipped,
        "total_frames": total_frames, "size_mb": size_mb, "dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="parent output dir; shard writes to <out>/shard_<k>/")
    ap.add_argument("--shard", type=int, default=0, help="this shard index k")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards N")
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="cap entries per shard (testing); -1 = all")
    ap.add_argument("--cache-index", default=DEFAULT_CACHE,
                    help="JSON path to cache/reuse the pool index ('' to disable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-subdir", default=None,
                    help="override shard subdir name (default shard_<k>)")
    args = ap.parse_args()

    assert 0 <= args.shard < args.num_shards, "shard must be in [0, num_shards)"

    out_parent = Path(args.out)
    out_parent.mkdir(parents=True, exist_ok=True)
    cache_index = args.cache_index if args.cache_index else None
    if cache_index:
        Path(cache_index).parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] building SOMABonesSeedDataset "
          f"(cache_index={cache_index})...", flush=True)
    t0 = time.time()
    ds = build_dataset(args.seed, cache_index)
    print(f"[export] dataset built in {time.time()-t0:.1f}s; "
          f"pools natural={len(ds._pools['natural'])} "
          f"single={len(ds._pools['single'])} multi={len(ds._pools['multi'])} "
          f"virtual_len={len(ds)}", flush=True)

    uniq = enumerate_unique_entries(ds)
    print(f"[export] unique entries (union, deduped) = {len(uniq)}", flush=True)

    subdir = args.shard_subdir or f"shard_{args.shard}"
    out_dir = out_parent / subdir
    stats = export_shard(ds, uniq, args.shard, args.num_shards, out_dir,
                         args.max_samples)

    # quick self-verify: mmap-open the result.
    feat = np.load(out_dir / "features.npy", mmap_mode="r")
    assert feat.shape == (stats["total_frames"], DIM), \
        f"npy shape {feat.shape} != ({stats['total_frames']},{DIM})"
    print(f"[export] verified features.npy mmap shape={feat.shape} dtype={feat.dtype}",
          flush=True)


if __name__ == "__main__":
    main()
