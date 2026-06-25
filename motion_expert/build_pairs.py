"""Phase 0: build (text, motion_uniego) window pairs from the caption manifest + uniego_rep.

For each captioned, `usable` window in `manifest_video.jsonl`, emit a pair referencing the
uniego features slice `features[start_frame:end_frame]` of that sequence. Train/val split is
taken from the existing sequence-level `train_test_split.json` (NOT invented here).

Pair record (one JSON per line):
  {"uuid": "Sxx/seq", "uniego_path": ".../uniego_rep/Sxx/seq.npz",
   "start": int, "end": int, "caption": str}

Run (cosmos env):
  python motion_expert/build_pairs.py
Outputs:
  motion_expert/pairs_train.jsonl
  motion_expert/pairs_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np
from numpy.lib import format as npformat

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
UNIEGO_ROOT = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"
SPLIT_JSON = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"
HERE = os.path.dirname(os.path.abspath(__file__))


def npz_member_len(npz_path: str, key: str = "features") -> int:
    """Read the row count of `key` from an (uncompressed) .npz without decompressing data."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(key + ".npy") as f:
            npformat.read_magic(f)
            shape, _fortran, _dtype = npformat.read_array_header_1_0(f)
    return int(shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--uniego_root", default=UNIEGO_ROOT)
    ap.add_argument("--split", default=SPLIT_JSON)
    ap.add_argument("--min_frames", type=int, default=32, help="drop windows shorter than this")
    ap.add_argument("--out_dir", default=HERE)
    args = ap.parse_args()

    split = json.load(open(args.split))
    train_set, val_set = set(split["train"]), set(split["test"])

    len_cache: dict[str, int] = {}
    train_rows, val_rows = [], []
    n_win = n_nouniego = n_notsplit = n_short = n_oob = n_nocap = n_ambig = n_noground = 0

    for line in open(args.manifest):
        rec = json.loads(line)
        uuid = rec["uuid"]
        bucket = train_rows if uuid in train_set else (val_rows if uuid in val_set else None)
        if bucket is None:
            n_notsplit += len(rec.get("t2w_windows", []))
            continue
        npz = os.path.join(args.uniego_root, uuid + ".npz")
        if not os.path.isfile(npz):
            n_nouniego += len(rec.get("t2w_windows", []))
            continue
        if uuid not in len_cache:
            len_cache[uuid] = npz_member_len(npz)
        ulen = len_cache[uuid]

        for w in rec.get("t2w_windows", []):
            n_win += 1
            if not w.get("usable", False):
                continue
            # floor: require a ground offset; drop ambiguous (GT multi-level / estimated-unreliable)
            go = w.get("ground_offset_y")
            if go is None:
                n_noground += 1
                continue
            if w.get("ambiguous") or w.get("est_ambiguous"):
                n_ambig += 1
                continue
            cap = (w.get("caption") or "").strip()
            if not cap:
                n_nocap += 1
                continue
            s, e = int(w["start_frame"]), int(w["end_frame"])
            if e > ulen:                # uniego features must cover the window
                e = min(e, ulen)
            if e - s < args.min_frames:
                n_short += 1
                continue
            if s < 0 or e > ulen:
                n_oob += 1
                continue
            bucket.append({"uuid": uuid, "uniego_path": npz, "start": s, "end": e,
                           "caption": cap, "ground_offset_y": float(go)})

    for name, rows in [("train", train_rows), ("val", val_rows)]:
        out = os.path.join(args.out_dir, f"pairs_{name}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[pairs] {name}: {len(rows)} windows -> {out}")
    print(f"[pairs] manifest windows scanned (in-split, uniego-present): {n_win} | "
          f"dropped: not-in-split={n_notsplit} no-uniego={n_nouniego} no-ground={n_noground} "
          f"ambiguous={n_ambig} no-caption={n_nocap} short(<{args.min_frames})={n_short} oob={n_oob}")


if __name__ == "__main__":
    main()
