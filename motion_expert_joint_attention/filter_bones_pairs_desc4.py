"""Filter BONES pair JSONLs so overview/natural uses only content_natural_desc_4.

This is a fast post-process for already-built/validated BONES pair files. It keeps all
single-timeline and multi-timeline rows, and removes only natural/overview rows whose caption is
not the fourth overview caption for that filename.
"""
from __future__ import annotations

import argparse
import csv
import json
import os


WEKA_RUNS = "/weka/jungbin/cosmos_motion_ft_runs/joint_attention"
WEKA_NATURAL_CSV = "/weka/jungbin/seed/metadata/seed_metadata_v004.csv"


def _row_key(row: dict) -> tuple[str, int, int, str]:
    return (
        row["uniego_path"],
        int(row["start"]),
        int(row["end"]),
        row["caption"],
    )


def _desc4_by_file(natural_csv: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(natural_csv, newline="") as f:
        for row in csv.DictReader(f):
            desc4 = (row.get("content_natural_desc_4") or "").strip()
            if desc4:
                out[row["filename"]] = desc4
    return out


def _natural_drop_keys(index_json: str, natural_csv: str) -> set[tuple[str, int, int, str]]:
    """Keys for natural rows that should be removed from an already-built JSONL."""
    desc4 = _desc4_by_file(natural_csv)
    raw = json.load(open(index_json))
    drop: set[tuple[str, int, int, str]] = set()
    for entry in raw.get("natural", []):
        text = (entry.get("text") or "").strip()
        if not text or desc4.get(entry["filename"]) == text:
            continue
        sf = int(round(float(entry["seg_start_sec"]) * 20))
        ef = int(round(float(entry["seg_end_sec"]) * 20))
        # build_bones_pairs.py clamps/caps these before writing. Natural rows always start at 0;
        # the end may be capped to max_clip_sec=10s -> 200 frames.
        sf = max(0, sf)
        ef = min(max(sf, ef), sf + 200)
        drop.add((entry["motion_path"], sf, ef, text))
    return drop


def filter_split(split: str, src: str, dst: str, index_json: str, natural_csv: str) -> None:
    drop = _natural_drop_keys(index_json, natural_csv)
    tmp = dst + f".desc4tmp.{os.getpid()}"
    n_in = n_out = n_drop = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if _row_key(row) in drop:
                n_drop += 1
                continue
            fout.write(json.dumps(row) + "\n")
            n_out += 1
    os.replace(tmp, dst)
    print(f"[{split}] in={n_in} out={n_out} dropped_natural_non_desc4={n_drop} -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default=WEKA_RUNS)
    ap.add_argument("--natural_csv", default=WEKA_NATURAL_CSV)
    ap.add_argument("--backup_suffix", default="all4_before_desc4_20260706",
                    help="source file suffix: bones_pairs_<split>.<suffix>.jsonl")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = os.path.join(args.runs_dir, f"bones_pairs_{split}.{args.backup_suffix}.jsonl")
        dst = os.path.join(args.runs_dir, f"bones_pairs_{split}.jsonl")
        index_json = os.path.join(args.runs_dir, f"bones_index_{split}.json")
        filter_split(split, src, dst, index_json, args.natural_csv)


if __name__ == "__main__":
    main()
