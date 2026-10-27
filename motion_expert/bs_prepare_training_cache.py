"""Build the unchanged POC training index and enumerate its exact raw captions."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import socket

from bs_dataset import BonesSeedUniegoDataset, NATURAL_CSV, MEAN_PATH, STD_PATH
from bs_normalization import load_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--viz-split", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--captions", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("POC preparation must run on an owned compute allocation")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, _, normalization = load_normalization(MEAN_PATH, STD_PATH)
    print(f"[prepare] normalization={json.dumps(normalization, sort_keys=True)}", flush=True)
    dataset = BonesSeedUniegoDataset(args.train_split, cache_index=args.index, seed=0)
    captions = {""}
    counts = {}
    for source, entries in dataset._pools.items():
        counts[source] = len(entries)
        captions.update(entry.text for entry in entries)
    # Include held-out raw natural captions for deterministic in-training visuals.
    # No caption rewriting or model-based conditioning is introduced here.
    names = {Path(line.strip()).name for line in Path(args.viz_split).read_text().splitlines()
             if line.strip() and not line.strip().endswith("_M")}
    with open(NATURAL_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            caption = (row.get("content_natural_desc_1") or "").strip()
            if row["filename"] in names and caption:
                captions.add(caption)
    target = Path(args.captions)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(captions)) + "\n")
    temporary.replace(target)
    report = {"pools": counts, "unique_captions": len(captions),
              "train_split": args.train_split, "viz_split": args.viz_split,
              "index": args.index, "normalization": normalization}
    target.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[prepare] {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
