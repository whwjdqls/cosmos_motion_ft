"""Build kimodo-style segment metadata JSONL from NymeriaPlus narration CSVs.

For every converted kimodo motion NPZ (in --motion-root), find the matching
sequence's narration/ dir. Parse each CSV present, map narration seconds to
body-frame indices, emit one JSONL row per (segment, narration-type).

JSONL row schema (one per segment):
  {
    "filename":       <kimodo motion stem, e.g. "20230710_s0_barbara_norman_act4_ebaqa8">,
    "subject":        "S11",
    "source":         "atomic_action" | "activity_summarization" | "motion_narration",
    "seg_start_sec":  float,  # body-frame-local seconds (i.e. (ts_us - ts_us[0])/1e6)
    "seg_end_sec":    float,
    "start_frame":    int,    # frame index into the 20-fps motion NPZ
    "end_frame":      int,
    "text":           str,    # primary caption; for motion_narration this is the joined 4 columns
    "extras":         { ... } # any extra columns (e.g. annotator id)
  }

Sequences with no narration (94 empty zips) get NO rows emitted; kimodo's
loader can either filter to motion+image-only training or skip them entirely.

Outputs:
  --out-dir/metadata_all.jsonl         # all rows from all sequences (every narration source)
  --out-dir/metadata_per_subject/*.jsonl
  --out-dir/coverage_summary.json      # per-subject row counts + sequences with no narration
  --out-dir/metadata_atomic_action.jsonl              # source=="atomic_action" subset
  --out-dir/metadata_per_subject_atomic_action/*.jsonl
  --out-dir/coverage_atomic_action.json
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def find_narration_csvs(seq_dir: Path) -> dict[str, Path]:
    d = seq_dir / "narration"
    if not d.is_dir():
        return {}
    out = {}
    for nm in ["atomic_action.csv", "activity_summarization.csv", "motion_narration.csv"]:
        p = d / nm
        if p.is_file():
            out[nm[:-4]] = p
    return out


# motion_narration has 4 description columns; we join them.
MN_COLS = [
    "Describe my body posture",
    "Describe my hands/arms motion",
    "Describe my legs/feet motion",
    "Describe my focus attention",
]


def parse_csv_rows(p: Path, source: str) -> list[dict]:
    rows = list(csv.DictReader(open(p)))
    out = []
    for r in rows:
        try:
            ss = float(r["start_time"])
            es = float(r["end_time"])
        except (KeyError, ValueError):
            continue
        if source == "motion_narration":
            parts = [r.get(c, "").strip() for c in MN_COLS]
            text = " | ".join(p for p in parts if p)
        elif source == "atomic_action":
            text = r.get("Describe my atomic actions", "").strip()
        elif source == "activity_summarization":
            text = r.get("Describe my activity", "").strip()
        else:
            text = ""
        extras = {k: r[k] for k in ("annotator", "request_id", "gaia_id") if k in r}
        out.append({"start_sec_abs": ss, "end_sec_abs": es, "text": text, "extras": extras})
    return out


def align_to_body(rows: list[dict], body_ts_us: np.ndarray) -> list[dict]:
    """Map narration time (in its own epoch, seconds) to body frame indices.

    Treats both timelines as having dt-consistent clocks with possibly
    different starts. Uses the per-CSV-collection's first row's start_time
    as the narration epoch zero.
    """
    if not rows:
        return []
    narr_zero_us = int(rows[0]["start_sec_abs"] * 1e6)
    body_start_us = int(body_ts_us[0])
    body_end_us = int(body_ts_us[-1])
    out = []
    for r in rows:
        rel_start_us = int(r["start_sec_abs"] * 1e6) - narr_zero_us
        rel_end_us = int(r["end_sec_abs"] * 1e6) - narr_zero_us
        target_start_us = body_start_us + rel_start_us
        target_end_us = body_start_us + rel_end_us
        # clip to body range
        if target_end_us < body_start_us or target_start_us > body_end_us:
            continue
        target_start_us = max(target_start_us, body_start_us)
        target_end_us = min(target_end_us, body_end_us)
        sf = int(np.searchsorted(body_ts_us, target_start_us))
        ef = int(np.searchsorted(body_ts_us, target_end_us))
        sf = max(0, min(sf, len(body_ts_us) - 1))
        ef = max(0, min(ef, len(body_ts_us) - 1))
        if ef <= sf:
            continue
        out.append({
            "seg_start_sec": round((target_start_us - body_start_us) / 1e6, 3),
            "seg_end_sec": round((target_end_us - body_start_us) / 1e6, 3),
            "start_frame": sf,
            "end_frame": ef,
            "text": r["text"],
            "extras": r["extras"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo_proportional"))
    ap.add_argument("--nymeria-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo_proportional/metadata"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata_per_subject").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata_per_subject_atomic_action").mkdir(parents=True, exist_ok=True)
    all_jsonl_p = args.out_dir / "metadata_all.jsonl"
    atomic_jsonl_p = args.out_dir / "metadata_atomic_action.jsonl"

    seqs = sorted(args.motion_root.glob("S*/*.npz"))
    # skip the batch summary file
    seqs = [p for p in seqs if p.name != "_batch_summary.json"]
    print(f"[scan] {len(seqs)} kimodo motion NPZs under {args.motion_root}")

    counters_per_subj = defaultdict(Counter)
    atomic_per_subj = Counter()
    seqs_no_narration = []
    seqs_with = 0
    total_rows = 0
    total_atomic = 0

    per_subj_files = {}
    atomic_subj_files = {}
    all_f = open(all_jsonl_p, "w")
    atomic_f = open(atomic_jsonl_p, "w")

    for npz_path in seqs:
        subj = npz_path.parent.name  # e.g. "S11"
        seq_name = npz_path.stem
        seq_dir = args.nymeria_root / subj / seq_name
        if not seq_dir.is_dir():
            seqs_no_narration.append((subj, seq_name, "seq_dir_missing"))
            continue

        # Load timestamps
        with np.load(npz_path) as z:
            body_ts_us = z["timestamps_us"].astype(np.int64)

        csvs = find_narration_csvs(seq_dir)
        if not csvs:
            seqs_no_narration.append((subj, seq_name, "no_narration_dir_or_empty"))
            continue

        seqs_with += 1
        # output file handles per subject
        if subj not in per_subj_files:
            per_subj_files[subj] = open(args.out_dir / "metadata_per_subject" / f"{subj}.jsonl", "w")
        subj_f = per_subj_files[subj]

        for source, p in csvs.items():
            raw_rows = parse_csv_rows(p, source)
            aligned = align_to_body(raw_rows, body_ts_us)
            counters_per_subj[subj][source] += len(aligned)
            total_rows += len(aligned)
            for r in aligned:
                row = {
                    "filename": seq_name,
                    "subject": subj,
                    "source": source,
                    **r,
                }
                line = json.dumps(row) + "\n"
                all_f.write(line)
                subj_f.write(line)
                if source == "atomic_action":
                    atomic_f.write(line)
                    if subj not in atomic_subj_files:
                        atomic_subj_files[subj] = open(
                            args.out_dir / "metadata_per_subject_atomic_action" / f"{subj}.jsonl", "w")
                    atomic_subj_files[subj].write(line)
                    atomic_per_subj[subj] += 1
                    total_atomic += 1

    all_f.close()
    atomic_f.close()
    for f in per_subj_files.values():
        f.close()
    for f in atomic_subj_files.values():
        f.close()

    # coverage summary
    coverage = {
        "total_motion_npzs": len(seqs),
        "sequences_with_narration": seqs_with,
        "sequences_without_narration": len(seqs_no_narration),
        "total_rows_emitted": total_rows,
        "rows_per_subject_per_source": {
            s: dict(c) for s, c in counters_per_subj.items()
        },
        "sequences_without_narration_examples": seqs_no_narration[:20],
    }
    json.dump(coverage, open(args.out_dir / "coverage_summary.json", "w"), indent=2)

    atomic_coverage = {
        "total_rows_atomic_action": total_atomic,
        "rows_per_subject": dict(sorted(atomic_per_subj.items())),
    }
    json.dump(atomic_coverage, open(args.out_dir / "coverage_atomic_action.json", "w"), indent=2)

    print(f"\n=== done ===")
    print(f"  motion NPZs           : {len(seqs)}")
    print(f"  with narration        : {seqs_with}")
    print(f"  without narration     : {len(seqs_no_narration)}")
    print(f"  total rows emitted    : {total_rows}")
    print(f"  atomic_action rows    : {total_atomic}")
    print(f"\nper-subject row counts (by source):")
    for s, c in sorted(counters_per_subj.items()):
        print(f"  {s}: {dict(c)}")
    print(f"\nwrote {all_jsonl_p}")
    print(f"wrote {args.out_dir}/coverage_summary.json")


if __name__ == "__main__":
    main()
