"""Run floor-height extraction across ALL NymeriaPlus subjects.

Thin driver over `extract_floor.process_seq` (which is per-sequence and already
encapsulates floor resolution + multi-floor majority vote + root validation). For
each subject `<Sxx>` under `--nymeria-root` that has downloaded object boxes
(`<seq>/objects/boxy/`), this:
  - runs `process_seq` for every sequence (root validation uses each seq's co-located
    SOMA fit `<seq>/body/xdata_soma.npz`; falls back to a kimodo motion NPZ under
    `--motion-root-base/<Sxx>` only when no SOMA fit exists),
  - writes the same per-subject `<Sxx>_floor.{json,csv}` as `extract_floor.py`,
  - emits combined `all_floor.{json,csv}` across every subject.

SOMA fits cover ~all sequences across all subjects, so nearly every sequence gets
root validation. A seq with neither a SOMA fit nor a kimodo motion gets a floor
height with `validation = null` (and multi-floor falls back to largest footprint).

USAGE
  python extract_floor_all.py \
      --nymeria-root     /weka/jungbin/nymeriaplus \
      --out-dir          /weka/jungbin/nymeriaplus_kimodo/floor
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

from extract_floor import process_seq, _fmt

CSV_COLS = ["subject", "seq", "status", "n_floor_instances", "primary_floor_uid",
            "floor_z", "floor_surface_z", "root_dz_min", "root_dz_mean",
            "root_dz_max", "validation_ok", "perframe_dz_min", "perframe_dz_mean",
            "perframe_dz_max", "ok_perframe"]


def _row(subj: str, r: dict) -> list:
    v = r.get("validation") or {}
    return [subj, r["seq"], r["status"], r.get("n_floor_instances", ""),
            r.get("primary_floor_uid", ""),
            _fmt(r.get("floor_z")), _fmt(r.get("floor_surface_z")),
            _fmt(v.get("root_dz_min")), _fmt(v.get("root_dz_mean")),
            _fmt(v.get("root_dz_max")), v.get("ok", ""),
            _fmt(v.get("perframe_dz_min")), _fmt(v.get("perframe_dz_mean")),
            _fmt(v.get("perframe_dz_max")), v.get("ok_perframe", "")]


def write_csv(path: Path, rows: list[list]):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_COLS)
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nymeria-root", type=Path, default=Path("/weka/jungbin/nymeriaplus"))
    ap.add_argument("--motion-root-base", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/motions"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/floor"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name for p in args.nymeria_root.glob("S*")
                      if p.is_dir() and any(p.glob("*/objects/boxy")))
    print(f"[scan] {len(subjects)} subjects with object boxes: {subjects}")

    all_records: list[dict] = []
    all_rows: list[list] = []
    for subj in subjects:
        objects_root = args.nymeria_root / subj
        motion_root = args.motion_root_base / subj
        seqs = sorted({p.parent.parent.name
                       for p in objects_root.glob("*/objects/boxy")})
        recs = [process_seq(s, objects_root, motion_root) for s in seqs]

        json.dump(recs, open(args.out_dir / f"{subj}_floor.json", "w"), indent=2)
        write_csv(args.out_dir / f"{subj}_floor.csv", [_row(subj, r) for r in recs])

        ok = sum(1 for r in recs if r["status"] == "ok")
        val_ok = sum(1 for r in recs if (r.get("validation") or {}).get("ok"))
        val_pf = sum(1 for r in recs if (r.get("validation") or {}).get("ok_perframe"))
        val_n = sum(1 for r in recs if r.get("validation"))
        multi = sum(1 for r in recs if r.get("n_floor_instances", 0) > 1)
        print(f"  {subj}: {len(seqs):3d} seq | floor {ok:3d} | multi {multi:2d} "
              f"| val_ok {val_ok}/{val_n} | perframe {val_pf}/{val_n}")

        for r in recs:
            r2 = {"subject": subj, **r}
            all_records.append(r2)
            all_rows.append(_row(subj, r))

    json.dump(all_records, open(args.out_dir / "all_floor.json", "w"), indent=2)
    write_csv(args.out_dir / "all_floor.csv", all_rows)

    n = len(all_records)
    ok = sum(1 for r in all_records if r["status"] == "ok")
    multi = sum(1 for r in all_records if r.get("n_floor_instances", 0) > 1)
    val_ok = sum(1 for r in all_records if (r.get("validation") or {}).get("ok"))
    val_pf = sum(1 for r in all_records if (r.get("validation") or {}).get("ok_perframe"))
    val_n = sum(1 for r in all_records if r.get("validation"))
    no_floor = [f"{r['subject']}/{r['seq']}" for r in all_records
                if r["status"] != "ok"]
    print(f"\n=== ALL subjects ===")
    print(f"  subjects        : {len(subjects)}")
    print(f"  sequences       : {n}")
    print(f"  with floor      : {ok}/{n}")
    print(f"  multi-floor     : {multi}")
    print(f"  root val OK     : {val_ok}/{val_n} (single-floor)")
    print(f"  per-frame val OK: {val_pf}/{val_n} (multi-level aware)")
    print(f"  no floor        : {len(no_floor)} {no_floor[:10]}"
          f"{' ...' if len(no_floor) > 10 else ''}")
    print(f"\nwrote {args.out_dir}/all_floor.json and .csv (+ per-subject files)")


if __name__ == "__main__":
    main()
