"""Per-atomic-action-slice GT floor height for NymeriaPlus -> kimodo.

For every atomic_action slice (from metadata_atomic_action.jsonl), decide which GT
floor the slice is on and emit the grounding offset, so downstream grounding uses the
real room floor instead of the per-sequence min-foot heuristic (which mis-aligns
sitting/lying motions against the egocentric video).

Reuses floor/extract_floor.py: `_per_frame_floor`, `pick_primary`, `SUPPORT_TOL`.
Floor geometry comes from the already-extracted all_floor.json (Z-up SLAM frame).
The root trajectory per slice is the kimodo NPZ `root_positions` (Y-up), converted
back to Z-up for the footprint/below tests:

    kimodo (x,y,z) = (soma_x, soma_z, -soma_y)   =>   soma = (kx, -kz, ky)

The grounding offset is `floor_surface_z = floor_z + half_thickness` (Z-up). Because
kimodo Y == SOMA Z, this scalar is exactly what to subtract from kimodo
`root_positions[:,1]` to put that slice's floor at y=0 (Y only; xz untouched).

Multi-floor (e.g. stairs) slices are FLAGGED (`ambiguous=true`) and grounded to the
MAJORITY floor (the one supporting the most frames in the slice).

Output: metadata_atomic_action_floor.jsonl (one row per slice) + per-subject dir,
mirroring the atomic_action metadata layout.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

# reuse the floor helpers (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_floor import _per_frame_floor, SUPPORT_TOL  # noqa: E402

AMBIG_FRAC = 0.20  # a floor LEVEL counts as "present" in a slice if it supports >= this frac of grounded frames
LEVEL_TOL = 0.15   # floor boxes whose surfaces are within this (m) are the SAME physical level
                   # (the dataset tiles one floor with several boxes at identical height; without
                   #  this, a single-floor slice spanning >=2 boxes was falsely flagged ambiguous)

DEFAULT_META = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action.jsonl")
DEFAULT_FLOOR = Path("/weka/jungbin/nymeriaplus_kimodo/floor/all_floor.json")
DEFAULT_MOTION_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
DEFAULT_OUT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl")


def kimodo_root_to_soma(root_y: np.ndarray) -> np.ndarray:
    """(T,3) kimodo Y-up root -> (T,3) SOMA Z-up: soma = (kx, -kz, ky)."""
    return np.column_stack([root_y[:, 0], -root_y[:, 2], root_y[:, 1]])


def assign_slice_floor(boxes: list[dict], roots_soma: np.ndarray) -> dict:
    """Majority GT floor for a slice's Z-up root trajectory + ambiguity info."""
    cand_z, has_floor = _per_frame_floor(boxes, roots_soma)
    n = len(roots_soma)
    support_frac = float(has_floor.mean()) if n else 0.0
    if not has_floor.any():
        return {"floor_status": "no_support", "support_frac": support_frac}

    nearest = np.argmax(cand_z[has_floor], axis=1)            # idx into boxes per grounded frame
    counts = np.bincount(nearest, minlength=len(boxes)).astype(float)
    grounded = counts.sum()
    fracs = counts / grounded

    def surf(j):
        return float(boxes[j]["floor_z"] + boxes[j]["half_thickness"])

    # per-box detail (kept for inspection / back-compat)
    floors = [{"uid": boxes[j]["uid"], "floor_surface_z": surf(j),
               "frac_frames": round(float(fracs[j]), 4)}
              for j in range(len(boxes)) if fracs[j] > 0.0]
    floors.sort(key=lambda f: -f["frac_frames"])

    # group boxes into LEVELS by surface height (so same-floor tiling != ambiguous)
    idxs = sorted((j for j in range(len(boxes)) if fracs[j] > 0.0), key=surf)
    groups: list[dict] = []
    for j in idxs:
        if groups and abs(surf(j) - groups[-1]["anchor"]) <= LEVEL_TOL:
            groups[-1]["boxes"].append(j); groups[-1]["frac"] += float(fracs[j])
        else:
            groups.append({"anchor": surf(j), "boxes": [j], "frac": float(fracs[j])})
    groups.sort(key=lambda g: -g["frac"])
    maj_group = groups[0]
    rep = max(maj_group["boxes"], key=lambda j: fracs[j])     # representative box of the majority level
    surface = surf(rep)
    present_levels = [g for g in groups if g["frac"] >= AMBIG_FRAC]
    levels = [{"surface_z": round(float(np.mean([surf(j) for j in g["boxes"]])), 5),
               "frac_frames": round(g["frac"], 4),
               "uids": [boxes[j]["uid"] for j in g["boxes"]]}
              for g in groups]

    b = boxes[rep]
    return {
        "floor_status": "ok",
        "floor_uid": b["uid"],
        "floor_z": float(b["floor_z"]),
        "half_thickness": float(b["half_thickness"]),
        "floor_surface_z": surface,
        "ground_offset_y": surface,                 # kimodo root_positions[:,1] -= this (majority LEVEL)
        "ambiguous": bool(len(present_levels) > 1),  # >=2 distinct floor LEVELS, not boxes
        "n_floors_in_slice": int(len(present_levels)),
        "support_frac": support_frac,
        "floors": floors,                            # per-box
        "levels": levels,                            # per-physical-level (height-grouped)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--floor-json", type=Path, default=DEFAULT_FLOOR)
    ap.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--subjects", nargs="*", default=None, help="limit to these S-codes")
    args = ap.parse_args()

    floor_recs = {r["seq"]: r for r in json.load(open(args.floor_json))}
    print(f"[floor] {len(floor_recs)} sequences in {args.floor_json.name}")

    slices = [json.loads(ln) for ln in open(args.meta) if ln.strip()]
    if args.subjects:
        slices = [s for s in slices if s["subject"] in set(args.subjects)]
    # group by (subject, filename) to load each NPZ once
    by_seq: dict[tuple, list] = {}
    for s in slices:
        by_seq.setdefault((s["subject"], s["filename"]), []).append(s)
    print(f"[slices] {len(slices)} atomic_action slices over {len(by_seq)} sequences")

    rows = []
    stats = {"ok": 0, "no_floor": 0, "no_support": 0, "ambiguous": 0, "no_motion": 0}
    root_cache: dict[str, np.ndarray] = {}
    for (subj, fn), seq_slices in by_seq.items():
        rec = floor_recs.get(fn)
        npz = args.motion_root / subj / f"{fn}.npz"
        roots_soma = None
        if npz.is_file():
            d = np.load(npz, allow_pickle=True)
            roots_soma = kimodo_root_to_soma(d["root_positions"].astype(np.float64))
        for s in seq_slices:
            base = {k: s[k] for k in ("filename", "subject", "start_frame", "end_frame", "text")}
            if rec is None or rec.get("status") != "ok" or not rec.get("floor_instances"):
                rows.append({**base, "floor_status": "no_floor"}); stats["no_floor"] += 1; continue
            if roots_soma is None:
                rows.append({**base, "floor_status": "no_motion"}); stats["no_motion"] += 1; continue
            sf = max(0, int(s["start_frame"])); ef = min(len(roots_soma), int(s["end_frame"]))
            if ef - sf < 1:
                rows.append({**base, "floor_status": "no_support", "support_frac": 0.0})
                stats["no_support"] += 1; continue
            info = assign_slice_floor(rec["floor_instances"], roots_soma[sf:ef])
            rows.append({**base, **info})
            stats[info["floor_status"]] = stats.get(info["floor_status"], 0) + 1
            if info.get("ambiguous"):
                stats["ambiguous"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # per-subject
    per_dir = args.out.parent / "metadata_per_subject_atomic_action_floor"
    per_dir.mkdir(parents=True, exist_ok=True)
    by_subj: dict[str, list] = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    for subj, rs in by_subj.items():
        with open(per_dir / f"{subj}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    print(f"[done] wrote {len(rows)} rows -> {args.out}")
    print(f"  status: ok={stats['ok']} no_floor={stats['no_floor']} "
          f"no_support={stats['no_support']} no_motion={stats['no_motion']}  "
          f"ambiguous(among ok)={stats['ambiguous']}")


if __name__ == "__main__":
    main()
