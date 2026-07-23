"""Extract per-sequence floor height from NymeriaPlus 3D object bounding boxes.

For each sequence under --objects-root that has `objects/boxy/` (downloaded by
`fetch_objects.py`), find the annotated **Floor** instance(s) and report the
ground-plane height in the SLAM world frame, then validate against the motion
root trajectory.

WORLD FRAME / "UP" AXIS
-----------------------
The NymeriaPlus SLAM world frame is gravity-aligned with **+z up**. This is
confirmed empirically here: every Floor box's plane normal lies along world z
(|n_z| > 0.9), and the body root (Hips) trajectory sits *above* the floor in z.
We therefore take height = world-z, and `floor_z = floor box center z`. (The
floor slab is ~0.12 m thick; `floor_surface_z = floor_z + half_thickness`.)

FLOOR RESOLUTION
----------------
The floor is NOT always object_uid 0. We resolve floor instances via
`instances.json` (category == "Floor"). A sequence may contain several Floor
instances at different heights (multi-level capture); we pick the *supporting*
floor for the actor by majority vote over per-frame assignment:
  per frame, the supporting floor = the nearest floor *below the hip* (highest
  floor_z with root_z >= floor_z - SUPPORT_TOL) whose world-XY footprint contains
  the root; the primary floor is the one supporting the most frames.
This correctly handles STACKED floors (overlapping XY footprints, different z):
a higher level only wins on frames where the hip is actually up on it, so XY
overlap alone never decides which level the actor is on. Fallback when no motion
is available: the floor with the largest footprint. All floor instances and
their heights are still emitted so callers can do per-frame assignment if needed.

INPUTS
  objects: <objects-root>/<seq>/objects/boxy/{instances.json, scene_objects.csv, 3dbb.csv}
  root   : <objects-root>/<seq>/body/xdata_soma.npz (key `transl` (T,3), world meters),
           the co-located SOMA fit -- subsampled to ~20 fps for validation. Falls back to
           the kimodo motion npz <motion-root>/<seq>.npz (`root_positions`) if no SOMA fit.

  scene_objects.csv : object_uid, timestamp[ns], t_wo_{x,y,z}[m], q_wo_{w,x,y,z}
                      (t_wo / q_wo = pose of object in world; static objects use timestamp -1)
  3dbb.csv          : object_uid, timestamp[ns], p_local_obj_{x,y,z}{min,max}[m]

OUTPUTS (--out-dir)
  <Sxx>_floor.json : list of per-seq records (all floor instances + primary + validation)
  <Sxx>_floor.csv  : flat one-row-per-seq table (seq, floor_z, floor_surface_z, dz stats, ok)

USAGE
  python extract_floor.py \
      --objects-root /weka/jungbin/nymeriaplus/S04 \
      --out-dir      /weka/jungbin/nymeriaplus_kimodo/floor
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

# A root frame "supports on" a floor if it is at most this far below floor_z
# (allows lying/sitting where the hip dips toward the ground).
SUPPORT_TOL = 0.30
# Plausible mean hip-above-floor band for the validation OK flag.
DZ_MEAN_OK = (0.20, 1.20)
# Subsample the (full-rate) soma root trajectory to ~this fps for validation.
TARGET_FPS = 20.0


def _load_roots(objects_root: Path, seq: str, motion_root: Path | None) -> np.ndarray | None:
    """Root (Hips) world trajectory for floor validation, in the SLAM world frame.

    Primary source: the co-located SOMA fit `<seq>/body/xdata_soma.npz` (`transl`,
    subsampled to ~TARGET_FPS via `timestamps_us`). This is the same SLAM world
    frame as the object boxes (verified: identical to the kimodo `root_positions`,
    which is just a 20-fps copy of this transl). Falls back to the kimodo motion
    npz `<motion_root>/<seq>.npz` (`root_positions`) when no SOMA fit exists.
    """
    soma_p = objects_root / seq / "body" / "xdata_soma.npz"
    if soma_p.exists():
        d = np.load(soma_p)
        tr = d["transl"].astype(np.float64)
        ts = d["timestamps_us"] if "timestamps_us" in d.files else None
        if ts is not None and len(ts) > 1:
            dt = float(np.median(np.diff(ts))) / 1e6  # s per frame
            stride = max(1, round((1.0 / dt) / TARGET_FPS)) if dt > 0 else 1
            tr = tr[::stride]
        return tr
    if motion_root is not None:
        mp = motion_root / f"{seq}.npz"
        if mp.exists():
            return np.load(mp)["root_positions"].astype(np.float64)
    return None


def quat_wxyz_to_R(w, x, y, z):
    """Quaternion (w,x,y,z), world<-object, to a 3x3 rotation matrix."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rows_by_uid(csv_path: Path) -> dict[str, list[str]]:
    out = {}
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if row:
                out[row[0].strip()] = row
    return out


def floor_boxes(boxy_dir: Path) -> list[dict]:
    """All Floor instances with world-frame geometry."""
    inst = json.load(open(boxy_dir / "instances.json"))
    floor_uids = [k for k, v in inst.items()
                  if str(v.get("category", "")).lower() == "floor"]
    so = _rows_by_uid(boxy_dir / "scene_objects.csv")
    bb = _rows_by_uid(boxy_dir / "3dbb.csv")

    boxes = []
    for uid in floor_uids:
        if uid not in so or uid not in bb:
            continue
        t = np.array([float(so[uid][2]), float(so[uid][3]), float(so[uid][4])])
        q = [float(so[uid][5]), float(so[uid][6]), float(so[uid][7]), float(so[uid][8])]
        e = [float(v) for v in bb[uid][2:8]]  # xmin,xmax,ymin,ymax,zmin,zmax
        R = quat_wxyz_to_R(*q)
        # 8 local corners -> world
        cl = np.array([[e[0], e[2], e[4]], [e[1], e[2], e[4]],
                       [e[0], e[3], e[4]], [e[1], e[3], e[4]],
                       [e[0], e[2], e[5]], [e[1], e[2], e[5]],
                       [e[0], e[3], e[5]], [e[1], e[3], e[5]]])
        cw = cl @ R.T + t
        center = R @ np.array([(e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[4] + e[5]) / 2]) + t
        normal = R @ np.array([0.0, 1.0, 0.0])  # floor local up
        boxes.append({
            "uid": uid,
            "floor_z": float(center[2]),
            "half_thickness": float((cw[:, 2].max() - cw[:, 2].min()) / 2),
            "footprint": [float(cw[:, 0].min()), float(cw[:, 0].max()),
                          float(cw[:, 1].min()), float(cw[:, 1].max())],
            "footprint_area": float((cw[:, 0].max() - cw[:, 0].min()) *
                                    (cw[:, 1].max() - cw[:, 1].min())),
            "normal_along_z": float(abs(normal[2])),
        })
    return boxes


def _xy_inside(footprint, xy):
    xmin, xmax, ymin, ymax = footprint
    return (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)


def _per_frame_floor(boxes: list[dict], roots: np.ndarray):
    """For each frame, the *nearest floor below the hip*.

    Among floors whose world-XY footprint contains the root and whose height is
    at/below the hip (within SUPPORT_TOL, to allow lying/sitting), take the highest.
    Returns (cand_z, has_floor) where cand_z[t,j] = floor_z if floor j supports frame
    t else -inf, and has_floor[t] marks frames with >=1 qualifying floor.
    """
    xy, z = roots[:, :2], roots[:, 2]
    cand_z = np.full((len(roots), len(boxes)), -np.inf)
    for j, b in enumerate(boxes):
        support = _xy_inside(b["footprint"], xy) & (b["floor_z"] <= z + SUPPORT_TOL)
        cand_z[support, j] = b["floor_z"]
    return cand_z, np.isfinite(cand_z).any(axis=1)


def pick_primary(boxes: list[dict], roots: np.ndarray | None) -> dict:
    """Choose the supporting floor for the actor (or largest footprint if no motion).

    The primary floor is the per-frame nearest-floor-below (see `_per_frame_floor`)
    that supports the most frames (majority vote).

    This resolves STACKED floors (same XY footprint, different z): a higher floor
    only wins on frames whose hip is actually up on that level, so the vote tracks
    which level the actor is really on -- XY overlap alone never decides it.
    """
    if len(boxes) == 1:
        return boxes[0]
    if roots is None:
        return max(boxes, key=lambda b: b["footprint_area"])

    cand_z, has_floor = _per_frame_floor(boxes, roots)
    if not has_floor.any():  # actor never over any floor footprint
        return max(boxes, key=lambda b: b["footprint_area"])
    nearest_below = np.argmax(cand_z, axis=1)[has_floor]  # highest qualifying floor per frame
    counts = np.bincount(nearest_below, minlength=len(boxes))
    return boxes[int(counts.argmax())]


def process_seq(seq: str, objects_root: Path, motion_root: Path | None = None) -> dict:
    boxy = objects_root / seq / "objects" / "boxy"
    if not boxy.is_dir():
        return {"seq": seq, "status": "no_objects"}
    boxes = floor_boxes(boxy)
    if not boxes:
        return {"seq": seq, "status": "no_floor_instance"}

    roots = _load_roots(objects_root, seq, motion_root)

    primary = pick_primary(boxes, roots)
    rec = {
        "seq": seq,
        "status": "ok",
        "up_axis": "+z",
        "n_floor_instances": len(boxes),
        "floor_instances": boxes,
        "primary_floor_uid": primary["uid"],
        "floor_z": primary["floor_z"],
        "floor_surface_z": primary["floor_z"] + primary["half_thickness"],
        "normal_along_z": primary["normal_along_z"],
    }
    if roots is not None:
        dz = roots[:, 2] - primary["floor_z"]
        val = {
            "n_frames": int(len(roots)),
            "root_dz_min": float(dz.min()),
            "root_dz_mean": float(dz.mean()),
            "root_dz_max": float(dz.max()),
            "ok": bool(DZ_MEAN_OK[0] <= dz.mean() <= DZ_MEAN_OK[1]
                       and dz.min() > -0.40 and primary["normal_along_z"] > 0.9),
        }
        # Per-frame metric: dz against the nearest floor BELOW the hip each frame.
        # For multi-level captures the single primary floor can't fit every frame
        # (the actor changes storeys), but per-frame grounding does -- this is the
        # quality number to trust when n_floor_instances > 1.
        cand_z, has_floor = _per_frame_floor(boxes, roots)
        if has_floor.any():
            pf = roots[has_floor, 2] - cand_z[has_floor].max(axis=1)
            val.update({
                "n_frames_grounded": int(has_floor.sum()),
                "perframe_dz_min": float(pf.min()),
                "perframe_dz_mean": float(pf.mean()),
                "perframe_dz_max": float(pf.max()),
                "ok_perframe": bool(DZ_MEAN_OK[0] <= pf.mean() <= DZ_MEAN_OK[1]
                                    and pf.min() > -0.40
                                    and primary["normal_along_z"] > 0.9),
            })
        else:
            val.update({"n_frames_grounded": 0, "perframe_dz_min": None,
                        "perframe_dz_mean": None, "perframe_dz_max": None,
                        "ok_perframe": False})
        rec["validation"] = val
    else:
        rec["validation"] = None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-root", type=Path, required=True,
                    help="Subject dir, e.g. /weka/jungbin/nymeriaplus/S04")
    ap.add_argument("--motion-root", type=Path, default=None,
                    help="kimodo motions dir, fallback root source when a seq has no "
                         "SOMA fit (validation otherwise uses <seq>/body/xdata_soma.npz)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/floor"))
    args = ap.parse_args()

    subj = args.objects_root.name
    seqs = sorted(p.parent.parent.name
                  for p in args.objects_root.glob("*/objects/boxy"))
    print(f"[scan] {subj}: {len(seqs)} sequences with object boxes")

    records = [process_seq(s, args.objects_root, args.motion_root) for s in seqs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out_dir / f"{subj}_floor.json", "w"), indent=2)

    cols = ["seq", "status", "n_floor_instances", "primary_floor_uid",
            "floor_z", "floor_surface_z", "root_dz_min", "root_dz_mean",
            "root_dz_max", "validation_ok", "perframe_dz_min", "perframe_dz_mean",
            "perframe_dz_max", "ok_perframe"]
    with open(args.out_dir / f"{subj}_floor.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in records:
            v = r.get("validation") or {}
            wr.writerow([r["seq"], r["status"], r.get("n_floor_instances", ""),
                         r.get("primary_floor_uid", ""),
                         _fmt(r.get("floor_z")), _fmt(r.get("floor_surface_z")),
                         _fmt(v.get("root_dz_min")), _fmt(v.get("root_dz_mean")),
                         _fmt(v.get("root_dz_max")), v.get("ok", ""),
                         _fmt(v.get("perframe_dz_min")), _fmt(v.get("perframe_dz_mean")),
                         _fmt(v.get("perframe_dz_max")), v.get("ok_perframe", "")])

    ok = sum(1 for r in records if r["status"] == "ok")
    val_ok = sum(1 for r in records if (r.get("validation") or {}).get("ok"))
    val_pf = sum(1 for r in records if (r.get("validation") or {}).get("ok_perframe"))
    val_n = sum(1 for r in records if r.get("validation"))
    multi = sum(1 for r in records if r.get("n_floor_instances", 0) > 1)
    print(f"\n=== {subj} floor extraction ===")
    print(f"  sequences with floor : {ok}/{len(records)}")
    print(f"  multi-floor sequences: {multi}")
    print(f"  root validation OK   : {val_ok}/{val_n} (single-floor)  "
          f"{val_pf}/{val_n} (per-frame)")
    print(f"\n  {'seq':40s} {'floors':>6} {'floor_z':>9} {'dz_mean':>8} {'ok':>4}")
    for r in records:
        v = r.get("validation") or {}
        print(f"  {r['seq']:40s} {r.get('n_floor_instances',''):>6} "
              f"{_fmt(r.get('floor_z')):>9} {_fmt(v.get('root_dz_mean')):>8} "
              f"{str(v.get('ok','')):>4}")
    print(f"\nwrote {args.out_dir}/{subj}_floor.json and .csv")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else ""


if __name__ == "__main__":
    main()
