"""Foot-skating + fallback-floor estimation for the per-slice GT-floor metadata.

Enriches `metadata_atomic_action_floor.jsonl` in place:

1. `ok` slices (GT floor present): compute foot-skating (cm/s during contact,
   measured relative to the slice's GT floor) -> `foot_skating_cms`,
   `n_contact_frames`; mark `usable=true`, `floor_source="gt"`.

2. Calibrate the floor<->foot gap from `ok` slices: how far the GT floor sits below
   the lowest toe / ankle (the "foot wrist"). This gives a percentile estimator
   `floor ~= percentile_P(per-frame min foot height) - C`.

3. `no_floor` / `no_support` slices (no usable GT floor box): estimate ONE floor per
   *sequence* from the whole-sequence foot trajectory using that estimator. For each
   such slice, if the root travels a large horizontal distance (so a single-floor
   estimate is unreliable -- like a stairs / multi-floor slice), flag it
   `usable=false`, `floor_status="estimated_ambiguous"`. Otherwise ground it with the
   estimated floor (`floor_source="estimated"`, `usable=true`) and compute its
   foot-skating relative to that estimated floor.

Also writes a stats JSON for the README.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.motion_rep.feature_utils import compute_vel_xyz

FPS = 20.0
s30 = SOMASkeleton30(); s77 = SOMASkeleton77()
IDX30 = [s77.bone_order_names.index(n) for n in s30.bone_order_names]
F30 = list(s30.foot_joint_idx)              # [LeftFoot, LeftToeBase, RightFoot, RightToeBase]
ANK = [F30[0], F30[2]]                       # ankles ("foot wrist")
TOE = [F30[1], F30[3]]                        # toes

META = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl")
MROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
STATS = Path("/home/jungbin_cho/.claude/jobs/d53e5f26/tmp/floor_skating_stats.json")

PCTL_CANDS = [0.0, 0.5, 1.0, 2.0, 5.0]       # percentile candidates for the floor estimator


def fk_posed30(d) -> torch.Tensor:
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
    B = lrm.shape[0]
    lrm30 = s30.from_SOMASkeleton77(lrm)
    nj30 = nj[IDX30].unsqueeze(0).expand(B, -1, -1)
    _, posed30, _ = s30.fk(lrm30, root, neutral_joints=nj30)
    return posed30                            # (B,30,3)


def slice_skating(posed30_slice: torch.Tensor, ground_offset: float) -> np.ndarray:
    """Foot speeds (cm/s) during contact, floor put at y=0 via ground_offset."""
    T = posed30_slice.shape[0]
    if T < 2:
        return np.zeros(0)
    p = posed30_slice.clone(); p[:, :, 1] -= float(ground_offset)
    vel = compute_vel_xyz(p.unsqueeze(0), FPS, lengths=torch.tensor([T]))[0]
    fc = foot_detect_from_pos_and_vel(p.unsqueeze(0), vel.unsqueeze(0), s30, 0.15, 0.10)[0].numpy()
    pp = p.numpy(); out = []
    for k in range(4):
        foot = pp[:, F30[k]]
        sp = np.linalg.norm(foot[1:] - foot[:-1], axis=-1) * FPS * 100.0
        c = (fc[1:, k] > 0.5) & (fc[:-1, k] > 0.5)
        out.append(sp[c])
    return np.concatenate(out) if out else np.zeros(0)


def horiz_travel(root_xyz: np.ndarray) -> float:
    """Horizontal (xz) bounding-box diagonal of the root over the slice, metres."""
    if len(root_xyz) < 2:
        return 0.0
    dx = root_xyz[:, 0].max() - root_xyz[:, 0].min()
    dz = root_xyz[:, 2].max() - root_xyz[:, 2].min()
    return float(np.hypot(dx, dz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=META)
    ap.add_argument("--motion-root", type=Path, default=MROOT)
    ap.add_argument("--limit-seqs", type=int, default=0, help="debug: process only N sequences")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta) if l.strip()]
    by_seq: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_seq[(r["subject"], r["filename"])].append(i)
    seqs = list(by_seq)
    if args.limit_seqs:
        seqs = seqs[:args.limit_seqs]
    print(f"[load] {len(rows)} rows / {len(by_seq)} sequences", flush=True)

    # ---- pass 1: ok-slice skating + calibration, cache per-seq min-foot for estimator ----
    ok_skate = []                  # per-ok-slice mean cm/s (only slices with contact)
    calib_gap_toe, calib_gap_ank, calib_gap_min = [], [], []   # (min foot height over slice) - floor
    seq_minfoot_pctl = {}          # seq -> {p: percentile of per-frame min-foot height}
    seq_single_gt = {}             # seq -> GT floor surface if single-floor (one floor_uid)
    ok_travel_amb, ok_travel_noamb = [], []   # horizontal travel of ok slices

    for n, key in enumerate(seqs):
        subj, fn = key
        npz = args.motion_root / subj / f"{fn}.npz"
        if not npz.is_file():
            continue
        d = np.load(npz, allow_pickle=True)
        root_xyz = d["root_positions"].astype(np.float64)
        posed30 = fk_posed30(d)
        perframe_minfoot = posed30[:, F30, 1].min(dim=1).values.numpy()   # (T,)
        seq_minfoot_pctl[key] = {p: float(np.percentile(perframe_minfoot, p)) for p in PCTL_CANDS}

        ok_uids = set()
        for i in by_seq[key]:
            r = rows[i]
            if r.get("floor_status") != "ok":
                continue
            ok_uids.add(r.get("floor_uid"))
            r["est_ambiguous"] = False
            sf = max(0, int(r["start_frame"])); ef = min(posed30.shape[0], int(r["end_frame"]))
            off = float(r["ground_offset_y"])
            # calibration: how far the floor sits below the lowest foot joints in this slice
            if ef - sf >= 1:
                seg = posed30[sf:ef]
                calib_gap_toe.append(float(seg[:, TOE, 1].min()) - off)
                calib_gap_ank.append(float(seg[:, ANK, 1].min()) - off)
                calib_gap_min.append(float(seg[:, F30, 1].min()) - off)
            tv = horiz_travel(root_xyz[sf:ef])
            (ok_travel_amb if r.get("ambiguous") else ok_travel_noamb).append(tv)
            sk = slice_skating(posed30[sf:ef], off)
            r["floor_source"] = "gt"; r["usable"] = True; r["est_ambiguous"] = False
            r["foot_skating_cms"] = round(float(sk.mean()), 3) if sk.size else None
            r["n_contact_frames"] = int(sk.size)
            if sk.size:
                ok_skate.append(float(sk.mean()))
        if len(ok_uids) == 1 and None not in ok_uids:
            # single GT floor across the sequence -> usable as estimator calibration target
            gts = [float(rows[i]["ground_offset_y"]) for i in by_seq[key] if rows[i].get("floor_status") == "ok"]
            seq_single_gt[key] = float(np.median(gts))
        if (n + 1) % 100 == 0:
            print(f"  pass1 {n+1}/{len(seqs)} seqs", flush=True)

    # ---- choose the floor estimator from single-floor ok sequences ----
    calib = []
    for key, gt in seq_single_gt.items():
        pct = seq_minfoot_pctl.get(key)
        if pct:
            calib.append((gt, pct))
    best = None
    for p in PCTL_CANDS:
        errs = np.array([pct[p] - gt for gt, pct in calib])
        med = float(np.median(errs)); mad = float(np.median(np.abs(errs - med)))
        if best is None or mad < best["mad"]:
            best = {"pctl": p, "C": med, "mad": mad}
    P, C = best["pctl"], best["C"]
    print(f"[estimator] floor ~= percentile_{P}(min-foot) - {C:+.4f}  (MAD={best['mad']:.4f}, "
          f"n_calib={len(calib)})", flush=True)

    # ---- horizontal-travel threshold for flagging estimated slices ----
    noamb = np.array(ok_travel_noamb); amb = np.array(ok_travel_amb)
    T_thresh = float(np.percentile(noamb, 95)) if noamb.size else 2.0
    _pa = lambda a, q: float(np.percentile(a, q)) if a.size else float("nan")
    print(f"[travel] ok non-amb horiz travel m: p50={_pa(noamb,50):.2f} "
          f"p95={_pa(noamb,95):.2f}  | ok amb: p50={_pa(amb,50):.2f} "
          f"p10={_pa(amb,10):.2f}  -> T={T_thresh:.2f} m", flush=True)

    # ---- pass 2: estimate floor for no_floor / no_support sequences ----
    est_skate = []
    est_used = est_flagged = 0
    deferred = [k for k in seqs if any(rows[i].get("floor_status") in ("no_floor", "no_support")
                                       for i in by_seq[k])]
    for n, key in enumerate(deferred):
        subj, fn = key
        npz = args.motion_root / subj / f"{fn}.npz"
        if not npz.is_file():
            continue
        d = np.load(npz, allow_pickle=True)
        root_xyz = d["root_positions"].astype(np.float64)
        posed30 = fk_posed30(d)
        est_floor = seq_minfoot_pctl[key][P] - C
        for i in by_seq[key]:
            r = rows[i]
            if r.get("floor_status") not in ("no_floor", "no_support"):
                continue
            sf = max(0, int(r["start_frame"])); ef = min(posed30.shape[0], int(r["end_frame"]))
            tv = horiz_travel(root_xyz[sf:ef])
            r["floor_source"] = "estimated"
            r["est_floor_surface_z"] = round(est_floor, 5)
            r["horiz_travel_m"] = round(tv, 3)
            if tv > T_thresh:
                r["usable"] = False           # large horizontal travel on a single estimated floor
                r["est_ambiguous"] = True      # floor_status stays as GT provenance (no_floor/no_support)
                r["foot_skating_cms"] = None; r["n_contact_frames"] = 0
                est_flagged += 1
                continue
            r["usable"] = True
            r["est_ambiguous"] = False
            r["ground_offset_y"] = round(est_floor, 5)
            sk = slice_skating(posed30[sf:ef], est_floor)
            r["foot_skating_cms"] = round(float(sk.mean()), 3) if sk.size else None
            r["n_contact_frames"] = int(sk.size)
            est_used += 1
            if sk.size:
                est_skate.append(float(sk.mean()))
        if (n + 1) % 50 == 0:
            print(f"  pass2 {n+1}/{len(deferred)} seqs", flush=True)

    # ---- write enriched metadata + per-subject mirror ----
    with open(args.meta, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_dir = args.meta.parent / "metadata_per_subject_atomic_action_floor"
    per_dir.mkdir(parents=True, exist_ok=True)
    by_subj = defaultdict(list)
    for r in rows:
        by_subj[r["subject"]].append(r)
    for subj, rs in by_subj.items():
        with open(per_dir / f"{subj}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    def stat(a):
        a = np.asarray(a, float)
        if not a.size:
            return None
        return {"n_slices": int(a.size), "mean": round(float(a.mean()), 3),
                "median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3),
                "p10": round(float(np.percentile(a, 10)), 3)}

    out = {
        "ok_skating_cms": stat(ok_skate),
        "estimated_usable_skating_cms": stat(est_skate),
        "floor_foot_gap_m": {
            "toe_minus_floor": {"median": round(float(np.median(calib_gap_toe)), 4),
                                 "p10": round(float(np.percentile(calib_gap_toe, 10)), 4),
                                 "p90": round(float(np.percentile(calib_gap_toe, 90)), 4)},
            "ankle_minus_floor": {"median": round(float(np.median(calib_gap_ank)), 4),
                                   "p10": round(float(np.percentile(calib_gap_ank, 10)), 4),
                                   "p90": round(float(np.percentile(calib_gap_ank, 90)), 4)},
            "minfoot_minus_floor": {"median": round(float(np.median(calib_gap_min)), 4),
                                     "p10": round(float(np.percentile(calib_gap_min, 10)), 4),
                                     "p90": round(float(np.percentile(calib_gap_min, 90)), 4)},
        },
        "estimator": {"percentile": P, "offset_C": round(C, 5), "mad_m": round(best["mad"], 5),
                       "n_calib_seqs": len(calib),
                       "formula": f"est_floor = percentile_{P}(per-frame min-foot height) - ({C:+.5f})"},
        "horiz_travel_threshold_m": round(T_thresh, 3),
        "ok_travel_m": {"nonamb_p50": round(float(np.percentile(noamb, 50)), 3),
                         "nonamb_p95": round(float(np.percentile(noamb, 95)), 3),
                         "amb_p10": round(float(np.percentile(amb, 10)), 3),
                         "amb_p50": round(float(np.percentile(amb, 50)), 3)},
        "counts": {"ok": int(sum(1 for r in rows if r.get("floor_status") == "ok")),
                    "estimated_usable": est_used, "estimated_ambiguous_flagged": est_flagged},
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(STATS, "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(out, indent=2))
    print(f"\nwrote enriched {args.meta}\nwrote stats {STATS}")


if __name__ == "__main__":
    main()
