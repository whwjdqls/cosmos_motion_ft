#!/usr/bin/env python
"""NymeriaPlus per-sequence floor calibration + bad-window drop list (offline precompute).

WHY (the problem)
-----------------
NymeriaPlus motion comes from a SOMA fit whose feet sit a roughly CONSTANT ~5-7 cm BELOW the
GT room floor for a given sequence (a per-sequence fit bias, not a per-window floor error).
The per-window ``ground_offset_y`` in ``manifest_video.jsonl`` encodes the GT (multi-)floor
LEVEL of each captioned window (majority level in the slice), so after GT grounding
(``grounded_y = features[..., j*9+7] - ground_offset_y``, exactly
``nymeria_joint_dataset._load_motion`` / ``uniego_layout.ground_features``) the feet still
penetrate the floor by the per-seq fit bias. BONES-SEED, the other motion source in the
7-task mixture, is grounded by a per-seq min-foot convention that leaves contacted feet
~+3 cm ABOVE y=0 (ankle/toe joint centers have physical radius). The two sources therefore
sat ~6-8 cm apart in the shared normalized motion space.

THE CALIBRATION (validated on 48 seqs; /tmp/floorcalib/floor_calib_analysis.py + pass2-4.py)
--------------------------------------------------------------------------------------------
Per-sequence additive delta applied ON TOP of the per-window GT ``ground_offset_y``::

    d_minc(seq) = median over the seq's usable captioned windows' contact frames of
                  min(contacted-foot-joint y AFTER GT grounding)      # per-frame min
    c0          = BONES's own median per-frame min-contacted-joint y  # stance-height convention
    delta_seq   = d_minc(seq) - c0

so that ``grounded_y_calibrated = y - (ground_offset_y + delta_seq)`` places NymeriaPlus
stance frames at the SAME height convention as BONES (contacted feet at ~c0 above the floor
before the shift, ~0+stance offset after — i.e. window min-foot median moves from ~-5.5 cm to
~+1 cm). Sequences with no contact frames at all fall back to the GLOBAL median of the
per-seq deltas. The delta is a FIT-BIAS correction only: the GT per-window multi-floor level
(``ground_offset_y``) is fully preserved; estimated-floor sequences self-correct (their
floor-estimation bias is absorbed into d_minc); windows whose estimated floor picked a
metres-off level land in the drop list below.

Contact channels: ``features[:, 279:283] > 0.5`` map (in order) to SOMA-30 joints
``[24 LeftFoot, 25 LeftToeBase, 28 RightFoot, 29 RightToeBase]``; each joint's grounded
height is its Y translation channel ``j*9 + 7``.

WINDOW DROPS (recorded, not applied here — the dataset skips them at index-build time)
--------------------------------------------------------------------------------------
1. ``wrong_floor``:  n_contact >= 10  AND  |window contact-median - (delta_seq + c0)| > 0.20 m
   — metres-scale floor-SELECTION errors (mostly estimated-floor multi-level slices);
   was 1.3% of windows on the 48-seq sample.
2. ``residual_penetration``: window min-foot-y AFTER calibration < -0.20 m — deep SOMA fit
   failures the constant delta cannot fix; was ~0.65% on the 48-seq sample.
Both are computed from the raw Y channels directly (no decode needed). Expected total ~2%.

OUTPUT (one JSON, consumed by ``nymeria_joint_dataset.NymeriaJointDataset``)
----------------------------------------------------------------------------
::

    { "c0": float,                       # BONES stance-height reference (m)
      "global_delta": float,             # fallback for no-contact seqs (m)
      "deltas": {uuid: delta_seq},       # per-seq additive correction (m)
      "dropped_windows": {uuid: [[ws, we, reason], ...]},   # raw manifest start/end frames
      "stats": {...before/after summary...},
      "provenance": {date, script, criteria, ...} }

The dataset folds the correction into each index entry's stored offset:
``entry["off"] = ground_offset_y + delta_seq`` (the calibrated TOTAL vertical shift — a
future camera<->motion absolute alignment must use this total), keeping the original pair
recoverable via ``entry["off_gt"]`` / ``entry["delta"]``.

RUN (CPU only, kimodo env, ssh to a node — NOT srun; see cpu_jobs_ssh_not_srun)::

    ssh a3ultravis-a3ultranodeset-1 \
      "cd /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention && \
       ~/miniforge3/envs/kimodo/bin/python precompute_floor_calibration.py"

Full 713-seq scan takes ~5-15 min with the default worker pool.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------------------------------------
# Constants (mirror the validated 48-seq analysis + nymeria_joint_dataset exactly)
# ---------------------------------------------------------------------------------------------
MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
UNIEGO = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"
BONES_JSONL = "/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_pairs_train.jsonl"
OUT_JSON = "/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/floor_calibration.json"

# SOMA-30: contact channels [279:283] map (in order) to these joints; Y channel = j*9+7.
FOOT_J = [24, 25, 28, 29]                 # LeftFoot, LeftToeBase, RightFoot, RightToeBase
FOOT_Y = [j * 9 + 7 for j in FOOT_J]      # [223, 232, 259, 268]
HEAD_Y = 6 * 9 + 7                        # 61 (Head joint 6) — for the lying/extent gate

WRONG_FLOOR_MIN_CONTACT = 10              # frames with any contact needed to trust con_y_med
WRONG_FLOOR_TOL = 0.20                    # m; |con_y_med - (delta_seq + c0)| beyond this = drop
RESIDUAL_PEN_THRESH = -0.20               # m; min-foot AFTER calibration below this = drop
N_BONES_CLIPS = 2500                      # unique BONES clips for c0 (>= 2000 required)


# ---------------------------------------------------------------------------------------------
# c0: BONES stance-height reference
# ---------------------------------------------------------------------------------------------
def compute_c0(bones_jsonl: str, n_clips: int):
    """c0 = median (over >= n_clips unique BONES clips' contact frames) of the per-frame
    min contacted-foot-joint y. Also returns the BONES per-clip min-foot distribution
    (the AFTER-calibration target reference in the stats table)."""
    paths, seen = [], set()
    with open(bones_jsonl) as f:
        for line in f:
            r = json.loads(line)
            p = r["uniego_path"]
            if p not in seen:
                seen.add(p)
                paths.append((p, int(r["start"]), int(r["end"])))
            if len(paths) >= n_clips:
                break
    minc_all, minfoot = [], []
    n_used = 0
    for p, s, e in paths:
        if not os.path.isfile(p):
            continue
        with np.load(p) as d:
            F = d["features"][s:e]
        fy = F[:, FOOT_Y]
        c = F[:, 279:283] > 0.5
        anyc = c.any(axis=1)
        if anyc.sum():
            minc_all.append(np.where(c, fy, np.inf)[anyc].min(axis=1))
        minfoot.append(float(fy.min()))
        n_used += 1
    minc = np.concatenate(minc_all)
    c0 = float(np.median(minc))
    return c0, np.array(minfoot), n_used, int(minc.size), minc


# ---------------------------------------------------------------------------------------------
# Per-sequence measurement (one worker per seq; numpy only)
# ---------------------------------------------------------------------------------------------
def measure_seq(job):
    """job = (uuid, uni_path, nb, windows[list of (ws, we, off)]).

    Returns (uuid, d_minc | None, win_rows) where each win_row carries everything the drop
    criteria + stats need. Grounding replicates nymeria_joint_dataset._load_motion exactly:
    grounded_y = features[..., j*9+7] - ground_offset_y."""
    uuid, uni, nb, wins = job
    try:
        with np.load(uni) as d:
            F = d["features"][:]
    except Exception as e:  # noqa: BLE001 — a corrupt npz must not kill the pool
        return uuid, None, [], f"{type(e).__name__}: {e}"
    Tn = F.shape[0]
    fy_raw = F[:, FOOT_Y]
    hy_raw = F[:, HEAD_Y]
    con = F[:, 279:283] > 0.5

    minc_events = []
    rows = []
    for ws, we, off in wins:
        hi = min(we, nb, Tn)
        if hi - ws <= 0 or off is None:
            continue
        fy = fy_raw[ws:hi] - off                       # (t, 4) grounded foot heights
        hy = hy_raw[ws:hi] - off
        c = con[ws:hi]
        anyc = c.any(axis=1)
        n_con = int(anyc.sum())
        ev = fy[c]
        if n_con:
            minc_events.append(np.where(c, fy, np.inf)[anyc].min(axis=1))
        extent = hy - fy.min(axis=1)                   # head - min foot per frame
        rows.append(dict(
            ws=int(ws), we=int(we),
            minfoot=float(fy.min()),
            n_contact=n_con,
            con_y_med=(float(np.median(ev)) if ev.size else None),
            frac_lying=float((extent < 1.2).mean()),
        ))
    d_minc = float(np.median(np.concatenate(minc_events))) if minc_events else None
    return uuid, d_minc, rows, None


# ---------------------------------------------------------------------------------------------
# Stats helpers (same format as the 48-seq analysis reports 2/4)
# ---------------------------------------------------------------------------------------------
def dist_stats(a: np.ndarray, lying: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_cm": round(float(np.median(a)) * 100, 2),
        "p10_cm": round(float(np.percentile(a, 10)) * 100, 2),
        "p25_cm": round(float(np.percentile(a, 25)) * 100, 2),
        "p75_cm": round(float(np.percentile(a, 75)) * 100, 2),
        "p90_cm": round(float(np.percentile(a, 90)) * 100, 2),
        "p99_cm": round(float(np.percentile(a, 99)) * 100, 2),
        "within_3cm": round(float((np.abs(a) < 0.03).mean()), 4),
        "below_5cm": round(float((a < -0.05).mean()), 4),
        "below_20cm": round(float((a < -0.20).mean()), 4),
        "above_15cm": round(float((a > 0.15).mean()), 4),
        "lying_median_cm": (round(float(np.median(a[lying])) * 100, 2) if lying.any() else None),
    }


def print_dist(name: str, s: dict):
    print(f"  {name:28s} n={s['n']:6d} median={s['median_cm']:+6.1f}cm "
          f"p10={s['p10_cm']:+6.1f} p90={s['p90_cm']:+6.1f} p99={s['p99_cm']:+6.1f} | "
          f"within±3 {s['within_3cm']:.1%}  <-5 {s['below_5cm']:.1%}  "
          f"<-20 {s['below_20cm']:.2%}  >+15 {s['above_15cm']:.2%} | "
          f"lying med={s['lying_median_cm']}cm")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--uniego_root", default=UNIEGO)
    ap.add_argument("--bones_jsonl", default=BONES_JSONL)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--n_bones", type=int, default=N_BONES_CLIPS)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # ---- c0 from BONES (recomputed over >= 2000 clips) ------------------------------------
    print(f"[c0] sampling {args.n_bones} unique BONES clips from {args.bones_jsonl} ...",
          flush=True)
    c0, b_minfoot, n_bones_used, n_bones_events, b_minc = compute_c0(
        args.bones_jsonl, args.n_bones)
    print(f"[c0] c0 = {c0 * 100:+.2f} cm  (median per-frame min-contacted-joint y; "
          f"{n_bones_used} clips, {n_bones_events} contact frames; "
          f"p10 {np.percentile(b_minc, 10) * 100:+.1f} p90 {np.percentile(b_minc, 90) * 100:+.1f})",
          flush=True)

    # ---- manifest scan: build per-seq jobs ---------------------------------------------------
    jobs = []
    n_seqs_manifest = 0
    n_missing_npz = 0
    n_win_no_off = 0
    with open(args.manifest) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            if not uuid:
                continue
            n_seqs_manifest += 1
            uni = os.path.join(args.uniego_root, f"{uuid}.npz")
            if not os.path.isfile(uni):
                n_missing_npz += 1
                continue
            nb = int(rec.get("nb_frames", 0))
            wins = []
            for w in rec.get("t2w_windows", []):
                if not w.get("usable", False) or not w.get("caption"):
                    continue
                off = w.get("ground_offset_y", None)
                if off is None:
                    n_win_no_off += 1
                    continue
                wins.append((int(w["start_frame"]), int(w["end_frame"]), float(off)))
            jobs.append((uuid, uni, nb, wins))
    print(f"[scan] {n_seqs_manifest} manifest seqs; {len(jobs)} with uniego npz "
          f"({n_missing_npz} missing npz); {n_win_no_off} usable+captioned windows skipped "
          f"(no ground_offset_y)", flush=True)

    # ---- per-seq measurement (parallel) ------------------------------------------------------
    d_minc = {}
    win_rows = defaultdict(list)
    seq_errors = {}
    with Pool(args.workers) as pool:
        for i, (uuid, dm, rows, err) in enumerate(pool.imap_unordered(measure_seq, jobs, 4)):
            if err:
                seq_errors[uuid] = err
                continue
            d_minc[uuid] = dm
            win_rows[uuid] = rows
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(jobs)} seqs", flush=True)
    if seq_errors:
        print(f"[scan] {len(seq_errors)} seqs failed to load: "
              f"{list(seq_errors.items())[:3]}", flush=True)

    # ---- per-seq deltas + global fallback ----------------------------------------------------
    deltas = {u: (dm - c0) for u, dm in d_minc.items() if dm is not None}
    no_contact = [u for u, dm in d_minc.items() if dm is None]
    dv = np.array(list(deltas.values()))
    global_delta = float(np.median(dv))
    print(f"\n[delta] per-seq delta = d_minc - c0 over {len(deltas)}/{len(d_minc)} seqs "
          f"with contacts (no-contact fallback seqs: {len(no_contact)})")
    print(f"[delta] median={np.median(dv) * 100:+.2f}cm  p10={np.percentile(dv, 10) * 100:+.1f} "
          f"p90={np.percentile(dv, 90) * 100:+.1f}  min={dv.min() * 100:+.1f} "
          f"max={dv.max() * 100:+.1f}   GLOBAL fallback = {global_delta * 100:+.2f}cm", flush=True)

    # ---- window drops + before/after stats ---------------------------------------------------
    dropped = defaultdict(list)
    n_drop = {"wrong_floor": 0, "residual_penetration": 0}
    before, after, lying, kept_mask = [], [], [], []
    for uuid, rows in win_rows.items():
        d = deltas.get(uuid, global_delta)
        for r in rows:
            b = r["minfoot"]
            a = b - d
            before.append(b)
            after.append(a)
            lying.append(r["frac_lying"] > 0.5)
            reason = None
            if (r["n_contact"] >= WRONG_FLOOR_MIN_CONTACT and r["con_y_med"] is not None
                    and abs(r["con_y_med"] - (d + c0)) > WRONG_FLOOR_TOL):
                reason = "wrong_floor"
            elif a < RESIDUAL_PEN_THRESH:
                reason = "residual_penetration"
            if reason is not None:
                n_drop[reason] += 1
                dropped[uuid].append([r["ws"], r["we"], reason])
            kept_mask.append(reason is None)
    before = np.array(before)
    after = np.array(after)
    lying = np.array(lying)
    kept = np.array(kept_mask)
    n_win = int(before.size)
    n_drop_total = int((~kept).sum())

    print(f"\n== WINDOW MIN-FOOT-Y DISTRIBUTION (n={n_win} captioned windows, "
          f"{len(win_rows)} seqs) ==")
    s_bones = dist_stats(b_minfoot, np.zeros(b_minfoot.size, dtype=bool))
    s_before = dist_stats(before, lying)
    s_after = dist_stats(after, lying)
    s_after_kept = dist_stats(after[kept], lying[kept])
    print_dist("BONES target (per-clip)", s_bones)
    print_dist("BEFORE (GT floor only)", s_before)
    print_dist("AFTER  delta calibration", s_after)
    print_dist("AFTER  + drops removed", s_after_kept)

    print(f"\n== DROPS ==")
    print(f"  wrong_floor          : {n_drop['wrong_floor']:5d} "
          f"({n_drop['wrong_floor'] / n_win:.2%})")
    print(f"  residual_penetration : {n_drop['residual_penetration']:5d} "
          f"({n_drop['residual_penetration'] / n_win:.2%})")
    print(f"  TOTAL                : {n_drop_total:5d} ({n_drop_total / n_win:.2%}) "
          f"across {len(dropped)} seqs")

    stats = {
        "n_seqs_manifest": n_seqs_manifest,
        "n_seqs_measured": len(win_rows),
        "n_seqs_missing_npz": n_missing_npz,
        "n_seqs_with_contacts": len(deltas),
        "n_seqs_no_contact_fallback": len(no_contact),
        "no_contact_uuids": no_contact,
        "n_windows_measured": n_win,
        "n_windows_no_ground_offset": n_win_no_off,
        "bones_reference": {"n_clips": n_bones_used, "n_contact_frames": n_bones_events,
                            "minfoot": s_bones},
        "delta_per_seq_cm": {"median": round(float(np.median(dv)) * 100, 2),
                             "p10": round(float(np.percentile(dv, 10)) * 100, 2),
                             "p90": round(float(np.percentile(dv, 90)) * 100, 2),
                             "min": round(float(dv.min()) * 100, 2),
                             "max": round(float(dv.max()) * 100, 2)},
        "minfoot_before": s_before,
        "minfoot_after": s_after,
        "minfoot_after_kept": s_after_kept,
        "drops": {"wrong_floor": n_drop["wrong_floor"],
                  "residual_penetration": n_drop["residual_penetration"],
                  "total": n_drop_total,
                  "frac": round(n_drop_total / n_win, 5),
                  "n_seqs_affected": len(dropped)},
        "seq_load_errors": seq_errors,
    }
    provenance = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "motion_expert_joint_attention/precompute_floor_calibration.py",
        "manifest": args.manifest,
        "uniego_root": args.uniego_root,
        "bones_jsonl": args.bones_jsonl,
        "n_bones_clips_requested": args.n_bones,
        "criteria": {
            "delta": "delta_seq = median(per-frame min contacted-foot-joint grounded y) - c0; "
                     "fallback = global median of per-seq deltas",
            "c0": "BONES median per-frame min-contacted-joint y",
            "contact_joints": {"channels": "[279:283]", "joints": FOOT_J,
                               "names": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]},
            "wrong_floor": f"n_contact >= {WRONG_FLOOR_MIN_CONTACT} and "
                           f"|window contact-median - (delta_seq + c0)| > {WRONG_FLOOR_TOL} m",
            "residual_penetration": f"window min-foot-y after calibration < "
                                    f"{RESIDUAL_PEN_THRESH} m",
            "dropped_windows_format": "[start_frame, end_frame, reason] (raw manifest frames)",
        },
    }

    payload = {
        "c0": c0,
        "global_delta": global_delta,
        "deltas": {u: float(v) for u, v in sorted(deltas.items())},
        "dropped_windows": {u: v for u, v in sorted(dropped.items())},
        "stats": stats,
        "provenance": provenance,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, args.out)
    print(f"\n[out] wrote {args.out}  "
          f"(c0={c0 * 100:+.2f}cm, global_delta={global_delta * 100:+.2f}cm, "
          f"{len(deltas)} deltas, {n_drop_total} dropped windows)")


if __name__ == "__main__":
    main()
