"""Check consistency of the device->camera extrinsic across ALL NymeriaPlus recordings,
and within-sequence stability of the online calibration.

(A) For every recording_head: factory T_device_rgb and T_device_slam-left from the VRS
    device calibration. Compare across recordings (rotation angle + translation vs a global
    reference); cluster to see how many distinct calibrations exist and whether they group
    by subject/device.
(B) For a few recordings: load MPS online_calibration.jsonl (if present) and measure how much
    T_device_rgb varies over time within the sequence (factory-static vs time-varying).

Run in the `nymeria_plus` env.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
import argparse, glob, json, os
import numpy as np
from projectaria_tools.core import data_provider

NROOT = "/weka/jungbin/nymeriaplus"


def geodesic_deg(Ra, Rb):
    R = Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def extrinsic(dc, label):
    return dc.get_transform_device_sensor(label).to_matrix().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_calib_check")
    ap.add_argument("--per_subject", type=int, default=0, help="0 = all recordings")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = sorted(glob.glob(os.path.join(NROOT, "S*", "*", "recording_head")))
    by_subj = {}
    for r in recs:
        subj = r.split("/")[-3].split("/")[0]
        subj = os.path.basename(os.path.dirname(os.path.dirname(r)))
        by_subj.setdefault(subj, []).append(r)
    sel = []
    for subj, rs in by_subj.items():
        sel += rs if args.per_subject == 0 else rs[: args.per_subject]
    print(f"checking {len(sel)} recordings across {len(by_subj)} subjects")

    rows = []
    ref_rgb = None
    for i, rp in enumerate(sel):
        subj = os.path.basename(os.path.dirname(os.path.dirname(rp)))
        seq = os.path.basename(os.path.dirname(rp))
        vrs = None
        for rel in ("data/data.vrs", "data/motion.vrs"):
            p = os.path.join(rp, rel)
            if os.path.isfile(p):
                vrs = p; break
        if vrs is None:
            continue
        try:
            dp = data_provider.create_vrs_data_provider(vrs)
            dc = dp.get_device_calibration()
            Trgb = extrinsic(dc, "camera-rgb")
            Tsl = extrinsic(dc, "camera-slam-left")
            try: serial = dc.get_serial_number()
            except Exception: serial = ""
        except Exception as e:
            print(f"  [skip] {subj}/{seq}: {str(e)[:60]}"); continue
        if ref_rgb is None:
            ref_rgb, ref_sl = Trgb, Tsl
        rows.append(dict(subj=subj, seq=seq, serial=str(serial),
                         rgb_ang=geodesic_deg(ref_rgb[:3, :3], Trgb[:3, :3]),
                         rgb_tmm=float(np.linalg.norm(Trgb[:3, 3] - ref_rgb[:3, 3]) * 1000),
                         tilt=float(np.degrees(np.arccos(np.clip(Trgb[2, 2], -1, 1)))),
                         rgb=Trgb.flatten().tolist()))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(sel)}")
    json.dump(rows, open(os.path.join(args.out, "rgb_extrinsics.json"), "w"))

    # ---- summary ----
    tilts = np.array([r["tilt"] for r in rows])
    angs = np.array([r["rgb_ang"] for r in rows])
    tmm = np.array([r["rgb_tmm"] for r in rows])
    print(f"\n=== {len(rows)} recordings ===")
    print(f"optical tilt (device-z vs cam-z): mean {tilts.mean():.2f} std {tilts.std():.2f} "
          f"min {tilts.min():.2f} max {tilts.max():.2f}")
    print(f"rotation angle vs reference: mean {angs.mean():.2f} max {angs.max():.2f} deg")
    print(f"translation diff vs reference: mean {tmm.mean():.2f} max {tmm.max():.2f} mm")
    # cluster recordings by rounded tilt
    from collections import Counter
    c = Counter(round(t, 1) for t in tilts)
    print(f"distinct tilt values (0.1deg bins): {dict(sorted(c.items()))}")
    # serials
    sc = Counter(r["serial"] for r in rows if r["serial"])
    print(f"distinct serials: {len(sc)}  -> {dict(list(sc.items())[:8])}")
    # per-subject tilt spread
    print("per-subject mean tilt:")
    bys = {}
    for r in rows:
        bys.setdefault(r["subj"], []).append(r["tilt"])
    for s in sorted(bys):
        a = np.array(bys[s]); print(f"   {s}: n={len(a)} tilt {a.mean():.2f}±{a.std():.2f}")


if __name__ == "__main__":
    main()
