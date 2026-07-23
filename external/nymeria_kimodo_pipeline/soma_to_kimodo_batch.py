"""Batch driver: convert every NymeriaPlus sequence with body/xdata_soma.npz
into kimodo-format motion NPZs.

Output layout:
  /weka/jungbin/nymeriaplus_kimodo/motions/{Sxx}/{seq_name}.npz

Idempotent: skips a sequence if the output NPZ exists and is non-empty.
SOMALayer is built ONCE and reused (orient + parent_T are read-only constants).
"""
from __future__ import annotations
import argparse, glob, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
if str(SOMA_ROOT) not in sys.path: sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer
from soma.geometry.rig_utils import apply_joint_orient_local
from soma.geometry.transforms import rotvec_to_matrix


def pick_indices_at_fps(timestamps_us: np.ndarray, target_fps: float) -> np.ndarray:
    t0, t1 = int(timestamps_us[0]), int(timestamps_us[-1])
    n_out = max(1, int((t1 - t0) / 1e6 * target_fps))
    query = np.linspace(t0, t1, n_out).astype(np.int64)
    idx = np.searchsorted(timestamps_us, query)
    return np.unique(np.clip(idx, 0, len(timestamps_us) - 1))


def convert_one(soma_npz: Path, out_path: Path, target_fps: float,
                orient_77: torch.Tensor, opt_77: torch.Tensor,
                expected_joint_names: list[str]) -> dict:
    z = np.load(soma_npz)
    poses = z["poses"].astype(np.float32)
    transl = z["transl"].astype(np.float32)
    ts = z["timestamps_us"].astype(np.int64)
    jn = list(z["joint_names"])
    if jn != expected_joint_names:
        raise ValueError("joint name list does not match SOMALayer rig[1:]")

    idx = pick_indices_at_fps(ts, target_fps)
    poses_s = poses[idx]
    transl_s = transl[idx]
    ts_s = ts[idx]
    T = poses_s.shape[0]

    R_rel = rotvec_to_matrix(torch.from_numpy(poses_s).reshape(-1, 3)).reshape(T, 77, 3, 3)
    local_rot_mats = apply_joint_orient_local(R_rel, orient_77, opt_77).numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        local_rot_mats=local_rot_mats,
        root_positions=transl_s.astype(np.float32),
        timestamps_us=ts_s.astype(np.int64),
        fps=np.int64(target_fps),
        source_seq=str(soma_npz.parent.parent.name),
        source_subject=str(soma_npz.parent.parent.parent.name),
    )
    return {
        "in_frames": int(poses.shape[0]),
        "out_frames": int(T),
        "duration_sec": round(float((ts_s[-1] - ts_s[0]) / 1e6), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nymeria-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/weka/jungbin/nymeriaplus_kimodo/motions"))
    ap.add_argument("--target-fps", type=float, default=20.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    soma = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                     device=args.device, mode="warp")
    expected_jn = list(soma.rig_data["joint_names"])[1:]  # drop Root
    orient_77 = soma._t_pose_orient.detach().cpu()[1:]
    opt_77 = soma._t_pose_orient_parent_T.detach().cpu()[1:]
    print(f"[init] SOMALayer loaded; J=77 after dropping Root; "
          f"joint_names[:5]={expected_jn[:5]}")

    seqs = sorted(args.nymeria_root.glob("S*/*/body/xdata_soma.npz"))
    print(f"[scan] {len(seqs)} xdata_soma.npz found under {args.nymeria_root}")
    if args.limit:
        seqs = seqs[: args.limit]

    n_done = n_skip = n_fail = 0
    t_total = time.perf_counter()
    summary = []
    for i, soma_npz in enumerate(seqs, 1):
        subj = soma_npz.parent.parent.parent.name
        seq_name = soma_npz.parent.parent.name
        out_path = args.out_root / subj / f"{seq_name}.npz"

        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            n_skip += 1
            continue
        try:
            t0 = time.perf_counter()
            meta = convert_one(soma_npz, out_path, args.target_fps, orient_77, opt_77, expected_jn)
            dt = time.perf_counter() - t0
            n_done += 1
            if i % 20 == 0 or i == len(seqs):
                print(f"  [{i}/{len(seqs)}] {subj}/{seq_name[:38]:38s}  "
                      f"out_frames={meta['out_frames']:6d}  {dt:.2f}s")
            summary.append({"subj": subj, "seq": seq_name, **meta, "elapsed_s": round(dt, 3)})
        except Exception as exc:
            n_fail += 1
            print(f"  [{i}/{len(seqs)}] FAIL {subj}/{seq_name}: {exc!r}")
            traceback.print_exc()

    args.out_root.mkdir(parents=True, exist_ok=True)
    sum_path = args.out_root / "_batch_summary.json"
    json.dump(
        {"summary": summary, "totals": {"done": n_done, "skipped": n_skip, "failed": n_fail}},
        open(sum_path, "w"),
        indent=2,
    )
    dt_all = time.perf_counter() - t_total
    print(f"\n=== done in {dt_all:.1f}s ===")
    print(f"  converted: {n_done}")
    print(f"  skipped  : {n_skip}")
    print(f"  failed   : {n_fail}")
    print(f"  summary written to {sum_path}")


if __name__ == "__main__":
    main()
