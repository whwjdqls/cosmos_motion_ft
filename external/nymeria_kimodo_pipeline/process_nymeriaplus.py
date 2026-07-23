#!/usr/bin/env python
"""Batch-process NymeriaPlus sequences:

  1. Convert each sequence's SMPL motion -> SOMA pose params, saved next to the
     SMPL file (`<seq>/body/xdata_soma.npz`) with metadata
     (`<seq>/body/xdata_soma_meta.json`, incl. per-vertex error stats).
  2. (optional) Delete the recording-WRIST SLAM data (trajectory + camera
     params + semidense + summary) -- the `recording_lwrist` / `recording_rwrist`
     folders. Head and observer recordings are never touched. (--delete-wrist-slam)
  3. (optional) Delete the unused MHR mesh `body/xdata_mhr.glb`, but only for
     sequences that already have `xdata_soma.npz`. (--delete-mhr)

Designed to be re-run as more Nymeria data is downloaded:
  * Sequences are discovered by the presence of `body/xdata_smpl_neutral.npz`
    anywhere under --root, so any S-folder layout works.
  * Conversion is idempotent: a sequence with `xdata_soma.npz` is skipped
    unless --overwrite. Deletion is idempotent: already-removed wrist dirs are
    skipped.

Parallel runs (do NOT reshuffle the data into shard folders -- use --shard):
  Launch one process per GPU, each a disjoint strided shard. e.g. 4-way:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i nohup python process_nymeriaplus.py \
        --root /weka/jungbin/nymeriaplus/S07 --delete-wrist-slam --delete-mhr \
        --shard $i/4 > ~/_s07_shard$i.log 2>&1 &
    done
  Shards process disjoint sequences (no write races); each is independently
  resumable thanks to the idempotent skip.

MUST run in the `soma` conda env on a GPU node (tmux 1):
    conda activate soma
    CUDA_VISIBLE_DEVICES=<free-gpu> python process_nymeriaplus.py \
        --root /weka/jungbin/nymeriaplus/S11 \
        --smpl-model-path /home/jungbin_cho/SMPL_NEUTRAL.pkl \
        --delete-wrist-slam

==============================================================================
NOTES / LESSONS LEARNED  (read before changing this script)
==============================================================================

CRITICAL BUG (fixed) -- per-identity state leak:
  Reusing ONE SOMALayer/PoseInversion across sequences with different identities
  silently corrupts every sequence after the first: only the 1st sequence in a
  process gets correct error (~0.4cm); the rest get inflated error (median 2-11cm)
  or NaN. No exception is raised, so `failed=0` does NOT mean good output.
  => convert_sequence() rebuilds a FRESH SOMALayer + PoseInversion every call.
     `smpl_model` (smplx) is stateless w.r.t. betas and is reused.
  => ALWAYS verify by aggregating the per-vertex errors from the meta JSONs
     (expect median <~0.6cm, no NaN). Do not trust the run summary alone.

Identity-fit-to-person is native: identity_model_type="smpl" +
  inv.prepare_identity(betas) maps the subject's SMPL shape onto SOMA. We pass
  the per-sequence MEDIAN betas (NymeriaPlus stores betas as 5 piecewise-constant
  windows -- a pipeline artifact, not real time-varying shape).

Error stats: torch.quantile() caps at ~16.7M (2^24) elements and throws on the
  full per-vertex error tensor. _err_stats_cm() uses exact mean/max reductions +
  a strided numpy subsample for median/p99.

Environment gotchas (soma env):
  * Run on tmux 1 = GPU node (a3ultranodeset-0). The login node has NO GPU, so
    torch.cuda tests there are false negatives.
  * torch must match the driver: driver is CUDA 12.8, so torch==2.11.0+cu128
    (the env originally shipped 2.12.0+cu130 -> cuda unavailable). After changing
    torch, purge leftover unsuffixed/`*-cu13` CUDA-13 nvidia-* pip pkgs (cu12/cu13
    share the nvidia/ namespace dirs) and force-reinstall torch cu128.
  * chumpy 0.70 is patched in-place (chumpy/__init__.py) for py3.11 + numpy>=1.24
    (inspect.getargspec shim + re-added numpy aliases) so smplx can load the pkl.
  * SMPL_NEUTRAL.pkl is symlinked into SOMA-X/assets/SMPL/ (SOMALayer's SMPL
    identity model looks for it there).

Output `body/xdata_soma.npz` keys (via soma.io.save_soma_npz, SOMA-X compatible):
  poses (N,77,3) rotvec T-pose-RELATIVE, transl (N,3) m, identity_coeffs (1,10),
  joint_names, joint_orient, rotation_repr, absolute_pose=False, unit, keep_root,
  + extra: timestamps_us (N,) -- native rate, 1:1 with the source SMPL frames.

Wrist deletion targets ONLY recording_lwrist/ + recording_rwrist/ (pure mps/slam,
  no VRS); guarded to skip any wrist dir that unexpectedly contains a .vrs.
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

import smplx  # noqa: E402
from soma.geometry.rig_utils import remove_joint_orient_local  # noqa: E402
from soma.geometry.transforms import matrix_to_rotvec  # noqa: E402
from soma.io import save_soma_npz  # noqa: E402
from soma.pose_inversion import PoseInversion  # noqa: E402
from soma.soma import SOMALayer  # noqa: E402

SMPL_REL = Path("body/xdata_smpl_neutral.npz")
SOMA_REL = Path("body/xdata_soma.npz")
META_REL = Path("body/xdata_soma_meta.json")
MHR_REL = Path("body/xdata_mhr.glb")
WRIST_DIRS = ("recording_lwrist", "recording_rwrist")
SCRIPT_VERSION = "1.1"  # 1.1: recenter large-magnitude transl before SOMA fit (fixes S17)
# Threshold for "world-scale" transl that degrades fp32 precision in the SOMA
# inversion. NymeriaPlus indoor captures are ~±5-10 m; S17 sits at ~±300 m.
TRANSL_RECENTER_THRESHOLD_M = 50.0


def _err_stats_cm(err) -> dict:
    """Per-vertex error stats in cm. mean/max exact (cheap reductions);
    median/p99 from a strided subsample (torch.quantile caps at ~16.7M elems)."""
    flat = err.reshape(-1)
    n = flat.numel()
    stride = max(1, n // 2_000_000)
    sub = flat[::stride].numpy()
    return {
        "mean": round(float(flat.mean()) * 100, 4),
        "median": round(float(np.median(sub)) * 100, 4),
        "p99": round(float(np.percentile(sub, 99)) * 100, 4),
        "max": round(float(flat.max()) * 100, 4),
        "subsample_n": int(sub.size),
    }


def find_sequences(root: Path) -> list[Path]:
    return sorted({p.parents[1] for p in root.rglob(SMPL_REL.name)
                   if p.parent.name == "body"})


def convert_sequence(seq: Path, smpl_model, args) -> dict:
    """Convert one sequence's SMPL -> SOMA. Returns metadata dict.

    A fresh SOMALayer + PoseInversion is built per sequence: reusing one
    instance across different identities leaks state (inflated error / NaNs)."""
    device = args.device
    soma = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                     device=device, mode="warp")
    inv = PoseInversion(soma, low_lod=True)

    npz = np.load(seq / SMPL_REL)
    betas_all = npz["betas"].astype(np.float32)
    body_pose = npz["body_pose"].astype(np.float32)
    global_orient = npz["global_orient"].astype(np.float32)
    transl = npz["transl"].astype(np.float32)
    ts_us = npz["timestamps"].astype(np.int64)  # native rate, 1:1 with output
    N = body_pose.shape[0]

    # Recenter large-magnitude transl. SOMA pose inversion's residual is
    # sub-cm, but fp32 vertex coordinates lose ~1cm of precision when world
    # positions reach ~300m (S17 outdoor capture). Subtracting a constant
    # offset preserves the per-frame relative motion the optimizer fits, then
    # the offset is added back to root_transl before saving so the output
    # trajectory stays in the original world frame.
    if np.abs(transl).max() > TRANSL_RECENTER_THRESHOLD_M:
        transl_origin = transl.mean(0).astype(np.float32)
        transl = transl - transl_origin
        print(f"    recenter: |transl|max={np.abs(transl + transl_origin).max():.1f}m "
              f"> {TRANSL_RECENTER_THRESHOLD_M:.0f}m; subtracting origin {transl_origin}")
    else:
        transl_origin = None

    # Shape is physically constant; npz stores 5 piecewise-constant windows.
    betas_med = np.median(betas_all, axis=0).astype(np.float32)[None]  # (1,10)
    betas_id = torch.from_numpy(betas_med).to(device)
    inv.prepare_identity(betas_id)

    rot, root_t, errs = [], [], []
    bs = args.batch_size
    t0 = time.perf_counter()
    for s in range(0, N, bs):
        e = min(s + bs, N)
        with torch.no_grad():
            out = smpl_model(
                betas=betas_id.expand(e - s, -1),
                body_pose=torch.from_numpy(body_pose[s:e]).to(device),
                global_orient=torch.from_numpy(global_orient[s:e]).to(device),
                transl=torch.from_numpy(transl[s:e]).to(device),
            )
        r = inv.fit(out.vertices, body_iters=args.body_iters,
                    finger_iters=args.finger_iters, full_iters=args.full_iters,
                    autograd_iters=args.autograd_iters, autograd_lr=args.autograd_lr)
        rot.append(r["rotations"].cpu())
        root_t.append(r["root_translation"].cpu())
        errs.append(r["per_vertex_error"].cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(rot, 0)
    root_transl = torch.cat(root_t, 0)
    if transl_origin is not None:
        root_transl = root_transl + torch.from_numpy(transl_origin)
    err = torch.cat(errs, 0)

    # absolute -> T-pose-relative rotvec
    odev = soma._t_pose_orient.device
    rel = remove_joint_orient_local(rotations.to(odev), soma._t_pose_orient,
                                    soma._t_pose_orient_parent_T)
    poses_rotvec = matrix_to_rotvec(rel.reshape(-1, 3, 3)).reshape(
        rotations.shape[0], rotations.shape[1], 3).cpu()

    save_soma_npz(
        seq / SOMA_REL, poses_rotvec, root_transl.clone(),
        joint_names=list(soma.rig_data["joint_names"]),
        identity_model_type=soma.identity_model_type,
        identity_coeffs=betas_id.cpu(),
        joint_orient=soma._t_pose_orient,
        unit="meters", keep_root=False,
        extra_arrays={"timestamps_us": ts_us},
    )

    meta = {
        "script_version": SCRIPT_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_smpl": str(SMPL_REL),
        "output_soma": str(SOMA_REL),
        "num_frames": int(N),
        "fps_native": True,
        "duration_sec": round(float((ts_us[-1] - ts_us[0]) / 1e6), 3),
        "identity_model_type": soma.identity_model_type,
        "identity_coeffs_median_betas": [round(float(x), 5) for x in betas_med[0]],
        "solver": {"body_iters": args.body_iters, "finger_iters": args.finger_iters,
                   "full_iters": args.full_iters, "autograd_iters": args.autograd_iters},
        "per_vertex_error_cm": _err_stats_cm(err),
        "fit_seconds": round(dt, 2),
        "fit_fps": round(N / dt, 1),
        "smpl_model": str(args.smpl_model_path),
    }
    with open(seq / META_REL, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def delete_wrist_slam(seq: Path, dry_run: bool) -> int:
    """Remove recording_lwrist/recording_rwrist (wrist SLAM). Returns bytes freed.
    Safety: only acts on the two wrist dirs, and only if they contain NO .vrs."""
    freed = 0
    for name in WRIST_DIRS:
        d = seq / name
        if not d.is_dir():
            continue
        if any(d.rglob("*.vrs")):
            print(f"    SKIP {name}: contains .vrs (unexpected) -- not deleting")
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        freed += size
        if dry_run:
            print(f"    [dry-run] would delete {name}/ ({size/1e6:.0f} MB)")
        else:
            shutil.rmtree(d)
            print(f"    deleted {name}/ ({size/1e6:.0f} MB)")
    return freed


def delete_mhr(seq: Path, dry_run: bool) -> int:
    """Remove body/xdata_mhr.glb (MHR mesh, unused). Returns bytes freed.
    Only called when a SOMA file already exists (so we never drop MHR before a
    successful SMPL->SOMA conversion)."""
    f = seq / MHR_REL
    if not f.is_file():
        return 0
    size = f.stat().st_size
    if dry_run:
        print(f"    [dry-run] would delete {MHR_REL.name} ({size/1e6:.0f} MB)")
    else:
        f.unlink()
        print(f"    deleted {MHR_REL.name} ({size/1e6:.0f} MB)")
    return size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/weka/jungbin/nymeriaplus/S11"))
    p.add_argument("--smpl-model-path", type=Path, default=Path("/home/jungbin_cho/SMPL_NEUTRAL.pkl"))
    p.add_argument("--overwrite", action="store_true", help="Re-convert even if xdata_soma.npz exists.")
    p.add_argument("--delete-wrist-slam", action="store_true", help="Delete wrist recording SLAM dirs.")
    p.add_argument("--delete-mhr", action="store_true",
                   help="Delete body/xdata_mhr.glb (only when xdata_soma.npz exists).")
    p.add_argument("--dry-run", action="store_true", help="Preview deletions without removing.")
    p.add_argument("--skip-convert", action="store_true", help="Only do deletion step.")
    p.add_argument("--limit", type=int, default=None, help="Process only first N sequences.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--body-iters", type=int, default=2)
    p.add_argument("--finger-iters", type=int, default=0)
    p.add_argument("--full-iters", type=int, default=1)
    p.add_argument("--autograd-iters", type=int, default=0)
    p.add_argument("--autograd-lr", type=float, default=5e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default=None,
                   help="Process a strided subset for parallel runs, e.g. '0/4' = "
                        "shard 0 of 4. Launch one process per GPU with shards "
                        "0/N..(N-1)/N; each handles a disjoint set of sequences.")
    args = p.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    seqs = find_sequences(args.root)
    print(f"[root] {args.root}  ->  {len(seqs)} sequence(s) discovered")
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not (0 <= si < sn):
            raise SystemExit(f"--shard {args.shard}: need 0 <= index < count")
        seqs = seqs[si::sn]  # strided: balances long/short sequences across shards
        print(f"[shard] {si}/{sn}  ->  {len(seqs)} sequence(s) this shard")
    if args.limit:
        seqs = seqs[: args.limit]
        print(f"[limit] {args.limit}  ->  {len(seqs)} sequence(s)")
    if args.device != "cuda":
        print("WARNING: CUDA unavailable; running on CPU (slow).")

    smpl_model = None
    if not args.skip_convert:
        # smpl_model is stateless w.r.t. identity (betas passed per forward) so
        # it is reused; SOMALayer/PoseInversion are rebuilt per sequence inside
        # convert_sequence to avoid cross-identity state leakage.
        smpl_model = smplx.create(
            model_type="smpl", model_path=str(args.smpl_model_path),
            use_pca=False, flat_hand_mean=True, batch_size=1).to(args.device)

    n_conv = n_skip = n_fail = 0
    freed_total = 0
    freed_mhr = 0
    for i, seq in enumerate(seqs, 1):
        print(f"\n[{i}/{len(seqs)}] {seq.name}")
        # ---- convert ----
        if not args.skip_convert:
            if (seq / SOMA_REL).exists() and (seq / META_REL).exists() and not args.overwrite:
                print("    convert: SKIP (already converted)")
                n_skip += 1
            else:
                try:
                    m = convert_sequence(seq, smpl_model, args)
                    e = m["per_vertex_error_cm"]
                    print(f"    convert: {m['num_frames']} frames, "
                          f"err mean={e['mean']}cm median={e['median']}cm max={e['max']}cm "
                          f"({m['fit_fps']} fps)")
                    n_conv += 1
                except Exception as exc:
                    print(f"    convert: FAILED -- {exc!r}")
                    n_fail += 1
        # ---- delete wrist slam ----
        if args.delete_wrist_slam:
            freed_total += delete_wrist_slam(seq, args.dry_run)
        # ---- delete MHR (guarded: only if SOMA conversion exists) ----
        if args.delete_mhr:
            if (seq / SOMA_REL).exists():
                freed_mhr += delete_mhr(seq, args.dry_run)
            else:
                print("    delete-mhr: SKIP (no xdata_soma.npz yet)")

    print(f"\n=== summary ===")
    print(f"converted={n_conv}  skipped={n_skip}  failed={n_fail}")
    tag = "would free" if args.dry_run else "freed"
    if args.delete_wrist_slam:
        print(f"wrist-slam deletion: {tag} {freed_total/1e9:.2f} GB")
    if args.delete_mhr:
        print(f"mhr deletion: {tag} {freed_mhr/1e9:.2f} GB")


if __name__ == "__main__":
    main()
