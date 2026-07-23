"""Shape-aware NymeriaPlus -> kimodo motion converter (CORRECTED).

Produces the kimodo motion NPZ (local_rot_mats + root_positions @ 20 fps) PLUS a
per-actor ``neutral_joints (77,3)`` rest skeleton, in kimodo's Y-up frame and
FK-decodable by kimodo SOMASkeleton (so it visualizes/trains like BONES-SEED).

Why this differs from soma_to_kimodo_single.py (which is BROKEN for kimodo):
  The old converter baked SOMA-X's joint orient via apply_joint_orient_local.
  Those local rotations are NOT in kimodo's local-rotation convention, so kimodo
  FK folds the body up (head-foot collapses to ~0.1 m vs ~1.2 m standing). The
  capture world is also Z-up while kimodo is Y-up. Diagnosed by comparing kimodo
  FK head-above-feet for BONES-SEED (1.24 m, ok) vs old Nymeria (0.13 m, crumpled)
  vs SOMA-X ground truth (1.20 m, up=+Z).

Corrected pipeline (per sequence):
  1. SOMA-X true world joint transforms T_world via the skinning pose (Z-up):
        R_posed[f,j] (animated), R_rest[j] (T-pose), rest positions N[j].
  2. Per-joint rest->posed global rotation:  G_rel = R_posed @ R_rest^T.
  3. Rotate the posed side Z-up -> Y-up:      G_rel_y = R_z2y @ G_rel,
     with R_z2y = [[1,0,0],[0,0,1],[0,-1,0]]  i.e. (x,y,z)->(x,z,-y).
  4. kimodo local rotations:  local[j] = G_rel_y[parent]^T @ G_rel_y[j]
     (root: parent=I). This IS kimodo.skeleton.transforms.global_rots_to_local_rots,
     reimplemented inline so the converter needs only the `soma` env.
  5. neutral_joints = rest positions N (pelvis-centered, Y-up).
  6. root_positions = R_z2y @ hips_world_pos.
  Validated: kimodo FK(local, root, neutral) reproduces the actor standing in Y-up
  with head-foot matching the SOMA-X ground truth (asserted per sequence).

Input  : /weka/jungbin/nymeriaplus/{Sxx}/{seq}/body/xdata_soma.npz
Output : /weka/jungbin/nymeriaplus_kimodo_proportional/{Sxx}/{seq}.npz
  local_rot_mats (T,77,3,3) f32, root_positions (T,3) f32, timestamps_us (T,) i64,
  fps () i64, source_seq (), source_subject (), neutral_joints (77,3) f32,
  identity_coeffs (1,10) f32.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer  # noqa: E402

DEFAULT_INPUT_ROOT = Path("/weka/jungbin/nymeriaplus")
DEFAULT_OUTPUT_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
KIMODO_BUFFERS = Path("/weka/jungbin/kimodo_caches/_somaskel77_buffers.npz")
SOMA_REL = Path("body/xdata_soma.npz")
TARGET_FPS = 20.0
R_Z2Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # Z-up -> Y-up
HEADFOOT_MIN_M = 0.8   # per-seq assert: posed head must sit >= this above feet in +Y (mean)


def pick_indices_at_fps(timestamps_us: np.ndarray, target_fps: float) -> np.ndarray:
    t0, t1 = int(timestamps_us[0]), int(timestamps_us[-1])
    n_out = max(1, int((t1 - t0) / 1e6 * target_fps))
    query = np.linspace(t0, t1, n_out).astype(np.int64)
    idx = np.searchsorted(timestamps_us, query)
    idx = np.clip(idx, 0, len(timestamps_us) - 1)
    return np.unique(idx)


def global_to_local(G, parents, root_idx):
    """Inline kimodo global_rots_to_local_rots: local[j] = G[parent]^T @ G[j]."""
    B, J = G.shape[0], G.shape[1]
    par_G = G[:, parents].copy()                 # (B,J,3,3)
    par_G[:, root_idx] = np.eye(3)
    return np.einsum("bjnm,bjno->bjmo", par_G, G)  # par_G^T @ G


def fk_positions(local, root_pos, neutral, parents, root_idx):
    """Minimal kimodo FK (root_positions_is_global=True) for the assert."""
    B, J = local.shape[0], local.shape[1]
    neutral = neutral - neutral[root_idx]
    gr = np.zeros((B, J, 3, 3)); pos = np.zeros((B, J, 3))
    for j in range(J):
        p = parents[j]
        if p < 0 or j == root_idx:
            gr[:, j] = local[:, j]
        else:
            gr[:, j] = gr[:, p] @ local[:, j]
            pos[:, j] = pos[:, p] + np.einsum("bij,j->bi", gr[:, p], neutral[j] - neutral[p])
    return pos + root_pos[:, None]


class Converter:
    def __init__(self, device="cpu", ground=True):
        self.ground = ground   # subtract per-seq min-foot from stored Y (feet-at-0)
        self.layer = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                               device=device, mode="warp")
        # capture T_world from the skinning pose
        self._cap = {}
        _orig = self.layer.batched_skinning.pose
        def _patched(*a, **k):
            k["return_transforms"] = True
            out = _orig(*a, **k)
            self._cap["T"] = out[1].detach()
            return out
        self.layer.batched_skinning.pose = _patched
        # kimodo joint hierarchy (order matches SOMA joint_names[1:])
        b = np.load(KIMODO_BUFFERS, allow_pickle=True)
        self.parents = b["joint_parents"].astype(int)
        self.root_idx = int(b["root_idx"])
        self.names = [str(x) for x in b["bone_order_names"]]
        self.head = self.names.index("Head")
        self.lf = self.names.index("LeftFoot"); self.rf = self.names.index("RightFoot")
        # lowest joints for floor estimation (toes if present, else feet)
        floor_names = [n for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")
                       if n in self.names]
        self.floor_j = [self.names.index(n) for n in floor_names]

    def convert_one(self, soma_npz: Path, out_path: Path, target_fps: float) -> dict:
        z = np.load(soma_npz, allow_pickle=True)
        poses = z["poses"].astype(np.float32)
        transl = z["transl"].astype(np.float32)
        timestamps_us = z["timestamps_us"].astype(np.int64)
        identity_coeffs = z["identity_coeffs"].astype(np.float32)

        self.layer.prepare_identity(torch.from_numpy(identity_coeffs).float())
        idx = pick_indices_at_fps(timestamps_us, target_fps)
        with torch.no_grad():
            self.layer.pose(torch.from_numpy(poses[idx]),
                            transl=torch.from_numpy(transl[idx]), pose2rot=True)
            Tw_posed = self._cap["T"].cpu().numpy()           # (B,78,4,4)
            self.layer.pose(torch.zeros(1, 77, 3), transl=torch.zeros(1, 3), pose2rot=True)
            Tw_rest = self._cap["T"].cpu().numpy()            # (1,78,4,4)

        R_posed = Tw_posed[:, 1:, :3, :3].astype(np.float64)  # drop Root -> 77
        R_rest = Tw_rest[0, 1:, :3, :3].astype(np.float64)
        N = Tw_rest[0, 1:, :3, 3].astype(np.float64)          # rest positions (Y-up template)
        hips_world = Tw_posed[:, 1, :3, 3].astype(np.float64) # joint idx1 = Hips

        G_rel = np.einsum("bjmn,jno->bjmo", R_posed, np.transpose(R_rest, (0, 2, 1)))
        G_rel_y = np.einsum("mn,bjno->bjmo", R_Z2Y, G_rel)
        local = global_to_local(G_rel_y, self.parents, self.root_idx)   # (B,77,3,3)
        root_y = hips_world @ R_Z2Y.T                                   # (B,3)
        neutral = (N - N[self.root_idx]).astype(np.float32)            # (77,3) Y-up, centered

        # floor_offset = whole-sequence min foot/toe height (Y-up). Recorded in the
        # NPZ so grounding is reversible/toggleable downstream. Only SUBTRACTED from
        # the stored Y when self.ground is True (BONES-SEED feet-at-0 convention);
        # with --no-ground the stored height is the raw rotated SLAM-world height.
        foot_world = Tw_posed[:, [j + 1 for j in self.floor_j], :3, 3].astype(np.float64)  # (B,k,3) Z-up
        foot_y = (foot_world @ R_Z2Y.T)[..., 1]                        # (B,k) Y-up height
        floor_offset = float(foot_y.min())
        ground_shift = floor_offset if self.ground else 0.0
        root_y[:, 1] -= ground_shift

        # --- per-seq assert (activity-independent): kimodo FK must reproduce the
        # SOMA true joint positions (rotated Z->Y, same ground shift). Robust to
        # whether the actor stands, sits, leans or lies down.
        P_soma = Tw_posed[:, 1:, :3, 3].astype(np.float64)            # (B,77,3) Z-up true joints
        expected = P_soma @ R_Z2Y.T                                   # -> Y-up
        expected[:, :, 1] -= ground_shift
        sel = np.linspace(0, local.shape[0] - 1, min(60, local.shape[0])).astype(int)
        posed = fk_positions(local[sel], root_y[sel], neutral.astype(np.float64),
                             self.parents, self.root_idx)
        geo_err = float(np.abs(posed - expected[sel]).max())
        if not (np.isfinite(local).all() and geo_err < 0.05):
            raise ValueError(
                f"FK geo-check failed for {soma_npz.parent.parent.name}/{soma_npz.parent.name}: "
                f"max |FK - rotated SOMA joints| = {geo_err:.3e} m (>0.05); finite={np.isfinite(local).all()}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            local_rot_mats=local.astype(np.float32),
            root_positions=root_y.astype(np.float32),
            timestamps_us=timestamps_us[idx].astype(np.int64),
            fps=np.int64(target_fps),
            source_seq=str(soma_npz.parent.parent.name),
            source_subject=str(soma_npz.parent.parent.parent.name),
            neutral_joints=neutral,
            identity_coeffs=identity_coeffs,
            floor_offset=np.float32(floor_offset),   # per-seq min foot Y (Y-up); see `grounded`
            grounded=np.bool_(self.ground),          # whether floor_offset was subtracted from Y
        )
        return {"out_frames": int(local.shape[0]), "geo_err_m": geo_err,
                "height": float(np.ptp(neutral[:, 1])), "floor_offset": floor_offset}


def discover(input_root: Path):
    return sorted(p.parent.parent for p in input_root.rglob(SOMA_REL.name)
                  if p.match(f"*/{SOMA_REL}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--target-fps", type=float, default=TARGET_FPS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--single", type=Path, default=None)
    ap.add_argument("--no-ground", action="store_true",
                    help="store raw rotated SLAM-world height (do NOT subtract per-seq "
                         "min-foot). floor_offset is still saved in the NPZ for later use.")
    args = ap.parse_args()

    print(f"[load] SOMALayer ({args.device})  ground={not args.no_ground}")
    conv = Converter(device=args.device, ground=not args.no_ground)

    if args.single is not None:
        seqs = [args.single]
    else:
        seqs = discover(args.input_root)[args.shard_id::args.num_shards]
        if args.limit:
            seqs = seqs[: args.limit]
    print(f"[run] {len(seqs)} sequence(s)  shard {args.shard_id}/{args.num_shards}")

    ok = skip = fail = 0
    t0 = time.time()
    for i, seq in enumerate(seqs, 1):
        out_path = args.output_root / seq.parent.name / f"{seq.name}.npz"
        if out_path.exists() and not args.overwrite:
            skip += 1
            continue
        soma_npz = seq / SOMA_REL
        if not soma_npz.is_file():
            fail += 1; print(f"  [missing] {soma_npz}"); continue
        try:
            m = conv.convert_one(soma_npz, out_path, args.target_fps)
            ok += 1
            if i % 25 == 0 or i <= 3:
                print(f"  [{i}/{len(seqs)}] {seq.parent.name}/{seq.name} "
                      f"frames={m['out_frames']} geo_err={m['geo_err_m']:.1e} h={m['height']:.2f}")
        except Exception as e:
            fail += 1; print(f"  [FAIL] {seq.parent.name}/{seq.name}: {e}")
    print(f"\n[done] ok={ok} skip={skip} fail={fail} in {time.time()-t0:.0f}s -> {args.output_root}")


if __name__ == "__main__":
    main()
