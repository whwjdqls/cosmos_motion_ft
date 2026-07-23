"""Diagnose the SOMA-world -> kimodo-frame mapping using the uniform body, then
validate that mapping reproduces an actor's posed joints under kimodo FK.

Hypothesis: kimodo SOMASkeleton77's canonical neutral_joints are the SOMA
neutral-identity (betas=0) T-pose joints, expressed in a kimodo frame that
differs from SOMA world by a single global rotation R (e.g. up-axis convention).
If so:
  R = orthogonal-Procrustes( soma_uniform_tpose_centered -> kimodo_canon_centered )
and per-actor neutrals in kimodo frame are  N = R @ (soma_actor_tpose - root).

We then check kimodo FK(local_rot_mats, root_pos, N) reproduces the actor's
SOMA-X posed joints after mapping them through R about the root.
"""
from __future__ import annotations
import argparse, importlib.util, sys, types, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
KIMODO_ROOT = Path("/home/jungbin_cho/kimodo_open")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer  # noqa: E402
BUFS = "/weka/jungbin/kimodo_caches/_somaskel77_buffers.npz"


def load_kin():
    pk = types.ModuleType("kimodo"); ps = types.ModuleType("kimodo.skeleton"); pt = types.ModuleType("kimodo.tools")
    def eb(**k):
        def w(f):
            def i(*a, **kw): return f(*a, **kw)
            return i
        return w
    pt.ensure_batched = eb
    sys.modules.update({"kimodo": pk, "kimodo.skeleton": ps, "kimodo.tools": pt})
    spec = importlib.util.spec_from_file_location(
        "kimodo.skeleton.kinematics", KIMODO_ROOT / "kimodo/skeleton/kinematics.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tpose_joints(layer, ic):
    layer.prepare_identity(torch.from_numpy(ic).float())
    with torch.no_grad():
        out = layer.pose(torch.zeros(1, 77, 3), transl=torch.zeros(1, 3), pose2rot=True)
    return out["joints"][0].cpu().numpy().astype(np.float64)


def orthogonal_procrustes_R(A, B):
    """Best rotation R (3x3, det=+1) s.t. R@A.T ~ B.T for centered point sets A,B (N,3)."""
    H = A.T @ B                      # 3x3
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--kimodo", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=200)
    args = ap.parse_args()

    layer = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl", device="cpu", mode="warp")
    bufs = np.load(BUFS, allow_pickle=True)
    kimodo_canon = bufs["neutral_joints"].astype(np.float64)       # (77,3)
    parents_77 = bufs["joint_parents"].astype(int).tolist()
    root_idx = int(bufs["root_idx"])

    # 1. uniform identity (betas=0) T-pose
    soma_uni = tpose_joints(layer, np.zeros((1, 10), np.float32))   # (77,3)
    A = soma_uni - soma_uni[root_idx]
    B = kimodo_canon - kimodo_canon[root_idx]
    R = orthogonal_procrustes_R(A, B)
    resid = np.abs((A @ R.T) - B)
    print(f"[frame map] orthogonal Procrustes residual: max={resid.max():.4e} m  mean={resid.mean():.4e} m")
    print(f"[frame map] R=\n{np.round(R,3)}")
    # also a per-bone-length comparison uniform vs kimodo
    def bonelens(J):
        return np.array([np.linalg.norm(J[j]-J[parents_77[j]]) for j in range(len(J)) if parents_77[j] >= 0])
    bl_u, bl_k = bonelens(soma_uni), bonelens(kimodo_canon)
    print(f"[frame map] uniform vs kimodo bone-length max|Δ|={np.abs(bl_u-bl_k).max():.4e} m "
          f"(scale ratio median={np.median(bl_k/np.clip(bl_u,1e-9,None)):.4f})")

    # 2. actor identity + posed ground truth
    z = np.load(args.seq_dir / "body" / "xdata_soma.npz", allow_pickle=True)
    ic = z["identity_coeffs"].astype(np.float32)
    poses = z["poses"].astype(np.float32); transl = z["transl"].astype(np.float32)
    ts_o = z["timestamps_us"].astype(np.int64)
    soma_actor = tpose_joints(layer, ic)                            # (77,3) SOMA frame

    conv = np.load(args.kimodo)
    lrm_all = conv["local_rot_mats"].astype(np.float32); rp_all = conv["root_positions"].astype(np.float32)
    ts_c = conv["timestamps_us"].astype(np.int64)
    oi = np.clip(np.searchsorted(ts_o, ts_c), 0, len(ts_o)-1)
    n = min(args.n_frames, len(ts_c))
    sc = np.linspace(0, len(ts_c)-1, n, dtype=np.int64); so = oi[sc]

    layer.prepare_identity(torch.from_numpy(ic).float())
    with torch.no_grad():
        soma_posed = layer.pose(torch.from_numpy(poses[so]), transl=torch.from_numpy(transl[so]),
                                pose2rot=True)["joints"].cpu().numpy().astype(np.float64)  # (n,77,3) SOMA frame

    # map ground-truth posed joints into kimodo frame, about the per-frame root
    root_w = soma_posed[:, root_idx:root_idx+1, :]                  # (n,1,3)
    soma_posed_kimodo = (soma_posed - root_w) @ R.T + rp_all[sc][:, None, :]

    # 3. actor neutrals in kimodo frame = R @ (soma_actor centered)
    N = ((soma_actor - soma_actor[root_idx]) @ R.T).astype(np.float32)

    kin = load_kin()
    class S: pass
    skel = S(); skel.joint_parents = torch.tensor(parents_77); skel.neutral_joints = torch.from_numpy(N); skel.root_idx = root_idx
    _, posed_k, _ = kin.fk(torch.from_numpy(lrm_all[sc]), torch.from_numpy(rp_all[sc]), skel, root_positions_is_global=True)
    posed_k = posed_k.numpy().astype(np.float64)

    diff = np.abs(posed_k - soma_posed_kimodo)
    print(f"\n[validate] kimodo FK vs R-mapped SOMA posed: "
          f"root={np.abs(diff[:,root_idx]).max():.3e}  non-root max={diff[:, [i for i in range(77) if i!=root_idx]].max():.3e}  mean={diff.mean():.3e} m")
    jn = list(layer.rig_data["joint_names"])[1:]
    pj = diff.max(axis=(0,2)); worst = np.argsort(-pj)[:5]
    print("   worst-5:", [(jn[i], round(float(pj[i]),4)) for i in worst])


if __name__ == "__main__":
    main()
