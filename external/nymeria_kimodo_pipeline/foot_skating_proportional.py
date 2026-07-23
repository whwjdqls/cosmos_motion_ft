"""Foot skating over the Nymeria proportional set, same method as the BONES-SEED
proportional skating check: foot 3D speed (cm/s) during contact frames, pooled
over all sequences. Nymeria NPZs don't store posed_joints/foot_contacts, so we
FK with the actor neutrals and detect contacts with the same heuristic
motion_rep uses (foot_detect_from_pos_and_vel, vel_thres=0.15, height=0.10).
"""
import argparse, glob
import numpy as np, torch
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.motion_rep.feature_utils import compute_vel_xyz

s30, s77 = SOMASkeleton30(), SOMASkeleton77()
IDX30 = [s77.bone_order_names.index(n) for n in s30.bone_order_names]
FIDX = s30.foot_joint_idx  # [LeftFoot, LeftToeBase, RightFoot, RightToeBase]


def skating_speeds(npz_path, fps=20.0):
    d = np.load(npz_path, allow_pickle=True)
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))[IDX30]
    lrm30 = s30.from_SOMASkeleton77(lrm)
    T = lrm30.shape[0]
    _, posed, _ = s30.fk(lrm30, root, neutral_joints=nj.unsqueeze(0).expand(T, -1, -1))  # (T,30,3)
    lengths = torch.tensor([T])
    vel = compute_vel_xyz(posed.unsqueeze(0), fps, lengths=lengths)[0]
    fc = foot_detect_from_pos_and_vel(posed.unsqueeze(0), vel.unsqueeze(0), s30, 0.15, 0.10)[0].numpy()  # (T,4)
    p = posed.numpy()
    out = []
    for k in range(4):
        foot = p[:, FIDX[k]]
        speed = np.linalg.norm(foot[1:] - foot[:-1], axis=-1) * fps * 100.0  # cm/s (3D)
        contact = (fc[1:, k] > 0.5) & (fc[:-1, k] > 0.5)
        out.append(speed[contact])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/weka/jungbin/nymeriaplus_kimodo_proportional")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.root}/**/*.npz", recursive=True))[args.shard_id::args.num_shards]
    pool = []
    per_seq_mean = []
    for i, f in enumerate(files, 1):
        try:
            sp = skating_speeds(f)
            if sp.size:
                pool.append(sp); per_seq_mean.append(float(sp.mean()))
        except Exception as e:
            print(f"  [skip] {f}: {e}")
        if i % 25 == 0:
            print(f"  shard{args.shard_id}: {i}/{len(files)} done")
    pool = np.concatenate(pool) if pool else np.zeros(0, np.float32)
    np.savez(args.out, pool=pool.astype(np.float32), per_seq_mean=np.array(per_seq_mean, np.float32),
             n_seq=len(per_seq_mean))
    print(f"shard{args.shard_id}: {len(per_seq_mean)} seqs, {pool.size} contact-frames, "
          f"mean={pool.mean() if pool.size else 0:.2f} cm/s -> {args.out}")


if __name__ == "__main__":
    main()
