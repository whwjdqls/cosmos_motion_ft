"""Prep forward_dynamics inputs (image + optional text + camera movement -> video).

Reuses the already-extracted baseline samples. For each sample:
  - converts GT camera npz -> 9D camera pseudo-action [T,9] (Cosmos-exact)
  - writes camera_action.json (as-is) and camera_action_x8.json (translation*8)
  - emits fd_input.jsonl records conditioning on the GT first frame + caption + that action path

Also re-patches i2v_input.jsonl to square aspect (1,1) so the generated framing matches the
square egocentric GT.
"""
from __future__ import annotations

import argparse, json, os

import numpy as np

from camera_to_action import camera_poses_to_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
    ap.add_argument("--image_size", type=int, default=480)
    ap.add_argument("--trans_scale", type=float, default=8.0)
    args = ap.parse_args()
    samples = os.path.join(args.root, "samples")
    names = sorted(os.listdir(samples))

    fd_records = []
    for name in names:
        sdir = os.path.join(samples, name)
        meta = json.load(open(os.path.join(sdir, "meta.json")))
        d = np.load(os.path.join(sdir, "gt_camera.npz"))
        act = camera_poses_to_action(d["cam_world_pos"], d["cam_world_rot"])  # (T-1,9)
        T = act.shape[0]
        json.dump(act.tolist(), open(os.path.join(sdir, "camera_action.json"), "w"))
        act8 = act.copy(); act8[:, :3] *= args.trans_scale
        json.dump(act8.tolist(), open(os.path.join(sdir, "camera_action_x8.json"), "w"))

        first_frame = os.path.join(sdir, "first_frame.png")
        common = dict(model_mode="forward_dynamics", vision_path=first_frame,
                      prompt=meta["caption"], domain_name="camera_pose", view_point="ego_view",
                      action_chunk_size=T, image_size=args.image_size, fps=int(round(meta["fps"])),
                      num_steps=30, guidance=1.0, shift=10.0, seed=0)
        fd_records.append({**common, "name": name,
                           "action_path": os.path.join(sdir, "camera_action.json")})
        fd_records.append({**common, "name": name + "_camx8",
                           "action_path": os.path.join(sdir, "camera_action_x8.json")})
        print(f"  {name}: action {act.shape}  GT|Δt|={np.linalg.norm(act[:,:3],axis=1).mean():.3f} "
              f"x8|Δt|={np.linalg.norm(act8[:,:3],axis=1).mean():.3f}")

    fd_path = os.path.join(args.root, "fd_input.jsonl")
    with open(fd_path, "w") as f:
        for r in fd_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {fd_path} ({len(fd_records)} records)")

    # patch i2v to square aspect
    i2v_path = os.path.join(args.root, "i2v_input.jsonl")
    if os.path.exists(i2v_path):
        recs = [json.loads(l) for l in open(i2v_path) if l.strip()]
        for r in recs:
            r["aspect_ratio"] = "1,1"
        with open(i2v_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"patched {i2v_path} -> aspect_ratio 1,1")


if __name__ == "__main__":
    main()
