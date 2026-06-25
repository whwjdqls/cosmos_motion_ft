"""Build forward_dynamics inputs using the corrected Cosmos camera frame:
rgb-optical + Rz(-90deg) (= OpenCV of the upright video; best directional match to the
pretrained model, cosine 0.51->0.70).

For each baseline sample:
  - gt_camera_opencv.npz (T_world_rgb) -> rotate optical axis by -90 -> gt_camera_cosmos.npz
  - 9D camera action (Cosmos-exact) -> camera_action_cosmos.json (+ _x8 translation-scaled)
  - fd record conditioning on the GT first frame + caption + that action
Run in the `cosmos` env (imports cosmos_framework via camera_to_action).
"""
from __future__ import annotations
import json, os
import numpy as np
from camera_to_action import camera_poses_to_action

RZ_NEG90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])  # right-mult => -90 about optical z


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
    ap.add_argument("--x8_only", action="store_true")
    A = ap.parse_args()
    ROOT = A.root
    samples = os.path.join(ROOT, "samples")
    fd_records = []
    for name in sorted(os.listdir(samples)):
        sdir = os.path.join(samples, name)
        d = np.load(os.path.join(sdir, "gt_camera_opencv.npz"))
        pos = d["cam_world_pos"].astype(np.float64)
        rot = d["cam_world_rot"].astype(np.float64) @ RZ_NEG90  # optical-axis -90
        np.savez(os.path.join(sdir, "gt_camera_cosmos.npz"),
                 cam_world_pos=pos.astype(np.float32), cam_world_rot=rot.astype(np.float32))

        act = camera_poses_to_action(pos, rot)                       # (T-1,9)
        json.dump(act.tolist(), open(os.path.join(sdir, "camera_action_cosmos.json"), "w"))
        act8 = act.copy(); act8[:, :3] *= 8.0
        json.dump(act8.tolist(), open(os.path.join(sdir, "camera_action_cosmos_x8.json"), "w"))

        meta = json.load(open(os.path.join(sdir, "meta.json")))
        common = dict(model_mode="forward_dynamics",
                      vision_path=os.path.join(sdir, "first_frame.png"),
                      prompt=meta["caption"], domain_name="camera_pose", view_point="ego_view",
                      action_chunk_size=act.shape[0], image_size=480,
                      fps=int(round(meta["fps"])), num_steps=30, guidance=1.0, shift=10.0, seed=0)
        if not A.x8_only:
            fd_records.append({**common, "name": name,
                               "action_path": os.path.join(sdir, "camera_action_cosmos.json")})
        fd_records.append({**common, "name": name + "_camx8",
                           "action_path": os.path.join(sdir, "camera_action_cosmos_x8.json")})
        print(f"  {name}: cosmos-frame action {act.shape}  |Δt|={np.linalg.norm(act[:,:3],axis=1).mean():.4f}")

    fd_path = os.path.join(ROOT, "fd_cosmos_input.jsonl")
    with open(fd_path, "w") as f:
        for r in fd_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {fd_path} ({len(fd_records)} records)")


if __name__ == "__main__":
    main()
