"""Build inference inputs for the 3 tasks from HELD-OUT TEST sequences (train_test_split.json).

Uses the preprocessed upright-RGB camera (camera_rgb), so GT camera + actions are in the same
frame the model was trained on. For N test windows emits:
  <OUT>/samples/<name>/{first_frame.png, gt_clip.mp4, gt_camera_cosmos.npz, camera_action.json, meta.json}
  <OUT>/invdyn_input.jsonl   (video -> camera)
  <OUT>/fd_input.jsonl       (image + action -> video)
  <OUT>/policy_input.jsonl   (image -> action + video)
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import imageio.v3 as iio
from nymeria_camera_rgb_dataset import rel_action_from_window, _rgb_path
from nymeria_camera_dataset import decode_window_pyav

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
SPLIT = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"


def pick_test_windows(n, num_frames, seed=0):
    test = set(json.load(open(SPLIT))["test"])
    cand = []
    for line in open(MANIFEST):
        r = json.loads(line)
        if r.get("uuid") not in test or not r.get("camera_path"):
            continue
        rgb = _rgb_path(r["camera_path"])
        if not os.path.isfile(rgb):
            continue
        nb = int(r.get("nb_frames", 0))
        for w in r.get("t2w_windows", []):
            cap = (w.get("caption") or "").lower()
            if w.get("usable") and w["start_frame"] + num_frames <= nb and \
               any(k in cap for k in ("walk", "turn")):
                cand.append((r["uuid"], int(w["start_frame"]), r["vision_path"], rgb, w["caption"]))
                break  # one window per test seq
    random.Random(seed).shuffle(cand)
    return cand[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_eval")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=97)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--image_size", type=int, default=480)
    args = ap.parse_args()
    picks = pick_test_windows(args.n, args.num_frames)
    print(f"picked {len(picks)} TEST-split windows")
    sdir_root = os.path.join(args.out, "samples"); os.makedirs(sdir_root, exist_ok=True)
    inv, fd, pol = [], [], []
    chunk = args.num_frames - 1

    for i, (uuid, s, vis, rgb, cap) in enumerate(picks):
        name = f"t{i}_{uuid.replace('/', '_')}"
        sd = os.path.join(sdir_root, name); os.makedirs(sd, exist_ok=True)
        frames = decode_window_pyav(vis, s, args.num_frames, args.fps)
        iio.imwrite(os.path.join(sd, "first_frame.png"), frames[0])
        iio.imwrite(os.path.join(sd, "gt_clip.mp4"), frames, fps=int(round(args.fps)), codec="libx264")
        d = np.load(rgb)
        pos = d["cam_world_pos_upright"][s:s + args.num_frames].astype(np.float32)
        rot = d["cam_world_rot_upright"][s:s + args.num_frames].astype(np.float32)
        np.savez(os.path.join(sd, "gt_camera_cosmos.npz"), cam_world_pos=pos, cam_world_rot=rot)
        act = rel_action_from_window(pos, rot)  # (T-1,9)
        json.dump(act.tolist(), open(os.path.join(sd, "camera_action.json"), "w"))
        json.dump({"uuid": uuid, "start_frame": s, "caption": cap,
                   "num_frames": args.num_frames, "fps": args.fps},
                  open(os.path.join(sd, "meta.json"), "w"), indent=2)

        gt_clip = os.path.join(sd, "gt_clip.mp4"); ff = os.path.join(sd, "first_frame.png")
        ap_json = os.path.join(sd, "camera_action.json")
        inv.append(dict(model_mode="inverse_dynamics", name=name, vision_path=gt_clip,
                        domain_name="camera_pose", view_point="ego_view", action_chunk_size=chunk,
                        image_size=args.image_size, fps=int(round(args.fps)),
                        num_steps=30, guidance=1.0, shift=10.0, seed=0, prompt=""))
        fd.append(dict(model_mode="forward_dynamics", name=name, vision_path=ff, prompt=cap,
                       action_path=ap_json, domain_name="camera_pose", view_point="ego_view",
                       action_chunk_size=chunk, image_size=args.image_size, fps=int(round(args.fps)),
                       num_steps=30, guidance=1.0, shift=10.0, seed=0))
        pol.append(dict(model_mode="policy", name=name, vision_path=ff, prompt=cap,
                        domain_name="camera_pose", view_point="ego_view", action_chunk_size=chunk,
                        image_size=args.image_size, fps=int(round(args.fps)),
                        num_steps=30, guidance=1.0, shift=10.0, seed=0))
        print(f"  [{i}] {name}  '{cap[:55]}'")

    for fn, recs in (("invdyn_input.jsonl", inv), ("fd_input.jsonl", fd), ("policy_input.jsonl", pol)):
        with open(os.path.join(args.out, fn), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(f"wrote invdyn/fd/policy jsonl to {args.out}")


if __name__ == "__main__":
    main()
