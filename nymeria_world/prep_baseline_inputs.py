"""Prepare zero-shot baseline inference inputs from NymeriaPlus instances.

For N diverse windows, emit:
  <OUT>/samples/<name>/first_frame.png   # i2v conditioning image (GT frame 0, full 640^2)
  <OUT>/samples/<name>/gt_clip.mp4       # frame-exact GT clip (NUM_FRAMES @ fps)
  <OUT>/samples/<name>/gt_camera.npz     # cam_world_pos/rot slice for the window
  <OUT>/samples/<name>/meta.json
  <OUT>/i2v_input.json                    # image2video records (text+image -> video)
  <OUT>/invdyn_input.json                 # inverse_dynamics records (video -> camera action)

Frame-exact decode (PyAV) guarantees gt_clip / gt_camera / model input all align.
"""
from __future__ import annotations

import argparse
import json
import os

import imageio.v3 as iio
import numpy as np

from nymeria_camera_dataset import decode_window_pyav

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"


def pick_windows(n: int, num_frames: int, skip_subjects=()):
    seen, picks = set(skip_subjects), []
    for line in open(MANIFEST):
        r = json.loads(line)
        if r["subject"] in seen or not r.get("camera_path"):
            continue
        nb = int(r.get("nb_frames", 0))
        for w in r.get("t2w_windows", []):
            cap = (w.get("caption") or "").lower()
            if w.get("usable") and w["start_frame"] + num_frames <= nb and \
               any(k in cap for k in ("walk", "turn")):
                picks.append((r["uuid"], int(w["start_frame"]), r["vision_path"],
                              r["camera_path"], w["caption"]))
                seen.add(r["subject"])
                break
        if len(picks) >= n:
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=97)  # 4*24+1
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--resolution", type=int, default=480)   # i2v gen res
    ap.add_argument("--image_size", type=int, default=480)   # invdyn resize
    ap.add_argument("--skip_subjects", type=str, default="")  # comma list, e.g. S01,S02,S03
    args = ap.parse_args()

    skip = tuple(s for s in args.skip_subjects.split(",") if s)
    picks = pick_windows(args.n, args.num_frames, skip)
    print(f"picked {len(picks)} windows")
    samples_dir = os.path.join(args.out, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    i2v_records, invdyn_records = [], []
    chunk = args.num_frames - 1  # action_chunk_size: NUM_FRAMES frames -> NUM_FRAMES-1 deltas

    for i, (uuid, start, vis, cam, caption) in enumerate(picks):
        name = f"s{i}_{uuid.replace('/', '_')}"
        sdir = os.path.join(samples_dir, name)
        os.makedirs(sdir, exist_ok=True)

        # frame-exact GT clip + first frame
        frames = decode_window_pyav(vis, start, args.num_frames, args.fps)  # (T,H,W,3) uint8
        iio.imwrite(os.path.join(sdir, "first_frame.png"), frames[0])
        gt_clip = os.path.join(sdir, "gt_clip.mp4")
        iio.imwrite(gt_clip, frames, fps=int(round(args.fps)), codec="libx264")

        # GT camera slice
        d = np.load(cam)
        pos = d["cam_world_pos"][start:start + args.num_frames].astype(np.float32)
        rot = d["cam_world_rot"][start:start + args.num_frames].astype(np.float32)
        np.savez(os.path.join(sdir, "gt_camera.npz"), cam_world_pos=pos, cam_world_rot=rot)

        json.dump({"uuid": uuid, "start_frame": start, "caption": caption,
                   "num_frames": args.num_frames, "fps": args.fps},
                  open(os.path.join(sdir, "meta.json"), "w"), indent=2)

        # i2v record: condition on GT first frame + caption, generate NUM_FRAMES
        i2v_records.append({
            "model_mode": "image2video",
            "name": name,
            "vision_path": os.path.join(sdir, "first_frame.png"),
            "prompt": caption,
            "num_frames": args.num_frames,
            "fps": int(round(args.fps)),
            "resolution": str(args.resolution),
            "num_steps": 35, "guidance": 6.0, "shift": 10.0, "seed": 0,
        })

        # inverse_dynamics record: GT clip -> predicted camera action
        invdyn_records.append({
            "model_mode": "inverse_dynamics",
            "name": name,
            "vision_path": gt_clip,
            "domain_name": "camera_pose",
            "view_point": "ego_view",
            "action_chunk_size": chunk,
            "image_size": args.image_size,
            "fps": int(round(args.fps)),
            "num_steps": 30, "guidance": 1.0, "shift": 10.0, "seed": 0,
            "prompt": "",
        })
        print(f"  [{i}] {name}  start={start}  frames={frames.shape}  '{caption[:60]}...'")

    # one record per line (.jsonl): the inference loader treats .json as a single record
    with open(os.path.join(args.out, "i2v_input.jsonl"), "w") as f:
        for r in i2v_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "invdyn_input.jsonl"), "w") as f:
        for r in invdyn_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out}/i2v_input.jsonl  and  invdyn_input.jsonl")


if __name__ == "__main__":
    main()
