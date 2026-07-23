"""Stage B — build the Cosmos-ready training manifest that joins, per sequence:
  ego VIDEO (Stage A mp4)  <->  CAMERA trajectory sidecar  <->  MOTION npz  <->
  the per-slice atomic-action TEXT + GT/estimated floor grounding.

Output is shaped like Cosmos' SFT video JSONL
(`cosmos_framework/data/vfm/local_datasets/sft_dataset.py`): one JSON record per
sequence (one mp4), carrying the per-window list `t2w_windows` with `start_frame`/
`end_frame`/caption. We extend each window with the extra modalities this project
trains jointly: per-window `ground_offset_y` (the floor height to subtract — "removing
the floor height"), `floor_source`/`usable`/`ambiguous` flags, foot-skating, and
sequence-level `motion_path` + `camera_path` so a dataloader can pull the aligned
369-d kimodo motion and head-camera pose for the same frame window.

A training window therefore resolves to:
  video frames  mp4[start:end]                              (Cosmos VAE encodes these)
  motion        motion_npz: FK -> 369-d kimodo, grounded by  root_y -= ground_offset_y
  camera        camera_npz: cam_world_pos/rot[start:end], Z->Y, grounded by same offset
all 1:1 frame-aligned at 20 fps.

Only `usable==true` slices are emitted as windows, and each window is clipped to the
video's valid frame range (`valid_start..valid_end` from the Stage-A sidecar).

Inputs:
  --video-root  .../video/{Sxx}/{seq}.json   (Stage-A sidecars; presence => video exists)
  --floor       .../metadata/metadata_atomic_action_floor.jsonl
  --camera-root .../camera/{Sxx}/{seq}.npz   (optional; null if absent)
  --motion-root .../{Sxx}/{seq}.npz

Outputs (under --video-root):
  manifest_video.jsonl         one record per sequence-with-video (Cosmos SFT shape)
  manifest_video_slices.jsonl  flat: one record per usable window (convenience)
  manifest_video_stats.json
Env: numpy only (run in `kimodo`); no VRS / no GPU.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

MROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", type=Path, default=MROOT / "video")
    ap.add_argument("--floor", type=Path,
                    default=MROOT / "metadata" / "metadata_atomic_action_floor.jsonl")
    ap.add_argument("--camera-root", type=Path, default=MROOT / "camera")
    ap.add_argument("--motion-root", type=Path, default=MROOT)
    ap.add_argument("--min-frames", type=int, default=5,
                    help="drop windows shorter than this after clipping to valid range")
    args = ap.parse_args()

    # 1) sequences that have a Stage-A video sidecar
    vid_meta = {}
    for j in sorted(args.video_root.glob("S*/*.json")):
        m = json.loads(j.read_text())
        vid_meta[(m["subject"], m["filename"])] = m
    print(f"[scan] {len(vid_meta)} sequences with extracted video")

    # 2) usable atomic-action slices grouped by sequence
    slices_by_seq = defaultdict(list)
    n_rows = n_usable = 0
    with open(args.floor) as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            if not r.get("usable", False):
                continue
            n_usable += 1
            slices_by_seq[(r["subject"], r["filename"])].append(r)
    print(f"[scan] floor rows={n_rows} usable={n_usable}")

    seq_records, slice_records = [], []
    n_windows = n_clipped = n_dropped = 0
    for (subj, seq), m in sorted(vid_meta.items()):
        vstart, vend = int(m["valid_start"]), int(m["valid_end"])
        cam = args.camera_root / subj / f"{seq}.npz"
        motion = args.motion_root / subj / f"{seq}.npz"
        windows = []
        for r in sorted(slices_by_seq.get((subj, seq), []), key=lambda x: x["start_frame"]):
            sf, ef = int(r["start_frame"]), int(r["end_frame"])
            csf, cef = max(sf, vstart), min(ef, vend + 1)   # clip to valid video range
            if cef - csf < args.min_frames:
                n_dropped += 1
                continue
            if (csf, cef) != (sf, ef):
                n_clipped += 1
            w = {
                "start_frame": csf, "end_frame": cef,
                "caption": r.get("text", ""),
                "ground_offset_y": r.get("ground_offset_y"),
                "floor_source": r.get("floor_source"),
                "floor_status": r.get("floor_status"),
                "usable": True,
                "ambiguous": bool(r.get("ambiguous", False)),
                "est_ambiguous": bool(r.get("est_ambiguous", False)),
                "n_floors_in_slice": r.get("n_floors_in_slice"),
                "foot_skating_cms": r.get("foot_skating_cms"),
            }
            windows.append(w)
            slice_records.append({
                "uuid": f"{subj}/{seq}", "subject": subj, "filename": seq,
                "vision_path": m["vision_path"],
                "camera_path": str(cam) if cam.exists() else None,
                "motion_path": str(motion) if motion.exists() else None,
                "width": m["width"], "height": m["height"], "framerate": m["framerate"],
                **w,
            })
            n_windows += 1
        if not windows:
            continue
        seq_records.append({
            "uuid": f"{subj}/{seq}", "subject": subj, "filename": seq,
            "vision_path": m["vision_path"],
            "width": m["width"], "height": m["height"],
            "framerate": m["framerate"], "nb_frames": m["nb_frames"],
            "duration": m["duration"],
            "valid_start": vstart, "valid_end": vend, "n_invalid": m.get("n_invalid", 0),
            "camera_path": str(cam) if cam.exists() else None,
            "motion_path": str(motion) if motion.exists() else None,
            "t2w_windows": windows,
        })

    out_seq = args.video_root / "manifest_video.jsonl"
    out_sl = args.video_root / "manifest_video_slices.jsonl"
    with open(out_seq, "w") as f:
        for rec in seq_records:
            f.write(json.dumps(rec) + "\n")
    with open(out_sl, "w") as f:
        for rec in slice_records:
            f.write(json.dumps(rec) + "\n")

    n_cam = sum(1 for r in seq_records if r["camera_path"])
    stats = {
        "n_sequences_with_video": len(vid_meta),
        "n_sequences_in_manifest": len(seq_records),
        "n_sequences_with_camera": n_cam,
        "n_windows": n_windows, "n_windows_clipped_to_valid": n_clipped,
        "n_windows_dropped_too_short": n_dropped,
        "floor_rows": n_rows, "floor_usable": n_usable,
        "manifest_video": str(out_seq), "manifest_video_slices": str(out_sl),
    }
    (args.video_root / "manifest_video_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
