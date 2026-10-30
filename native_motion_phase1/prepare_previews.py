"""Prepare four fixed held-out cases using the existing Phase-3 motion transform."""
import json
import os
from pathlib import Path
import numpy as np
from nymeria_joint_dataset import NymeriaJointDataset
from dataset import humanize_caption
from native_phase_training.evaluate_inverse_forward import _read_video_rgb, _resize_gt_like_native


def main():
    root = Path(os.environ["WEKA_ROOT"]) / "nymeriaplus_kimodo_proportional"
    source, dest = Path(os.environ["NATIVEP1_PREVIEW_INPUT_DIR"]), Path(os.environ["MOTIONP1_PREVIEW_DATA"])
    dest.mkdir(parents=True, exist_ok=True)
    dataset = NymeriaJointDataset(
        manifest_path=str(root / "video/manifest_video.jsonl"), split_file=str(root / "train_test_split.json"),
        split="test", num_frames=97, aligned_num_frames=97, fps=20,
        task_weights={"video2motion": .5, "motimg2video": .5}, bones_text2motion_frac=0,
        cfg_dropout=0, prefer_latents=False, train=False, uniego_root=os.environ["NYMERIA_UNIEGO_ROOT"],
        floor_calibration_json=str(root / "metadata/floor_calibration.json"), seed=0)
    rows = [json.loads(s) for s in (source / "fd_input.jsonl").read_text().splitlines() if s.strip()]
    split = json.loads((root / "train_test_split.json").read_text())
    output = []
    for task, index in zip(("text2motion", "textimg2motion", "video2motion", "motion2video"), (0, 23, 46, 70)):
        case = rows[index]["name"].removesuffix("_forward_dynamics")
        folder = source / "samples" / case
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["uuid"] in split["test"] and meta["uuid"] not in split["train"]
        match = next(r for r in dataset._index if r["uuid"] == meta["uuid"] and r["s"] == meta["start_frame"])
        motion, neutral, pad = dataset._load_motion(match["uni"], match["s"], match["off"], avail=97)
        assert not pad.any() and np.isfinite(motion).all()
        rgb = _resize_gt_like_native(_read_video_rgb(folder / "gt_clip.mp4"), 256, 256)
        assert rgb.shape == (97, 256, 256, 3)
        np.savez(dest / (task + ".npz"), motion=motion, neutral=neutral, rgb=rgb)
        output.append(dict(task=task, case=case, uuid=meta["uuid"], start=meta["start_frame"],
                           caption=humanize_caption(meta["caption"]), bundle=task + ".npz"))
    (dest / "cases.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
