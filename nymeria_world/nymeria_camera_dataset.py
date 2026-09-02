"""Map-style NymeriaPlus ego-video + text + camera-action dataset.

Emits the EXACT raw-sample contract that Cosmos's `ActionTransformPipeline` consumes
(same as `DROIDLeRobotDataset._build_result`), so it plugs into the native action SFT
path unchanged:

    base sample (this class) -> ActionSFTDataset(base, ActionTransformPipeline, resolution)

Per-sample dict returned by __getitem__:
    video            uint8  [3, T, H, W]   (model does /127.5 - 1 internally)
    action           float  [T-1, 9]       (camera pseudo-action; pipeline pads to max_action_dim)
    ai_caption       str                    (NymeriaPlus narration for the window)
    conditioning_fps long   scalar          (20)
    mode             str                    forward_dynamics | inverse_dynamics | policy
    domain_id        long   scalar          2  (camera_pose)
    viewpoint        str                    "ego_view"

Reads `manifest_video.jsonl` (Cosmos-SFT shape): one record/sequence with `vision_path`
(mp4), `camera_path` (npz), and `t2w_windows[]` (each has start/end_frame + caption + usable).
Video windows are decoded on demand with PyAV (frame-accurate seek; the mp4s are ~17k frames).
Camera actions come from `camera_to_action.camera_poses_to_action` (matches Cosmos exactly).
"""
from __future__ import annotations

import functools
import json
import random
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch.utils.data import Dataset

from camera_to_action import DOMAIN_ID, camera_poses_to_action  # local module

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


@functools.lru_cache(maxsize=16)
def _load_camera_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["cam_world_pos"].astype(np.float32), d["cam_world_rot"].astype(np.float32)


def decode_window_pyav(path: str, start_frame: int, num_frames: int, fps: float) -> np.ndarray:
    """Frame-accurate windowed decode -> (num_frames, H, W, 3) uint8 RGB.

    Seeks to the nearest keyframe before `start_frame`, decodes forward, and keeps frames
    whose integer index falls in [start_frame, start_frame+num_frames)."""
    want = set(range(start_frame, start_frame + num_frames))
    out: dict[int, np.ndarray] = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        # target presentation timestamp for start_frame, seek backward to a keyframe
        target_pts = int(start_frame / fps / tb)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            idx = int(round(float(frame.pts * tb) * fps))
            if idx in want:
                out[idx] = frame.to_ndarray(format="rgb24")  # HWC uint8
            if idx >= start_frame + num_frames - 1 and len(out) >= num_frames:
                break
    frames = [out[i] for i in range(start_frame, start_frame + num_frames) if i in out]
    if len(frames) != num_frames:
        raise RuntimeError(
            f"{Path(path).name}: decoded {len(frames)}/{num_frames} frames at start={start_frame}"
        )
    return np.stack(frames, axis=0)  # (T,H,W,3)


class NymeriaPlusCameraActionDataset(Dataset):
    """Base map-style dataset; wrap with ActionSFTDataset for training."""

    def __init__(
        self,
        manifest_path: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl",
        num_frames: int = 33,            # Wan VAE wants T = 4N+1 (33 = 4*8+1)
        fps: float = 20.0,
        mode: str = "joint",             # "joint" -> random per-sample mode; or a fixed mode
        viewpoint: str = "ego_view",
        require_usable: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_frames % 4 != 1:
            raise ValueError(f"num_frames must be 4N+1 for the Wan VAE (got {num_frames})")
        if mode != "joint" and mode not in _MODE_CHOICES:
            raise ValueError(f"mode must be 'joint' or one of {_MODE_CHOICES}")
        self._num_frames = int(num_frames)
        self._fps = float(fps)
        self._mode = mode
        self._viewpoint = viewpoint
        self._rng = random.Random(seed)

        # Flatten manifest -> one training sample per usable window with >= num_frames frames.
        self._index: list[dict[str, Any]] = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                cam = resolve_legacy_path(rec.get("camera_path"))
                vis = resolve_legacy_path(rec.get("vision_path"))
                if not cam or not vis:
                    continue
                nb = int(rec.get("nb_frames", 0))
                for w in rec.get("t2w_windows", []):
                    if require_usable and not w.get("usable", False):
                        continue
                    s = int(w["start_frame"])
                    if s + self._num_frames > nb:
                        continue
                    cap = w.get("caption")
                    if not cap:
                        continue
                    self._index.append(
                        {"vision_path": vis, "camera_path": cam, "start_frame": s, "caption": cap}
                    )
        if max_samples is not None:
            self._index = self._index[:max_samples]

    def __len__(self) -> int:
        return len(self._index)

    @property
    def fps(self) -> float:
        return self._fps

    def _choose_mode(self) -> str:
        return self._rng.choice(_MODE_CHOICES) if self._mode == "joint" else self._mode

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._index[idx]
        s, T = item["start_frame"], self._num_frames

        # --- video: (T,H,W,3) uint8 -> [3,T,H,W] ---
        frames = decode_window_pyav(item["vision_path"], s, T, self._fps)
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # [3,T,H,W] uint8

        # --- camera action: poses[s:s+T] -> (T-1, 9) ---
        pos, rot = _load_camera_npz(item["camera_path"])
        act = camera_poses_to_action(pos[s : s + T], rot[s : s + T])       # (T-1, 9)
        action = torch.from_numpy(act).float()

        return {
            "video": video,
            "action": action,
            "ai_caption": item["caption"],
            "conditioning_fps": torch.tensor(int(round(self._fps)), dtype=torch.long),
            "mode": self._choose_mode(),
            "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
            "viewpoint": self._viewpoint,
        }


# --------------------------------------------------------------------------------------
# No-GPU smoke: base contract + run through the REAL ActionTransformPipeline
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_frames", type=int, default=33)
    ap.add_argument("--resolution", type=str, default="256")
    ap.add_argument("--n", type=int, default=4, help="samples to pull")
    ap.add_argument("--mode", type=str, default="joint")
    args = ap.parse_args()

    base = NymeriaPlusCameraActionDataset(num_frames=args.num_frames, mode=args.mode)
    print(f"[index] {len(base)} usable windows (>= {args.num_frames} frames)")

    # tokenizer_config=None -> skip text tokenization (no model needed for this smoke)
    pipe = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        max_action_dim=64,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
    )
    ds = ActionSFTDataset(base, pipe, args.resolution)

    for i in range(args.n):
        b = base[i]
        print(f"\n--- base[{i}] mode={b['mode']} ---")
        print(f"  video {tuple(b['video'].shape)} {b['video'].dtype} range[{int(b['video'].min())},{int(b['video'].max())}]")
        print(f"  action {tuple(b['action'].shape)} {b['action'].dtype} "
              f"trans|.|max={b['action'][:,:3].norm(dim=1).max():.3f} rot6d[{b['action'][:,3:].min():.3f},{b['action'][:,3:].max():.3f}]")
        print(f"  domain_id={int(b['domain_id'])} fps={int(b['conditioning_fps'])} viewpoint={b['viewpoint']}")
        print(f"  caption: {b['ai_caption'][:90]}...")
        t = ds[i]
        sp = t["sequence_plan"]
        print(f"  [piped] video {tuple(t['video'].shape)} action "
              f"{None if t['action'] is None else tuple(t['action'].shape)} "
              f"raw_action_dim={None if t.get('raw_action_dim') is None else int(t['raw_action_dim'])}")
        print(f"  [piped] sequence_plan: {sp.as_dict()}")
        print(f"  [piped] image_size={t.get('image_size')}")
    print("\n[ok] base dataset + ActionTransformPipeline produce the Cosmos action contract.")
from runtime_paths import resolve_legacy_path
