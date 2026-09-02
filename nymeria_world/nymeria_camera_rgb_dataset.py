"""NymeriaPlus camera-only Phase-2 dataset (uses PREPROCESSED upright-RGB camera poses).

Emits the Cosmos action-SFT contract for a 4-task mixture, all from the SAME aligned
(video, camera) windows:

  1. inverse_dynamics  : video -> camera action            (NO text)
  2. forward_dynamics  : image + action [+text] -> video
  3. policy            : image -> action + video [+text]
  4. image2video       : image [+text] -> video            (no action)

"[+text]" = the caption is used, dropped 10% of the time (ActionTransformPipeline
cfg_dropout_rate=0.1) so the empty prompt is a valid CFG-null at inference. For
inverse_dynamics the caption is always empty (the task has no instruction).

Camera action comes from the preprocessed UPRIGHT RGB poses
(`camera_rgb/{Sxx}/{seq}.npz`, see nymeria_kimodo_pipeline/camera/CAMERA_RGB_PREPROCESS.md),
NOT the raw device frame. Per-frame relative action = Cosmos-exact pose_abs_to_rel(rot6d,
backward_framewise).

Wrap with ActionSFTDataset(base, ActionTransformPipeline(cfg_dropout_rate=0.1, ...), resolution).
"""
from __future__ import annotations
import functools, json, os, random, signal
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

try:  # Cosmos-3 Nano framework layout
    from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel
except ModuleNotFoundError:  # Cosmos-3 Edge framework layout
    from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
from camera_to_action import DOMAIN_ID  # camera_pose -> 2

# A single corrupt/undecodable video makes one rank's PyAV worker hang -> stalls the
# collective on every rank -> distributed deadlock (0% GPU util, no crash). Guard the
# per-window decode with a hard timeout + skip-to-next-window so no sample can stall a rank.
_DECODE_TIMEOUT = int(os.environ.get("NYMERIA_DECODE_TIMEOUT", "120"))  # seconds per window
_MAX_SKIP = int(os.environ.get("NYMERIA_MAX_SKIP", "64"))               # consecutive bad windows to skip


class _DecodeTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _DecodeTimeout(f"decode exceeded {_DECODE_TIMEOUT}s")
from nymeria_camera_dataset import decode_window_pyav

# 4-task mixture (weights). inverse_dynamics has no text; others use 10% CFG dropout.
MODE_WEIGHTS = {
    "forward_dynamics": 0.40,
    "inverse_dynamics": 0.25,
    "policy": 0.20,
    "image2video": 0.15,
}
# diagnostics: drop tasks listed in $NYMERIA_DROP_MODES (comma-sep) from the mixture.
_drop = {m for m in os.environ.get("NYMERIA_DROP_MODES", "").split(",") if m}
if _drop:
    MODE_WEIGHTS = {k: v for k, v in MODE_WEIGHTS.items() if k not in _drop}


@functools.lru_cache(maxsize=32)
def _load_rgb_cam(path: str):
    d = np.load(path)
    return d["cam_world_pos_upright"].astype(np.float32), d["cam_world_rot_upright"].astype(np.float32)


def _rgb_path(camera_path: str) -> str:
    return camera_path.replace("/camera/", "/camera_rgb/")


def rel_action_from_window(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """upright poses (T,3)+(T,3,3) -> (T-1,9) Cosmos-exact camera action."""
    T = len(pos)
    P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return pose_abs_to_rel(P.astype(np.float64), rotation_format="rot6d",
                           pose_convention="backward_framewise").astype(np.float32)


class NymeriaCameraRGBDataset(Dataset):
    def __init__(
        self,
        manifest_path: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl",
        num_frames: int = 33,
        fps: float = 20.0,
        mode: str = "mixture",          # "mixture" or a fixed mode
        require_usable: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
        require_rgb_cam: bool = True,    # only keep windows whose camera_rgb npz exists
        split: str = "all",             # "train" | "test" | "all" (per-sequence)
        split_file: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json",
    ):
        super().__init__()
        if num_frames % 4 != 1:
            raise ValueError(f"num_frames must be 4N+1 (got {num_frames})")
        self._num_frames = int(num_frames)
        self._fps = float(fps)
        self._mode = mode
        self._modes = list(MODE_WEIGHTS); self._mw = [MODE_WEIGHTS[m] for m in self._modes]
        self._rng = random.Random(seed)
        # per-sequence train/test split (hold out whole recordings; no window-level leakage)
        keep_uuids = None
        if split not in ("all", None):
            sp = json.load(open(split_file))
            assert split in sp, f"split '{split}' not in {split_file}"
            keep_uuids = set(sp[split])
        self._index: list[dict[str, Any]] = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                cam = resolve_legacy_path(rec.get("camera_path"))
                vis = resolve_legacy_path(rec.get("vision_path"))
                nb = int(rec.get("nb_frames", 0))
                if not cam or not vis:
                    continue
                if keep_uuids is not None and rec.get("uuid") not in keep_uuids:
                    continue
                rgb = _rgb_path(cam)
                if require_rgb_cam and not os.path.isfile(rgb):
                    continue
                for w in rec.get("t2w_windows", []):
                    if require_usable and not w.get("usable", False):
                        continue
                    s = int(w["start_frame"])
                    if s + self._num_frames > nb or not w.get("caption"):
                        continue
                    self._index.append({"vis": vis, "rgb": rgb, "s": s, "cap": w["caption"]})
        if max_samples is not None:
            self._index = self._index[:max_samples]

    def __len__(self):
        return len(self._index)

    @property
    def fps(self):
        return self._fps

    def _choose_mode(self):
        if self._mode != "mixture":
            return self._mode
        return self._rng.choices(self._modes, weights=self._mw, k=1)[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        n = len(self._index)
        # Deterministically skip to the next window on any decode failure/timeout (corrupt video,
        # short camera npz, NaN action). Runs in the DataLoader worker's main thread, so SIGALRM works.
        has_alarm = hasattr(signal, "SIGALRM")
        last_err = None
        for attempt in range(_MAX_SKIP):
            j = (idx + attempt) % n
            it = self._index[j]; s, T = it["s"], self._num_frames
            try:
                if has_alarm:
                    _old = signal.signal(signal.SIGALRM, _on_alarm); signal.alarm(_DECODE_TIMEOUT)
                frames = decode_window_pyav(it["vis"], s, T, self._fps)     # (T,H,W,3) uint8
                pos, rot = _load_rgb_cam(it["rgb"])
                if has_alarm:
                    signal.alarm(0); signal.signal(signal.SIGALRM, _old)
                act = rel_action_from_window(pos[s:s + T], rot[s:s + T])    # (T-1,9)
                if frames.shape[0] != T or act.shape[0] != T - 1 or not np.isfinite(act).all():
                    raise ValueError(f"bad shapes/NaN: frames {frames.shape} act {act.shape}")
            except Exception as e:  # noqa: BLE001 — skip ANY bad window so it can't stall the collective
                if has_alarm:
                    signal.alarm(0)
                last_err = e
                if attempt < 3 or attempt % 16 == 0:
                    print(f"[NymeriaCameraRGB] skip bad window {it['vis']}@{s} "
                          f"({type(e).__name__}: {e}); attempt {attempt + 1}/{_MAX_SKIP}", flush=True)
                continue
            video = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
            caption = "" if mode == "inverse_dynamics" else it["cap"]
            return {
                "video": video,
                "action": torch.from_numpy(act).float(),
                "ai_caption": caption,
                "conditioning_fps": torch.tensor(int(round(self._fps)), dtype=torch.long),
                "mode": mode,
                "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
                "viewpoint": "ego_view",
            }
        raise RuntimeError(f"no decodable window after {_MAX_SKIP} attempts from idx {idx}: {last_err}")


def get_nymeria_camera_sft_dataset(
    *,
    manifest_path: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl",
    num_frames: int = 33,
    fps: float = 20.0,
    mode: str = "mixture",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    max_samples: int | None = None,
    split: str = "train",
):
    """Factory: base NymeriaPlus camera dataset -> ActionTransformPipeline (4-task mixture, 10% CFG)."""
    from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    base = NymeriaCameraRGBDataset(manifest_path=manifest_path, num_frames=num_frames,
                                   fps=fps, mode=mode, max_samples=max_samples, split=split)
    pipe = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,           # 10% text dropout -> CFG null
        max_action_dim=max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
    )
    return ActionSFTDataset(base, pipe, resolution)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    base = NymeriaCameraRGBDataset()
    print(f"[index] {len(base)} usable windows with preprocessed camera_rgb")
    from collections import Counter
    ds = get_nymeria_camera_sft_dataset()
    c = Counter()
    for i in range(a.n):
        b = base[i]; c[b["mode"]] += 1
        t = ds[i]; sp = t["sequence_plan"]
        act = None if t["action"] is None else tuple(t["action"].shape)
        print(f"  [{i}] mode={b['mode']:16s} video={tuple(b['video'].shape)} action={act} "
              f"cap='{b['ai_caption'][:40]}' has_action={sp.has_action} cond_vis={sp.condition_frame_indexes_vision[:3]}")
    print("mode counts:", dict(c))
from runtime_paths import resolve_legacy_path
