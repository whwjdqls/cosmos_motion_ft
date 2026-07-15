"""5-modality aligned NymeriaPlus loader for the 7-task joint-attention model.

This is the DATA seam of ``DESIGN_7TASK.md`` section 4. It emits, per ALIGNED
NymeriaPlus ``(uuid, start, T=33)`` window, the SUPERSET row carrying every modality a
task could need::

    {
      "caption":         str               (blanked per task + 10% CFG; see task_plan text_policy)
      "video_frames":    uint8 [3,T,H,W]   (raw pixels; present iff no precomputed latents)
      "video_latents":   float32 [C,T_lat,h,w]  (precomputed Wan-VAE latents; preferred over frames)
      "reasoner_image":  uint8 [3,H,W]      (optional frame-0 image for reasoner-side image tasks)
      "first_frame_image": <folded into video frame 0 -- the "image" is video latent/frame 0 clean>
      "camera_action":   float32 [T-1,9]   (Cosmos camera pseudo-action; UN-normalized, domain 2)
      "motion":          float32 [T,283]   (z-scored uniego rep, frame-0 canonicalized, grounded)
      "neutral_joints":  float32 [30,3]    (centered shape cue)
      "motion_pad_mask": bool   [T]        (True = padded frame, excluded from loss)
      "mode":            str               (one of task_plan.TASKS)
      "domain_id":       long              (== CAMERA_DOMAIN_ID = 2)
      "source":          str               ("nymeria" | "bones")
    }

The KEY enabling fact (5-modality report): for every NymeriaPlus ``(uuid, start)`` window all of
{caption, first-frame image, video, camera action, motion, neutral_joints} are co-available and
frame-aligned (713/713 seqs, 0 missing). So ONE window of length ``T=33`` (4N+1 for the Wan VAE)
sliced at the SAME ``(uuid, start)`` feeds ALL 7 tasks:

  * video / camera  -- reuse ``NymeriaCameraRGBDataset``'s manifest index, ``decode_window_pyav``,
                       ``_load_rgb_cam`` and ``rel_action_from_window`` VERBATIM (the source of
                       truth for video+camera + the camera world-model tasks 1-3).
  * motion          -- reuse ``dataset.py``'s uniego loading: ``features[s:s+T]`` from the per-seq
                       ``uniego_rep`` npz, grounded with the window's ``ground_offset_y``,
                       frame-0-canonicalized, z-scored with the SAME 283-d stats, + centered
                       ``neutral_joints``. SAME ``(uuid, start)`` -> the motion is the exact same
                       window as the video/camera.

MODE ROUTING (MOTION-WEIGHTED mixture, ``task_plan.TASK_WEIGHTS``):
  Each sample first picks a ``mode`` from the task weights. Two things then decide where its data
  comes from: (1) which SOURCE (NymeriaPlus vs BONES) can supply the modalities the task needs, and
  (2) for NymeriaPlus, which INDEX (native motion-window vs T-frame video-aligned) it draws from.

  TWO NYMERIAPLUS INDICES (built together over the same captioned/usable manifest windows):
    * ``_t2m_index`` -- NATIVE motion-window index: ONE entry per captioned ~100-frame (~5 s @ 20 fps)
      atomic-action window, kept at its native span and later padded+masked to T. It is
      T-INDEPENDENT (no ``s + T <= we`` length test), so EVERY captioned window survives at any T.
      Used only by NymeriaPlus ``text2motion``. The caption correctly supervises its native
      motion span without requiring a matching video window.
    * ``_index`` -- T-frame VIDEO-ALIGNED index: non-overlapping ``T``-frame sub-windows with
      ``s + T <= we`` so video / camera / motion all share the SAME ``(uuid, start)``. Used by every
      task that REQUIRES frame alignment: inverse_dynamics / forward_dynamics / policy (video<->camera
      aligned), motimg2video (motion<->video aligned), video2motion (video<->motion aligned). These
      COLLAPSE to a handful at large T (a captioned window is only ~100 frames).

  The router picks ``_t2m_index`` only for text2motion; every other task uses ``_index``.

  ALIGNMENT INVARIANT: text2motion needs no video alignment. NymeriaPlus textimg2motion uses the
  same fixed-T aligned motion/video window and sends frame 0 to the reasoner. Tasks that pack
  video<->motion or video<->camera together also use ``_index``.

  TWO DATA SOURCES (``task_plan.TASK_SOURCES``): NymeriaPlus windows carry ALL 5 modalities so they
  serve every task; BONES-SEED windows are motion-only (no image / video / camera) so they can ONLY
  serve ``text2motion``. BONES = TEXT2MOTION ONLY -- every BONES row is always ``text2motion``, and it
  can never serve any task that needs image/video/camera. A fraction of ``text2motion`` mass is routed
  to the (large) BONES stream (``bones_text2motion_frac``) for extra text-conditioning coverage; the
  rest of ``text2motion`` and ALL other tasks draw NymeriaPlus. The loader enforces this availability
  by construction.

  Both reasoner-image and historical generator-image TI2M use ``_index``; only the image encoding
  path differs.

NORMALIZATION SPACES are kept STRICTLY per-modality (DESIGN_7TASK.md section 4) -- never
cross-normalize: motion is z-scored (uniego283 stats), camera is un-normalized (Cosmos-exact),
video lives in raw-pixel or Wan-VAE-latent space. Only ``motion`` is z-scored here.

The trainer derives every per-token CLEAN/NOISED ``condition_mask`` from ``mode`` via
``task_plan.resolve_sample`` -- this dataset only supplies the raw aligned modalities + the chosen
mode; it does NOT noise anything.

FLOOR CALIBRATION (see README "Floor calibration" + ``precompute_floor_calibration.py``):
the SOMA fit penetrates the GT floor by a roughly CONSTANT per-seq bias (~5-7 cm), while BONES
grounds contacted feet ~+3 cm above y=0 -- so the two motion sources sat ~6-8 cm apart in the
shared space. A precomputed per-seq ``delta_seq = d_minc(seq) - c0`` (loaded from
``config.FLOOR_CALIBRATION_JSON``) is folded into each index entry's stored offset at INDEX BUILD
time: ``entry["off"] = ground_offset_y + delta_seq``, and ``ground_features`` then subtracts the
TOTAL. The original pair stays recoverable via ``entry["off_gt"]`` (the GT per-window multi-floor
level) and ``entry["delta"]`` -- IMPORTANT: any future camera<->motion ABSOLUTE alignment must use
the calibrated TOTAL ``off``, which is the full vertical world->grounded shift applied to motion.
Windows in the precomputed drop list (``wrong_floor`` metre-scale floor-selection errors,
``residual_penetration`` deep fit failures, and ``extreme_y`` outliers; about 5-6% of usable
NymeriaPlus windows in the current train/test manifests) are SKIPPED in BOTH indices. If the json
is missing we WARN loudly and proceed uncalibrated (backward compat). BONES samples are untouched.
"""
from __future__ import annotations

import json
import os
import random
import signal
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config
import task_plan
from uniego_layout import FEAT_DIM, canonicalize_frame0, ground_features

# Pull the verified NymeriaPlus video+camera loaders from nymeria_world (source of truth). We add
# its dir to sys.path so this repo can import the camera dataset / decode helpers verbatim.
_NYMERIA_WORLD = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
if _NYMERIA_WORLD not in sys.path:
    sys.path.insert(0, _NYMERIA_WORLD)
from nymeria_camera_rgb_dataset import (  # noqa: E402  (path-injected import)
    _DECODE_TIMEOUT,
    _MAX_SKIP,
    _load_rgb_cam,
    _on_alarm,
    _rgb_path,
    decode_window_pyav,
    rel_action_from_window,
)
from camera_to_action import DOMAIN_ID  # noqa: E402  (camera_pose -> 2)

# Reuse the existing 283-d uniego BONES source factory for the text2motion-only BONES stream.
from dataset import N_NEUTRAL_JOINTS, humanize_caption, make_bones  # noqa: E402

assert DOMAIN_ID == config.CAMERA_DOMAIN_ID, (
    f"camera domain mismatch: camera_to_action={DOMAIN_ID} vs config={config.CAMERA_DOMAIN_ID}"
)


# ----------------------------------------------------------------------------
# Precomputed Wan-VAE latent path convention. MUST match precompute_latents.out_path
# EXACTLY (it is the writer; this is the reader). For uuid "Sxx/<seq>" and window start s:
#   {root}/{Sxx}/{Sxx__<seq>}_{s}.npz     (uuid_safe = uuid.replace("/", "__"))
# A uuid with no "/" goes under a "_misc" subdir (mirrors out_path's fallback).
# ----------------------------------------------------------------------------
def latent_path(uuid: str, start: int, root: str = config.VIDEO_LATENT_ROOT) -> str:
    uuid_safe = uuid.replace("/", "__")
    subj = uuid.split("/")[0] if "/" in uuid else "_misc"
    return os.path.join(root, subj, f"{uuid_safe}_{int(start)}.npz")


def _load_latents(path: str) -> np.ndarray:
    """Load a precomputed packed latent stack (C, T_lat, h, w) float32 from {path}.

    precompute_latents.py saves Wan2.2-VAE latents channel-first as (z_dim, T_lat, h, w);
    gen_heads.encode_video_latents consumes that exact (C, T_lat, h, w) layout, so we return
    it verbatim (no transpose)."""
    with np.load(path) as d:
        key = "latents" if "latents" in d else d.files[0]
        return d[key].astype(np.float32)


def uses_native_motion_index(mode: str, *, needs_video: bool, needs_camera: bool) -> bool:
    """Whether a Nymeria task can use its complete native captioned motion span."""
    return (
        mode == "text2motion"
        and not needs_video
        and not needs_camera
    )


class NymeriaJointDataset(Dataset):
    """Aligned 5-modality NymeriaPlus loader, sliced at T=33 (4N+1) windows.

    The index is built EXACTLY like ``NymeriaCameraRGBDataset`` (same manifest, same usable /
    rgb-cam filters, same per-sequence split) but additionally requires the ``uniego_rep`` npz so
    every window carries motion too. Each captioned ``t2w_window`` (100-frame) is sliced into one
    or more ``T``-frame sub-windows so motion/video/camera all share the SAME ``(uuid, start)``.

    Args:
        manifest_path / split_file / split: per-sequence NymeriaPlus split (hold out whole
            recordings; no window-level leakage) -- same contract as NymeriaCameraRGBDataset.
        num_frames: output/padding length T (must be 4N+1 when generator video is active).
        aligned_num_frames: optional task-specific valid length for aligned windows. This may
            differ from ``num_frames`` only for Phase-2 T2M + reasoner-image TI2M, where T2M keeps
            its 200-frame capacity while TI2M uses the 97-frame Nymeria video window.
        fps: video fps (20).
        task_weights: per-mode sampling weights (defaults to the motion-weighted config mixture).
        bones_text2motion_frac: fraction of ``text2motion`` mass routed to the BONES stream (the
            rest of ``text2motion`` + ALL other tasks draw NymeriaPlus windows). 0 disables BONES.
        cfg_dropout: per-sample prob of blanking an instruction caption to "" (train only). Tasks
            whose ``text_policy == "empty"`` (inverse_dynamics / video2motion) ALWAYS use "".
        prefer_latents: load precomputed Wan-VAE latents instead of raw frames when present AND
            they match the current window length T (T_lat == (T-1)//4+1). A cache built at a
            different T is ignored -> raw frames are emitted for a live VAE encode in the trainer.
        force_on_the_fly: never read the precomputed latent cache; always emit raw frames so the
            trainer VAE-encodes every window live (used to force the on-the-fly path at any T).
        reasoner_image_for_textimg: use corrected reasoner-side image conditioning for TI2M.
        reasoner_image_size: square size returned for that frame-0 image. The released Cosmos Nano
            processor supports 256x256 and emits 64 merged visual tokens at this size.
        train: random window start within a 100-frame slice + caption CFG-drop when True; center /
            no drop otherwise.
        require_rgb_cam / require_usable: window filters (same as the camera dataset).
        uniego_root: root of the per-seq uniego_rep npzs (``{root}/{uuid}.npz``).
        floor_calibration_json: precomputed per-seq floor deltas + drop list (see module
            docstring / precompute_floor_calibration.py). None or a missing file -> loud
            warning + uncalibrated behavior (entry "off" stays the raw GT ground_offset_y).
        max_samples / seed: cap + RNG seed for reproducibility.
        bones_kwargs: extra kwargs forwarded to ``dataset.make_bones`` for the BONES source.
    """

    def __init__(
        self,
        *,
        manifest_path: str = config.NYMERIA_MANIFEST,
        split_file: str = config.NYMERIA_SPLIT_FILE,
        split: str = "train",
        num_frames: int = config.VIDEO_NUM_FRAMES,
        aligned_num_frames: Optional[int] = None,
        fps: float = 20.0,
        task_weights: Optional[Dict[str, float]] = None,
        bones_text2motion_frac: float = 0.5,
        cfg_dropout: float = 0.10,
        prefer_latents: bool = True,
        force_on_the_fly: bool = False,
        reasoner_image_for_textimg: bool = False,
        reasoner_image_size: Optional[int] = 256,
        latent_root: str = config.VIDEO_LATENT_ROOT,
        train: bool = True,
        require_rgb_cam: bool = True,
        require_usable: bool = True,
        uniego_root: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep",
        floor_calibration_json: Optional[str] = config.FLOOR_CALIBRATION_JSON,
        max_samples: Optional[int] = None,
        seed: int = 0,
        bones_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        # The Wan VAE's 4x temporal compression needs a 4N+1 pixel window, so the 4N+1 constraint
        # is required ONLY when a video/camera generator-vision task is in the mixture. Pure
        # text2motion carries NO video, and reasoner-image textimg2motion only decodes frame 0 for
        # the Qwen-VL reasoner, so ANY T is allowed for those cases.
        tw_probe = dict(task_weights or config.TASK_WEIGHTS)
        tw_probe = {m: float(w) for m, w in tw_probe.items() if w > 0.0}
        needs_4n1 = any(
            task_plan.build_task_plan(m).has_gen
            and not (bool(reasoner_image_for_textimg) and m == "textimg2motion")
            for m in tw_probe
        )
        if needs_4n1 and num_frames % 4 != 1:
            raise ValueError(
                f"num_frames must be 4N+1 for video/camera tasks (Wan VAE); got {num_frames} "
                f"with tasks {list(tw_probe)}"
            )
        self._num_frames = int(num_frames)
        self._aligned_num_frames = (
            self._num_frames if aligned_num_frames is None else int(aligned_num_frames)
        )
        if not 1 <= self._aligned_num_frames <= self._num_frames:
            raise ValueError(
                "aligned_num_frames must be in [1, num_frames], got "
                f"{self._aligned_num_frames} for num_frames={self._num_frames}"
            )
        if self._aligned_num_frames != self._num_frames:
            unsupported = set(tw_probe) - {"text2motion", "textimg2motion"}
            if unsupported or not bool(reasoner_image_for_textimg):
                raise ValueError(
                    "task-specific aligned_num_frames is supported only for Phase-2 "
                    "text2motion + reasoner-side textimg2motion; got unsupported tasks "
                    f"{sorted(unsupported)} reasoner_image_for_textimg={reasoner_image_for_textimg}"
                )
        self._fps = float(fps)
        self._train = bool(train)
        self._cfg_dropout = float(cfg_dropout)
        self._prefer_latents = bool(prefer_latents)
        self._force_on_the_fly = bool(force_on_the_fly)
        self._reasoner_image_for_textimg = bool(reasoner_image_for_textimg)
        self._reasoner_image_size = (
            None if reasoner_image_size is None else int(reasoner_image_size)
        )
        if self._reasoner_image_size is not None and self._reasoner_image_size <= 0:
            raise ValueError(
                f"reasoner_image_size must be positive or None, got {reasoner_image_size}"
            )
        self._latent_root = latent_root
        self._uniego_root = uniego_root
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._worker_rng = None
        self._worker_rng_key = None

        # ---- motion stats (z-score) + 283-d stats shared with the PoC / dataset.py ----------
        self._mean = np.load(config.MOTION_STATS_MEAN).astype(np.float32)
        self._std = np.load(config.MOTION_STATS_STD).astype(np.float32)
        assert self._mean.shape == (FEAT_DIM,) and self._std.shape == (FEAT_DIM,), (
            f"motion stats must be ({FEAT_DIM},); got mean={self._mean.shape} std={self._std.shape}"
        )

        # ---- task mixture: split NymeriaPlus-eligible vs BONES-eligible mass --------------------
        tw = dict(task_weights or config.TASK_WEIGHTS)
        tw = {m: float(w) for m, w in tw.items() if w > 0.0}
        if not tw:
            raise ValueError("task_weights has no positive-weight tasks")
        self._modes = list(tw)
        self._mode_w = [tw[m] for m in self._modes]
        # of text2motion's mass, route `bones_text2motion_frac` to the BONES stream.
        self._bones_frac = float(bones_text2motion_frac)

        # ---- per-sequence split (whole recordings) ----------------------------------------------
        keep_uuids = None
        if split not in ("all", None):
            sp = json.load(open(split_file))
            assert split in sp, f"split '{split}' not in {split_file}"
            keep_uuids = set(sp[split])

        # ---- per-seq floor calibration + drop list (precompute_floor_calibration.py) ------------
        # deltas[uuid] (m) is folded into every index entry's "off" below; dropped windows are
        # skipped in BOTH indices. Missing file -> WARN + uncalibrated (backward compat).
        self._floor_deltas: Dict[str, float] = {}
        self._floor_global_delta = 0.0
        self._calibrated = False
        _drop_map: Dict[str, Dict[tuple, str]] = {}
        if floor_calibration_json and os.path.isfile(floor_calibration_json):
            with open(floor_calibration_json) as f:
                _fc = json.load(f)
            self._floor_deltas = {u: float(v) for u, v in _fc.get("deltas", {}).items()}
            self._floor_global_delta = float(_fc.get("global_delta", 0.0))
            for _u, _lst in _fc.get("dropped_windows", {}).items():
                _drop_map[_u] = {(int(e[0]), int(e[1])): (str(e[2]) if len(e) > 2 else "unknown")
                                 for e in _lst}
            self._calibrated = True
        else:
            print(f"[NymeriaJoint] *** WARNING: floor calibration file missing "
                  f"({floor_calibration_json!r}) -- proceeding UNCALIBRATED: NymeriaPlus feet "
                  f"will penetrate the floor by the per-seq SOMA fit bias (~5-7 cm) and sit "
                  f"~6-8 cm below the BONES convention; wrong-floor windows are NOT dropped. "
                  f"Run precompute_floor_calibration.py (kimodo env, CPU). ***", flush=True)
        _n_dropped: Dict[str, int] = {}          # reason -> n captioned manifest windows dropped
        _own_delta_uuids: set = set()            # kept-window uuids with their own per-seq delta
        _fallback_uuids: set = set()             # kept-window uuids using the global fallback

        # ---- two indices over captioned/usable manifest windows --------------------------------
        #   self._index      : T-frame ALIGNED sub-windows (s + T <= min(we, nb)) -- for tasks that
        #                      need a full co-aligned video/camera/motion window (video/camera and
        #                      motion-in-aligned). These COLLAPSE to a handful at large T because a
        #                      captioned atomic-action window is only ~100 frames (~5 s @ 20 fps).
        #   self._t2m_index  : ONE entry per captioned atomic-action window at its NATIVE span, for
        #                      NymeriaPlus T2M. It is T-INDEPENDENT:
        #                      every usable+captioned window is included and later padded+masked to
        #                      T. `avail` = usable motion length (min(we, nb) - ws, ~100). A 5 s
        #                      caption then correctly matches its ~100 valid frames; we NEVER slide
        #                      a long window (which would stretch the caption over the wrong frames).
        self._index: List[Dict[str, Any]] = []
        self._t2m_index: List[Dict[str, Any]] = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                uuid = rec.get("uuid")
                vis = rec.get("vision_path")
                cam = rec.get("camera_path")
                nb = int(rec.get("nb_frames", 0))
                if not uuid or not vis or not cam:
                    continue
                if keep_uuids is not None and uuid not in keep_uuids:
                    continue
                rgb = _rgb_path(cam)
                if require_rgb_cam and not os.path.isfile(rgb):
                    continue
                uni = os.path.join(uniego_root, f"{uuid}.npz")
                if not os.path.isfile(uni):
                    continue
                for w in rec.get("t2w_windows", []):
                    if require_usable and not w.get("usable", False):
                        continue
                    cap = w.get("caption")
                    if not cap:
                        continue
                    ws, we = int(w["start_frame"]), int(w["end_frame"])
                    # ---- floor calibration: skip precomputed bad windows -------------------
                    _reason = _drop_map.get(uuid, {}).get((ws, we))
                    if _reason is not None:
                        _n_dropped[_reason] = _n_dropped.get(_reason, 0) + 1
                        continue
                    # ---- fold the per-seq delta into the stored offset ---------------------
                    # "off" is the CALIBRATED TOTAL vertical shift (GT per-window multi-floor
                    # ground_offset_y + per-seq SOMA-fit delta); _load_motion/ground_features
                    # subtracts the total. The original pair stays recoverable via
                    # "off_gt"/"delta" (a camera<->motion absolute alignment needs the total).
                    off_gt = w.get("ground_offset_y", None)
                    delta = 0.0
                    if self._calibrated and off_gt is not None:
                        if uuid in self._floor_deltas:
                            delta = self._floor_deltas[uuid]
                            _own_delta_uuids.add(uuid)
                        else:
                            delta = self._floor_global_delta
                            _fallback_uuids.add(uuid)
                    off = (float(off_gt) + delta) if off_gt is not None else None
                    hi = min(we, nb)                       # last usable frame (exclusive)
                    avail = hi - ws                        # native usable motion length (~100)
                    if avail <= 0:
                        continue
                    # (a) ALIGNED index: normally aligned-T == output T. Phase-2 may keep output
                    #     T=200 for full T2M while using aligned-T=97 for reasoner-image TI2M;
                    #     _load_motion pads those 97 valid rows to output T and masks the tail.
                    s = ws
                    while s + self._aligned_num_frames <= hi:
                        self._index.append({
                            "uuid": uuid, "vis": vis, "rgb": rgb, "uni": uni,
                            "s": int(s), "avail": self._aligned_num_frames,
                            "cap": cap, "off": off,
                            "off_gt": off_gt, "delta": delta,
                        })
                        s += self._aligned_num_frames
                    # (b) NATIVE text2motion index: one entry per captioned window, T-independent.
                    #     `avail` motion frames are loaded (<= T) then padded+masked up to T.
                    self._t2m_index.append({
                        "uuid": uuid, "vis": vis, "rgb": rgb, "uni": uni,
                        "s": int(ws), "avail": int(avail), "cap": cap, "off": off,
                        "off_gt": off_gt, "delta": delta,
                    })
        # ---- floor-calibration init log: drops (by reason) + delta coverage ---------------------
        if self._calibrated:
            _n_drop_total = sum(_n_dropped.values())
            print(f"[NymeriaJoint] floor calibration ON ({floor_calibration_json}): dropped "
                  f"{_n_drop_total} captioned windows {_n_dropped or {}} (each removes 1 "
                  f"_t2m_index entry + its _index sub-windows); delta coverage: "
                  f"{len(_own_delta_uuids)} seqs own delta, {len(_fallback_uuids)} seqs global "
                  f"fallback ({self._floor_global_delta:+.4f} m); "
                  f"aligned_T={self._aligned_num_frames} output_T={self._num_frames} "
                  f"_index={len(self._index)} _t2m_index={len(self._t2m_index)}", flush=True)
        # `_index` may legitimately be empty at large T (no >=T captioned span) -- text2motion then
        # runs entirely off `_t2m_index` + BONES, so only require SOMETHING to sample from.
        if not self._index and not self._t2m_index:
            raise RuntimeError(
                f"NymeriaJointDataset: empty index (manifest={manifest_path} split={split} "
                f"T={self._num_frames})"
            )
        if max_samples is not None:
            self._index = self._index[:max_samples]
            self._t2m_index = self._t2m_index[:max_samples]

        # ---- optional BONES text2motion stream (motion-only) ------------------------------------
        self._bones = None
        if self._bones_frac > 0.0 and "text2motion" in tw:
            try:
                bk = dict(bones_kwargs or {})
                bk.setdefault("cfg_dropout", cfg_dropout)
                # Cap BONES clips at T so no BONES motion exceeds T. UniegoPairsDataset with a fixed
                # T random-crops (train) / center-crops longer clips to exactly T and keeps shorter
                # ones ragged (<= T) with an all-valid pad mask; the collate then pads the batch to
                # its max (<= T). This keeps the batched nymeria (padded to T) + bones (<= T) motion
                # at Tmax == T so `motion` is (B, T, 283), with the pad mask marking padded frames.
                bk.setdefault("T", self._num_frames)
                self._bones = make_bones(split=("train" if train else "val"), **bk)
            except Exception as e:  # noqa: BLE001 -- BONES is optional; degrade to nymeria-only.
                print(f"[NymeriaJoint] BONES source unavailable ({type(e).__name__}: {e}); "
                      f"text2motion will draw NymeriaPlus only", flush=True)
                self._bones = None

    def __len__(self) -> int:
        # `_index` counts video-ALIGNED windows, which collapse to a handful at large T (a captioned
        # atomic-action window is ~100 frames, so few carry a full >=T aligned span: 119632 @ T=97
        # -> 94 @ T=201). The NATIVE text2motion pool is `_t2m_index` (one entry per captioned
        # window, T-independent, ~tens of thousands) plus the BONES stream. __getitem__ maps any idx
        # via `(idx+attempt) % len(pool)`, so report the LARGEST live pool -- otherwise a tiny length
        # split across ranks gives < batch_size samples/rank -> 0 batches -> the train loop spins on
        # an empty iterator (the T=201 startup grind, 0 steps / 126% CPU).
        n = max(len(self._index), len(self._t2m_index))
        if self._bones is not None:
            n = max(n, len(self._bones))
        return n

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def has_bones(self) -> bool:
        return self._bones is not None

    # -- mode selection -----------------------------------------------------------------------
    def _active_rng(self) -> random.Random:
        """Return a worker-unique RNG instead of cloning one stream into every worker."""
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            return self._rng
        key = (int(worker.id), int(worker.seed))
        if self._worker_rng is None or self._worker_rng_key != key:
            self._worker_rng_key = key
            self._worker_rng = random.Random(int(worker.seed) + 1_000_003 * self._seed)
        return self._worker_rng

    def _choose_mode(self) -> str:
        return self._active_rng().choices(self._modes, weights=self._mode_w, k=1)[0]

    def _route_to_bones(self, mode: str) -> bool:
        """A text2motion sample is routed to the BONES stream with prob ``bones_text2motion_frac``
        (only NymeriaPlus carries video/camera, so every other task MUST stay on NymeriaPlus)."""
        return (
            mode == "text2motion"
            and self._bones is not None
            and self._active_rng().random() < self._bones_frac
        )

    # -- caption policy (task text_policy + 10% CFG) ------------------------------------------
    def _apply_caption_policy(self, mode: str, caption: str) -> str:
        plan = task_plan.build_task_plan(mode)
        if plan.caption_always_empty:                     # inverse_dynamics / video2motion
            return ""
        if self._train and self._active_rng().random() < self._cfg_dropout:  # 10% CFG drop
            return ""
        return humanize_caption(caption)  # "C is ..." -> "A person is ..." (Nymeria convention)

    # -- motion (uniego 283-d): ground -> canon frame0 -> z-score -> PAD to T -----------------
    def _load_motion(self, uni_path: str, s: int, off: Optional[float],
                     avail: Optional[int] = None):
        """Load the motion window at ``[s : s + min(T, avail)]``, z-score it, then zero-pad to T.

        ``off`` is the CALIBRATED TOTAL vertical shift stored in the index entry
        (``ground_offset_y + delta_seq`` when floor calibration is loaded; the raw GT
        ``ground_offset_y`` otherwise) -- ``ground_features`` subtracts it wholesale.

        ``avail`` is the number of usable motion frames actually present for this window (native
        atomic-action span, ~100). We load ``k = min(T, avail)`` frames (falling back to whatever
        the npz slice yields when ``avail`` is None, e.g. an aligned window that requires exactly
        T), process them, and zero-pad up to T. The returned ``pad`` mask is True (=padded,
        excluded from the rectified-flow loss) for frames ``[k : T]``. When ``avail >= T`` this is
        the previous full-T behavior (all valid). We NEVER raise on a short window -- pad instead."""
        T = self._num_frames
        want = T if avail is None else min(T, int(avail))
        with np.load(uni_path) as npz:
            feats = npz["features"][s:s + want].astype(np.float32)   # [k, 283], k <= T
            nj = npz["neutral_joints"].astype(np.float32)            # [30, 3]
        k = int(feats.shape[0])                                       # frames actually present
        if k == 0:
            raise ValueError(f"motion window empty: 0 frames at {uni_path}@{s}")
        if off is not None:
            feats = ground_features(feats, off)                      # feet -> floor (y~0)
        feats = canonicalize_frame0(feats)                           # window starts canonically
        feats = (feats - self._mean) / self._std                     # z-score (uniego283 stats)
        # FEATURE-SANITY GUARD (defense-in-depth vs "loss bomb" windows -- run 2727 was destroyed
        # at step 14140 by ONE extreme batch, loss=34.6, weights damaged for ~4k steps). A healthy
        # z-scored window is |z| <~ 8; metre-scale mis-grounding / degenerate fits produce |z| in
        # the tens. Raising here routes the window through __getitem__'s skip-to-next retry loop
        # (same guard style as the decode-timeout path), so a bad row can never reach the loss.
        zmax = float(np.abs(feats).max())
        if zmax > 20.0:
            raise ValueError(f"extreme motion features |z|max={zmax:.1f} at {uni_path}@{s}")
        if k < T:                                                    # zero-pad the tail up to T
            pad_rows = np.zeros((T - k, FEAT_DIM), dtype=np.float32)
            feats = np.concatenate([feats, pad_rows], axis=0)        # [T, 283]
        nj = nj - nj.mean(axis=0, keepdims=True)                     # center; scale = size cue
        assert nj.shape == (N_NEUTRAL_JOINTS, 3), f"neutral_joints must be (30,3); got {nj.shape}"
        pad = np.zeros((T,), dtype=bool)
        pad[k:] = True                                               # True = padded (no loss)
        return feats, nj, pad

    # -- video: precomputed latents (preferred) or raw decoded frames -------------------------
    def _resize_reasoner_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize a CHW uint8 frame before it reaches the frozen Qwen visual tower."""
        size = self._reasoner_image_size
        if size is None or tuple(image.shape[-2:]) == (size, size):
            return image.contiguous()
        return F.interpolate(
            image.float().unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0).round().clamp_(0, 255).to(torch.uint8).contiguous()

    def _load_video(self, uuid: str, vis: str, s: int, *, need_reasoner_image: bool = False):
        """Return (video_frames, video_latents, reasoner_image).

        Prefers precomputed Wan-VAE latents at ``latent_path(uuid, s)`` -- but ONLY when they
        match the current window length ``T`` (expected ``T_lat = (T-1)//4 + 1``). If the cache
        is absent OR was precomputed at a different ``T`` (e.g. cache is T=33 -> T_lat=9 but we
        now train at T=97 -> T_lat=25), we fall back to a raw PyAV window decode (guarded by
        SIGALRM like the camera dataset) so the trainer can VAE-encode the frames live at the
        current ``T``. When ``force_on_the_fly`` is set we always decode raw frames."""
        expected_t_lat = (self._num_frames - 1) // 4 + 1
        if self._prefer_latents and not self._force_on_the_fly:
            lp = latent_path(uuid, s, self._latent_root)
            if os.path.isfile(lp):
                lat = _load_latents(lp)                           # (C, T_lat, h, w)
                if lat.shape[1] == expected_t_lat:                # cache matches current T
                    reasoner_image = None
                    if need_reasoner_image:
                        frames = decode_window_pyav(vis, s, self._num_frames, self._fps)
                        reasoner_image = torch.from_numpy(
                            np.ascontiguousarray(frames[0])
                        ).permute(2, 0, 1).contiguous()
                        reasoner_image = self._resize_reasoner_image(reasoner_image)
                    return None, torch.from_numpy(np.ascontiguousarray(lat)).float(), reasoner_image
                # else: cache is for a DIFFERENT T -> decode raw frames + encode live.
        # raw-frame fallback (heavy; matches NymeriaCameraRGBDataset decode contract)
        frames = decode_window_pyav(vis, s, self._num_frames, self._fps)  # (T,H,W,3) uint8
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
        reasoner_image = (
            self._resize_reasoner_image(video[:, 0]) if need_reasoner_image else None
        )
        return video, None, reasoner_image

    def _load_reasoner_image(self, vis: str, s: int) -> torch.Tensor:
        """Decode only frame 0 for reasoner-image textimg2motion.

        This path intentionally bypasses Wan-latent lookup and the 4N+1 video window contract:
        the generator receives no image/video rows for corrected textimg2motion.
        """
        frames = decode_window_pyav(vis, s, 1, self._fps)  # (1,H,W,3) uint8
        image = torch.from_numpy(np.ascontiguousarray(frames[0])).permute(2, 0, 1).contiguous()
        return self._resize_reasoner_image(image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mode = self._choose_mode()

        # ---- BONES text2motion fast path (no video/camera; reuse the uniego BONES source) -------
        if self._route_to_bones(mode):
            b = self._bones[idx % len(self._bones)]
            cap = self._apply_caption_policy("text2motion", b["caption"])
            return {
                "mode": "text2motion",
                "source": "bones",
                "caption": cap,
                "motion": b["motion"],
                "neutral_joints": b["neutral_joints"],
                "motion_pad_mask": b["motion_pad_mask"],
                "video_frames": None,
                "video_latents": None,
                "reasoner_image": None,
                "camera_action": None,
                "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
            }

        plan = task_plan.build_task_plan(mode)
        reasoner_textimg = self._reasoner_image_for_textimg and mode == "textimg2motion"
        needs_video = (plan.video.present or plan.image.present) and not reasoner_textimg
        needs_reasoner_image = reasoner_textimg
        needs_camera = plan.camera.present
        needs_motion = plan.motion.present
        T = self._num_frames

        # T2M draws from the native caption span. NymeriaPlus TI2M uses the fixed-T aligned
        # video/motion window even when frame 0 is encoded by the reasoner rather than generator.
        use_native_motion = uses_native_motion_index(
            mode,
            needs_video=needs_video,
            needs_camera=needs_camera,
        )
        pool = self._t2m_index if use_native_motion else self._index
        n = len(pool)
        if n == 0:
            raise RuntimeError(
                f"NymeriaJoint: no windows for mode={mode} at T={T} "
                f"(use_native_motion={use_native_motion}); _index={len(self._index)} "
                f"_t2m={len(self._t2m_index)}"
            )

        # Deterministically skip to the next window on any decode/load failure (corrupt video,
        # short npz, NaN action) so no sample can stall a distributed rank -- same guard as
        # NymeriaCameraRGBDataset (runs in the worker main thread, so SIGALRM applies).
        has_alarm = hasattr(signal, "SIGALRM")
        last_err = None
        for attempt in range(_MAX_SKIP):
            it = pool[(idx + attempt) % n]
            s = it["s"]
            try:
                if has_alarm and (needs_video or needs_reasoner_image):
                    _old = signal.signal(signal.SIGALRM, _on_alarm)
                    signal.alarm(_DECODE_TIMEOUT)

                video_frames = video_latents = reasoner_image = None
                if needs_video:
                    video_frames, video_latents, reasoner_image = self._load_video(
                        it["uuid"], it["vis"], s, need_reasoner_image=False
                    )
                elif needs_reasoner_image:
                    reasoner_image = self._load_reasoner_image(it["vis"], s)

                camera_action = None
                if needs_camera:
                    pos, rot = _load_rgb_cam(it["rgb"])
                    act = rel_action_from_window(pos[s:s + T], rot[s:s + T])  # (T-1,9)
                    if act.shape[0] != T - 1 or not np.isfinite(act).all():
                        raise ValueError(f"bad camera action {act.shape}")
                    camera_action = torch.from_numpy(np.ascontiguousarray(act)).float()

                motion = neutral = pad = None
                if needs_motion:
                    feats, nj, pad_np = self._load_motion(
                        it["uni"], s, it["off"], avail=it.get("avail"))
                    motion = torch.from_numpy(np.ascontiguousarray(feats))
                    neutral = torch.from_numpy(np.ascontiguousarray(nj))
                    pad = torch.from_numpy(pad_np)

                if has_alarm and (needs_video or needs_reasoner_image):
                    signal.alarm(0); signal.signal(signal.SIGALRM, _old)
            except Exception as e:  # noqa: BLE001 -- skip ANY bad window so it can't stall a rank.
                if has_alarm:
                    signal.alarm(0)
                last_err = e
                if attempt < 3 or attempt % 16 == 0:
                    print(f"[NymeriaJoint] skip bad window {it['uuid']}@{s} mode={mode} "
                          f"({type(e).__name__}: {e}); attempt {attempt + 1}/{_MAX_SKIP}",
                          flush=True)
                continue

            caption = self._apply_caption_policy(mode, it["cap"])
            return {
                "mode": mode,
                "source": "nymeria",
                "caption": caption,
                "motion": motion,
                "neutral_joints": neutral,
                "motion_pad_mask": pad,
                "video_frames": video_frames,
                "video_latents": video_latents,
                "reasoner_image": reasoner_image,
                "camera_action": camera_action,
                "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
            }
        raise RuntimeError(
            f"NymeriaJoint: no usable window after {_MAX_SKIP} attempts from idx {idx} "
            f"(mode={mode}): {last_err}"
        )


# ============================================================================
# Collate: per-modality batch-max padding. Modalities absent for a sample stay None and are
# only stacked when EVERY sample in the batch carries them (homogeneous batches are the common
# case once the trainer groups by mode; mixed batches keep the per-sample lists so the model
# can pack each sample's present modalities independently).
# ============================================================================
def collate_joint(batch: List[dict]) -> dict:
    """Pad each present modality to the batch max; keep mode/source/caption as per-sample lists.

    Output keys (per-sample lists unless noted):
        mode, source, caption  : List[str] length B
        domain_id              : long [B]
        motion                 : float32 [B, Tmax, 283]  (zeros in pad) OR None if no sample has it
        neutral_joints         : float32 [B, 30, 3]      OR None
        motion_pad_mask        : bool   [B, Tmax]        (True = pad) OR None
        camera_action          : float32 [B, Tc, 9]      (zeros in pad) OR None
        camera_pad_mask        : bool   [B, Tc]          OR None
        video_latents          : List[Optional[Tensor]]  (ragged latent grids kept per-sample)
        video_frames           : List[Optional[Tensor]]  (ragged pixel windows kept per-sample)

    Motion / camera frame counts are uniform (T / T-1) for NymeriaPlus windows but BONES motion
    windows are ragged, so motion is padded to the batch max with a pad mask (matching dataset.py).
    """
    B = len(batch)
    modes = [b["mode"] for b in batch]
    sources = [b["source"] for b in batch]
    captions = [b["caption"] for b in batch]
    domain_id = torch.stack([b["domain_id"] for b in batch], dim=0)

    out: Dict[str, Any] = {
        "mode": modes,
        "source": sources,
        "caption": captions,
        "domain_id": domain_id,
        # ragged generator modalities: keep per-sample (None for samples without them).
        "video_latents": [b.get("video_latents") for b in batch],
        "video_frames": [b.get("video_frames") for b in batch],
        "reasoner_image": [b.get("reasoner_image") for b in batch],
    }

    # ---- motion (batch-max padded, like dataset.collate) ----
    if any(b.get("motion") is not None for b in batch):
        lens = [b["motion"].shape[0] if b.get("motion") is not None else 0 for b in batch]
        Tmax = max(lens)
        motion = torch.zeros((B, Tmax, FEAT_DIM), dtype=torch.float32)
        pad_mask = torch.ones((B, Tmax), dtype=torch.bool)
        nj = torch.zeros((B, N_NEUTRAL_JOINTS, 3), dtype=torch.float32)
        for i, b in enumerate(batch):
            if b.get("motion") is None:
                continue
            m = lens[i]
            motion[i, :m] = b["motion"]
            pad_mask[i, :m] = b["motion_pad_mask"]
            nj[i] = b["neutral_joints"]
        out["motion"] = motion
        out["motion_pad_mask"] = pad_mask
        out["neutral_joints"] = nj
    else:
        out["motion"] = out["motion_pad_mask"] = out["neutral_joints"] = None

    # ---- camera action (batch-max padded; (T-1,9)) ----
    if any(b.get("camera_action") is not None for b in batch):
        clens = [b["camera_action"].shape[0] if b.get("camera_action") is not None else 0
                 for b in batch]
        Tc = max(clens)
        cam = torch.zeros((B, Tc, config.CAMERA_RAW_ACTION_DIM), dtype=torch.float32)
        cam_pad = torch.ones((B, Tc), dtype=torch.bool)
        for i, b in enumerate(batch):
            if b.get("camera_action") is None:
                continue
            c = clens[i]
            cam[i, :c] = b["camera_action"]
            cam_pad[i, :c] = False
        out["camera_action"] = cam
        out["camera_pad_mask"] = cam_pad
    else:
        out["camera_action"] = out["camera_pad_mask"] = None

    return out


# ============================================================================
# __main__ self-check: build the dataset and print a few rows' modality shapes + mode counts.
# ============================================================================
if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="rows to draw")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--no_bones", action="store_true")
    ap.add_argument("--no_latents", action="store_true")
    a = ap.parse_args()

    ds = NymeriaJointDataset(
        split=a.split,
        max_samples=a.max_samples,
        bones_text2motion_frac=0.0 if a.no_bones else 0.5,
        prefer_latents=not a.no_latents,
        seed=0,
    )
    print(f"[index] {len(ds)} aligned T={config.VIDEO_NUM_FRAMES} NymeriaPlus windows  "
          f"(has_bones={ds.has_bones})")
    print(f"[mixture] modes={ds._modes}")

    def _shape(x):
        if x is None:
            return None
        return tuple(x.shape)

    rows = []
    mode_counter: Counter = Counter()
    src_counter: Counter = Counter()
    for i in range(a.n):
        r = ds[i]
        rows.append(r)
        mode_counter[r["mode"]] += 1
        src_counter[r["source"]] += 1
        print(
            f"  [{i}] mode={r['mode']:16s} src={r['source']:7s} "
            f"motion={_shape(r['motion'])} cam={_shape(r['camera_action'])} "
            f"vid_lat={_shape(r['video_latents'])} vid_frm={_shape(r['video_frames'])} "
            f"nj={_shape(r['neutral_joints'])} cap='{(r['caption'] or '')[:32]}'"
        )

    print("\nmode counts :", dict(mode_counter))
    print("source counts:", dict(src_counter))

    # collate a small homogeneous-ish batch to exercise the padding path.
    batch = collate_joint(rows)
    print("\n[collate_joint] batch keys + shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:16s}: {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, list):
            shapes = [None if x is None else (tuple(x.shape) if torch.is_tensor(x) else x) for x in v]
            print(f"  {k:16s}: list[{len(v)}] sample0={shapes[0]}")
        else:
            print(f"  {k:16s}: {v}")
    print("\nOK: NymeriaJointDataset built + sampled + collated.")
