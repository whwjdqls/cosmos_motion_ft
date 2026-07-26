#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline Wan2.2-VAE video-latent precompute for the 7-task joint-attention model.

For every USABLE NymeriaPlus ``(uuid, start, T=33)`` window — the exact set the
7-task trainer will draw video/camera tasks from — this script:

  1. decodes the ego-video window with PyAV (``decode_window_pyav``, reused from
     ``nymeria_world``),
  2. resizes + reflection-pads it to the training resolution and normalizes to
     ``[-1, 1]`` THROUGH THE SAME ``ActionTransformPipeline`` the native camera
     world-model uses (so the pixels fed to the VAE are byte-identical to what
     training would feed),
  3. runs the Wan2.2-VAE encode (``Wan2pt2VAEInterface``, the same tokenizer the
     Nano model instantiates) -> ``(z_dim, T_lat, h, w)`` latents, cropping the
     reflection padding off in latent space exactly like
     ``omni_mot_model._remove_padding_from_latent``,
  4. computes the ``(T-1, 9)`` camera pseudo-action
     (``rel_action_from_window``, Cosmos-exact, un-normalized),
  5. saves ``{latents, camera_action, image_size, ...}`` to a SHARDED, RESUMABLE
     ``.npz`` keyed by ``(uuid, start, T)`` under the precomputed-latent root on
     ``/weka``.

The trainer (``nymeria_joint_dataset.py``) then only runs ``vae2llm``/patchify on
these latents — no per-step VAE — which is the whole point of precomputing (see
``DESIGN_7TASK.md`` section 2).

This is a SINGLE-GPU offline job (run it in the ``cosmos`` env on a node). It is
idempotent: existing ``.npz`` outputs are skipped, so it can be re-run / resumed /
sharded across many processes via ``--num_shards`` / ``--shard_id``.

Run (cosmos env, one GPU on a node — NOT the login node)::

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
    unset LD_LIBRARY_PATH
    export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
    export PYTHONPATH=/home/jungbin_cho/cosmos-framework:\
/home/jungbin_cho/cosmos_motion_ft/nymeria_world:\
/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
    cd /home/jungbin_cho/cosmos-framework
    python /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/precompute_latents.py \
        --limit 8                         # quick smoke subset
    # full run (optionally sharded across N processes):
    python .../precompute_latents.py --num_shards 8 --shard_id 0   # one per GPU/process

Output schema (one ``.npz`` per window), keyed ``{uuid_safe}_{start}.npz`` under a
per-subject subdir, with ``uuid_safe = uuid.replace("/", "__")``::

    latents        float16  (z_dim, T_lat, h, w)   Wan2.2-VAE latents, padding-cropped
    camera_action  float32  (T-1, 9)               Cosmos-exact relative camera pseudo-action
    image_size     float32  (4,)                   [target_h, target_w, orig_h, orig_w] (pixel space)
    uuid           str                              "Sxx/<seq>"
    start          int                              window start frame
    T              int                              window length (33)
    fps            float                            20.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

# The 7-task config (paths, T, resolution) and the nymeria_world loaders.
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft")
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/nymeria_world")
sys.path.insert(0, "/home/jungbin_cho/cosmos-framework")

import config as C  # noqa: E402  (motion_expert_joint_attention/config.py)
from native_phase_training.latent_cache_contract import (  # noqa: E402
    CACHE_CONTRACT_KIND,
    CACHE_CONTRACT_VERSION,
    LatentCacheContract,
    ensure_latent_cache_contract,
    validate_cached_sample,
)
from nymeria_camera_rgb_dataset import (  # noqa: E402
    _load_rgb_cam,
    _rgb_path,
    rel_action_from_window,
)
from nymeria_camera_dataset import decode_window_pyav  # noqa: E402


# ----------------------------------------------------------------------------
# Index: exactly the windows the 7-task trainer can draw video/camera tasks from.
# Mirrors NymeriaCameraRGBDataset.__init__ (manifest + per-sequence split +
# usable + caption + preprocessed camera_rgb present + fits in nb_frames), but
# keeps (uuid, start) so each window gets a stable on-disk key.
# ----------------------------------------------------------------------------
def build_index(
    manifest_path: str,
    split_file: str,
    split: str,
    num_frames: int,
    require_usable: bool = True,
    require_rgb_cam: bool = True,
) -> list[dict[str, Any]]:
    keep_uuids = None
    if split not in ("all", None):
        sp = json.load(open(split_file))
        assert split in sp, f"split '{split}' not in {split_file}"
        keep_uuids = set(sp[split])

    index: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            cam, vis = rec.get("camera_path"), rec.get("vision_path")
            nb = int(rec.get("nb_frames", 0))
            if not cam or not vis:
                continue
            if keep_uuids is not None and uuid not in keep_uuids:
                continue
            rgb = _rgb_path(cam)
            if require_rgb_cam and not os.path.isfile(rgb):
                continue
            for w in rec.get("t2w_windows", []):
                if require_usable and not w.get("usable", False):
                    continue
                if not w.get("caption"):
                    continue
                # MATCH NymeriaJointDataset's sub-window slicing EXACTLY: slide
                # non-overlapping num_frames sub-windows inside the captioned slice,
                # clamped so s + num_frames <= min(window_end, nb_frames). Encoding only
                # the first sub-window would leave the trainer's later sub-windows without a
                # precomputed latent (silent raw-decode fallback), so we enumerate them all.
                ws, we = int(w["start_frame"]), int(w["end_frame"])
                hi = min(we, nb)
                s = ws
                while s + num_frames <= hi:
                    index.append({"uuid": uuid, "vis": vis, "rgb": rgb, "s": s})
                    s += num_frames
    return index


def build_explicit_index(
    manifest_path: str,
    split_file: str,
    split: str,
    num_frames: int,
    windows_json: str,
) -> list[dict[str, Any]]:
    """Build an index from an explicit JSON list of ``{"uuid", "start"}`` windows.

    This is for evaluation sets whose starts do not necessarily lie on the trainer's
    non-overlapping precompute grid. It still validates the sequence split, RGB-camera
    file, video path, and window length against the manifest.
    """
    rows = json.load(open(windows_json))
    want = {(r["uuid"], int(r["start"])): i for i, r in enumerate(rows)}

    keep_uuids = None
    if split not in ("all", None):
        sp = json.load(open(split_file))
        assert split in sp, f"split '{split}' not in {split_file}"
        keep_uuids = set(sp[split])

    index: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            cam, vis = rec.get("camera_path"), rec.get("vision_path")
            nb = int(rec.get("nb_frames", 0))
            if not uuid or not cam or not vis:
                continue
            if keep_uuids is not None and uuid not in keep_uuids:
                continue
            rgb = _rgb_path(cam)
            if not os.path.isfile(rgb):
                continue
            for (u, s), order in want.items():
                if u == uuid and s + num_frames <= nb:
                    index.append({"uuid": uuid, "vis": vis, "rgb": rgb, "s": s, "_order": order})

    index.sort(key=lambda it: it["_order"])
    print(f"[precompute] windows_json: matched {len(index)}/{len(want)} requested windows",
          flush=True)
    return index


def out_path(root: str, uuid: str, start: int) -> str:
    """Per-subject sharded output path: <root>/<Sxx>/<seq>_<start>.npz."""
    uuid_safe = uuid.replace("/", "__")
    subj = uuid.split("/")[0] if "/" in uuid else "_misc"
    return os.path.join(root, subj, f"{uuid_safe}_{start}.npz")


def validate_existing_cached_file(path: str, contract: LatentCacheContract) -> None:
    """Reject a corrupt or incompatible file before treating it as resumable."""

    with np.load(path) as record:
        latents = record["latents"]
        camera_action = record["camera_action"]
        image_size = record["image_size"]
        record_t = int(record["T"])
        record_fps = float(record["fps"])
    validate_cached_sample(
        contract,
        latents=latents,
        camera_action=camera_action,
        image_size=image_size,
        context=path,
    )
    if record_t != contract.num_frames:
        raise ValueError(f"{path}: T={record_t} != {contract.num_frames}")
    if abs(record_fps - contract.fps) > 1e-6:
        raise ValueError(f"{path}: fps={record_fps} != {contract.fps}")


def deduplicate_physical_windows(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode one file per physical window even when captions duplicate rows."""

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in index:
        key = (str(item["uuid"]), int(item["s"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


# ----------------------------------------------------------------------------
# VAE: load the SAME Wan2.2 tokenizer the Nano model uses, then encode one
# resized+padded window exactly as omni_mot_model.get_data_and_condition would.
# ----------------------------------------------------------------------------
def load_vae(
    vae_path: str,
    resolution: str,
    num_frames: int,
    device: str,
    *,
    rank_local: bool = False,
):
    """Instantiate Wan2pt2VAEInterface with the NANO tokenizer config, pinned to
    encode this window length exactly (encode_exact_durations=[T], mirroring
    world_camera_nymeria_nano which sets the same on the model tokenizer).

    ``rank_local=True`` is reserved for rank-0-only checkpoint visualization.
    Wan's constructor normally broadcasts the newly loaded VAE to every rank,
    which deadlocks or mismatches collectives when only rank 0 is visualizing.
    Rank 0 already loads the complete local checkpoint, so that one broadcast
    can be disabled safely for this case.  All regular callers retain the
    framework's synchronized construction.
    """
    from cosmos_framework.model.vfm.tokenizers import (
        wan2pt2_vae_4x16x16 as wan_vae_module,
    )

    original_sync = None
    if rank_local:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            raise RuntimeError("rank-local VAE construction is only valid on global rank 0")
        original_sync = wan_vae_module.sync_model_states
        wan_vae_module.sync_model_states = lambda *_args, **_kwargs: None

    try:
        vae = wan_vae_module.Wan2pt2VAEInterface(
            bucket_name="",                       # local file (skip s3:// prefix)
            vae_path=vae_path,
            chunk_duration=93,
            keep_decoder_cache=False,
            use_streaming_encode=False,
            encode_chunk_frames={"256": 68, "480": 24, "720": 12},  # NANO default
            encode_exact_durations=[num_frames],  # exact length, no padding inflation
            spatial_compression_factor=16,
            temporal_compression_factor=4,
        )
    finally:
        if original_sync is not None:
            wan_vae_module.sync_model_states = original_sync
    return vae


def make_pipeline():
    """ActionTransformPipeline that ONLY does the spatial resize+pad on 'video'
    (no tokenizer, no CFG, no action padding logic we need) — the exact resize/pad
    the camera world-model feeds the VAE."""
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    return ActionTransformPipeline(
        pad_keys=["video"],
        keep_aspect_ratio=True,
        tokenizer_config=None,     # no text tokenization -> no model needed
        cfg_dropout_rate=0.0,
        max_action_dim=C.CAMERA_MAX_ACTION_DIM,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
    )


def encode_window(vae, pipe, resolution, frames_uint8: np.ndarray, device: str):
    """frames (T,H,W,3) uint8 -> (latents (z,T_lat,h,w) fp16, image_size (4,) fp32).

    Replicates omni_mot_model.get_data_and_condition's vision path:
      resize+reflection-pad -> /127.5 - 1 -> VAE.encode -> crop padding in latent.
    """
    video = torch.from_numpy(frames_uint8).permute(3, 0, 1, 2).contiguous()  # [3,T,H,W] uint8

    # 1. resize + reflection-pad to the resolution bucket; attaches image_size.
    sample = pipe({"video": video, "mode": "image2video", "ai_caption": ""}, resolution)
    vid = sample["video"]                         # [3,T,Hp,Wp], still uint8 (pipeline only resizes/pads)
    image_size = sample["image_size"].float()     # (4,) [target_h, target_w, orig_h, orig_w]

    # 2. uint8 -> [-1, 1] float, exactly as _normalize_video_databatch_inplace.
    assert vid.dtype == torch.uint8, f"expected uint8 from pipeline, got {vid.dtype}"
    vid = vid.to(device=device, dtype=torch.float32) / 127.5 - 1.0
    vid = vid.unsqueeze(0)                          # [1,3,T,Hp,Wp]

    # 3. Wan2.2-VAE encode -> [1, z_dim, T_lat, Hp//16, Wp//16].
    with torch.no_grad():
        latent = vae.encode(vid).contiguous().float()

    # 4. crop reflection padding in latent space (mirrors _remove_padding_from_latent).
    spatial_factor = vae.spatial_compression_factor
    orig_h = int(image_size[2].item())
    orig_w = int(image_size[3].item())
    h_lat = max(orig_h // spatial_factor, 1)
    w_lat = max(orig_w // spatial_factor, 1)
    latent = latent[:, :, :, :h_lat, :w_lat].contiguous()   # [1, z, T_lat, h, w]

    latents = latent[0].to(torch.float16).cpu().numpy()      # (z, T_lat, h, w)
    return latents, image_size.cpu().numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=C.NYMERIA_MANIFEST)
    ap.add_argument("--split_file", default=C.NYMERIA_SPLIT_FILE)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"],
                    help="which sequences to encode (default all: train+test windows)")
    ap.add_argument("--out_root", default=C.VIDEO_LATENT_ROOT)
    ap.add_argument("--num_frames", type=int, default=C.VIDEO_NUM_FRAMES)
    ap.add_argument("--fps", type=float, default=float(C.FPS))
    ap.add_argument("--resolution", default="256")
    ap.add_argument(
        "--model_resolution_tier",
        default=None,
        help="model tier recorded in the cache contract (for example 720)",
    )
    ap.add_argument(
        "--expected_image_hw",
        type=int,
        default=None,
        help="strict square transformed image size, required with --write_cache_contract",
    )
    ap.add_argument(
        "--expected_latent_hw",
        type=int,
        default=None,
        help="strict square latent size, required with --write_cache_contract",
    )
    ap.add_argument(
        "--write_cache_contract",
        action="store_true",
        help="atomically create/validate immutable cache metadata under out_root",
    )
    ap.add_argument(
        "--fail_on_error",
        action="store_true",
        help="return nonzero if any assigned window fails instead of accepting a partial shard",
    )
    ap.add_argument("--vae_path",
                    default=os.environ.get("WAN_VAE_PATH",
                                           "/weka/jungbin/wan22_vae/Wan2.2_VAE.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the window index into this many disjoint shards")
    ap.add_argument("--shard_id", type=int, default=0, help="which shard this process handles")
    ap.add_argument("--limit", type=int, default=None,
                    help="encode at most this many windows (quick smoke subset)")
    ap.add_argument("--windows_json", default=None,
                    help="optional explicit JSON list of {uuid,start} windows to encode")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-encode even if the .npz already exists (default: skip-existing)")
    args = ap.parse_args()

    if args.num_frames % 4 != 1:
        ap.error(f"num_frames must be 4N+1 for the Wan VAE (got {args.num_frames})")
    if not (0 <= args.shard_id < args.num_shards):
        ap.error(f"shard_id {args.shard_id} out of range for num_shards {args.num_shards}")
    if args.write_cache_contract and (
        args.model_resolution_tier is None
        or args.expected_image_hw is None
        or args.expected_latent_hw is None
    ):
        ap.error(
            "--write_cache_contract requires --model_resolution_tier, "
            "--expected_image_hw, and --expected_latent_hw"
        )

    print(f"[precompute] manifest={args.manifest}", flush=True)
    print(f"[precompute] out_root={args.out_root}  T={args.num_frames}  res={args.resolution}", flush=True)

    if args.windows_json:
        index = build_explicit_index(args.manifest, args.split_file, args.split,
                                     args.num_frames, args.windows_json)
    else:
        index = build_index(args.manifest, args.split_file, args.split, args.num_frames)
    n_source_rows = len(index)
    index = deduplicate_physical_windows(index)
    n_total_all = len(index)
    print(
        f"[precompute] source rows={n_source_rows}; unique physical windows={n_total_all}; "
        f"duplicate rows={n_source_rows - n_total_all}",
        flush=True,
    )
    # deterministic shard slice (stride so each shard sees a spread of sequences).
    index = index[args.shard_id :: args.num_shards]
    if args.limit is not None:
        index = index[: args.limit]
    print(f"[precompute] {n_total_all} usable windows total; "
          f"shard {args.shard_id}/{args.num_shards} -> {len(index)} windows"
          + (f" (limited to {args.limit})" if args.limit is not None else ""), flush=True)

    if len(index) == 0:
        print("[precompute] nothing to do.", flush=True)
        return

    os.makedirs(args.out_root, exist_ok=True)
    cache_contract = None
    if args.write_cache_contract:
        cache_contract = LatentCacheContract(
            schema_version=CACHE_CONTRACT_VERSION,
            kind=CACHE_CONTRACT_KIND,
            source_manifest=os.path.abspath(args.manifest),
            split_file=os.path.abspath(args.split_file),
            split=args.split,
            source_window_count=n_source_rows,
            expected_file_count=n_total_all,
            num_frames=args.num_frames,
            fps=args.fps,
            spatial_transform_resolution=str(args.resolution),
            model_resolution_tier=str(args.model_resolution_tier),
            expected_image_hw=(args.expected_image_hw, args.expected_image_hw),
            expected_latent_shape=(
                48,
                1 + (args.num_frames - 1) // 4,
                args.expected_latent_hw,
                args.expected_latent_hw,
            ),
            expected_camera_shape=(args.num_frames - 1, 9),
            latent_dtype="float16",
            vae_path=os.path.abspath(args.vae_path),
            num_shards=args.num_shards,
            limit_per_shard=args.limit,
        )
        contract_path = ensure_latent_cache_contract(args.out_root, cache_contract)
        print(f"[precompute] cache_contract={contract_path}", flush=True)

    # Load the VAE + the resize/pad pipeline once.
    t0 = time.time()
    vae = load_vae(args.vae_path, args.resolution, args.num_frames, args.device)
    pipe = make_pipeline()
    print(f"[precompute] Wan2.2-VAE loaded ({vae.model.count_param()/1e6:.1f}M params) "
          f"in {time.time()-t0:.1f}s; encoding...", flush=True)

    n_done = n_skip = n_fail = 0
    bytes_written = 0
    latent_shape = None
    t_start = time.time()

    for i, it in enumerate(index):
        uuid, vis, rgb, s = it["uuid"], it["vis"], it["rgb"], it["s"]
        T = args.num_frames
        dst = out_path(args.out_root, uuid, s)

        if (not args.overwrite) and os.path.isfile(dst):
            if cache_contract is not None and args.fail_on_error:
                try:
                    validate_existing_cached_file(dst, cache_contract)
                except Exception as error:  # noqa: BLE001 - repair invalid resume artifacts
                    print(
                        f"[precompute] REPAIR {uuid}@{s}: existing cache is invalid "
                        f"({type(error).__name__}: {error})",
                        flush=True,
                    )
                else:
                    n_skip += 1
                    if (i + 1) % args.log_every == 0:
                        _progress(i, index, n_done, n_skip, n_fail, bytes_written, t_start)
                    continue
            else:
                n_skip += 1
                if (i + 1) % args.log_every == 0:
                    _progress(i, index, n_done, n_skip, n_fail, bytes_written, t_start)
                continue

        try:
            frames = decode_window_pyav(vis, s, T, args.fps)          # (T,H,W,3) uint8
            pos, rot = _load_rgb_cam(rgb)
            act = rel_action_from_window(pos[s : s + T], rot[s : s + T])  # (T-1,9)
            if frames.shape[0] != T or act.shape[0] != T - 1 or not np.isfinite(act).all():
                raise ValueError(f"bad shapes/NaN: frames {frames.shape} act {act.shape}")

            latents, image_size = encode_window(vae, pipe, args.resolution, frames, args.device)
            if cache_contract is not None:
                validate_cached_sample(
                    cache_contract,
                    latents=latents,
                    camera_action=act,
                    image_size=image_size,
                    context=f"{uuid}@{s}",
                )
            elif not np.isfinite(latents).all():
                raise ValueError("non-finite latents")
            latent_shape = latents.shape
        except Exception as e:  # noqa: BLE001 — skip any bad window, keep going (resumable)
            n_fail += 1
            print(f"[precompute] SKIP {uuid}@{s} ({type(e).__name__}: {e})", flush=True)
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + f".tmp.{os.getpid()}.npz"  # unique per-proc atomic write (no cross-shard race on shared dst)
        np.savez(
            tmp,
            latents=latents,
            camera_action=act.astype(np.float32),
            image_size=image_size.astype(np.float32),
            uuid=uuid,
            start=np.int64(s),
            T=np.int64(T),
            fps=np.float32(args.fps),
        )
        os.replace(tmp, dst)
        bytes_written += os.path.getsize(dst)
        n_done += 1

        if (i + 1) % args.log_every == 0 or (i + 1) == len(index):
            _progress(i, index, n_done, n_skip, n_fail, bytes_written, t_start)

    dt = time.time() - t_start
    print("=" * 72, flush=True)
    print(f"[precompute] DONE shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  windows handled : {len(index)}", flush=True)
    print(f"  newly written   : {n_done}", flush=True)
    print(f"  skipped existing: {n_skip}", flush=True)
    print(f"  failed/skipped  : {n_fail}", flush=True)
    if latent_shape is not None:
        print(f"  latent shape    : {tuple(latent_shape)} (z_dim, T_lat, h, w) fp16", flush=True)
    print(f"  bytes written   : {bytes_written:,} ({bytes_written/1e9:.3f} GB)", flush=True)
    print(f"  elapsed         : {dt:.1f}s ({dt/max(n_done,1):.2f}s / new window)", flush=True)
    print(f"  out_root        : {args.out_root}", flush=True)
    if args.fail_on_error and n_fail:
        raise RuntimeError(
            f"strict precompute failed: shard {args.shard_id}/{args.num_shards} "
            f"had {n_fail} failed windows"
        )


def _progress(i, index, n_done, n_skip, n_fail, bytes_written, t_start):
    dt = time.time() - t_start
    rate = (i + 1) / max(dt, 1e-9)
    eta = (len(index) - (i + 1)) / max(rate, 1e-9)
    print(f"[precompute] {i+1}/{len(index)}  new={n_done} skip={n_skip} fail={n_fail}  "
          f"{bytes_written/1e9:.3f}GB  {rate:.1f} win/s  ETA {eta/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
