"""Thin GENERATOR-modality I/O adapter for the 7-task joint-attention model.

`GenHeads` encodes / decodes the three GENERATOR-carried modalities -- video latents,
the single first-frame IMAGE (a 1-frame video latent), and the camera pseudo-action --
by CALLING the frozen `cosmos.net`'s OWN params and methods. It reimplements nothing:
every projection routes through `net.vae2llm` / `net.llm2vae` / `net.action2llm` /
`net.llm2action` / `net.action_modality_embed` / `net.patchify_and_pack_latents` /
`net.unpatchify_and_unpack_latents` / `net.time_embedder`, so the generator tokens are
bit-identical to what `Cosmos3VFMNetwork._encode_vision` / `_encode_action` produce.

These tokens flow through the existing `_moe_gen` weight pathway automatically: the model
inserts them at `gen_idx` rows of the packed sequence and `MoTJointLayer.forward` already
routes any non-empty `gen_idx` through `q/k/v/o_proj_moe_gen` + `mlp_moe_gen` + the gen
layernorms (LoRA'd when `gen_lora`, fully trainable when `gen_full`). `GenHeads` owns NO
trainable params of its own -- it is a pure call adapter over the frozen net.

What each modality looks like in the packed sequence
----------------------------------------------------
* video : Wan2.2-VAE latents `[C, T_lat, h, w]` (PRECOMPUTED offline, see
          ``precompute_latents.py``) -> ``patchify_and_pack_latents`` -> ``vae2llm`` ->
          per-noised-frame timestep bias. One token PER LATENT PATCH; the latent grid is
          ``(T_lat, h//p, w//p)`` with ``p == net.latent_patch_size`` (2). 3D-mRoPE
          position ids come straight from that grid via ``get_3d_mrope_ids_vae_tokens``.
* image : literally video latent FRAME 0, always CLEAN (condition_mask=1, no timestep
          bias). When a task lists ``image`` but no ``video`` (textimg2motion) we pack ONLY
          that single clean latent frame (``T_lat == 1``). Encode-only -- never decoded.
* camera: relative SE(3) pseudo-action ``(T-1, 9)`` -> ``pad_action_to_max_dim(64)`` ->
          ``action2llm(per_token_domain_id=2)`` -> ``+ action_modality_embed`` -> per-noised
          -frame timestep bias. One token per action frame; 1x1 spatial grid; packed at the
          VISION part's START temporal offset (parallel to vision, native Cosmos parity) with
          ``start_frame_offset=1`` so action[0] aligns with vision frame 1.

The CONDITIONING MECHANISM is the single per-token ``condition_mask`` (1=CLEAN -> sigma 0,
no timestep bias, excluded from loss; 0=NOISED -> timestep bias, supervised), exactly as in
``task_plan.py`` / ``motion_heads.py``. We never normalize camera (Cosmos-exact); video lives
in VAE latent space; only motion (elsewhere) is z-scored.

Public surface (consumed by ``joint_motion_model.forward`` and ``sample.py``)
-----------------------------------------------------------------------------
* ``encode_video_latents(latents, condition_mask, sigma)``  -> GenTokens
* ``encode_image_latent(frame0_latent, sigma)``             -> GenTokens (1 clean frame)
* ``encode_camera(action_9, condition_mask, sigma, domain_id=2)`` -> GenTokens
* ``decode_video(gen_hidden, grid, noisy_frame_idx, orig_shape)`` -> latents (tasks 2,3,6)
* ``decode_camera(gen_hidden, n_noisy, domain_id=2)``       -> ``(N, 9)`` (tasks 1,3)
* ``build_gen_segment(resolved, *, video_latents, camera_action, sigma)`` -> GenSegment
  (the packed gen rows + their ``[3, N_gen]`` mRoPE ids + per-token condition_mask + the
  per-modality decode bookkeeping), given a ``task_plan.ResolvedPlan``.

All tensors are produced on ``self.device`` in ``self.dtype`` (bf16) for the hidden tokens;
the timestep embed is computed in fp32 (autocast off) then cast back, matching Cosmos.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

# `pad_action_to_max_dim` (Cosmos-exact zero-pad 9->64) lives in nymeria_world; reuse it
# verbatim rather than reimplementing the pad.
_NYMERIA_WORLD = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
if _NYMERIA_WORLD not in sys.path:
    sys.path.insert(0, _NYMERIA_WORLD)
from camera_to_action import pad_action_to_max_dim  # noqa: E402

# 3D-mRoPE id builders -- the SAME helpers Cosmos's sequence packer uses, so the gen tokens
# get position ids identical to the native path.
from cosmos_framework.model.vfm.mot.unified_3dmrope_utils import (  # noqa: E402
    get_3d_mrope_ids_vae_tokens,
)

import task_plan as TP  # noqa: E402  (CAMERA_DOMAIN_ID, CAMERA_RAW_DIM, ResolvedPlan)


# ============================================================================
# Return containers
# ============================================================================
@dataclass
class GenTokens:
    """Encoded tokens for ONE generator modality (video | image | camera).

    tokens
        ``[N, hidden]`` (bf16) packed tokens, ready to scatter into the packed sequence at
        this modality's ``gen_idx`` rows.
    mrope_ids
        ``[3, N]`` LONG 3D-mRoPE position ids (T/H/W) for these tokens, with a temporal
        offset already applied (so they continue after the reasoner / earlier gen segments).
        The model feeds these straight into ``cosmos.rope`` for the gen rows' cos/sin.
    condition_mask
        ``[N]`` bool: True == CLEAN (no timestep bias, excluded from loss), False == NOISED.
    next_temporal_offset
        The temporal offset the NEXT packed segment should start from (``max(t)+1``). NOTE:
        camera runs temporally PARALLEL to the vision part (native Cosmos parity, see
        ``build_gen_segment``), so the segment-level next offset is the max over parts.
    grid
        ``(grid_t, grid_h, grid_w)`` token grid (for video/image: the patchified latent grid;
        for camera: ``(T, 1, 1)``). Needed to unpatchify on decode.
    n_frames
        Number of temporal frames this modality contributes (T_lat for video/image, T-1 for
        camera). The per-frame condition is broadcast across the spatial grid in ``tokens``.
    noisy_frame_idx
        ``[n_noised_frames]`` LONG indices (into the modality's own frame axis) of the NOISED
        frames -- the frames the decode places predictions back into.
    """
    tokens: torch.Tensor
    mrope_ids: torch.Tensor
    condition_mask: torch.Tensor
    next_temporal_offset: int | float
    grid: tuple[int, int, int]
    n_frames: int
    noisy_frame_idx: torch.Tensor


@dataclass
class GenSegment:
    """The whole packed generator segment for ONE sample (image/video/camera concatenated).

    tokens
        ``[N_gen, hidden]`` bf16 -- the gen rows in packed order (the order the model lays
        them out between the reasoner rows and the motion rows).
    mrope_ids
        ``[3, N_gen]`` long -- 3D-mRoPE ids for those rows (temporal axis continues from the
        reasoner offset; the camera part shares the VISION part's temporal window -- native
        Cosmos action<->vision parallelism -- while image/video pack sequentially).
    condition_mask
        ``[N_gen]`` bool -- per-token CLEAN/NOISED.
    next_temporal_offset
        offset for whatever packs after the gen segment (the motion segment):
        ``max`` over the parts' next offsets, so downstream tokens never collide.
    parts
        per-modality ``GenTokens`` keyed by name (``"video"`` | ``"image"`` | ``"camera"``),
        in the packed order, carrying the row offsets + decode bookkeeping. The model uses
        these to slice the post-attention gen hidden back per modality and call
        ``decode_video`` / ``decode_camera`` for the SUPERVISED ones only.
    offsets
        ``{name: (start, end)}`` -- the half-open row range each modality occupies within
        ``tokens`` (and within the sample's gen_idx rows).
    """
    tokens: torch.Tensor
    mrope_ids: torch.Tensor
    condition_mask: torch.Tensor
    next_temporal_offset: int | float
    parts: dict[str, GenTokens] = field(default_factory=dict)
    offsets: dict[str, tuple[int, int]] = field(default_factory=dict)


# ============================================================================
# GenHeads
# ============================================================================
class GenHeads:
    """Call adapter over the frozen ``cosmos.net`` generator I/O (no params of its own)."""

    def __init__(self, cosmos, dtype: torch.dtype = torch.bfloat16, device=None):
        """``cosmos`` is a ``cosmos_loader.FrozenCosmos`` (we use ``cosmos.net`` + ``cosmos.rope``)."""
        self.cosmos = cosmos
        self.net = cosmos.net
        self.dtype = dtype
        self.device = device or getattr(cosmos, "device", "cuda")

        # Cache the frozen generator-I/O config off the net (set when vision_gen / action_gen).
        self.patch = int(getattr(self.net, "latent_patch_size", 2))            # p (spatial)
        self.latent_channel = int(getattr(self.net, "latent_channel", 16))     # C
        self.patch_latent_dim = int(getattr(self.net, "patch_latent_dim",
                                            self.patch * self.patch * self.latent_channel))
        self.timestep_scale = float(getattr(self.net, "timestep_scale", 0.001))
        self.tcf_vision = int(getattr(self.net, "temporal_compression_factor_vision", 4))
        self.base_fps = float(getattr(self.net, "base_fps", 24.0))
        self.action_dim = int(getattr(self.net, "action_dim", 64))             # max_action_dim (zero-pad target)
        self.camera_domain_id = TP.CAMERA_DOMAIN_ID
        self.camera_raw_dim = TP.CAMERA_RAW_DIM

    # ------------------------------------------------------------------ timestep bias
    def _timestep_bias(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma[...] -> additive timestep embed ``[*, hidden]`` (fp32 embedder, cast to dtype).

        Mirrors ``_encode_vision`` / ``_encode_action``: ``time_embedder(sigma * timestep_scale)``
        computed under fp32 autocast, then cast back to the hidden dtype.
        """
        s = sigma.reshape(-1).float() * self.timestep_scale
        with torch.autocast(device_type=torch.device(self.device).type, enabled=True,
                            dtype=torch.float32):
            emb = self.net.time_embedder(s)            # [*, hidden] fp32
        return emb.to(self.dtype)

    # ------------------------------------------------------------------ video / image encode
    def encode_video_latents(
        self,
        latents: torch.Tensor,            # [C, T_lat, h, w]  Wan2.2-VAE latents (single sample)
        condition_mask: torch.Tensor,    # [T_lat] bool: True == CLEAN frame
        sigma: torch.Tensor,             # scalar | [1] flow time/sigma for the NOISED frames
        *,
        temporal_offset: int | float = 0,
        fps: Optional[float] = None,
    ) -> GenTokens:
        """Encode one video latent stack -> packed gen tokens (mirrors ``net._encode_vision``).

        Steps (all via frozen net params):
          1. ``patchify_and_pack_latents([latent], [(T_lat,h,w)])`` -> ``[Np, patch_latent_dim]``
             with the latent grid ``(T_lat, h//p, w//p)`` flattened T-major.
          2. ``vae2llm`` -> ``[Np, hidden]``.
          3. per-NOISED-frame timestep bias: ``time_embedder(sigma*scale)`` broadcast across the
             spatial grid of each noised latent frame (clean frames get NO bias). Implemented by
             building a per-PATCH mask from the per-frame ``~condition_mask``.
          4. 3D-mRoPE ids from the grid (``get_3d_mrope_ids_vae_tokens``), temporal axis offset.

        ``condition_mask`` is per-LATENT-FRAME (length ``T_lat``); it is broadcast across the
        ``h//p * w//p`` patches of each frame to produce the per-TOKEN condition mask.
        """
        assert latents.dim() == 4, f"latents must be [C,T,h,w], got {tuple(latents.shape)}"
        C, T_lat, h, w = latents.shape
        p = self.patch
        latents = latents.to(self.device)
        # patchify_and_pack_latents expects a list of [1,C,T,H,W] tensors + their (t,h,w) shapes.
        packed_patches, orig_shapes = self.net.patchify_and_pack_latents(
            [latents.unsqueeze(0)], [(T_lat, h, w)]
        )  # packed_patches: [Np, patch_latent_dim]; orig_shapes: [(T_lat,h,w)]
        tokens = self.net.vae2llm(packed_patches.to(self.dtype))         # [Np, hidden]

        grid_h, grid_w = ((h + p - 1) // p), ((w + p - 1) // p)
        spatial = grid_h * grid_w
        Np = tokens.shape[0]
        assert Np == T_lat * spatial, (Np, T_lat, spatial)

        cond_frame = condition_mask.to(self.device).bool().view(T_lat)   # [T_lat]
        noisy_frame = ~cond_frame                                        # [T_lat]
        # per-token (per-patch) condition mask: broadcast the per-frame flag across the grid.
        cond_token = cond_frame.view(T_lat, 1).expand(T_lat, spatial).reshape(Np)

        # timestep bias on the NOISED frames' patches only.
        if noisy_frame.any():
            bias = self._timestep_bias(sigma.reshape(1))[0]             # [hidden] (one sigma)
            noisy_token = (~cond_token).view(Np, 1).to(self.dtype)
            tokens = tokens + noisy_token * bias.view(1, -1)

        mrope_ids, next_off = get_3d_mrope_ids_vae_tokens(
            grid_t=T_lat, grid_h=grid_h, grid_w=grid_w,
            temporal_offset=temporal_offset,
            reset_spatial_indices=True,
            fps=fps, base_fps=self.base_fps,
            temporal_compression_factor=self.tcf_vision,
            base_temporal_compression_factor=self.tcf_vision,
            start_frame_offset=0,
        )  # mrope_ids: [3, Np]
        # FPS-modulated native Cosmos positions are float; do not quantize them.
        mrope_ids = mrope_ids.to(self.device)

        return GenTokens(
            tokens=tokens.to(self.dtype),
            mrope_ids=mrope_ids,
            condition_mask=cond_token.to(self.device),
            next_temporal_offset=next_off,
            grid=(T_lat, grid_h, grid_w),
            n_frames=T_lat,
            noisy_frame_idx=torch.nonzero(noisy_frame, as_tuple=False).view(-1),
        )

    def encode_image_latent(
        self,
        frame0_latent: torch.Tensor,     # [C, 1, h, w] OR [C, h, w]  the single clean image frame
        *,
        temporal_offset: int | float = 0,
    ) -> GenTokens:
        """Encode the single first-frame IMAGE latent, ALWAYS clean (no timestep bias, no loss).

        Used by tasks that list ``image`` but no ``video`` (textimg2motion): we pack exactly one
        CLEAN latent frame (``T_lat == 1``). Encode-only -- there is no ``llm2vae`` decode for it.
        """
        if frame0_latent.dim() == 3:
            frame0_latent = frame0_latent.unsqueeze(1)                  # [C,1,h,w]
        assert frame0_latent.shape[1] == 1, "image latent must be a single frame"
        cond = torch.ones(1, dtype=torch.bool, device=self.device)     # the 1 frame is clean
        # sigma is irrelevant for an all-clean frame (no bias added); pass 0.
        return self.encode_video_latents(
            frame0_latent, condition_mask=cond,
            sigma=torch.zeros(1, device=self.device),
            temporal_offset=temporal_offset, fps=None,
        )

    # ------------------------------------------------------------------ camera encode
    def encode_camera(
        self,
        action_9: torch.Tensor,          # [T-1, 9] raw relative SE(3) pseudo-action (un-normalized)
        condition_mask: torch.Tensor,    # [T-1] bool: True == CLEAN frame
        sigma: torch.Tensor,             # scalar | [1] flow time/sigma for the NOISED frames
        *,
        domain_id: int | None = None,
        temporal_offset: int | float = 0,
        fps: Optional[float] = None,
    ) -> GenTokens:
        """Encode camera action -> packed gen tokens (mirrors ``net._encode_action``).

        Steps (all via frozen net params):
          1. ``pad_action_to_max_dim(9 -> action_dim=64)`` (Cosmos-exact zero-pad; NO normalization).
          2. ``action2llm(x, per_token_domain_id=domain_id)`` -> ``[T-1, hidden]``.
          3. ``+ action_modality_embed``.
          4. per-NOISED-frame timestep bias (clean frames untouched).
          5. 3D-mRoPE ids: 1x1 spatial grid, ``temporal_compression_factor_action=1``,
             ``start_frame_offset=1`` (action[0] aligns with vision frame 1).
        """
        domain_id = self.camera_domain_id if domain_id is None else int(domain_id)
        assert action_9.dim() == 2 and action_9.shape[-1] == self.camera_raw_dim, \
            f"camera action must be [T-1,{self.camera_raw_dim}], got {tuple(action_9.shape)}"
        T = action_9.shape[0]

        # 1. pad 9 -> action_dim (64), via the verbatim nymeria_world helper (numpy in/out).
        padded = pad_action_to_max_dim(action_9.detach().cpu().numpy(), self.action_dim)
        x = torch.from_numpy(padded).to(self.device, dtype=self.dtype)  # [T, action_dim]

        per_token_domain = torch.full((T,), domain_id, dtype=torch.long, device=self.device)
        tokens = self.net.action2llm(x, per_token_domain)               # [T, hidden]
        tokens = tokens + self.net.action_modality_embed.view(1, -1).to(self.dtype)

        cond_frame = condition_mask.to(self.device).bool().view(T)      # [T]
        noisy_frame = ~cond_frame
        if noisy_frame.any():
            bias = self._timestep_bias(sigma.reshape(1))[0]             # [hidden]
            tokens = tokens + (~cond_frame).view(T, 1).to(self.dtype) * bias.view(1, -1)

        # action: 1x1 spatial grid, action temporal compression factor = 1, start_frame_offset=1.
        mrope_ids, next_off = get_3d_mrope_ids_vae_tokens(
            grid_t=T, grid_h=1, grid_w=1,
            temporal_offset=temporal_offset,
            reset_spatial_indices=True,
            fps=fps, base_fps=self.base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=self.tcf_vision,
            start_frame_offset=1,
        )  # [3, T]
        mrope_ids = mrope_ids.to(self.device)

        return GenTokens(
            tokens=tokens.to(self.dtype),
            mrope_ids=mrope_ids,
            condition_mask=cond_frame.to(self.device),
            next_temporal_offset=next_off,
            grid=(T, 1, 1),
            n_frames=T,
            noisy_frame_idx=torch.nonzero(noisy_frame, as_tuple=False).view(-1),
        )

    # ------------------------------------------------------------------ video decode
    def decode_video(
        self,
        gen_hidden_noisy: torch.Tensor,  # [N_noisy_patches, hidden] post-attn gen hidden @ noised rows
        grid: tuple[int, int, int],      # (T_lat, grid_h, grid_w) token grid for this sample
        noisy_frame_idx: torch.Tensor,   # [n_noisy] LONG indices of the noised latent frames
        original_latent_shape: tuple[int, int, int],  # (T_lat, h, w) pre-patch latent shape
    ) -> torch.Tensor:
        """Decode supervised video tokens -> predicted latents (mirrors ``net._decode_vision``).

        ``llm2vae(hidden)`` -> ``[N_noisy_patches, patch_latent_dim]`` -> ``unpatchify_and_unpack_latents``
        -> ``[1, C, T_lat, h, w]`` with predictions placed at the noised frames (clean frames 0).
        Used by tasks 2 (forward_dynamics), 3 (policy), 6 (motimg2video). The IMAGE (frame 0) and
        any condition-only video (task 7) are NEVER passed here.
        """
        preds = self.net.llm2vae(gen_hidden_noisy.to(self.dtype))       # [N_noisy_patches, patch_latent_dim]
        T_lat, gh, gw = grid
        out_list = self.net.unpatchify_and_unpack_latents(
            preds,
            token_shapes_vision=[(T_lat, gh, gw)],
            noisy_frame_indexes_vision=[noisy_frame_idx.to(self.device)],
            original_latent_shapes=[original_latent_shape],
        )  # list of [1, C, T_lat, h, w]
        return out_list[0]

    # ------------------------------------------------------------------ camera decode
    def decode_camera(
        self,
        gen_hidden_noisy: torch.Tensor,  # [N_noisy, hidden] post-attn gen hidden @ noised action rows
        *,
        domain_id: int | None = None,
    ) -> torch.Tensor:
        """Decode supervised camera tokens -> ``[N_noisy, 9]`` (mirrors ``net._decode_action``).

        ``llm2action(hidden, domain_id)`` -> ``[N_noisy, action_dim=64]`` then SLICE to the raw
        channels ``[:9]`` (the zero-pad channels are not supervised). Used by tasks 1
        (inverse_dynamics) and 3 (policy). The model passes only the NOISED action rows.
        """
        domain_id = self.camera_domain_id if domain_id is None else int(domain_id)
        N = gen_hidden_noisy.shape[0]
        per_token_domain = torch.full((N,), domain_id, dtype=torch.long, device=self.device)
        preds = self.net.llm2action(gen_hidden_noisy.to(self.dtype), per_token_domain)  # [N, action_dim]
        return preds[:, : self.camera_raw_dim]                          # [N, 9]

    # ------------------------------------------------------------------ build the gen segment
    def build_gen_segment(
        self,
        resolved: "TP.ResolvedPlan",
        *,
        video_latents: Optional[torch.Tensor] = None,   # [C, T_lat, h, w] (video OR image-only frame0)
        camera_action: Optional[torch.Tensor] = None,   # [T-1, 9] raw camera pseudo-action
        sigma: Optional[torch.Tensor] = None,           # scalar flow time for the NOISED gen frames
        temporal_offset: int | float = 0,
        fps: Optional[float] = None,
    ) -> Optional[GenSegment]:
        """Assemble the packed GENERATOR segment for ONE sample from its ``ResolvedPlan``.

        Reads the resolved per-modality CLEAN/NOISED masks (built by ``task_plan.resolve_sample``)
        and encodes exactly the present generator modalities in packed order
        ``[ image | video | camera ]`` (image and video are mutually exclusive in the resolved
        plan: when both ``image`` and ``video`` are present in a TASK, the resolver folds the
        image into the video stack as frame 0, so here at most one of {image, video} appears).

        Returns ``None`` when the task has no generator modality (text2motion) -- the caller then
        keeps ``gen_idx`` empty (the existing text->motion fast path). Otherwise returns a
        ``GenSegment`` with the concatenated gen rows, their ``[3, N_gen]`` mRoPE ids, the
        per-token condition mask, and the per-modality decode bookkeeping in ``parts`` / ``offsets``.

        Temporal-offset layout (NATIVE Cosmos parity -- sequence_packing.py packs actions at
        ``action_temporal_offset = vision_start_temporal_offset`` and does NOT advance the
        offset past them): the CAMERA part is packed at the SAME temporal offset the vision
        (image/video) part STARTED at, so camera tokens run temporally PARALLEL to the vision
        tokens. With ``encode_camera``'s ``start_frame_offset=1``, camera[i] then lands on
        temporal id ``vision_start + 1 + i`` -- action[0] aligns with vision frame 1, exactly
        the pretrained action<->video temporal prior. (Every camera-packing task in task_plan
        also packs video -- inverse/forward/policy -- so the vision start always exists; if a
        future task packed camera WITHOUT vision, the camera falls back to its own sequential
        offset.) The segment's ``next_temporal_offset`` is the MAX over all parts' next offsets
        (camera can extend past the video: T-1 action ids vs T_lat latent ids), so the motion
        segment packed after never collides in mRoPE time. ``temporal_offset`` should be the
        reasoner length (so gen ids continue after the reasoner causal ids).
        """
        if not resolved.has_gen:
            return None
        if sigma is None:
            sigma = torch.zeros(1, device=self.device)

        parts: dict[str, GenTokens] = {}
        token_chunks: list[torch.Tensor] = []
        mrope_chunks: list[torch.Tensor] = []
        cond_chunks: list[torch.Tensor] = []
        offsets: dict[str, tuple[int, int]] = {}
        off = temporal_offset          # sequential offset (advanced by vision parts only)
        next_off = temporal_offset     # max over all parts' next offsets (for the NEXT segment)
        vision_start_off: Optional[int] = None  # temporal offset the image/video part started at
        row = 0

        # Order matters for packing + decode slicing: image (clean frame0), then video, then camera.
        # (image and video do not co-occur as separate entries in the resolved plan.)
        for name in ("image", "video", "camera"):
            mr = resolved.modalities.get(name)
            if mr is None or not mr.present:
                continue

            cmask = torch.tensor(mr.condition_mask, dtype=torch.bool, device=self.device)
            if name == "image":
                assert video_latents is not None, "image task needs the frame-0 latent in video_latents"
                vision_start_off = off
                gt = self.encode_image_latent(video_latents[:, :1], temporal_offset=off)
                off = gt.next_temporal_offset
            elif name == "video":
                assert video_latents is not None, "video task needs video_latents [C,T_lat,h,w]"
                vision_start_off = off
                gt = self.encode_video_latents(
                    video_latents, condition_mask=cmask, sigma=sigma,
                    temporal_offset=off, fps=fps,
                )
                off = gt.next_temporal_offset
            else:  # camera
                assert camera_action is not None, "camera task needs camera_action [T-1,9]"
                # NATIVE parity (sequence_packing._pack_action_tokens): camera tokens are packed
                # at the VISION part's START offset and run temporally PARALLEL to it (the offset
                # is not advanced past them). encode_camera's start_frame_offset=1 then aligns
                # action[0] with vision frame 1. Fallback: camera-without-vision (no such task
                # today) keeps the sequential offset.
                cam_off = vision_start_off if vision_start_off is not None else off
                gt = self.encode_camera(
                    camera_action, condition_mask=cmask, sigma=sigma,
                    temporal_offset=cam_off, fps=fps,
                )
                if vision_start_off is None:
                    off = gt.next_temporal_offset

            n = gt.tokens.shape[0]
            token_chunks.append(gt.tokens)
            mrope_chunks.append(gt.mrope_ids)
            cond_chunks.append(gt.condition_mask)
            offsets[name] = (row, row + n)
            parts[name] = gt
            row += n
            next_off = max(next_off, gt.next_temporal_offset)

        tokens = torch.cat(token_chunks, dim=0)                         # [N_gen, hidden]
        mrope_ids = torch.cat(mrope_chunks, dim=1)                      # [3, N_gen]
        condition_mask = torch.cat(cond_chunks, dim=0)                  # [N_gen]
        return GenSegment(
            tokens=tokens.to(self.dtype),
            mrope_ids=mrope_ids,
            condition_mask=condition_mask,
            next_temporal_offset=next_off,
            parts=parts,
            offsets=offsets,
        )


__all__ = ["GenHeads", "GenTokens", "GenSegment"]
