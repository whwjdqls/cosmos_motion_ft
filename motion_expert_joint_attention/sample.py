"""Sampling for the 7-task joint-attention multimodal model (cosmos env).

A single packed-sequence model ``JointMotionModel`` (frozen reasoner + frozen/LoRA
generator + trainable ``_moe_motion`` motion expert) generates ONE of three target
modalities per task -- motion (283-d uniego), video (Wan2.2-VAE latents) or camera
(9-d relative SE(3) pseudo-action) -- by running the rectified-flow REVERSE process on
the NOISED target while every conditioning modality (clean image / video / camera /
text) is pinned clean. CFG runs two reasoner contexts (the real caption vs the empty
prompt) through the SAME live forward; the conditioning modalities are IDENTICAL in both
passes and only the text content differs (DESIGN_7TASK.md section 6).

The 7 tasks + their (conditioning -> target) modality contract (task_plan.py):

    inverse_dynamics   video                 -> camera     (no text)
    forward_dynamics   camera + text + image -> video
    policy             text + image          -> camera + video
    text2motion        text                  -> motion
    textimg2motion     text + image          -> motion
    motimg2video       motion + text + image -> video
    video2motion       video                 -> motion     (no text)

For each task this file exposes a clean ``sample_<task>(...)`` function returning the
generated modality in its native space (motion unnormalized to 283-d via decode_uniego;
video as Wan-VAE latents; camera as (T-1,9) raw action). ``sample_task(model, mode, ...)``
dispatches by mode. PER-MODALITY objectives (matches train.py): MOTION targets dispatch on
the ckpt's trained motion objective -- ``flow.sample_x0`` (DDIM-in-sigma, the default for
new x0-motion runs) vs ``flow.sample_velocity`` (old velocity-motion ckpts) -- via
``model.predict_closure``; VIDEO/CAMERA targets ALWAYS integrate velocity
(``flow.sample_velocity_masked`` / the policy Euler co-integration) with a target closure
that re-encodes the current noised iterate through the generator each step, matching the
pretrained Cosmos generator's native rectified flow.

The original text->motion CLI ``main()`` is preserved verbatim (it now calls
``sample_text2motion`` under the hood) so existing run.sh invocations keep working:

  ssh a3ultravis-a3ultranodeset-1 'CUDA_VISIBLE_DEVICES=0 bash \
     motion_expert_joint_attention/run.sh sample.py \
     --ckpt <run>/ckpt_step050000.pt --objective velocity --out <run>/samples_step50k'
Then decode + render with viz.py in the kimodo env using the manifest written here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import torch

import config
import flow
import task_plan as TP
from cosmos_loader import FrozenCosmos
from joint_motion_model import JointMotionModel
from uniego_layout import FEAT_DIM, N_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
UNIEGO_ROOT = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"

DEFAULT_PROMPTS = [
    "a person walks forward",
    "a person turns around and walks back",
    "a person sits down on a chair",
    "a person picks up an object from the floor",
    "a person waves their right hand",
    "a person stands still",
]


# ============================================================================
# Skeleton / prompt helpers (text->motion CLI; unchanged contract)
# ============================================================================
def load_default_skeleton() -> np.ndarray:
    """Centered neutral_joints (30,3) from the first S01 train actor (fixed size to sample)."""
    f = sorted(glob.glob(os.path.join(UNIEGO_ROOT, "S01", "*.npz")))[0]
    nj = np.load(f)["neutral_joints"].astype(np.float32)
    return nj - nj.mean(axis=0, keepdims=True)


def load_skeleton(npz_path: str | None) -> np.ndarray:
    if npz_path:
        nj = np.load(npz_path)["neutral_joints"].astype(np.float32)
        return nj - nj.mean(axis=0, keepdims=True)
    return load_default_skeleton()


def read_prompts(args) -> list[str]:
    if args.prompts_file:
        with open(args.prompts_file) as fh:
            prompts = [ln.strip() for ln in fh if ln.strip()]
        if not prompts:
            raise ValueError(f"--prompts_file {args.prompts_file} is empty")
        return prompts
    return args.prompts


def slugify(prompt: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in prompt)[:40]


# ============================================================================
# Shared: motion-stats (de)normalization + per-modality target sampling closures.
# ============================================================================
def _as_list(x, B: int):
    """Coerce a per-sample tensor/None into a length-B list of per-sample entries."""
    if x is None:
        return [None] * B
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _tokenize_pair(cosmos: FrozenCosmos, caption: str, *, always_empty: bool):
    """Return (cond_ids, null_ids) for one caption. ``always_empty`` tasks (inverse_dynamics /
    video2motion) pass "" for BOTH passes (no instruction => CFG is a no-op there)."""
    cond = "" if always_empty else (caption or "")
    return cosmos.tokenize(cond), cosmos.tokenize("")


# ---- MOTION target ----------------------------------------------------------
@torch.no_grad()
def _sample_motion_target(
    model: JointMotionModel,
    *,
    mode: str,
    caption: str,
    neutral_joints: torch.Tensor,    # [1,30,3]
    T: int,
    video_latents: Optional[torch.Tensor] = None,   # [C,T_lat,h,w] clean conditioning (tasks 5,7)
    reasoner_image: Optional[torch.Tensor] = None,  # [3,H,W] uint8 for corrected textimg2motion
    steps: int = 50,
    guidance: float = 2.5,
    objective: str | None = None,
    device=None,
    seed: int = 0,
) -> np.ndarray:
    """Reverse-process the NOISED motion while holding any conditioning image/video/text clean.

    Reuses ``model.predict_closure`` (it returns the motion prediction -- v-hat or x0-hat per
    the trained MOTION objective; the sampler is picked to match below) + the existing
    motion samplers. The clean conditioning modalities (image latent for textimg2motion; full
    video stack for video2motion) are packed IDENTICALLY in the cond and null passes -- only the
    reasoner text content differs. Returns normalized / z-scored motion ``[T,283]``.
    """
    device = device or neutral_joints.device
    neutral_joints = neutral_joints.to(device)
    obj = objective or model.objective
    plan = TP.build_task_plan(mode)

    motion_pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    noisy_frame_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    cond_ids, null_ids = _tokenize_pair(model.cosmos, caption,
                                        always_empty=plan.caption_always_empty)

    vlat = [video_latents.to(device)] if video_latents is not None else [None]
    modes = [mode]
    reasoner_cond = reasoner_null = None
    if model.textimg_condition == "reasoner" and mode == "textimg2motion":
        if reasoner_image is None:
            raise ValueError("sample_textimg2motion for textimg_condition=reasoner needs reasoner_image=[3,H,W]")
        reasoner_cond = model.cosmos.encode_reasoner_image_text(caption, reasoner_image)
        reasoner_null = model.cosmos.encode_reasoner_image_text("", reasoner_image)
        cond_ids = reasoner_cond["input_ids"].view(1, -1)
        null_ids = reasoner_null["input_ids"].view(1, -1)
        vlat = [None]

    predict_cond = model.predict_closure(
        input_ids_list=[cond_ids], neutral_joints=neutral_joints,
        motion_pad_mask=motion_pad_mask, noisy_frame_mask=noisy_frame_mask,
        modes=modes, video_latents=vlat,
        reasoner_inputs=[reasoner_cond] if reasoner_cond is not None else None,
    )
    # text_policy == "empty" tasks pass "" for BOTH passes -> the null forward would be
    # byte-identical to the cond one (CFG is exactly a no-op: v_u + g*(v_c - v_u) == v_c).
    # Skip it entirely (predict_null=None disables the second forward in the sampler).
    predict_null = None
    if not plan.caption_always_empty:
        predict_null = model.predict_closure(
            input_ids_list=[null_ids], neutral_joints=neutral_joints,
            motion_pad_mask=motion_pad_mask, noisy_frame_mask=noisy_frame_mask,
            modes=modes, video_latents=vlat,
            reasoner_inputs=[reasoner_null] if reasoner_null is not None else None,
        )

    sampler = flow.sample_velocity if obj == "velocity" else flow.sample_x0
    g_m = torch.Generator(device=device).manual_seed(seed)
    x = sampler(
        predict_cond, T=T, motion_dim=model.motion_dim, steps=steps,
        guidance=guidance, predict_null=predict_null,
        device=device, dtype=torch.float32, generator=g_m,
    )  # [1,T,283] normalized
    return x[0].cpu().numpy().astype(np.float32)


# ---- VIDEO target -----------------------------------------------------------
def _make_gen_target_closure(
    model: JointMotionModel,
    *,
    mode: str,
    target: str,                      # "video" | "camera"
    input_ids,                        # reasoner ids for THIS pass (cond or null)
    neutral_joints: Optional[torch.Tensor],
    motion: Optional[torch.Tensor],   # [1,T,283] CLEAN motion condition (motimg2video) else None
    motion_pad_mask: Optional[torch.Tensor],
    clean_video_latents: Optional[torch.Tensor],   # [C,T_lat,h,w] with clean frames at truth
    clean_camera_action: Optional[torch.Tensor],   # [T-1,9] clean action (forward_dynamics)
    cond_video_mask: Optional[torch.Tensor],       # [T_lat] True=CLEAN latent frame
    cond_camera_mask: Optional[torch.Tensor],      # [T-1] True=CLEAN action frame
    device,
):
    """Build ``fn(x_flat, t_b) -> v_flat`` for a VIDEO or CAMERA generation target.

    ``x_flat`` is the current noised iterate for the TARGET modality, flattened to ``[1, N, D]``
    (N=#noised target frames, D = latent-channels or 9). Each call: scatter the noised iterate
    back into a full per-frame tensor (clean frames at their truth values), run ``model.forward``
    with that tensor as the generator input (the model adds the timestep bias to the noised
    frames -- matching the native Cosmos noised-input contract), then gather the predicted
    velocity at the noised frames and re-flatten. Conditioning modalities (clean video/image/
    camera/motion + text) are packed identically across cond/null passes; only ``input_ids``
    differs for CFG.
    """
    modes = [mode]
    nj = neutral_joints

    if target == "video":
        assert clean_video_latents is not None and cond_video_mask is not None
        base = clean_video_latents.to(device)                 # [C,T_lat,h,w]
        C, T_lat, h, w = base.shape
        noised_frames = torch.nonzero(~cond_video_mask.to(device).bool(),
                                      as_tuple=False).view(-1)  # [n_noisy]
        # forward_dynamics conditions the generated video on a CLEAN camera action -- pass it
        # through so model.forward's resolver sees n_camera>0 (else it raises "packs camera but
        # n_camera=0"). Tasks with no camera modality (motimg2video) leave this None.
        cam_cond = ([clean_camera_action.to(device)] if clean_camera_action is not None
                    else [None])

        def fn(x_flat: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
            # x_flat: [1, n_noisy, C*h*w] noised latent for the noised frames.
            cur = base.clone()
            xv = x_flat[0].reshape(noised_frames.numel(), C, h, w).permute(1, 0, 2, 3)  # [C,n,h,w]
            cur[:, noised_frames] = xv.to(cur.dtype)
            out = model.forward(
                input_ids_list=[input_ids], x_t=motion, t_or_sigma=t_b,
                neutral_joints=nj, motion_pad_mask=motion_pad_mask,
                noisy_frame_mask=(torch.zeros(1, motion.shape[1], dtype=torch.bool, device=device)
                                  if motion is not None else None),
                modes=modes, video_latents=[cur], camera_action=cam_cond,
                return_dict=True,
            )
            pred = out["video_pred"][0]                        # [1,C,T_lat,h,w] (preds @ noised)
            vp = pred[0][:, noised_frames]                     # [C,n,h,w]
            vp = vp.permute(1, 0, 2, 3).reshape(1, noised_frames.numel(), C * h * w)
            return vp.to(x_flat.dtype)

        meta = dict(C=C, T_lat=T_lat, h=h, w=w, noised_frames=noised_frames,
                    D=C * h * w, N=int(noised_frames.numel()))
        return fn, meta

    if target == "camera":
        assert clean_camera_action is not None and cond_camera_mask is not None
        base = clean_camera_action.to(device)                 # [T-1,9]
        Tm1, D = base.shape
        noised_frames = torch.nonzero(~cond_camera_mask.to(device).bool(),
                                      as_tuple=False).view(-1)
        vlat = [clean_video_latents.to(device)] if clean_video_latents is not None else [None]

        def fn(x_flat: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
            cur = base.clone()
            cur[noised_frames] = x_flat[0].reshape(noised_frames.numel(), D).to(cur.dtype)
            out = model.forward(
                input_ids_list=[input_ids], x_t=motion, t_or_sigma=t_b,
                neutral_joints=nj, motion_pad_mask=motion_pad_mask,
                noisy_frame_mask=None,
                modes=modes, video_latents=vlat, camera_action=[cur],
                return_dict=True,
            )
            pred = out["camera_pred"][0]                       # [n_noisy, 9]
            return pred.reshape(1, noised_frames.numel(), D).to(x_flat.dtype)

        meta = dict(D=D, Tm1=Tm1, noised_frames=noised_frames, N=int(noised_frames.numel()))
        return fn, meta

    raise ValueError(f"_make_gen_target_closure: bad target {target!r}")


# ---- POLICY joint (video + camera) target ----------------------------------
def _make_policy_joint_closure(
    model: JointMotionModel,
    *,
    input_ids,                          # reasoner ids for THIS pass (cond or null)
    base_video: torch.Tensor,           # [C,T_lat,h,w] with the CLEAN frame 0 at truth
    noised_video_frames: torch.Tensor,  # [n_noisy] LONG indices of the noised latent frames
    camera_T: int,
    device,
):
    """Build ``fn(xv_flat, xc, t_b) -> (v_vid_flat, v_cam)`` for the policy JOINT target.

    ONE model call per step with BOTH current iterates packed exactly like training
    (train.step_loss policy branch): video frame 0 clean + frames 1.. = the video iterate,
    camera all frames = the camera iterate, ONE shared t. Returns the predicted velocities
    for both modalities so the sampler Euler-steps the pair together.
      xv_flat : [1, n_noisy, C*h*w] current noised video latents (noised frames only).
      xc      : [1, camera_T, 9]    current noised camera action (ALL frames noised).
    """
    C, T_lat, h, w = base_video.shape
    n_noisy = int(noised_video_frames.numel())

    def fn(xv_flat: torch.Tensor, xc: torch.Tensor, t_b: torch.Tensor):
        cur = base_video.clone()
        xv = xv_flat[0].reshape(n_noisy, C, h, w).permute(1, 0, 2, 3)   # [C,n,h,w]
        cur[:, noised_video_frames] = xv.to(cur.dtype)
        cam = xc[0].to(base_video.dtype)                                # [camera_T,9]
        out = model.forward(
            input_ids_list=[input_ids], x_t=None, t_or_sigma=t_b,
            neutral_joints=None, motion_pad_mask=None, noisy_frame_mask=None,
            modes=["policy"], video_latents=[cur], camera_action=[cam],
            return_dict=True,
        )
        vp = out["video_pred"][0][0][:, noised_video_frames]            # [C,n,h,w]
        v_vid = vp.permute(1, 0, 2, 3).reshape(1, n_noisy, C * h * w)
        v_cam = out["camera_pred"][0].reshape(1, camera_T, -1)          # all frames noised
        return v_vid.to(xv_flat.dtype), v_cam.to(xc.dtype)

    return fn


@torch.no_grad()
def _sample_gen_target(
    model: JointMotionModel,
    *,
    mode: str,
    target: str,                      # "video" | "camera"
    caption: str,
    clean_video_latents: Optional[torch.Tensor] = None,   # [C,T_lat,h,w] (frame0/all clean per task)
    clean_camera_action: Optional[torch.Tensor] = None,   # [T-1,9]
    motion: Optional[torch.Tensor] = None,                # [1,T,283] CLEAN motion (motimg2video)
    neutral_joints: Optional[torch.Tensor] = None,
    steps: int = 50,
    guidance: float = 2.5,
    device=None,
    seed: int = 0,
):
    """Reverse-process the NOISED video latents OR camera action while holding conditioning clean.

    Resolves the per-sample CLEAN/NOISED layout from ``task_plan.resolve_sample`` (so the same
    masking the trainer uses drives which frames are integrated), builds the target velocity
    closures for the cond + null reasoner contexts, and integrates only the noised frames with
    ``flow.sample_velocity_masked``. Returns the generated modality in its native space:
      video  -> Wan-VAE latents ``[C, T_lat, h, w]`` (clean frames at truth, noised frames generated)
      camera -> raw action ``[T-1, 9]`` (clean frames at truth, noised frames generated).
    """
    device = device or model.cosmos.device
    plan = TP.build_task_plan(mode)

    # Resolve which frames are clean vs noised for the target modality.
    t_lat = int(clean_video_latents.shape[1]) if clean_video_latents is not None else 0
    n_cam = int(clean_camera_action.shape[0]) if clean_camera_action is not None else 0
    mvm = None
    if plan.motion.present and motion is not None:
        mvm = [True] * int(motion.shape[1])
    resolved = TP.resolve_sample(
        mode, t_lat=t_lat, n_camera=n_cam,
        motion_valid_mask=mvm, has_shape_token=plan.motion.present,
    )
    tgt_res = resolved.modalities[target]
    cond_mask = torch.tensor(tgt_res.condition_mask, dtype=torch.bool, device=device)

    motion_pad_mask = None
    if motion is not None:
        motion = motion.to(device)
        motion_pad_mask = torch.zeros(1, motion.shape[1], dtype=torch.bool, device=device)
    nj = neutral_joints.to(device) if neutral_joints is not None else None

    cond_ids, null_ids = _tokenize_pair(model.cosmos, caption,
                                        always_empty=plan.caption_always_empty)

    common = dict(
        model=model, mode=mode, target=target, neutral_joints=nj, motion=motion,
        motion_pad_mask=motion_pad_mask, clean_video_latents=clean_video_latents,
        clean_camera_action=clean_camera_action,
        cond_video_mask=(cond_mask if target == "video" else None),
        cond_camera_mask=(cond_mask if target == "camera" else None),
        device=device,
    )
    predict_cond, meta = _make_gen_target_closure(input_ids=cond_ids, **common)
    # "empty"-text tasks (inverse_dynamics): cond == null byte-identically -> CFG is a no-op;
    # skip the redundant null forward (halves the per-step model calls).
    predict_null = None
    if not plan.caption_always_empty:
        predict_null, _ = _make_gen_target_closure(input_ids=null_ids, **common)

    # Build the flat clean tensor [1, N, D] over the NOISED frames (its values are ignored where
    # noised -- overwritten by sampled noise at init -- so any placeholder works; we pass zeros).
    N, D = meta["N"], meta["D"]
    x0_clean = torch.zeros(1, N, D, device=device, dtype=torch.float32)
    cond_flat = torch.zeros(1, N, dtype=torch.bool, device=device)  # all noised -> integrate all

    g = torch.Generator(device=device).manual_seed(seed)
    x = flow.sample_velocity_masked(
        predict_cond, x0_clean=x0_clean, condition_mask=cond_flat,
        steps=steps, guidance=guidance, predict_null=predict_null,
        device=device, dtype=torch.float32, generator=g,
    )  # [1, N, D] generated noised frames

    # Scatter the generated frames back into the full modality tensor (clean frames at truth).
    if target == "video":
        C, T_lat, h, w = meta["C"], meta["T_lat"], meta["h"], meta["w"]
        out = clean_video_latents.to(device).clone()           # [C,T_lat,h,w]
        nf = meta["noised_frames"]
        xv = x[0].reshape(nf.numel(), C, h, w).permute(1, 0, 2, 3)
        out[:, nf] = xv.to(out.dtype)
        return out.cpu().numpy().astype(np.float32)
    else:  # camera
        out = clean_camera_action.to(device).clone()           # [T-1,9]
        nf = meta["noised_frames"]
        out[nf] = x[0].reshape(nf.numel(), meta["D"]).to(out.dtype)
        return out.cpu().numpy().astype(np.float32)


# ============================================================================
# Per-task public eval functions (one per task; clean signatures for the eval harness).
# Each returns the generated modality in native space. Motion is returned NORMALIZED
# (z-scored) -- the caller un-normalizes with the 283-d stats (decode happens in viz.py).
# ============================================================================
def sample_text2motion(model, *, caption, neutral_joints, T, **kw) -> np.ndarray:
    """text -> motion [T,283] (z-scored). The existing trained path."""
    return _sample_motion_target(model, mode="text2motion", caption=caption,
                                 neutral_joints=neutral_joints, T=T, **kw)


def sample_textimg2motion(
    model, *, caption, neutral_joints, T, image_latent=None, reasoner_image=None, **kw
) -> np.ndarray:
    """text + image -> motion [T,283] (z-scored).

    New checkpoints should use ``reasoner_image`` as raw uint8 ``[3,H,W]`` frame-0 pixels.
    Historical/deprecated checkpoints may use ``image_latent`` as a clean generator latent.
    """
    if model.textimg_condition == "reasoner":
        return _sample_motion_target(model, mode="textimg2motion", caption=caption,
                                     neutral_joints=neutral_joints, T=T,
                                     reasoner_image=reasoner_image, **kw)
    if image_latent is None:
        raise ValueError("sample_textimg2motion needs image_latent for textimg_condition=generator")
    if image_latent.dim() == 3:
        image_latent = image_latent.unsqueeze(1)               # [C,1,h,w]
    return _sample_motion_target(model, mode="textimg2motion", caption=caption,
                                 neutral_joints=neutral_joints, T=T,
                                 video_latents=image_latent, **kw)


def sample_video2motion(model, *, neutral_joints, T, video_latents, **kw) -> np.ndarray:
    """video (all clean [C,T_lat,h,w]) -> motion [T,283] (z-scored). No text instruction."""
    return _sample_motion_target(model, mode="video2motion", caption="",
                                 neutral_joints=neutral_joints, T=T,
                                 video_latents=video_latents, **kw)


def sample_forward_dynamics(model, *, caption, image_latent, camera_action, T_lat=None, **kw):
    """camera + text + image -> video latents [C,T_lat,h,w].

    ``image_latent`` is the clean first frame ([C,h,w] or [C,1,h,w]); ``T_lat`` is the latent
    length to generate (frames 1.. are noised). We tile the clean frame 0 into a [C,T_lat,h,w]
    stack (frames 1.. are placeholders, overwritten during sampling)."""
    if image_latent.dim() == 4:
        image_latent = image_latent[:, 0]                      # [C,h,w]
    C, h, w = image_latent.shape
    if T_lat is None:
        raise ValueError("forward_dynamics needs T_lat (the latent length to generate)")
    base = image_latent.new_zeros(C, T_lat, h, w)
    base[:, 0] = image_latent
    return _sample_gen_target(model, mode="forward_dynamics", target="video", caption=caption,
                              clean_video_latents=base, clean_camera_action=camera_action, **kw)


def sample_inverse_dynamics(model, *, video_latents, **kw) -> np.ndarray:
    """video (all clean [C,T_lat,h,w]) -> camera action [T-1,9]. No text instruction.

    The camera frame count is T-1 for the T-frame pixel window; pass ``camera_T`` for the action
    length (defaults to 4*(T_lat-1) for a 4x temporally-compressed window: T = 4*(T_lat-1)+1)."""
    camera_T = kw.pop("camera_T", None)
    if camera_T is None:
        T_lat = int(video_latents.shape[1])
        camera_T = 4 * (T_lat - 1)                             # (T-1) for T = 4*(T_lat-1)+1
    cam0 = torch.zeros(camera_T, TP.CAMERA_RAW_DIM, device=video_latents.device)
    return _sample_gen_target(model, mode="inverse_dynamics", target="camera", caption="",
                              clean_video_latents=video_latents, clean_camera_action=cam0, **kw)


@torch.no_grad()
def sample_policy(model, *, caption, image_latent, T_lat, camera_T=None,
                  steps=50, guidance=2.5, device=None, seed=0):
    """text + image -> (camera action [T-1,9], video latents [C,T_lat,h,w]). JOINT generation.

    Both modalities are noised+generated; the image (frame 0) is the only clean condition.
    Training (train.step_loss policy branch) noises BOTH with the SAME t and runs ONE forward
    over the pair -- so the sampler CO-INTEGRATES them: ONE Euler ODE loop whose state is the
    PAIR (video frames 1.., camera all frames); each step calls the model ONCE with both
    current iterates packed exactly like training (video frame0 clean + frames1.. = iterate,
    camera all-frames = iterate, one shared t), reads both predicted velocities, and steps
    both. (Integrating them independently with zeros as the other modality is a never-trained
    distribution.) Returns a dict {'video': ..., 'camera': ...} in native space."""
    device = device or model.cosmos.device
    if image_latent.dim() == 4:
        image_latent = image_latent[:, 0]
    image_latent = image_latent.to(device)
    C, h, w = image_latent.shape
    base = image_latent.new_zeros(C, T_lat, h, w)
    base[:, 0] = image_latent
    if camera_T is None:
        camera_T = 4 * (T_lat - 1)

    # Resolve the training CLEAN/NOISED layout (video frame0 clean, camera all noised).
    resolved = TP.resolve_sample("policy", t_lat=T_lat, n_camera=camera_T)
    vmask = torch.tensor(resolved.modalities["video"].condition_mask,
                         dtype=torch.bool, device=device)               # [T_lat] True=CLEAN
    cmask = torch.tensor(resolved.modalities["camera"].condition_mask,
                         dtype=torch.bool, device=device)               # [camera_T]
    assert not cmask.any(), "policy camera must be fully noised (matches training)"
    noised_frames = torch.nonzero(~vmask, as_tuple=False).view(-1)      # [n_noisy]

    plan = TP.build_task_plan("policy")
    cond_ids, null_ids = _tokenize_pair(model.cosmos, caption,
                                        always_empty=plan.caption_always_empty)
    common = dict(model=model, base_video=base, noised_video_frames=noised_frames,
                  camera_T=camera_T, device=device)
    predict_cond = _make_policy_joint_closure(input_ids=cond_ids, **common)
    predict_null = None
    if guidance != 1.0 and not plan.caption_always_empty:
        predict_null = _make_policy_joint_closure(input_ids=null_ids, **common)

    # ONE Euler ODE over the pair (t: 1 -> 0), CFG applied per modality per step.
    n_v = int(noised_frames.numel())
    g = torch.Generator(device=device).manual_seed(seed)
    xv = torch.randn(1, n_v, C * h * w, device=device, dtype=torch.float32, generator=g)
    xc = torch.randn(1, camera_T, TP.CAMERA_RAW_DIM, device=device, dtype=torch.float32,
                     generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    for i in range(steps):
        t_b = ts[i].expand(1)
        dt = ts[i] - ts[i + 1]
        v_v, v_c = predict_cond(xv, xc, t_b)
        if guidance != 1.0 and predict_null is not None:
            u_v, u_c = predict_null(xv, xc, t_b)
            v_v = u_v + guidance * (v_v - u_v)
            v_c = u_c + guidance * (v_c - u_c)
        xv = xv - dt * v_v.to(xv.dtype)
        xc = xc - dt * v_c.to(xc.dtype)

    video = base.clone()                                                # [C,T_lat,h,w]
    video[:, noised_frames] = (
        xv[0].reshape(n_v, C, h, w).permute(1, 0, 2, 3).to(video.dtype))
    return {"video": video.cpu().numpy().astype(np.float32),
            "camera": xc[0].cpu().numpy().astype(np.float32)}


def sample_motimg2video(model, *, caption, image_latent, motion, neutral_joints, T_lat, **kw):
    """motion (all clean [1,T,283]) + text + image -> video latents [C,T_lat,h,w]."""
    if image_latent.dim() == 4:
        image_latent = image_latent[:, 0]
    C, h, w = image_latent.shape
    base = image_latent.new_zeros(C, T_lat, h, w)
    base[:, 0] = image_latent
    return _sample_gen_target(model, mode="motimg2video", target="video", caption=caption,
                              clean_video_latents=base, motion=motion,
                              neutral_joints=neutral_joints, **kw)


# ---- dispatch ---------------------------------------------------------------
def sample_task(model, mode: str, **kw):
    """Dispatch to the per-task sampler by ``mode`` (one of task_plan.TASKS).

    Each task takes the conditioning modalities it needs (see the per-task function docstrings):
      text2motion       (caption, neutral_joints, T)
      textimg2motion    (caption, neutral_joints, T, image_latent)
      video2motion      (neutral_joints, T, video_latents)
      forward_dynamics  (caption, image_latent, camera_action, T_lat)
      inverse_dynamics  (video_latents)
      policy            (caption, image_latent, T_lat)
      motimg2video      (caption, image_latent, motion, neutral_joints, T_lat)
    plus shared sampling kwargs (steps, guidance, seed, device, objective for motion tasks).
    Returns the generated modality (np.ndarray) or, for policy, a {'video','camera'} dict.
    """
    fns = {
        "text2motion": sample_text2motion,
        "textimg2motion": sample_textimg2motion,
        "video2motion": sample_video2motion,
        "forward_dynamics": sample_forward_dynamics,
        "inverse_dynamics": sample_inverse_dynamics,
        "policy": sample_policy,
        "motimg2video": sample_motimg2video,
    }
    if mode not in fns:
        raise KeyError(f"sample_task: unknown mode {mode!r}; expected one of {TP.TASKS}")
    return fns[mode](model, **kw)


# ============================================================================
# Model loading (shared by the CLI + the eval harness).
# ============================================================================
def load_joint_model(ckpt_path: str, *, device="cuda", objective_cli=None,
                     gen_lora=False, reasoner_lora=False, gen_full=False):
    """Build a FrozenCosmos + JointMotionModel and overlay the trainable ckpt delta by name.

    Returns (model, cosmos, ckpt_meta). Honors the train-scope toggles recorded in the ckpt args
    (so a gen-LoRA / gen-full run rebuilds the same trainable params before the overlay).

    The resolved ``objective`` is the MOTION objective only (per-modality design): it picks the
    motion-target sampler (``flow.sample_x0`` vs ``flow.sample_velocity``). Vision/camera targets
    ALWAYS integrate velocity (``sample_velocity_masked`` / the policy co-integration), matching
    the pretrained Cosmos generator's native rectified flow."""
    cosmos = FrozenCosmos(dtype=torch.bfloat16, device=device)
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck.get("args", {}) or {}
    # MOTION objective: the ckpt's recorded value wins (train.py saves vars(args)). Old
    # checkpoints that never recorded it were velocity-motion runs -> default "velocity" so
    # they keep sampling correctly via sample_velocity. objective_cli is only a fallback /
    # sanity cross-check, never an override.
    objective = a.get("objective") or objective_cli or "velocity"
    if objective_cli is not None and objective != objective_cli:
        print(f"[sample] WARNING: ckpt motion objective='{objective}' != "
              f"--objective='{objective_cli}'; using the ckpt's '{objective}'")
    # ---- ARCHITECTURE-AFFECTING ckpt args: rebuild the exact trained model shape. ----------
    # train.py saves vars(args); the JointMotionModel ctor knobs that change the parameter set,
    # shapes, or positional behavior must be replayed here. Without threading them, a checkpoint
    # can load successfully but evaluate under the wrong architecture/position convention.
    _arch_missing = [k for k in ("motion_intermediate", "motion_layer_stride") if k not in a]
    if _arch_missing:
        print(f"[sample] WARNING: ckpt args missing architecture keys {_arch_missing} (old "
              f"checkpoint?) -- falling back to config defaults "
              f"(motion_intermediate={config.MOTION_INTERMEDIATE_SIZE}, "
              f"motion_layer_stride={config.MOTION_LAYER_STRIDE}). If the run used non-default "
              f"values, the overlay below will skip mismatched tensors (watch the skipped count).")
    motion_intermediate = int(a.get("motion_intermediate", config.MOTION_INTERMEDIATE_SIZE))
    motion_layer_stride = int(a.get("motion_layer_stride", config.MOTION_LAYER_STRIDE))
    motion_mrope = str(a.get("motion_mrope", "legacy"))
    coupling = str(a.get("coupling", "joint"))
    textimg_condition = str(a.get("textimg_condition", "generator"))
    if textimg_condition == "generator":
        print("[sample][DEPRECATED] checkpoint uses textimg_condition=generator for textimg2motion. "
              "This historical path packs the image as a clean generator latent; new TI2M runs "
              "should use textimg_condition=reasoner.", flush=True)
    model = JointMotionModel(
        cosmos, objective=objective, motion_dim=FEAT_DIM,
        motion_intermediate_size=motion_intermediate,
        motion_layer_stride=motion_layer_stride,
        motion_mrope=motion_mrope,
        coupling=coupling,
        textimg_condition=textimg_condition,
        gen_lora=a.get("gen_lora", gen_lora),
        reasoner_lora=a.get("reasoner_lora", reasoner_lora),
        gen_full=a.get("gen_full", gen_full),
        freeze_gen=bool(a.get("freeze_gen", False)),
        freeze_motion=bool(a.get("freeze_motion", False)),
    ).to(device)

    def _load_saved_model_state(path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("model", payload) if isinstance(payload, dict) else payload

    # Bridge-only / frozen-specialist Phase-3 checkpoints save only trainable bridge tensors.
    # Recreate the actual training-time model by loading the recorded frozen specialists first,
    # then overlay this checkpoint's trainable delta below.
    init_gen = a.get("init_gen")
    if init_gen:
        if os.path.exists(init_gen):
            gsd = _load_saved_model_state(init_gen)
            n_load, n_miss, n_shape = model.load_gen_subset(gsd)
            print(f"[sample][init_gen] {init_gen}: loaded {n_load} gen keys "
                  f"(skipped missing={n_miss} shape-mismatch={n_shape})")
        else:
            print(f"[sample][WARN] ckpt records init_gen={init_gen!r}, but the file is missing; "
                  "evaluating without the frozen generator specialist.")
    init_motion = a.get("init_motion")
    if init_motion:
        if os.path.exists(init_motion):
            msd = _load_saved_model_state(init_motion)
            n_load, n_miss, n_shape = model.load_motion_subset(msd)
            print(f"[sample][init_motion] {init_motion}: loaded {n_load} motion keys "
                  f"(skipped missing={n_miss} shape-mismatch={n_shape})")
        else:
            print(f"[sample][WARN] ckpt records init_motion={init_motion!r}, but the file is missing; "
                  "evaluating without the frozen motion specialist.")

    state = ck.get("model", ck)
    # Overlay the trainable delta by name over BOTH the model's own params and cosmos.net's
    # (LoRA / gen_full params are saved with a `cosmos.net.` prefix by trainable_state_dict).
    own = dict(model.named_all_parameters())
    own.update(dict(model.named_buffers()))
    own.update({model._NET_PREFIX + n: b for n, b in model.cosmos.net.named_buffers()})
    loaded, skipped = 0, 0
    with torch.no_grad():
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(own[k].dtype))
                loaded += 1
            else:
                skipped += 1
    model.eval()
    print(f"[sample] loaded {ckpt_path} (step {ck.get('step')}) | "
          f"motion_objective={objective} (vision/camera targets: always velocity) | "
          f"motion_mrope={motion_mrope} | "
          f"coupling={coupling} textimg_condition={textimg_condition} | "
          f"overlaid {loaded} trainable tensors (skipped {skipped})")
    return model, cosmos, ck


# ============================================================================
# CLI: text->motion sampling (backward compatible).
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--prompts_file", default=None,
                    help="one prompt per line; overrides --prompts")
    ap.add_argument("--objective", choices=["velocity", "x0"], default=None,
                    help="MOTION-target sampler fallback for OLD ckpts whose saved args lack "
                         "'objective' (velocity=Euler ODE on v-hat; x0=DDIM-in-sigma on x0-hat). "
                         "The ckpt's recorded motion objective always wins when present; "
                         "vision/camera targets always sample velocity (per-modality design).")
    ap.add_argument("--T", type=int, default=96, help="number of motion frames to sample")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5, help="classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skeleton", default=None,
                    help="uniego npz to take neutral_joints from; default = first S01 train actor")
    # Stats live in the repo dir (same constant path train.py uses): uniego283_{mean,std}.npy.
    ap.add_argument("--mean", default=os.path.join(HERE, "uniego283_mean.npy"))
    ap.add_argument("--std", default=os.path.join(HERE, "uniego283_std.npy"))
    ap.add_argument("--ablation", choices=["cfg", "both"], default="cfg",
                    help="cfg=just the CFG-guided sample; both=ALSO save the null/uncond "
                         "(empty-prompt) sample for the does-text-matter test")
    args = ap.parse_args()

    dev = "cuda"
    mean = torch.from_numpy(np.load(args.mean)).float().to(dev)
    std = torch.from_numpy(np.load(args.std)).float().to(dev)

    model, cosmos, ck = load_joint_model(args.ckpt, device=dev, objective_cli=args.objective)
    objective = model.objective

    # ---- actor skeleton (shape token) ----
    nj = load_skeleton(args.skeleton)
    nj_t = torch.from_numpy(nj).float().to(dev)[None]  # [1,30,3]

    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for prompt in read_prompts(args):
        modes = [("cfg", args.cfg)]
        if args.ablation == "both":
            modes.append(("null", 1.0))   # empty-prompt, no guidance (does-text-matter test)

        for mode, guidance in modes:
            cap = prompt if mode == "cfg" else ""
            x_norm = sample_text2motion(
                model, caption=cap, neutral_joints=nj_t, T=args.T,
                steps=args.steps, guidance=guidance, objective=objective,
                device=dev, seed=args.seed,
            )  # [T,283] normalized
            x = (torch.from_numpy(x_norm).to(dev) * std + mean).cpu().numpy().astype(np.float32)
            name = f"{slugify(prompt)}__{mode}"
            np.save(os.path.join(args.out, name + ".npy"), x)
            manifest.append({
                "prompt": prompt, "mode": mode, "file": name + ".npy",
                "ckpt": args.ckpt, "objective": objective, "cfg": float(guidance),
                "T": args.T, "steps": args.steps,
            })
            print(f"  [{mode}] '{prompt[:40]}' -> {name}.npy  "
                  f"(range {x.min():.2f}..{x.max():.2f})")

    # ---- render-ready manifest (consumed by viz.py in the kimodo env) ----
    np.save(os.path.join(args.out, "skeleton_neutral_joints.npy"), nj)
    json.dump(
        {
            "n_joints": N_JOINTS, "feat_dim": FEAT_DIM,
            "ckpt": args.ckpt, "objective": objective, "step": ck.get("step"),
            "cfg": args.cfg, "T": args.T, "steps": args.steps,
            "skeleton": args.skeleton or "default_S01_neutral_joints",
            "samples": manifest,
        },
        open(os.path.join(args.out, "manifest.json"), "w"), indent=2,
    )
    print(f"[sample] wrote {len(manifest)} samples + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
