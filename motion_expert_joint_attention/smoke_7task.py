#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-process GPU smoke for the 7-task joint-attention model (cosmos env).

Builds the REAL model -- frozen Cosmos-3 Nano + trainable motion expert (`_moe_motion` +
`MotionHeads` + `norm_moe_motion`) + generator LoRA (`--gen_lora`, default ON here) -- and
exercises the load-bearing seams end to end on ONE GPU, WITHOUT needing the full `/weka`
dataset:

  1. TEXT2MOTION (no latents needed): a few steps on SYNTHETIC motion/captions through the
     real packed forward + per-task flow loss. Asserts:
       * loss is finite,
       * grad flows onto the active-trainable set (motion pathway + gen-LoRA adapters),
       * EVERY frozen param carries zero grad (the freeze routing did not leak) -- via
         `model.assert_frozen_grads_zero()`, which (after this audit) covers BOTH the model's
         own params AND `cosmos.net`'s LoRA / gen_full params.
  2. (optional) ONE VIDEO TASK (`video2motion` by default; `--video_mode`): only when
     `--latents_dir` points at a precompute_latents.py output subset. We pick one real
     `.npz`, build a B=1 batch with that latent + synthetic motion, and run one step,
     asserting finite loss + the same freeze partition with a NON-EMPTY generator segment
     (so the `gen_idx` flip + `_moe_gen`/gen-LoRA routing is actually exercised).

This mirrors what `train.py --smoke` does but is self-contained (synthetic data) so it can run
before any precompute / dataset build. The two share the SAME loss + freeze code paths.

Run (cosmos env, ONE GPU on a node -- NOT the login node)::

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
    unset LD_LIBRARY_PATH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/home/jungbin_cho/cosmos-framework:\
/home/jungbin_cho/cosmos_motion_ft/nymeria_world:\
/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
    cd /home/jungbin_cho/cosmos-framework
    python /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/smoke_7task.py \
        --gen_lora --steps 3
    # also exercise a video task against a precomputed-latent subset:
    python .../smoke_7task.py --gen_lora --latents_dir /weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config
import flow
import task_plan as TP
from cosmos_loader import FrozenCosmos
from decode_uniego_torch import decode_joints
from joint_motion_model import JointMotionModel
from uniego_layout import FEAT_DIM, N_JOINTS


# ----------------------------------------------------------------------------
# Synthetic batch builders (the SAME row keys / collate shapes as
# nymeria_joint_dataset.collate_joint, so step_loss is exercised verbatim).
# ----------------------------------------------------------------------------
def _synthetic_motion_batch(B: int, T: int, mode: str, device, rng):
    """A collate-shaped batch of `B` motion samples of `mode` (text2motion / textimg2motion)."""
    motion = torch.from_numpy(rng.standard_normal((B, T, FEAT_DIM)).astype(np.float32))
    nj = torch.from_numpy(rng.standard_normal((B, N_JOINTS, 3)).astype(np.float32))
    pad = torch.zeros((B, T), dtype=torch.bool)             # all valid
    return {
        "mode": [mode] * B,
        "source": ["synthetic"] * B,
        "caption": ["a person walks forward"] * B,
        "domain_id": torch.full((B,), config.CAMERA_DOMAIN_ID, dtype=torch.long),
        "motion": motion,
        "neutral_joints": nj,
        "motion_pad_mask": pad,
        "camera_action": None,
        "camera_pad_mask": None,
        "video_latents": [None] * B,
        "video_frames": [None] * B,
    }


def _video_batch_from_latent(npz_path: str, T: int, mode: str, device, rng):
    """A B=1 batch for a VIDEO-conditioned task from one precompute_latents .npz.

    `mode` defaults to video2motion (video clean -> motion target): needs the video latent +
    synthetic motion + neutral_joints. Other gen tasks (forward/inverse/policy/motimg2video)
    are also supported -- we always supply the latent + camera + motion so the resolver can pack
    whatever the task needs.
    """
    with np.load(npz_path) as d:
        lat = d["latents"].astype(np.float32)               # (C, T_lat, h, w)
        cam = d["camera_action"].astype(np.float32) if "camera_action" in d else None
    vlat = torch.from_numpy(np.ascontiguousarray(lat))      # [C,T_lat,h,w]
    plan = TP.build_task_plan(mode)

    motion = nj = pad = None
    if plan.motion.present:
        motion = torch.from_numpy(rng.standard_normal((1, T, FEAT_DIM)).astype(np.float32))
        nj = torch.from_numpy(rng.standard_normal((1, N_JOINTS, 3)).astype(np.float32))
        pad = torch.zeros((1, T), dtype=torch.bool)
    camera_action = camera_pad = None
    if plan.camera.present:
        Tc = cam.shape[0] if cam is not None else (T - 1)
        c = (torch.from_numpy(np.ascontiguousarray(cam)) if cam is not None
             else torch.zeros(Tc, config.CAMERA_RAW_ACTION_DIM))
        camera_action = c.unsqueeze(0)                       # [1,Tc,9]
        camera_pad = torch.zeros((1, Tc), dtype=torch.bool)

    return {
        "mode": [mode],
        "source": ["synthetic"],
        "caption": [""] if plan.caption_always_empty else ["a person walks forward"],
        "domain_id": torch.full((1,), config.CAMERA_DOMAIN_ID, dtype=torch.long),
        "motion": motion,
        "neutral_joints": nj,
        "motion_pad_mask": pad,
        "camera_action": camera_action,
        "camera_pad_mask": camera_pad,
        "video_latents": [vlat],
        "video_frames": [None],
    }


# ----------------------------------------------------------------------------
# step_loss: a COMPACT re-statement of train.step_loss's modality wiring, so the smoke
# exercises the exact noising/forward/loss contract without importing train.main().
# (It deliberately mirrors train.py; if you change one, change both.)
# ----------------------------------------------------------------------------
def step_loss(model, cosmos, batch, std, mean, dev, w):
    modes = batch["mode"]
    B = len(modes)
    input_ids_list = [cosmos.tokenize(c) for c in batch["caption"]]

    motion = batch["motion"]
    nj = batch["neutral_joints"]
    pad = batch["motion_pad_mask"]
    x_t = t_motion = target_motion = noisy_frame_mask = None
    x0 = None
    if motion is not None:
        x0 = motion.to(dev)
        nj = nj.to(dev)
        pad = pad.to(dev)
        valid = ~pad
        cond_motion = torch.ones_like(valid)
        for s in range(B):
            plan = TP.build_task_plan(modes[s])
            if plan.motion.present and plan.motion.clean_policy != "all":
                cond_motion[s] = pad[s]
        x_t, t_motion, target_motion, noised_motion = flow.add_noise_velocity_masked(x0, cond_motion)
        noisy_frame_mask = noised_motion & valid

    cam_dense = batch["camera_action"]
    cam_pad = batch["camera_pad_mask"]
    vid_list = batch["video_latents"]
    t_sample = t_motion if t_motion is not None else torch.rand(B, device=dev)

    noised_vid = [None] * B
    target_vid = [None] * B
    if cam_dense is not None:
        cam_dense = cam_dense.to(dev)
        cam_pad = cam_pad.to(dev)
    for s in range(B):
        plan = TP.build_task_plan(modes[s])
        ts = t_sample[s:s + 1]
        vlat = vid_list[s]
        if (plan.video.present or plan.image.present) and vlat is not None:
            vlat = vlat.to(dev)
            C, T_lat, h, ww = vlat.shape
            if plan.image.present and not plan.video.present:
                noised_vid[s] = vlat[:, :1]
            else:
                cmask_frames = torch.tensor(
                    TP._video_condition_mask(plan.video.clean_policy, T_lat),
                    dtype=torch.bool, device=dev)
                flat = vlat.permute(1, 0, 2, 3).reshape(1, T_lat, -1)
                nfx, _, tgt, _ = flow.add_noise_velocity_masked(flat, cmask_frames.view(1, T_lat), t=ts)
                noised_vid[s] = nfx.view(T_lat, C, h, ww).permute(1, 0, 2, 3).contiguous()
                if plan.video.supervised:
                    target_vid[s] = tgt.view(T_lat, C, h, ww)

    noised_cam_dense = camera_target = None
    if cam_dense is not None and any(TP.build_task_plan(m).camera.present for m in modes):
        Tc = cam_dense.shape[1]
        cond_cam = torch.ones((B, Tc), dtype=torch.bool, device=dev)
        for s in range(B):
            plan = TP.build_task_plan(modes[s])
            if not plan.camera.present:
                continue
            cond_cam[s] = (torch.ones(Tc, dtype=torch.bool, device=dev)
                           if plan.camera.clean_policy == "all" else cam_pad[s])
        noised_cam_dense, _, camera_target, _ = flow.add_noise_velocity_masked(
            cam_dense, cond_cam, t=t_sample)
    noised_cam_list = [None] * B
    if noised_cam_dense is not None:
        for s in range(B):
            if TP.build_task_plan(modes[s]).camera.present:
                keep = ~cam_pad[s]
                noised_cam_list[s] = noised_cam_dense[s][keep]

    out = model.forward(
        input_ids_list, x_t=x_t, t_or_sigma=t_sample,
        neutral_joints=nj if motion is not None else None,
        motion_pad_mask=pad if motion is not None else None,
        noisy_frame_mask=noisy_frame_mask,
        modes=modes, video_latents=noised_vid, camera_action=noised_cam_list,
        return_dict=True,
    )

    total = torch.zeros((), device=dev)
    sc = {}

    mp = out.get("motion_pred")
    if mp is not None and target_motion is not None and noisy_frame_mask.any():
        m = noisy_frame_mask
        while m.dim() < mp.dim():
            m = m.unsqueeze(-1)
        l_feat = (((mp - target_motion) ** 2) * m).sum() / m.expand_as(mp).sum().clamp(min=1)
        tb_ = t_motion.view(-1, *([1] * (x_t.dim() - 1)))
        x0_hat = x_t - tb_ * mp
        j_hat = decode_joints(x0_hat * std + mean)
        total = total + w["feat"] * l_feat
        sc["motion_feat"] = float(l_feat)

    vp = out.get("video_pred") or [None] * B
    for s in range(B):
        if vp[s] is None or target_vid[s] is None:
            continue
        plan = TP.build_task_plan(modes[s])
        C, T_lat, h, ww = vid_list[s].shape
        cmask = torch.tensor(TP._video_condition_mask(plan.video.clean_policy, T_lat),
                             dtype=torch.bool, device=dev)
        nfrm = ~cmask
        vp5 = vp[s][0].permute(1, 0, 2, 3)
        lv = flow.vision_flow_loss(vp5.reshape(1, T_lat, -1), target_vid[s].reshape(1, T_lat, -1),
                                   nfrm.view(1, T_lat), weight=w["vision"])
        total = total + lv
        sc["vision"] = float(lv)

    cp = out.get("camera_pred") or [None] * B
    for s in range(B):
        if cp[s] is None or camera_target is None:
            continue
        plan = TP.build_task_plan(modes[s])
        if not plan.camera.supervised:
            continue
        keep = ~cam_pad[s]
        tgt = camera_target[s][keep]
        n = cp[s].shape[0]
        nm = torch.ones((1, n), dtype=torch.bool, device=dev)
        lc = flow.camera_flow_loss(cp[s].unsqueeze(0), tgt[:n].unsqueeze(0), nm, weight=w["camera"])
        total = total + lc
        sc["camera"] = float(lc)

    return total, sc


# ----------------------------------------------------------------------------
# FIX-1 verification (native camera<->vision mRoPE parallelism): the CAMERA part of a
# built gen segment must start at the VIDEO part's START temporal offset (+1 for
# start_frame_offset=1, so action[0] aligns with vision frame 1), and the segment's
# next_temporal_offset must clear the max of both parts. Smoke-only -- not hot code.
# ----------------------------------------------------------------------------
def check_camera_offset(model, dev, temporal_offset=7, t_lat=9, n_cam=32):
    resolved = TP.resolve_sample("policy", t_lat=t_lat, n_camera=n_cam)
    C = model.gen.latent_channel                     # net latent channels (48 for Wan2.2)
    vlat = torch.randn(C, t_lat, 32, 32, device=dev)
    cam = torch.randn(n_cam, TP.CAMERA_RAW_DIM, device=dev)
    seg = model.gen.build_gen_segment(
        resolved, video_latents=vlat, camera_action=cam,
        sigma=torch.full((1,), 0.5, device=dev), temporal_offset=temporal_offset)
    vs, ve = seg.offsets["video"]
    cs, ce = seg.offsets["camera"]
    vid_t = seg.mrope_ids[0, vs:ve]
    cam_t = seg.mrope_ids[0, cs:ce]
    video_start = int(vid_t.min())
    assert video_start == temporal_offset, (video_start, temporal_offset)
    # camera runs PARALLEL to vision: its offset == the VIDEO part's START offset, so with
    # start_frame_offset=1 the first camera id is video_start + 1 (aligned to vision frame 1).
    assert int(cam_t.min()) == video_start + 1, \
        f"camera temporal start {int(cam_t.min())} != video start {video_start} + 1"
    assert int(cam_t.max()) == video_start + n_cam, (int(cam_t.max()), video_start + n_cam)
    expect_next = max(int(vid_t.max()), int(cam_t.max())) + 1
    assert seg.next_temporal_offset == expect_next, (seg.next_temporal_offset, expect_next)
    print(f"[smoke] camera-offset check OK: video t=[{video_start},{int(vid_t.max())}]  "
          f"camera t=[{int(cam_t.min())},{int(cam_t.max())}] (parallel, start=video_start+1)  "
          f"next_temporal_offset={seg.next_temporal_offset}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_lora", action="store_true", default=True,
                    help="inject generator LoRA (the first-run default; ON here).")
    ap.add_argument("--no_gen_lora", dest="gen_lora", action="store_false")
    ap.add_argument("--gen_full", action="store_true")
    ap.add_argument("--reasoner_lora", action="store_true")
    ap.add_argument("--freeze_motion", action="store_true",
                    help="PHASE-1: freeze the motion pathway; train ONLY the gen-LoRA. With a "
                         "--latents_dir this runs ONE camera task (inverse_dynamics by default via "
                         "--video_mode) and asserts finite loss + grad ONLY on gen-LoRA (zero on "
                         "motion + reasoner + gen-base).")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--T", type=int, default=config.VIDEO_NUM_FRAMES)
    ap.add_argument("--latents_dir", default=None,
                    help="precompute_latents output root; if set, also run one --video_mode step")
    ap.add_argument("--video_mode", nargs="*", default=["video2motion"],
                    help="gen task(s) to smoke against a real latent (default video2motion); "
                         "e.g. --video_mode policy inverse_dynamics forward_dynamics")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config.validate_train_scope({"gen_lora": args.gen_lora, "gen_full": args.gen_full})
    dev = args.device
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).float().to(dev)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).float().to(dev)
    w = dict(feat=1.0, vision=1.0, camera=config.ACTION_LOSS_WEIGHT)

    print("[smoke] building FrozenCosmos + JointMotionModel "
          f"(gen_lora={args.gen_lora} gen_full={args.gen_full} reasoner_lora={args.reasoner_lora})",
          flush=True)
    cosmos = FrozenCosmos(device=dev)
    model = JointMotionModel(
        cosmos, objective="velocity", motion_intermediate_size=config.MOTION_INTERMEDIATE_SIZE,
        motion_layer_stride=config.MOTION_LAYER_STRIDE,
        gen_lora=args.gen_lora, gen_full=args.gen_full, reasoner_lora=args.reasoner_lora,
        freeze_motion=args.freeze_motion,
    ).to(dev)
    model.freeze()
    model.train()
    if args.freeze_motion:
        model.assert_motion_frozen()
        print("[smoke] freeze_motion: motion pathway requires_grad=False (Phase-1 partition)",
              flush=True)

    trainable = model.trainable_parameters()
    n_train = sum(p.numel() for p in trainable)
    # sanity: gen_lora MUST surface trainable params that live under cosmos.net.
    net_trainable = [n for n, p in model.cosmos.net.named_parameters() if p.requires_grad]
    print(f"[smoke] trainable = {n_train/1e6:.2f}M ({len(trainable)} tensors); "
          f"cosmos.net trainable tensors = {len(net_trainable)}", flush=True)
    if args.gen_lora or args.gen_full or args.reasoner_lora:
        assert net_trainable, ("expected trainable params under cosmos.net for the active gen/"
                               "reasoner toggle, found NONE -- the LoRA/gen_full params are not "
                               "reaching the optimizer")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

    ok = True

    # FIX-1 verification: camera part of a built gen segment runs temporally PARALLEL to the
    # video part (native Cosmos parity). Cheap (one build_gen_segment on synthetic tensors).
    try:
        check_camera_offset(model, dev)
    except AssertionError as e:
        print(f"[smoke] camera-offset check FAILED: {e}", flush=True)
        ok = False

    # Under PHASE-1 (--freeze_motion) the motion pathway is frozen, so a text2motion step would
    # produce ZERO trainable grad (only motion carries a motion loss). Skip it and go straight to
    # a CAMERA-only task (inverse_dynamics by default) against real latents -- that is exactly the
    # Phase-1 regime (no motion tokens, gen-LoRA only).
    if args.freeze_motion and args.video_mode == ["video2motion"]:
        args.video_mode = ["inverse_dynamics"]

    # ---- 1) TEXT2MOTION (no latents) -- SKIPPED under --freeze_motion (no motion grad) ---------
    for step in range(0 if args.freeze_motion else args.steps):
        batch = _synthetic_motion_batch(args.batch_size, args.T, "text2motion", dev, rng)
        loss, sc = step_loss(model, cosmos, batch, std, mean, dev, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        train_grad = sum(p.grad.abs().sum().item()
                         for n, p in model.named_all_parameters()
                         if model._is_trainable_name(n) and p.grad is not None)
        try:
            model.assert_frozen_grads_zero()
            frozen_ok = True
        except AssertionError as e:
            print(f"[smoke] text2motion FROZEN-GRAD LEAK: {str(e)[:300]}", flush=True)
            frozen_ok = False
        fin = bool(torch.isfinite(loss).item())
        print(f"[smoke] text2motion step {step} loss={loss.item():.4f} finite={fin} "
              f"train_grad={train_grad:.3f} frozen_ok={frozen_ok} "
              f"{ {k: round(v, 3) for k, v in sc.items()} }", flush=True)
        ok = ok and fin and (train_grad > 0) and frozen_ok
        opt.step()

    # ---- 2) optional VIDEO / CAMERA task against a real precomputed latent ---------------
    if args.freeze_motion and not args.latents_dir:
        print("[smoke] --freeze_motion needs --latents_dir to exercise a camera task "
              "(no motion grad without it); PASS is vacuous otherwise", flush=True)
        ok = False
    if args.latents_dir:
        cands = sorted(glob.glob(os.path.join(args.latents_dir, "**", "*.npz"), recursive=True))
        if not cands:
            print(f"[smoke] WARNING: no .npz under {args.latents_dir}; skipping video task", flush=True)
        else:
            npz = cands[0]
            for vmode in args.video_mode:
                print(f"[smoke] video task '{vmode}' from {npz}", flush=True)
                batch = _video_batch_from_latent(npz, args.T, vmode, dev, rng)
                try:
                    loss, sc = step_loss(model, cosmos, batch, std, mean, dev, w)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    train_grad = sum(p.grad.abs().sum().item()
                                     for n, p in model.named_all_parameters()
                                     if model._is_trainable_name(n) and p.grad is not None)
                    model.assert_frozen_grads_zero()
                    # PHASE-1: grad must be ZERO on motion + reasoner + gen-base; nonzero ONLY on gen-LoRA.
                    motion_grad = gen_base_grad = 0.0
                    if args.freeze_motion:
                        for n, p in model.named_all_parameters():
                            if p.grad is None:
                                continue
                            g = p.grad.abs().sum().item()
                            if model._is_motion_name(n):
                                motion_grad += g
                            elif "_moe_gen" in n and "lora_" not in n.lower():
                                gen_base_grad += g
                        assert motion_grad == 0.0, f"freeze_motion LEAK: motion grad={motion_grad}"
                        assert gen_base_grad == 0.0, f"gen-base LEAK: gen_base grad={gen_base_grad}"
                    fin = bool(torch.isfinite(loss).item())
                    extra = (f" motion_grad={motion_grad:.1f} gen_base_grad={gen_base_grad:.1f}"
                             if args.freeze_motion else "")
                    print(f"[smoke] {vmode} loss={loss.item():.4f} finite={fin} "
                          f"train_grad={train_grad:.3f} frozen_ok=True{extra} "
                          f"{ {k: round(v, 3) for k, v in sc.items()} }", flush=True)
                    ok = ok and fin and (train_grad > 0)
                    opt.step()
                except Exception as e:  # noqa: BLE001
                    print(f"[smoke] {vmode} FAILED: {type(e).__name__}: {e}", flush=True)
                    ok = False

    print("[smoke] PASS" if ok else "[smoke] FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
