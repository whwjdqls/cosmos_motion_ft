"""DIAGNOSTIC ONLY (Phase-1 evidence, no fixes). Root-cause probe for the joint model's
noise-only text2motion samples at 10k despite dropping motion_feat.

Runs three tests against the SAME 10k checkpoint loaded exactly like sample.py:
  A) text-conditioning check (two captions, fixed x + sigma)
  B) denoiser/recon sanity: interpret the model output BOTH as x0-hat AND as velocity,
     across sigma in {0.1,0.3,0.5,0.7,0.9}, reporting MSE vs x0 and baselines.
  C) train-forward vs sample-forward path diff (runtime ||.|| on the same (cap,x,sigma)).

No edits to any non-diagnostic code. Prints raw numbers only.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import torch

import config
import flow
import task_plan as TP
from cosmos_loader import FrozenCosmos
from joint_motion_model import JointMotionModel
from uniego_layout import FEAT_DIM
from sample import load_joint_model, load_default_skeleton

CKPT = "/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_phase2_T200/ckpt_step010000.pt"
T = 200
DEV = "cuda"

torch.manual_seed(0)

print("=" * 78)
print("Loading model exactly as sample.py load_joint_model (objective from ckpt args)")
model, cosmos, ck = load_joint_model(CKPT, device=DEV, objective_cli="x0")
print(f"model.objective = {model.objective!r}   (ckpt args objective = {ck.get('args',{}).get('objective')!r})")
model.eval()

mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).float().to(DEV)
std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).float().to(DEV)

nj = load_default_skeleton()
nj_t = torch.from_numpy(nj).float().to(DEV)[None]  # [1,30,3]

# fixed noise x at motion_dim, T
gx = torch.Generator(device=DEV).manual_seed(1234)
x_fixed = torch.randn(1, T, FEAT_DIM, device=DEV, dtype=torch.float32, generator=gx)

def build_predict(caption):
    motion_pad_mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    noisy_frame_mask = torch.ones(1, T, dtype=torch.bool, device=DEV)
    ids = cosmos.tokenize(caption)
    return model.predict_closure(
        input_ids_list=[ids], neutral_joints=nj_t,
        motion_pad_mask=motion_pad_mask, noisy_frame_mask=noisy_frame_mask,
        modes=["text2motion"], video_latents=[None],
    )

# =====================================================================
print("\n" + "=" * 78)
print("TEST A — text conditioning during sampling (two captions, fixed x, sigma=0.5)")
capA = "a person is walking forward"
capB = "a person sits down and waves"
sigma_b = torch.full((1,), 0.5, device=DEV)
with torch.no_grad():
    pA = build_predict(capA)(x_fixed, sigma_b).float()
    pB = build_predict(capB)(x_fixed, sigma_b).float()
    # also the null/empty caption for reference
    pN = build_predict("")(x_fixed, sigma_b).float()
dAB = (pA - pB).norm().item()
dAN = (pA - pN).norm().item()
print(f"||out_A||            = {pA.norm().item():.4f}")
print(f"||out_B||            = {pB.norm().item():.4f}")
print(f"||out_null||         = {pN.norm().item():.4f}")
print(f"||out_A - out_B||    = {dAB:.4f}")
print(f"||out_A - out_null|| = {dAN:.4f}")
rel = dAB / (0.5 * (pA.norm().item() + pB.norm().item()) + 1e-8)
print(f"rel ||A-B|| / mean||.|| = {rel:.4f}")
print("Interpretation: near-0 => text NOT reaching motion expert; clearly>0 => text flows.")

# =====================================================================
print("\n" + "=" * 78)
print("TEST B — denoiser/recon sanity on a REAL (caption, motion) test pair")
print("Interpret model output BOTH as x0-hat AND as velocity (train supervises velocity).")

# Draw a real text2motion pair from the test set exactly as the dataset yields it.
from nymeria_joint_dataset import NymeriaJointDataset, collate_joint
vds = NymeriaJointDataset(
    split="test", num_frames=T, task_weights={"text2motion": 1.0},
    bones_text2motion_frac=0.5, cfg_dropout=0.0,
    prefer_latents=True, latent_root=config.VIDEO_LATENT_ROOT,
    force_on_the_fly=False, train=False, max_samples=4096, seed=0,
)
item = None
for i in range(len(vds)):
    it = vds[i]
    if it["mode"] == "text2motion" and it.get("motion") is not None and it.get("caption"):
        item = it
        break
assert item is not None, "no text2motion test item found"
cap = item["caption"]
x0 = torch.as_tensor(item["motion"]).float().to(DEV)[None]           # [1,Tm,283] z-scored
pad = torch.as_tensor(item["motion_pad_mask"]).bool().to(DEV)[None]  # [1,Tm]
nj_item = torch.as_tensor(item["neutral_joints"]).float().to(DEV)[None]
valid = ~pad
Tm = x0.shape[1]
print(f"real caption: {cap[:70]!r}")
print(f"Tm={Tm}  n_valid={int(valid.sum())}  ||x0 (valid)|| mean-energy MSE(0,x0) baseline below")

def mse_valid(a, b):
    m = valid.unsqueeze(-1).to(a.dtype)
    se = ((a - b) ** 2) * m
    return (se.sum() / m.expand_as(a).sum().clamp(min=1)).item()

baseline_zero = mse_valid(torch.zeros_like(x0), x0)  # ~mean energy (z-scored ~1)
print(f"MSE(0, x0) [mean energy baseline] = {baseline_zero:.4f}")

motion_pad_mask = pad
noisy_frame_mask = torch.ones(1, Tm, dtype=torch.bool, device=DEV)
predict = model.predict_closure(
    input_ids_list=[cosmos.tokenize(cap)], neutral_joints=nj_item,
    motion_pad_mask=motion_pad_mask, noisy_frame_mask=noisy_frame_mask,
    modes=["text2motion"], video_latents=[None],
)

print(f"\n{'sigma':>6} {'MSE(xsig,x0)':>13} {'MSE(x0hat_asOut,x0)':>20} {'MSE(x0hat_fromVel,x0)':>22}")
gnoise = torch.Generator(device=DEV).manual_seed(7)
for sigma in (0.1, 0.3, 0.5, 0.7, 0.9):
    # x0 convention noising matches how sample.py add_noise / flow.sample_x0 treat x_sigma:
    #   x_sigma = sigma*eps + (1-sigma)*x0   (== velocity forward with t=sigma too)
    eps = torch.randn(x0.shape, device=DEV, generator=gnoise)
    s = torch.full((1,), float(sigma), device=DEV)
    x_sigma = s.view(-1,1,1) * eps + (1.0 - s.view(-1,1,1)) * x0
    with torch.no_grad():
        out = predict(x_sigma, s).float()   # what the model actually emits
    # Interpretation 1: treat output AS x0-hat (what sample_x0 assumes).
    mse_as_x0 = mse_valid(out, x0)
    # Interpretation 2: treat output as VELOCITY v=eps-x0 (what training supervised).
    #   x0_hat = x_sigma - sigma * v_hat   (velocity ODE relation at t=sigma).
    x0_from_vel = x_sigma - s.view(-1,1,1) * out
    mse_from_vel = mse_valid(x0_from_vel, x0)
    mse_xsig = mse_valid(x_sigma, x0)
    print(f"{sigma:6.2f} {mse_xsig:13.4f} {mse_as_x0:20.4f} {mse_from_vel:22.4f}")

print("\nInterpretation:")
print(" - If MSE(x0hat_fromVel,x0) << MSE(x0hat_asOut,x0) and << MSE(xsig,x0): model is a")
print("   VELOCITY denoiser -> sampling with sample_x0 (treating output as x0) is WRONG.")
print(" - If MSE(x0hat_asOut,x0) is the small one: model outputs x0 (objective consistent).")

# Also report what the training loss actually measures (velocity target) at these sigma.
print("\n[train-loss view] MSE(model_out, velocity_target=eps-x0) — the quantity train.py minimizes:")
gnoise2 = torch.Generator(device=DEV).manual_seed(7)
for sigma in (0.1, 0.3, 0.5, 0.7, 0.9):
    eps = torch.randn(x0.shape, device=DEV, generator=gnoise2)
    s = torch.full((1,), float(sigma), device=DEV)
    x_sigma = s.view(-1,1,1) * eps + (1.0 - s.view(-1,1,1)) * x0
    vtgt = eps - x0
    with torch.no_grad():
        out = predict(x_sigma, s).float()
    print(f"  sigma={sigma:.2f}  MSE(out, eps-x0)={mse_valid(out, vtgt):.4f}")

# =====================================================================
print("\n" + "=" * 78)
print("TEST C — train-forward vs sample-forward path diff (runtime ||.||)")
# TRAIN path: exactly as train.step_loss builds it for text2motion.
#   x_t,t,target = flow.add_noise_velocity_masked(x0, cond_motion=~valid? -> see train)
# We reproduce the model.forward call train uses, then compare to the sample predict closure
# on the SAME (x_t, t).
sigma_c = torch.full((1,), 0.5, device=DEV)
gc = torch.Generator(device=DEV).manual_seed(99)
eps = torch.randn(x0.shape, device=DEV, generator=gc)
x_t = sigma_c.view(-1,1,1) * eps + (1.0 - sigma_c.view(-1,1,1)) * x0  # same forward both convs at t=sigma

# train.step_loss: cond_motion True=CLEAN; for text2motion clean only pad frames.
cond_motion = pad.clone()  # True where pad (clean), so valid frames get noised -> matches train
nfm_train = (~cond_motion) & valid  # noised AND valid (train's noisy_frame_mask)
with torch.no_grad():
    out_train = model.forward(
        input_ids_list=[cosmos.tokenize(cap)],
        x_t=x_t, t_or_sigma=sigma_c,
        neutral_joints=nj_item, motion_pad_mask=pad,
        noisy_frame_mask=nfm_train, modes=["text2motion"],
        video_latents=[None], camera_action=[None], return_dict=True,
    )["motion_pred"].float()
# sample path: predict closure uses noisy_frame_mask = ALL ones.
predict_c = model.predict_closure(
    input_ids_list=[cosmos.tokenize(cap)], neutral_joints=nj_item,
    motion_pad_mask=pad, noisy_frame_mask=torch.ones(1, Tm, dtype=torch.bool, device=DEV),
    modes=["text2motion"], video_latents=[None],
)
with torch.no_grad():
    out_sample = predict_c(x_t, sigma_c).float()
diff = mse_valid(out_train, out_sample)
dnorm = (out_train - out_sample)[valid.unsqueeze(-1).expand_as(out_train)].norm().item()
print(f"MSE(train_fwd, sample_fwd) over valid = {diff:.6e}")
print(f"||train_fwd - sample_fwd|| over valid = {dnorm:.6e}")
print("noisy_frame_mask: train uses (~cond & valid); sample uses ALL-ones. For text2motion")
print("all valid frames are noised in BOTH -> the masks should match on valid frames.")
print("(Any nonzero diff localizes a forward-path mismatch; ~0 means paths agree, so the")
print(" bug is the OBJECTIVE mismatch, not the forward.)")

print("\nDONE.")
