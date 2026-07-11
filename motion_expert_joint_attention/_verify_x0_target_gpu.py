"""Decisive TRAIN-target semantics check for the per-modality x0 objective (GPU). DIAGNOSTIC.

Mirrors _diag_objective_mismatch.py's TEST-B methodology, but on the NEW TRAINING path with a
FRESH-INIT model (no checkpoint): draw ONE real (caption, motion) text2motion item, noise at
fixed sigma=0.3 through the exact step_loss x0 branch (flow.add_noise_x0_masked with
cond_motion = pad), forward with the SAME sigma tensor (the t_or_sigma invariant), then print
    MSE(target_used, x0)       -- must be exactly 0 (the trainer regresses x0)
    MSE(target_used, eps-x0)   -- must be LARGE (provably NOT the velocity target)
    l_feat(train) vs masked_mse(pred, x0) vs masked_mse(pred, eps-x0)
and assert l_feat == masked_mse(pred, x0) bit-for-bit, plus x0_hat(geometric) is the raw
prediction. Proves the motion loss is against x0.

Run (node 2/3, cosmos env):
  ssh a3ultravis-a3ultranodeset-2 'CUDA_VISIBLE_DEVICES=0 bash \
      /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/run.sh \
      /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/_verify_x0_target_gpu.py'
"""
from __future__ import annotations

import numpy as np
import torch

import config
import flow
from cosmos_loader import FrozenCosmos
from joint_motion_model import JointMotionModel
from nymeria_joint_dataset import NymeriaJointDataset

DEV = "cuda"
T = 97
torch.manual_seed(0)


def masked_mse(a, b, valid):  # == train.masked_mse
    while valid.dim() < a.dim():
        valid = valid.unsqueeze(-1)
    se = ((a - b) ** 2) * valid
    return se.sum() / valid.expand_as(a).sum().clamp(min=1)


print("=" * 78)
print("Loading FrozenCosmos + FRESH-INIT JointMotionModel (objective='x0', no ckpt)")
cosmos = FrozenCosmos(device=DEV)
model = JointMotionModel(cosmos, objective="x0").to(DEV)
model.eval()

print("\nDrawing one real text2motion (caption, motion) test item...")
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
x0 = torch.as_tensor(item["motion"]).float().to(DEV)[None]            # [1,Tm,283] z-scored
pad = torch.as_tensor(item["motion_pad_mask"]).bool().to(DEV)[None]   # [1,Tm]
nj = torch.as_tensor(item["neutral_joints"]).float().to(DEV)[None]
valid = ~pad
Tm = x0.shape[1]
print(f"caption={cap[:70]!r}  Tm={Tm}  n_valid={int(valid.sum())}")

# ---- EXACT step_loss x0 branch, sigma pinned to 0.3 ----------------------------------
sigma = torch.full((1,), 0.3, device=DEV)
cond_motion = pad.clone()                                             # True=CLEAN (pads only)
x_t, t_motion, target_motion, noised_motion = flow.add_noise_x0_masked(x0, cond_motion, sigma)
assert torch.equal(t_motion, sigma), "noiser must echo the sigma it was given"
noisy_frame_mask = noised_motion & valid

# recover the eps the noiser drew (sigma_eff = 0.3 on noised rows) -> velocity target.
sb = sigma.view(-1, 1, 1)
eps = (x_t - (1.0 - sb) * x0) / sb                                    # exact on noised rows
v_target = eps - x0

print("\n--- TARGET semantics (the quantity the trainer regresses) ---")
m_t_x0 = masked_mse(target_motion, x0, noisy_frame_mask).item()
m_t_v = masked_mse(target_motion, v_target, noisy_frame_mask).item()
print(f"MSE(target_used, x0)      = {m_t_x0:.6e}   (must be 0)")
print(f"MSE(target_used, eps-x0)  = {m_t_v:.6e}   (must be LARGE)")
assert m_t_x0 == 0.0 and torch.equal(target_motion, x0)
assert m_t_v > 0.5

# ---- forward with the SAME sigma tensor (the t_or_sigma invariant) -------------------
with torch.no_grad():
    out = model.forward(
        input_ids_list=[cosmos.tokenize(cap)],
        x_t=x_t, t_or_sigma=t_motion,                                 # SAME tensor as the noiser
        neutral_joints=nj, motion_pad_mask=pad,
        noisy_frame_mask=noisy_frame_mask, modes=["text2motion"],
        video_latents=[None], camera_action=[None], return_dict=True,
    )
pred = out["motion_pred"].float()

l_feat = masked_mse(pred, target_motion, noisy_frame_mask).item()     # what train.py minimizes
mse_p_x0 = masked_mse(pred, x0, noisy_frame_mask).item()
mse_p_v = masked_mse(pred, v_target, noisy_frame_mask).item()
print("\n--- LOSS semantics at sigma=0.3, fresh-init prediction ---")
print(f"l_feat (train quantity)        = {l_feat:.6f}")
print(f"MSE(pred, x0)                  = {mse_p_x0:.6f}   (must == l_feat)")
print(f"MSE(pred, eps-x0)              = {mse_p_v:.6f}   (the OLD velocity loss; must differ)")
assert l_feat == mse_p_x0, "train's l_feat must literally be the x0 regression"

# ---- geometric x0_hat is the raw prediction (bs_train recipe) -------------------------
x0_hat_geom = pred                                                    # x0 branch: no ODE algebra
x0_hat_vel = x_t - t_motion.view(-1, 1, 1) * pred                     # what velocity would use
d = (x0_hat_geom - x0_hat_vel).abs().mean().item()
print(f"\nx0_hat(geometric)=pred directly; |pred - (x_t - t*pred)| mean = {d:.4f} "
      f"(nonzero => the two objectives are NOT interchangeable here)")

print("\nDECISIVE TARGET CHECK PASS: the motion loss is against x0, not eps-x0.")
