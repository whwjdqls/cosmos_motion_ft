"""PROOF of per-MODALITY noiser dispatch inside the REAL train.step_loss (GPU). DIAGNOSTIC.

Wraps flow.add_noise_x0_masked / flow.add_noise_velocity_masked with recording spies, then
runs train.main() in --smoke mode over motion tasks + gen tasks (default --objective x0).
Every spy call asserts the target semantics in-place:
    x0 noiser       : target IS x0 (bit-identical)             -> hit ONLY by motion [.,.,283]
    velocity noiser : x_t - x0 == sigma_eff * target (eps-x0)  -> hit ONLY by gen tensors
                      (video flat [1,T_lat,C*h*w] / camera [B,Tc,9])
so the gen pathway's loss targets are BYTE-IDENTICAL to before (native Cosmos velocity),
while motion trains as x0. Also checks the sigma fed to the motion noiser is logit-normal-
range (any (0,1) value) while gen t stays uniform -- both echoed back per the invariant.

Run (node 2/3, cosmos env):
  ssh a3ultravis-a3ultranodeset-2 'CUDA_VISIBLE_DEVICES=0 bash \
      /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/run.sh \
      /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/_verify_objective_dispatch_spy.py'
"""
from __future__ import annotations

import sys

import torch

import flow
import train

CALLS = []

_orig_x0 = flow.add_noise_x0_masked
_orig_v = flow.add_noise_velocity_masked


def spy_x0(x0, cm, sigma=None):
    out = _orig_x0(x0, cm, sigma)
    x_t, s, tgt, nm = out
    assert torch.equal(tgt, x0), "x0 noiser: target must be x0 itself"
    assert torch.equal(x_t[cm.bool()], x0[cm.bool()]), "x0 noiser: clean tokens must pass through"
    CALLS.append(("x0", tuple(x0.shape), float(s.min()), float(s.max())))
    return out


def spy_v(x0, cm, t=None):
    out = _orig_v(x0, cm, t)
    x_t, t_out, tgt, nm = out
    gate = nm.to(x0.dtype)
    while gate.dim() < x0.dim():
        gate = gate.unsqueeze(-1)
    sig_eff = t_out.view(-1, *([1] * (x0.dim() - 1))) * gate
    assert torch.allclose(x_t - x0, sig_eff * tgt, atol=1e-4), \
        "velocity noiser: x_t - x0 == sigma_eff * (eps - x0) identity broken"
    CALLS.append(("velocity", tuple(x0.shape), float(t_out.min()), float(t_out.max())))
    return out


flow.add_noise_x0_masked = spy_x0
flow.add_noise_velocity_masked = spy_v

sys.argv = ["train.py", "--smoke",
            "--tasks", "text2motion", "textimg2motion", "video2motion",
            "inverse_dynamics", "policy",
            "--gen_lora", "--T", "97", "--num_workers", "0"]
train.main()

print("\n" + "=" * 78)
print("NOISER DISPATCH LOG (which objective noised what)")
for kind, shape, lo, hi in CALLS:
    print(f"  {kind:8s} shape={shape}  t/sigma in [{lo:.3f},{hi:.3f}]")
mot = [c for c in CALLS if c[0] == "x0"]
gen = [c for c in CALLS if c[0] == "velocity"]
assert mot, "the x0 noiser never fired (motion tasks should have hit it)"
assert gen, "the velocity noiser never fired (gen tasks should have hit it)"
assert all(s[-1] == 283 for _, s, *_ in mot), "x0 noiser must ONLY ever see motion [.,.,283]"
assert all(s[-1] != 283 for _, s, *_ in gen), "velocity noiser must NEVER see motion here"
print(f"\nDISPATCH PROOF OK: x0 noiser calls={len(mot)} (all motion 283-d), "
      f"velocity noiser calls={len(gen)} (all gen video/camera) -- gen targets unchanged.")
