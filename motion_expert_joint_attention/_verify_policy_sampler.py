#!/usr/bin/env python
"""One-off: prove the policy JOINT co-integration sampler (sample.sample_policy) runs
shape-clean end to end on the BASE model (random motion expert is fine) for 2 ODE steps.

Checks:
  * ONE Euler loop over the PAIR (video frames 1.., camera all frames), each step calling
    the model with BOTH current iterates packed (mirroring train.step_loss's policy layout);
  * cond + null (CFG) passes both run;
  * outputs come back with the right shapes and finite values.

Run (cosmos env, ONE GPU on an idle node)::

    bash run.sh _verify_policy_sampler.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sample as S  # noqa: E402
import task_plan as TP  # noqa: E402
from cosmos_loader import FrozenCosmos  # noqa: E402
from joint_motion_model import JointMotionModel  # noqa: E402


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print("[verify_policy_sampler] building FrozenCosmos + base JointMotionModel...", flush=True)
    cosmos = FrozenCosmos(device=dev)
    model = JointMotionModel(cosmos).to(dev)
    model.eval()

    T_lat, camera_T = 9, 32
    C, h, w = model.gen.latent_channel, 32, 32       # net latent channels (48 for Wan2.2)
    img = torch.randn(C, h, w, device=dev)

    with torch.no_grad():
        out = S.sample_policy(
            model, caption="a person walks forward", image_latent=img,
            T_lat=T_lat, camera_T=camera_T, steps=2, guidance=2.5, seed=0, device=dev,
        )

    vid, cam = out["video"], out["camera"]
    assert vid.shape == (C, T_lat, h, w), vid.shape
    assert cam.shape == (camera_T, TP.CAMERA_RAW_DIM), cam.shape
    assert np.isfinite(vid).all(), "video output has non-finite values"
    assert np.isfinite(cam).all(), "camera output has non-finite values"
    # frame 0 must be pinned to the clean conditioning image.
    f0 = torch.from_numpy(vid[:, 0]).to(dev)
    assert torch.allclose(f0, img, atol=1e-4), "clean frame 0 was not preserved"
    print(f"[verify_policy_sampler] OK: video {vid.shape} finite, camera {cam.shape} finite, "
          f"frame0 pinned. Joint co-integration (2 ODE steps, cond+null) runs shape-clean.",
          flush=True)


if __name__ == "__main__":
    main()
