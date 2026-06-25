"""Phase 3: sample motion from text with a trained MotionExpert (cosmos env).

text → frozen reasoner H_R → MotionExpert (rectified-flow Euler sampling, CFG) → 283-D
motion → unnormalize → save .npy. Supports the POC ablation: condition on the real text
H_R ("cond") vs the empty/null H_R ("null") to test whether reasoner semantics matter.

Run (cosmos env, 1 GPU):
  ssh a3ultravis-a3ultranodeset-1 'CUDA_VISIBLE_DEVICES=0 bash motion_expert/run.sh sample.py \
     --ckpt <run>/ckpt_step050000.pt --out <run>/samples_step50k'
Then decode+render with viz.py in the kimodo env.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

import flow
from motion_expert import MotionExpert
from reasoner import D_REASONER, FrozenReasoner
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


def load_default_skeleton() -> np.ndarray:
    """Centered neutral_joints (30,3) from one train actor (fixed size for sampling)."""
    f = sorted(glob.glob(os.path.join(UNIEGO_ROOT, "S01", "*.npz")))[0]
    nj = np.load(f)["neutral_joints"].astype(np.float32)
    return nj - nj.mean(axis=0, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean", default=os.path.join(HERE, "stats", "uniego283_mean.npy"))
    ap.add_argument("--std", default=os.path.join(HERE, "stats", "uniego283_std.npy"))
    ap.add_argument("--ablation", choices=["cond", "null", "both"], default="both",
                    help="cond=text H_R; null=empty H_R (the hypothesis test); both=write each")
    ap.add_argument("--skeleton_npz", default=None, help="uniego npz to take neutral_joints from")
    args = ap.parse_args()

    dev = "cuda"
    mean = torch.from_numpy(np.load(args.mean)).float().to(dev)
    std = torch.from_numpy(np.load(args.std)).float().to(dev)

    reasoner = FrozenReasoner(dtype=torch.bfloat16, device=dev)

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck.get("args", {})
    model = MotionExpert(d=a.get("d", 512), n_layers=a.get("layers", 8), heads=a.get("heads", 8),
                         ffn=a.get("ffn", 2048), kv_dim=D_REASONER, motion_dim=FEAT_DIM).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[sample] loaded {args.ckpt} (step {ck.get('step')})")

    if args.skeleton_npz:
        nj = np.load(args.skeleton_npz)["neutral_joints"].astype(np.float32)
        nj = nj - nj.mean(axis=0, keepdims=True)
    else:
        nj = load_default_skeleton()
    nj_t = torch.from_numpy(nj).float().to(dev)

    os.makedirs(args.out, exist_ok=True)
    modes = ["cond", "null"] if args.ablation == "both" else [args.ablation]
    null_H = reasoner.null_H().unsqueeze(0).to(torch.float32)        # [1,T0,4096]
    null_pad = torch.zeros(1, null_H.shape[1], dtype=torch.bool, device=dev)

    manifest = []
    for prompt in args.prompts:
        H_cond, h_pad = reasoner.encode_text([prompt])              # [1,Tt,4096],[1,Tt]
        H_cond = H_cond.to(torch.float32)
        for mode in modes:
            H = H_cond if mode == "cond" else null_H
            hp = h_pad if mode == "cond" else null_pad
            g = torch.Generator(device=dev).manual_seed(args.seed)
            x0 = flow.sample_x0(
                model, H, hp, nj_t.unsqueeze(0), T=args.T, motion_dim=FEAT_DIM,
                steps=args.steps, guidance=(args.guidance if mode == "cond" else 1.0),
                H_null=null_H, null_pad_mask=null_pad, device=dev, dtype=torch.float32, generator=g,
            )                                                       # [1,T,283] normalized (x0-pred)
            x0 = (x0[0] * std + mean).cpu().numpy().astype(np.float32)  # unnormalize → [T,283]
            slug = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
            name = f"{slug}__{mode}"
            np.save(os.path.join(args.out, name + ".npy"), x0)
            manifest.append({"prompt": prompt, "mode": mode, "file": name + ".npy",
                             "T": args.T, "guidance": args.guidance if mode == "cond" else 1.0})
            print(f"  [{mode}] '{prompt[:40]}' -> {name}.npy  (range {x0.min():.2f}..{x0.max():.2f})")

    # save the skeleton + manifest for viz
    np.save(os.path.join(args.out, "skeleton_neutral_joints.npy"), nj)
    json.dump({"n_joints": N_JOINTS, "feat_dim": FEAT_DIM, "samples": manifest,
               "ckpt": args.ckpt, "step": ck.get("step")},
              open(os.path.join(args.out, "samples_manifest.json"), "w"), indent=2)
    print(f"[sample] wrote {len(manifest)} samples + manifest to {args.out}")


if __name__ == "__main__":
    main()
