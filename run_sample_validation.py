# SPDX-License-Identifier: OpenMDW-1.1
"""GPU validation driver for sample_motion.py: load the finetuned model ONCE, then
sample + decode + render several prompts. Keeps the 14GB model resident across
prompts (one srun allocation). Run from cosmos-framework, cosmos env.

    python run_sample_validation.py --ckpt <ckpt.pt> --frames 120 --steps 50 --cfg 2.5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_motion as sm
import train_motion_ft as T

OUT_DIR = os.environ.get("EVAL_OUT", "/home/jungbin_cho/cosmos_motion_ft/samples_step10k")
PROMPTS = [
    ("a person walks forward", "gen_walk"),
    ("a person waves their right hand", "gen_wave"),
    ("a person sits down on a chair", "gen_sit"),
    # BONES-SEED natural-caption style (closer to the training distribution)
    ("character picks up an object from the floor and then stands up straight", "gen_pickup"),
    ("a person turns around and walks back the other way", "gen_turn"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--shift", type=float, default=1.0,
                    help="rectified-flow schedule shift (1.0=original; Cosmos action ~5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dtype = torch.bfloat16

    t0 = time.time()
    net = sm.load_model(args.ckpt, dtype=dtype, verbose=True)
    proc = T.build_text_processor()
    print(f"[setup] model+proc ready in {time.time()-t0:.1f}s; proc={type(proc).__name__}",
          flush=True)

    def tokenize(texts):
        out = []
        for t in texts:
            tid = proc.tokenize_text(t)
            if len(tid) == 0:
                tid = [sm.SPECIAL_TOKENS["eos_token_id"]]
            out.append(tid)
        return out

    for prompt, name in PROMPTS:
        ts = time.time()
        x0 = sm.sample(net, tokenize, prompt, args.frames, args.steps, args.cfg,
                       dtype=dtype, seed=args.seed, shift=args.shift)
        x0_np = x0.cpu().numpy().astype(np.float32)
        prefix = os.path.join(OUT_DIR, name)
        np.save(prefix + ".npy", x0_np)
        finite = bool(np.isfinite(x0_np).all())
        print(f"\n=== {name}: {prompt!r} ===", flush=True)
        print(f"[sample] x0 {x0_np.shape} mean={x0_np.mean():.4f} std={x0_np.std():.4f} "
              f"min={x0_np.min():.3f} max={x0_np.max():.3f} finite={finite} "
              f"({time.time()-ts:.1f}s)", flush=True)

        joints, path, kind = sm.decode_and_render(
            x0_np, prefix,
            "/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz",
            "/weka/jungbin/seed/stats/soma_uniform_motions_20fps/",
            fps=20, title=f"GEN: {prompt}", is_normalized=True,
        )
        sz = os.path.getsize(path) if os.path.isfile(path) else -1
        print(f"[render] joints {joints.shape} jmin={joints.min():.3f} jmax={joints.max():.3f} "
              f"-> {kind} {path} ({sz} bytes)", flush=True)

    print(f"\n[done] all prompts in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
