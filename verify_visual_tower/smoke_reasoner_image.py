#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Smoke-test the isolated reasoner-side visual tower path.

Run on a GPU node in the ``cosmos`` env:

    python verify_visual_tower/smoke_reasoner_image.py

The script intentionally does not train. It builds FrozenCosmos, lazily attaches
the standalone ``vision_encoder`` weights, encodes one synthetic RGB image plus
text prompt, and prints tensor shapes needed by ``JointMotionModel.forward``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/jungbin_cho/cosmos_motion_ft")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_expert_joint_attention.cosmos_loader import FrozenCosmos  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--text", default="a person walks through a room")
    args = ap.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    cosmos = FrozenCosmos(dtype=dtype, device=args.device, verbose=True)

    # Synthetic deterministic RGB image. Qwen's processor only needs real pixel
    # structure for this smoke; semantic content is irrelevant.
    y = torch.linspace(0, 255, args.height, dtype=torch.float32).view(1, args.height, 1)
    x = torch.linspace(0, 255, args.width, dtype=torch.float32).view(1, 1, args.width)
    image = torch.cat([
        x.expand(1, args.height, args.width),
        y.expand(1, args.height, args.width),
        ((x + y) * 0.5).expand(1, args.height, args.width),
    ], dim=0).to(torch.uint8)

    out = cosmos.encode_reasoner_image_text(args.text, image)
    report = {
        "visual_tower_loaded": cosmos.visual_tower_loaded,
        "has_language_model_visual": hasattr(cosmos.net.language_model, "visual"),
        "input_ids_shape": list(out["input_ids"].shape),
        "inputs_embeds_shape": list(out["inputs_embeds"].shape),
        "position_ids_shape": list(out["position_ids"].shape),
        "visual_pos_count": int(out["visual_pos_mask"].sum().item()),
        "deepstack_shapes": [list(x.shape) for x in out["deepstack_visual_embeds"]],
        "dtype": str(out["inputs_embeds"].dtype),
        "device": str(out["inputs_embeds"].device),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
