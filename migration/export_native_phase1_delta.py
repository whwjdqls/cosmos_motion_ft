#!/usr/bin/env python
"""Export the Phase-1 EMA LoRA/action interface from a native Cosmos DCP.

The native checkpoint is about 90 GB because it contains the full network and
EMA copy. Phase 3 reads only generator LoRA tensors and the three camera/action
interface modules. This exporter writes exactly that loadable subset so a
portable Phase-3 restore does not require the full Phase-1 DCP.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_expert_joint_attention.checkpoint_utils import (
    load_joint_pt,
    load_native_gen_dcp,
)


FORMAT = "native_phase1_gen_delta_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", choices=("ema", "regular"), default="ema")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    model_dir = checkpoint / "model" if (checkpoint / "model").is_dir() else checkpoint
    metadata_path = model_dir / ".metadata"
    if not metadata_path.is_file():
        parser.error(f"missing native DCP metadata: {metadata_path}")

    state = load_native_gen_dcp(checkpoint, weights=args.weights)
    if not state:
        raise RuntimeError("native DCP selection returned no tensors")
    if not all(tensor.device.type == "cpu" for tensor in state.values()):
        raise RuntimeError("portable state must contain CPU tensors")

    payload = {
        "format": FORMAT,
        "native_weights": args.weights,
        "source_checkpoint": str(checkpoint),
        "source_model_metadata_sha256": sha256_file(metadata_path),
        "model": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent,
        prefix=args.output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        restored = load_joint_pt(temporary)
        if restored.keys() != state.keys():
            raise RuntimeError("portable checkpoint key round trip failed")
        for name, expected in state.items():
            if not torch.equal(restored[name], expected):
                raise RuntimeError(f"portable checkpoint tensor round trip failed: {name}")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    numel = sum(tensor.numel() for tensor in state.values())
    print(
        f"[phase1-delta] wrote {args.output} tensors={len(state)} "
        f"numel={numel} bytes={args.output.stat().st_size} "
        f"sha256={sha256_file(args.output)}"
    )


if __name__ == "__main__":
    main()
