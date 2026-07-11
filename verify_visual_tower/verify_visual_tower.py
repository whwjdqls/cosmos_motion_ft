#!/usr/bin/env python3
"""Verify whether local Cosmos3-Nano includes a reasoner visual tower.

This script intentionally separates three questions:

1. Does the local HF snapshot contain image/vision assets?
2. Does the language-model safetensor index contain Qwen reasoner ``visual`` weights?
3. Can the Cosmos/Qwen classes instantiate a reasoner model with ``.visual``?

It avoids loading the full 15B diffusion model by default.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/*"
)
QWEN_JSON = "/home/jungbin_cho/cosmos-framework/cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_snapshot() -> str:
    snaps = sorted(glob.glob(DEFAULT_GLOB))
    if not snaps:
        raise FileNotFoundError(f"no local snapshots matched {DEFAULT_GLOB}")
    # Match train_motion_ft.py behavior: sorted(...)[0].
    return snaps[0]


def summarize_weight_map(path: str) -> dict:
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    data = load_json(path)
    wm = data.get("weight_map", {})
    visual = [k for k in wm if "visual" in k]
    vision = [k for k in wm if "vision" in k]
    prefixes = Counter(k.split(".")[0] for k in wm)
    return {
        "exists": True,
        "path": path,
        "num_keys": len(wm),
        "top_prefixes": prefixes.most_common(20),
        "visual_key_count": len(visual),
        "vision_key_count": len(vision),
        "visual_key_examples": visual[:50],
        "vision_key_examples": vision[:50],
        "shards_for_visual": sorted({wm[k] for k in visual})[:20],
    }


def inspect_snapshot(snapshot: str) -> dict:
    files = {
        "top_index": os.path.join(snapshot, "model.safetensors.index.json"),
        "transformer_index": os.path.join(snapshot, "transformer", "diffusion_pytorch_model.safetensors.index.json"),
        "vision_encoder_weights": os.path.join(snapshot, "vision_encoder", "model.safetensors"),
        "vision_encoder_config": os.path.join(snapshot, "vision_encoder", "config.json"),
        "processor_config": os.path.join(snapshot, "preprocessor_config.json"),
        "chat_template": os.path.join(snapshot, "chat_template.json"),
        "root_config": os.path.join(snapshot, "config.json"),
        "transformer_config": os.path.join(snapshot, "transformer", "config.json"),
    }
    out = {
        "snapshot": snapshot,
        "files": {k: {"path": v, "exists": os.path.exists(v)} for k, v in files.items()},
        "top_index": summarize_weight_map(files["top_index"]),
        "transformer_index": summarize_weight_map(files["transformer_index"]),
    }
    for cfg_key in ("root_config", "transformer_config", "vision_encoder_config", "processor_config"):
        p = files[cfg_key]
        if os.path.exists(p):
            cfg = load_json(p)
            out[cfg_key] = {
                "keys": sorted(cfg.keys())[:80],
                "model_type": cfg.get("model_type"),
                "_class_name": cfg.get("_class_name"),
                "architectures": cfg.get("architectures"),
                "sample": {k: cfg.get(k) for k in sorted(cfg.keys())[:20]},
            }
    return out


def instantiate_probe() -> dict:
    """Instantiate Qwen classes on meta and report whether .visual exists."""
    import torch
    from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLMoTConfig
    from cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLVisionModel,
    )
    from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from cosmos_framework.model.vfm.vlm.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLTextForCausalLM

    out = {}
    with torch.device("meta"):
        mot_cfg = Qwen3VLMoTConfig.from_json_file(json_file=QWEN_JSON)
        mot_cfg.freeze_und = False
        mot_cfg.qk_norm_for_text = True
        mot_cfg.qk_norm_for_diffusion = True
        mot_cfg.tie_word_embeddings = True
        mot_cfg.use_moe = True
        text_lm = Qwen3VLTextForCausalLM(config=mot_cfg)
        out["Qwen3VLTextForCausalLM"] = {
            "has_visual": hasattr(text_lm, "visual"),
            "has_model_visual": hasattr(getattr(text_lm, "model", None), "visual"),
            "class": type(text_lm).__name__,
        }

        # Full Qwen3-VL configs are nested under the MoT wrapper.
        full_cfg = mot_cfg
        try:
            full_lm = Qwen3VLForConditionalGeneration._from_config(full_cfg)
            out["Qwen3VLForConditionalGeneration"] = {
                "ok": True,
                "has_visual": hasattr(full_lm, "visual"),
                "has_model_visual": hasattr(getattr(full_lm, "model", None), "visual"),
                "class": type(full_lm).__name__,
            }
        except Exception as e:  # noqa: BLE001
            out["Qwen3VLForConditionalGeneration"] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
        try:
            moe_lm = Qwen3VLMoeForConditionalGeneration._from_config(full_cfg)
            out["Qwen3VLMoeForConditionalGeneration"] = {
                "ok": True,
                "has_visual": hasattr(moe_lm, "visual"),
                "has_model_visual": hasattr(getattr(moe_lm, "model", None), "visual"),
                "class": type(moe_lm).__name__,
            }
        except Exception as e:  # noqa: BLE001
            out["Qwen3VLMoeForConditionalGeneration"] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
        try:
            vision_cfg = Qwen3VLVisionConfig.from_json_file(
                os.path.join(find_snapshot(), "vision_encoder", "config.json")
            )
            vision = Qwen3VLVisionModel._from_config(vision_cfg)
            out["Qwen3VLVisionModel_from_vision_encoder_config"] = {
                "ok": True,
                "class": type(vision).__name__,
                "num_params_meta": sum(p.numel() for p in vision.parameters()),
            }
        except Exception as e:  # noqa: BLE001
            out["Qwen3VLVisionModel_from_vision_encoder_config"] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--instantiate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    snapshot = args.snapshot or find_snapshot()
    report = inspect_snapshot(snapshot)
    if args.instantiate:
        report["instantiate_probe"] = instantiate_probe()

    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
