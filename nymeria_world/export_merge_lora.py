"""Merge a trainable-only LoRA+action-head delta DCP into the base DCP -> full DCP for inference.

State-dict level (no 16B model build). For each delta `*.lora_A.weight`:
    base[`*.weight`] += (alpha/rank) * (lora_B @ lora_A)
and the fine-tuned action heads (action2llm/llm2action/action_modality_embed) overwrite the base's.
Result is a plain Nano DCP (no LoRA modules) that native inference loads with lora_enabled=false.

Usage:
  python export_merge_lora.py --delta <run>/checkpoints/iter_NNNNNN/model --out <merged_dcp_dir>
"""
from __future__ import annotations
import argparse, os, shutil, tempfile
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

BASE_DCP = "/weka/jungbin/cosmos3_nano_dcp"
LORA_SCALE = 32.0 / 16.0  # alpha/rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="iter_*/model dir (trainable-only: lora + action heads)")
    ap.add_argument("--out", required=True, help="output merged DCP dir")
    ap.add_argument("--base", default=BASE_DCP)
    ap.add_argument("--tmp", default="/weka/jungbin/tmp_merge")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(os.path.join(args.out, "model"), exist_ok=True)

    base_pt = os.path.join(args.tmp, "base.pt"); delta_pt = os.path.join(args.tmp, "delta.pt")
    merged_pt = os.path.join(args.tmp, "merged.pt")
    if os.path.isfile(base_pt) and os.path.getsize(base_pt) > 1e9:
        print(f"[1/5] reuse {base_pt}")
    else:
        print(f"[1/5] base DCP -> {base_pt}"); dcp_to_torch_save(os.path.join(args.base, "model"), base_pt)
    print(f"[2/5] delta DCP -> {delta_pt}"); dcp_to_torch_save(args.delta, delta_pt)

    base = torch.load(base_pt, map_location="cpu", weights_only=False)
    delta = torch.load(delta_pt, map_location="cpu", weights_only=False)
    base = base.get("model", base) if isinstance(base, dict) and "model" in base else base
    delta = delta.get("model", delta) if isinstance(delta, dict) and "model" in delta else delta
    print(f"    base keys={len(base)}  delta keys={len(delta)}")

    n_merged = n_head = 0
    for k in list(delta.keys()):
        if k.startswith("net_ema."):
            continue  # EMA copies — use the regular net.* weights for inference
        if k.endswith(".lora_A.weight"):
            base_w = k.replace(".lora_A.weight", ".weight")
            kB = k.replace(".lora_A.", ".lora_B.")
            assert base_w in base, f"missing base weight {base_w}"
            A = delta[k].float(); B = delta[kB].float()           # (rank,in), (out,rank)
            dW = LORA_SCALE * (B @ A)                              # (out,in)
            base[base_w] = (base[base_w].float() + dW).to(base[base_w].dtype)
            n_merged += 1
        elif ".lora_B.weight" in k:
            continue
        else:
            # fine-tuned head (action2llm/llm2action/action_modality_embed): overwrite base
            assert k in base, f"head key {k} not in base"
            base[k] = delta[k].to(base[k].dtype); n_head += 1
    print(f"[3/5] merged {n_merged} LoRA layers + overwrote {n_head} head tensors")

    # inference builds the model with an EMA copy (net_ema.*) and loads strictly; mirror net->net_ema
    # so the load is satisfied (ema disabled at inference, so net is what's used).
    for k in [k for k in base if k.startswith("net.")]:
        base["net_ema." + k[len("net."):]] = base[k]
    print(f"    mirrored net->net_ema; total keys now {len(base)}")

    torch.save(base, merged_pt)
    print(f"[4/5] merged -> DCP {args.out}/model"); torch_save_to_dcp(merged_pt, os.path.join(args.out, "model"))
    # carry the config/checkpoint metadata so inference can build the model
    for f in ("checkpoint.json",):
        src = os.path.join(args.base, f)
        if os.path.isfile(src): shutil.copy(src, os.path.join(args.out, f))
    for f in os.listdir(os.path.join(args.base, "model")):
        if f.endswith(".json") or f == "config.json":
            shutil.copy(os.path.join(args.base, "model", f), os.path.join(args.out, "model", f))
    for p in (base_pt, delta_pt, merged_pt):
        try: os.remove(p)
        except OSError: pass
    print(f"[5/5] done -> {args.out}")


if __name__ == "__main__":
    main()
