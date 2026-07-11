"""Merge a joint-attention ``.pt`` generator LoRA checkpoint into a native Nano DCP.

The older ``export_merge_lora.py`` handles native Cosmos DCP trainable-only deltas.
Phase-1 joint-attention checkpoints are plain torch ``.pt`` files with keys like
``cosmos.net.language_model...lora_A.weight``. Native inference cannot overlay those
LoRA tensors, so this script applies them into the base DCP state dict and writes a
full DCP that can be sampled with ``lora_enabled=false``.

For ``ja_phase1_camera/ckpt_step200000.pt`` the effective LoRA alpha is 16 and rank is
16, so the default merge scale is 1.0. Pass ``--lora_scale`` for older experiments.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

BASE_DCP = "/weka/jungbin/cosmos3_nano_dcp"


def strip_joint_prefix(key: str) -> str:
    if key.startswith("cosmos.net."):
        return "net." + key[len("cosmos.net."):]
    if key.startswith("cosmos.net_ema."):
        return "net_ema." + key[len("cosmos.net_ema."):]
    if key.startswith("net.") or key.startswith("net_ema."):
        return key
    return key


def load_model_dict(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a state dict")
    return obj


def copy_metadata(base: str, out: str) -> None:
    for f in ("checkpoint.json",):
        src = os.path.join(base, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out, f))
    base_model = os.path.join(base, "model")
    out_model = os.path.join(out, "model")
    for f in os.listdir(base_model):
        if f.endswith(".json") or f == "config.json":
            shutil.copy(os.path.join(base_model, f), os.path.join(out_model, f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="joint-attention .pt checkpoint; omit to only mirror base net->net_ema")
    ap.add_argument("--out", required=True, help="output merged DCP directory")
    ap.add_argument("--base", default=BASE_DCP, help="base Nano DCP directory")
    ap.add_argument("--tmp", default="/weka/jungbin/tmp_merge_jointpt")
    ap.add_argument("--lora_scale", type=float, default=1.0, help="alpha/rank merge scale")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(os.path.join(args.out, "model"), exist_ok=True)
    base_pt = os.path.join(args.tmp, "base.pt")
    merged_pt = os.path.join(args.tmp, "merged.pt")

    if os.path.isfile(base_pt) and os.path.getsize(base_pt) > 1_000_000_000:
        print(f"[1/5] reuse {base_pt}", flush=True)
    else:
        print(f"[1/5] base DCP -> {base_pt}", flush=True)
        dcp_to_torch_save(os.path.join(args.base, "model"), base_pt)

    print(f"[2/5] load base" + (" + joint checkpoint" if args.ckpt else ""), flush=True)
    base_obj = torch.load(base_pt, map_location="cpu", weights_only=False)
    base = base_obj.get("model", base_obj) if isinstance(base_obj, dict) else base_obj
    delta = load_model_dict(args.ckpt) if args.ckpt else {}
    print(f"    base keys={len(base)}  delta keys={len(delta)}", flush=True)

    n_lora = 0
    n_direct = 0
    missing = []
    for src_k, tensor in list(delta.items()):
        k = strip_joint_prefix(src_k)
        if k.startswith("net_ema."):
            continue
        if k.endswith(".lora_A.weight"):
            base_w = k.replace(".lora_A.weight", ".weight")
            src_b = src_k.replace(".lora_A.", ".lora_B.")
            if base_w not in base or src_b not in delta:
                missing.append((src_k, base_w, src_b))
                continue
            A = tensor.float()
            B = delta[src_b].float()
            base[base_w] = (base[base_w].float() + args.lora_scale * (B @ A)).to(base[base_w].dtype)
            n_lora += 1
        elif ".lora_B.weight" in k:
            continue
        else:
            if k not in base:
                missing.append((src_k, k, None))
                continue
            base[k] = tensor.to(base[k].dtype)
            n_direct += 1

    if missing:
        preview = "\n".join(str(x) for x in missing[:20])
        raise KeyError(f"missing {len(missing)} merge keys; first entries:\n{preview}")
    print(f"[3/5] merged {n_lora} LoRA layers at scale {args.lora_scale:g}; "
          f"overwrote {n_direct} direct tensors", flush=True)

    for k in [k for k in base if k.startswith("net.")]:
        base["net_ema." + k[len("net."):]] = base[k]
    print(f"    mirrored net -> net_ema; total keys now {len(base)}", flush=True)

    torch.save(base, merged_pt)
    print(f"[4/5] merged torch save -> DCP {args.out}/model", flush=True)
    torch_save_to_dcp(merged_pt, os.path.join(args.out, "model"))
    copy_metadata(args.base, args.out)

    if not args.keep_tmp:
        try:
            os.remove(merged_pt)
        except OSError:
            pass
    print(f"[5/5] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
