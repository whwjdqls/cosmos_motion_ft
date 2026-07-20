"""Build a C45-compatible LLM2Vec cache for all Phase-2 evaluation captions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from shape_tmr_eval_common import DEFAULT_BUNDLE_ROOT, add_bundle_python_paths, sha256_file


def _digest_captions(captions: list[str]) -> str:
    value = "\0".join(captions).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _patch_peft_transformers_compat() -> bool:
    """Bridge one import-only PEFT mismatch in the pinned Cosmos environment.

    Current PEFT imports ``EmbeddingParallel`` unconditionally while checking whether
    adapter weights need tensor-parallel sharding. The installed Transformers version
    removed that name. This evaluator does not use a TP device mesh, so restoring the
    historical alias is sufficient and does not alter model weights or execution.
    """
    from transformers.integrations import tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return False
    if not hasattr(tensor_parallel, "ColwiseParallel"):
        raise RuntimeError(
            "cannot patch PEFT/Transformers compatibility: ColwiseParallel is unavailable"
        )
    tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--base-cache", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-captions", type=int, default=0)
    parser.add_argument("--parity-cases", type=int, default=4)
    parser.add_argument("--parity-min-cosine", type=float, default=0.999)
    args = parser.parse_args()

    bundle_root = add_bundle_python_paths(args.bundle_root)
    from kimodo.model import LLM2VecEncoder

    base_cache_path = Path(args.base_cache).resolve() if args.base_cache else (
        bundle_root / "artifacts" / "text" / "benchmark_llm2vec.pt"
    )
    captions_path = Path(args.captions_json).resolve()
    out_path = Path(args.out).resolve()
    requested = list(json.loads(captions_path.read_text()))
    if not requested or "" not in requested or len(requested) != len(set(requested)):
        raise ValueError("captions JSON must contain unique strings including the empty prompt")
    requested = sorted(requested)
    requested_digest = _digest_captions(requested)

    base = torch.load(base_cache_path, map_location="cpu", weights_only=False, mmap=True)
    base_captions = list(base["captions"])
    base_features = base["features"]
    if base_features.ndim != 2 or base_features.shape != (len(base_captions), 4096):
        raise ValueError(f"unexpected base cache shape {tuple(base_features.shape)}")
    base_index = {caption: index for index, caption in enumerate(base_captions)}
    missing = [caption for caption in requested if caption not in base_index]
    if args.max_new_captions > 0:
        missing = missing[: args.max_new_captions]

    if out_path.is_file():
        existing = torch.load(out_path, map_location="meta", weights_only=False, mmap=True)
        meta = existing.get("meta", {})
        expected_new = len(missing)
        if (
            meta.get("requested_caption_sha256") == requested_digest
            and int(meta.get("new_captions", -1)) == expected_new
            and meta.get("base_cache_sha256") == sha256_file(base_cache_path)
        ):
            print(f"[text-cache] reuse verified cache {out_path}", flush=True)
            return
        raise RuntimeError(f"existing output has incompatible provenance: {out_path}")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("Nymeria LLM2Vec cache construction requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if _patch_peft_transformers_compat():
        print(
            f"[text-cache][rank {rank}] installed import-only EmbeddingParallel compatibility alias",
            flush=True,
        )
    encoder = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path=(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
        ),
        dtype="bfloat16",
        llm_dim=4096,
        device=str(device),
    )

    parity_pool = [caption for caption in base_captions if caption][: args.parity_cases]
    parity_feature, _ = encoder(parity_pool)
    parity_feature = parity_feature[:, 0].float().cpu()
    expected = torch.stack([base_features[base_index[caption]].float() for caption in parity_pool])
    parity_cosine = F.cosine_similarity(parity_feature, expected, dim=-1)
    parity_max_abs = (parity_feature - expected).abs().amax(dim=-1)
    parity_min = float(parity_cosine.min().item())
    print(
        f"[text-cache][rank {rank}] bundle parity cosine min={parity_min:.8f} "
        f"max_abs={float(parity_max_abs.max().item()):.6g}",
        flush=True,
    )
    if parity_min < args.parity_min_cosine:
        raise RuntimeError(
            f"local LLM2Vec output does not match the bundle cache: cosine {parity_min:.8f} "
            f"< {args.parity_min_cosine}"
        )

    rank_positions = list(range(rank, len(missing), world))
    rank_captions = [missing[index] for index in rank_positions]
    encoded = []
    started = time.time()
    for start in range(0, len(rank_captions), args.batch_size):
        batch = rank_captions[start : start + args.batch_size]
        values, _ = encoder(batch)
        encoded.append(values[:, 0].float().cpu())
        done = min(start + len(batch), len(rank_captions))
        if done == len(rank_captions) or done % 100 < len(batch):
            print(
                f"[text-cache][rank {rank}] {done}/{len(rank_captions)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    features = torch.cat(encoded) if encoded else torch.empty(0, 4096, dtype=torch.float32)
    shard_dir = out_path.with_suffix(out_path.suffix + ".shards")
    shard_path = shard_dir / f"rank_{rank:04d}_of_{world:04d}.pt"
    _atomic_torch_save(
        {
            "rank": rank,
            "world": world,
            "positions": torch.tensor(rank_positions, dtype=torch.long),
            "captions": rank_captions,
            "features": features,
            "requested_caption_sha256": requested_digest,
        },
        shard_path,
    )
    del encoder
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()

    if rank == 0:
        merged_features = torch.empty(len(missing), 4096, dtype=torch.float32)
        seen = torch.zeros(len(missing), dtype=torch.bool)
        for shard_rank in range(world):
            path = shard_dir / f"rank_{shard_rank:04d}_of_{world:04d}.pt"
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if shard["requested_caption_sha256"] != requested_digest:
                raise RuntimeError(f"stale text-cache shard: {path}")
            positions = shard["positions"].long()
            merged_features[positions] = shard["features"].float()
            seen[positions] = True
        if not bool(seen.all()):
            raise RuntimeError(f"text-cache merge is missing {int((~seen).sum())} captions")

        all_captions = base_captions + missing
        all_features = torch.cat([base_features.float(), merged_features], dim=0)
        base_sha = sha256_file(base_cache_path)
        payload = {
            "captions": all_captions,
            "features": all_features,
            "meta": {
                "encoder_type": "llm2vec",
                "model": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                "dim": 4096,
                "n": len(all_captions),
                "base_captions": len(base_captions),
                "new_captions": len(missing),
                "requested_captions": len(requested),
                "requested_caption_sha256": requested_digest,
                "base_cache": str(base_cache_path),
                "base_cache_sha256": base_sha,
                "captions_json": str(captions_path),
                "captions_json_sha256": sha256_file(captions_path),
                "world_size": world,
                "bundle_parity_min_cosine": parity_min,
                "bundle_parity_max_abs": float(parity_max_abs.max().item()),
            },
        }
        _atomic_torch_save(payload, out_path)
        _atomic_json(payload["meta"], out_path.with_suffix(".provenance.json"))
        print(
            f"[text-cache] wrote {out_path}: base={len(base_captions)} "
            f"new={len(missing)} total={len(all_captions)}",
            flush=True,
        )

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
