"""Evaluate NVIDIA's released TMR-SOMA-RP-v1 on Kimodo GT test motions.

The released TMR model is a 30 fps SOMA30 evaluator. Our local benchmark copy is
the 20 fps testsuite, so this script upsamples GT posed joints to 30 fps before
encoding. Text uses the precomputed benchmark LLM2Vec cache to avoid loading the
raw LLM during evaluator checks.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
from kimodo.model.tmr import TMR
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
from kimodo.sanitize import sanitize_text
from kimodo.skeleton import SOMASkeleton30

from decode_uniego import decode_joints
from st_inline_eval import BENCH_TEXT_CACHE
from st_text_cache import LLM2VecCache

log = logging.getLogger("official_tmr_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


OFFICIAL_TMR_DIR = (
    "/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/"
    "snapshots/e427752ae3446dedba49e928c93ddc9f0e413401"
)
TESTSUITE_20FPS = "/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite"

PUBLISHED_GT_R3 = {
    "content/overview": 89.09,
    "content/timeline_single": 86.26,
    "content/timeline_multi": 88.47,
    "repetition/overview": 93.91,
    "repetition/timeline_single": 90.13,
    "repetition/timeline_multi": 94.49,
}
PUBLISHED_GT_FID = {k: 0.0 for k in PUBLISHED_GT_R3}
_SOMA30_SLICE_FROM77 = None


def soma30_slice_from77():
    global _SOMA30_SLICE_FROM77
    if _SOMA30_SLICE_FROM77 is None:
        soma30 = SOMASkeleton30()
        _SOMA30_SLICE_FROM77 = soma30.get_skel_slice(soma30.somaskel77)
    return _SOMA30_SLICE_FROM77


def build_official_tmr(model_dir: str, device: str) -> TMR:
    model_path = Path(model_dir)
    motion_rep = TMRMotionRep(
        skeleton=SOMASkeleton30(),
        fps=30,
        stats_path=str(model_path / "stats" / "motion"),
    )
    model = TMR.from_args(
        motion_rep=motion_rep,
        llm_shape=(1, 4096),
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        ckpt_folder=str(model_path / "last_weights"),
        device=device,
        sample_mean=True,
        unit_vector=True,
        compute_grads=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _resolve_uniego_case(cdir: Path, uniego_root: str):
    sm = json.load(open(cdir / "seed_motion.json", encoding="utf-8"))
    rel = sm["bvh_path"]
    rel = rel[4:] if rel.startswith("BVH/") else rel
    rel = rel[:-4] if rel.endswith(".bvh") else rel
    npz = Path(uniego_root) / f"{rel}.npz"
    return {
        "npz": npz,
        "crop_start": int(sm["crop_start_frame_index"]),
        "crop_end": int(sm["crop_end_frame_index"]),
    }


def load_cases(
    testsuite: str,
    split: str,
    group: str,
    text_cache: LLM2VecCache,
    limit: int,
    uniego_root: str | None = None,
):
    case_dirs = sorted(glob.glob(f"{testsuite}/{split}/text2motion/{group}/*"))
    cases, skipped = [], 0
    for cd in case_dirs:
        if limit > 0 and len(cases) >= limit:
            break
        cdir = Path(cd)
        try:
            text = sanitize_text(
                str(json.load(open(cdir / "meta.json", encoding="utf-8"))["text"])
            )
        except (OSError, json.JSONDecodeError, KeyError):
            skipped += 1
            continue
        if text not in text_cache:
            skipped += 1
            continue
        if uniego_root:
            try:
                motion = _resolve_uniego_case(cdir, uniego_root)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                skipped += 1
                continue
            if not motion["npz"].is_file():
                skipped += 1
                continue
            cases.append({"dir": cdir, "text": text, "motion": motion})
        else:
            npz = cdir / "gt_motion.npz"
            if not npz.is_file():
                skipped += 1
                continue
            cases.append({"dir": cdir, "text": text, "motion": {"npz": npz}})
    return cases, skipped


def resample_time(posed_joints: torch.Tensor, src_fps: float, tgt_fps: float) -> torch.Tensor:
    if src_fps == tgt_fps:
        return posed_joints
    t, j, c = posed_joints.shape
    new_t = max(1, round(t * tgt_fps / src_fps))
    x = posed_joints.reshape(t, j * c).permute(1, 0).unsqueeze(0)
    x = F.interpolate(x, size=new_t, mode="linear", align_corners=True)
    return x.squeeze(0).permute(1, 0).reshape(new_t, j, c)


def load_posed_joints(npz_path: Path, device: str, src_fps: float, tgt_fps: float) -> torch.Tensor:
    with np.load(npz_path, allow_pickle=False) as data:
        if "posed_joints" not in data:
            raise KeyError(f"missing posed_joints: {npz_path}")
        arr = np.asarray(data["posed_joints"], dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1 in {npz_path}, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected [T,J,3] posed_joints in {npz_path}, got {arr.shape}")
    return resample_time(torch.from_numpy(arr).to(device), src_fps, tgt_fps)


def load_uniego_joints(case: dict, device: str, src_fps: float, tgt_fps: float) -> torch.Tensor:
    motion = case["motion"]
    start = end = None
    with np.load(motion["npz"], allow_pickle=False) as data:
        n_total = int(data["features"].shape[0] if "features" in data else data["posed_joints"].shape[0])
        start = int(round(motion["crop_start"] * src_fps / 30.0))
        end = int(round(motion["crop_end"] * src_fps / 30.0))
        start = max(0, min(start, n_total))
        end = max(start, min(end, n_total))
        if "features" in data:
            win = torch.from_numpy(np.asarray(data["features"][start:end], dtype=np.float32)).unsqueeze(0)
            if win.shape[1] < 1:
                raise ValueError(f"empty motion window: {motion['npz']}")
            joints = decode_joints(win)[0]
        elif "posed_joints" in data:
            posed = np.asarray(data["posed_joints"][start:end], dtype=np.float32)
            if posed.shape[0] < 1:
                raise ValueError(f"empty motion window: {motion['npz']}")
            if posed.shape[1] == 77:
                posed = posed[:, soma30_slice_from77()]
            elif posed.shape[1] != 30:
                raise ValueError(f"expected 30 or 77 joints in {motion['npz']}, got {posed.shape}")
            joints = torch.from_numpy(posed)
        else:
            raise KeyError(f"{motion['npz']} has neither features nor posed_joints")
    joints = joints.to(device)
    return resample_time(joints, src_fps, tgt_fps)


@torch.no_grad()
def encode_group(model: TMR, text_cache: LLM2VecCache, cases: list[dict], args):
    motion_embs, text_embs = [], []
    for start in range(0, len(cases), args.batch_size):
        chunk = cases[start:start + args.batch_size]
        for c in chunk:
            if args.uniego_root:
                pj = load_uniego_joints(c, args.device, args.src_fps, args.tmr_fps)
            else:
                pj = load_posed_joints(c["motion"]["npz"], args.device, args.src_fps, args.tmr_fps)
            length_t = torch.as_tensor([pj.shape[0]], dtype=torch.long, device=args.device)
            # Upstream TMRMotionRep canonicalization is not batch-safe for this path.
            motion_vec = model.encode_motion(pj.unsqueeze(0), lengths=length_t, unit_vector=True)
            motion_embs.append(motion_vec.detach().cpu())

        llm = text_cache.batch([c["text"] for c in chunk])
        mask = torch.ones(llm.shape[0], llm.shape[1], dtype=torch.bool, device=args.device)
        text_vec = model.encode_text({"x": llm, "mask": mask}, unit_vector=True)
        text_embs.append(text_vec.detach().cpu())

    return torch.cat(motion_embs).numpy(), torch.cat(text_embs).numpy()


def plain_metrics(motion_emb: np.ndarray, text_emb: np.ndarray) -> dict[str, float]:
    sim = text_emb @ motion_emb.T
    ranks = (sim > sim.diagonal()[:, None]).sum(axis=1)
    tt = text_emb @ text_emb.T
    n = tt.shape[0]
    return {
        "plain_R01": float((ranks < 1).mean() * 100),
        "plain_R03": float((ranks < 3).mean() * 100),
        "plain_R05": float((ranks < 5).mean() * 100),
        "plain_R10": float((ranks < 10).mean() * 100),
        "plain_MedR": float(np.median(ranks) + 1),
        "dup_frac099": float(((tt > 0.99).sum() - n) / max(1, n * (n - 1))),
        "merge_frac096": float(((tt > 0.96).sum() - n) / max(1, n * (n - 1))),
        "motion_emb_std": float(np.linalg.norm(motion_emb.std(axis=0))),
        "text_emb_std": float(np.linalg.norm(text_emb.std(axis=0))),
    }


def eval_group(model: TMR, text_cache: LLM2VecCache, split: str, group: str, args) -> dict:
    key = f"{split}/{group}"
    cases, skipped = load_cases(
        args.testsuite, split, group, text_cache, args.cases,
        uniego_root=args.uniego_root,
    )
    if not cases:
        raise RuntimeError(f"no usable cases for {key}; skipped={skipped}")
    log.info("pool[%s]: %d cases (skipped %d)", key, len(cases), skipped)
    motion_emb, text_emb = encode_group(model, text_cache, cases, args)
    finite = np.isfinite(motion_emb).all(axis=1) & np.isfinite(text_emb).all(axis=1)
    skipped_nonfinite = int((~finite).sum())
    if skipped_nonfinite:
        log.warning("[%s] dropping %d non-finite embedding rows before metrics", key, skipped_nonfinite)
        motion_emb = motion_emb[finite]
        text_emb = text_emb[finite]
    if len(motion_emb) == 0:
        raise RuntimeError(f"all embeddings were non-finite for {key}")
    protocol = compute_tmr_retrieval_metrics(motion_emb, text_emb, gt_motion_emb=motion_emb)
    out = {k.split("/")[-1]: v for k, v in protocol.items() if "TMR/t2m_R/" in k}
    out.update({
        "FID_gen_gt": protocol["TMR/FID/gen_gt"],
        "FID_gt_text": protocol["TMR/FID/gt_text"],
        "n": len(cases),
        "skipped": skipped,
        "skipped_nonfinite": skipped_nonfinite,
        "published_GT_R03": PUBLISHED_GT_R3[key],
        "published_GT_FID": PUBLISHED_GT_FID[key],
    })
    out.update(plain_metrics(motion_emb, text_emb))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=OFFICIAL_TMR_DIR)
    parser.add_argument("--testsuite", default=TESTSUITE_20FPS)
    parser.add_argument("--text-cache", default=BENCH_TEXT_CACHE)
    parser.add_argument("--out", default=None)
    parser.add_argument("--uniego-root", default=None,
                        help="If set, resolve seed_motion.json paths into this uniego npz tree "
                             "instead of using testsuite gt_motion.npz.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--src-fps", type=float, default=20.0)
    parser.add_argument("--tmr-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cases", type=int, default=0, help="per group; <=0 means all")
    args = parser.parse_args()

    model = build_official_tmr(args.model_dir, args.device)
    text_cache = LLM2VecCache(args.text_cache, device=args.device)

    results = {}
    for split in ("content", "repetition"):
        for group in ("overview", "timeline_single", "timeline_multi"):
            key = f"{split}/{group}"
            results[key] = eval_group(model, text_cache, split, group, args)
            r = results[key]
            log.info(
                "[%s] n=%d R03=%.2f published=%.2f FID_gen_gt=%.6g plain_R03=%.2f plain_MedR=%.1f merge096=%.4f",
                key, r["n"], r["R03"], r["published_GT_R03"], r["FID_gen_gt"],
                r["plain_R03"], r["plain_MedR"], r["merge_frac096"],
            )

    payload = {
        "model_dir": args.model_dir,
        "testsuite": args.testsuite,
        "uniego_root": args.uniego_root,
        "src_fps": args.src_fps,
        "tmr_fps": args.tmr_fps,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
