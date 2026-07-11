"""CPU-only verification of the NATIVE (padded+masked) NymeriaPlus text2motion path.

Run from the cosmos-framework cwd with:
  export CUDA_VISIBLE_DEVICES=""
  PYTHONPATH=cosmos-framework:nymeria_world:motion_expert_joint_attention
Checks:
  1. T=200 is ALLOWED for a pure-text2motion mixture (4N+1 assert relaxed).
  2. `_t2m_index` is LARGE and ~T-independent (compare T=97 vs T=200).
  3. Sampled nymeria items -> motion (200,283), pad True on the padded tail (~100 valid);
     bones items likewise masked; no exclusions/crashes.
  4. DistributedSampler(8) + DataLoader(batch=32, drop_last=True) yields many batches/rank at
     T=200 with first-batch motion (32,200,283).
"""
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nymeria_joint_dataset import NymeriaJointDataset, collate_joint

TW = {"text2motion": 1.0}


def build(T):
    return NymeriaJointDataset(
        split="train", num_frames=T, task_weights=TW,
        bones_text2motion_frac=0.5, train=True, seed=0,
    )


def main():
    # ---- 1 + 2: T=200 allowed; _t2m_index size at T=97 vs T=200 -----------------------------
    ds97 = build(97)
    ds200 = build(200)   # would previously raise "num_frames must be 4N+1"
    print(f"[allowed]  T=200 constructed OK (4N+1 assert relaxed for pure text2motion)")
    print(f"[_index]     T=97 aligned windows = {len(ds97._index)}")
    print(f"[_index]     T=200 aligned windows = {len(ds200._index)}")
    print(f"[_t2m_index] T=97  = {len(ds97._t2m_index)}")
    print(f"[_t2m_index] T=200 = {len(ds200._t2m_index)}")
    print(f"[__len__]    T=200 = {len(ds200)}  has_bones={ds200.has_bones}"
          f"  bones={len(ds200._bones) if ds200.has_bones else 0}")
    assert len(ds200._t2m_index) > 10000, "expected tens of thousands of native windows"
    assert len(ds200._t2m_index) == len(ds97._t2m_index), \
        "native t2m index must be T-independent"

    # ---- 3: sample ~40 items; check shapes + valid-frame distribution -----------------------
    src_valid = {"nymeria": [], "bones": []}
    src_cnt = Counter()
    for i in range(40):
        r = ds200[i]
        src_cnt[r["source"]] += 1
        m = r["motion"]
        pad = r["motion_pad_mask"]
        Tm = m.shape[0]
        assert m.shape[1] == 283 and pad.shape[0] == Tm, f"[{i}] {r['source']} {tuple(m.shape)}"
        if r["source"] == "nymeria":
            # nymeria is ALWAYS zero-padded to exactly T with a pad mask marking the tail.
            assert Tm == 200, f"[{i}] nymeria motion={tuple(m.shape)} (must be T=200)"
            valid = int((~pad).sum())
            if valid < 200:
                assert torch.count_nonzero(m[valid:]) == 0, "nymeria padded tail must be zeros"
        else:
            # bones is ragged (<= T); collate pads it to batch-max. Individual item pad is all-valid.
            assert Tm <= 200, f"[{i}] bones motion={tuple(m.shape)} exceeds T"
            valid = int((~pad).sum())
        src_valid[r["source"]].append(valid)
    print(f"\n[sample40] sources = {dict(src_cnt)}")
    for s, v in src_valid.items():
        if v:
            va = np.array(v)
            print(f"[valid_frames:{s:7s}] n={len(v)} min={va.min()} "
                  f"median={int(np.median(va))} max={va.max()} mean={va.mean():.1f}")

    # ---- 4: DistributedSampler(8) + DataLoader batches/rank at T=200 ------------------------
    sampler = DistributedSampler(ds200, num_replicas=8, rank=0, shuffle=True, drop_last=True)
    dl = DataLoader(ds200, batch_size=32, sampler=sampler, drop_last=True,
                    num_workers=4, collate_fn=collate_joint)
    per_rank = len(sampler)                                   # samples this rank sees at T=200
    print(f"[DDP r0/8] sampler len (samples/rank) = {per_rank}  "
          f"=> {per_rank // 32} full batches/rank")
    MAX_BATCHES = 5                                           # enough to prove >0; full pass is slow
    nb = 0
    first = None
    for batch in dl:
        if first is None:
            first = tuple(batch["motion"].shape)
            # verify the collated pad mask marks padded frames (for BOTH nymeria and bones rows):
            # every zero motion frame with pad==False would be a masking bug.
            mot, pmask = batch["motion"], batch["motion_pad_mask"]
            nonzero_per_frame = mot.abs().sum(-1) > 0            # [B, T] True where frame has signal
            # a frame that is all-zero AND marked valid (pad False) is only OK if it is a genuine
            # clean frame; but our pads are always zero, so check: no frame is (valid & all-zero)
            # beyond the per-row valid count. Report the batched valid-frame span per source.
            srcs = batch["source"]
            vny = [int((~pmask[j]).sum()) for j in range(len(srcs)) if srcs[j] == "nymeria"]
            vbo = [int((~pmask[j]).sum()) for j in range(len(srcs)) if srcs[j] == "bones"]
            print(f"[batch0] nymeria valid span {min(vny) if vny else '-'}..{max(vny) if vny else '-'}"
                  f"  bones valid span {min(vbo) if vbo else '-'}..{max(vbo) if vbo else '-'}")
            # pad frames (mask True) must carry zero motion (zeros were written by collate/loader).
            assert torch.count_nonzero(mot[pmask]) == 0, "padded (mask=True) frames must be zeros"
        nb += 1
        if nb >= MAX_BATCHES:
            break
    print(f"\n[DDP r0/8] iterated {nb} batches (of {per_rank // 32}/rank)  first_batch_motion = {first}")
    assert per_rank // 32 > 0, "DistributedSampler yields 0 full batches/rank at T=200"
    assert nb > 0, "DataLoader yielded 0 batches at T=200"
    assert first == (32, 200, 283), f"first batch motion {first}"
    print("\nOK: native padded+masked text2motion path verified.")


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "run CPU-only (CUDA_VISIBLE_DEVICES=)"
    main()
