"""Diagnostic (read-only, no training changes): locate the T=201 startup grind.

Builds NymeriaJointDataset at T=97 (known-fast) then T=201 (grinds), timing the
index build + the first few __getitem__ + a collate. faulthandler dumps the live
stack every 45s so if any stage hangs we see EXACTLY where. CPU-only."""
import faulthandler, os, sys, time

sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
os.chdir("/home/jungbin_cho/cosmos-framework")
faulthandler.dump_traceback_later(45, repeat=True)  # if a stage hangs, print the stack

import config, task_plan  # noqa: E402
from nymeria_joint_dataset import NymeriaJointDataset  # noqa: E402

T2M = {"text2motion": 1.0}  # match Phase 2 exactly

for T in (97, 201):
    print(f"\n===== T={T} (text2motion-only) =====", flush=True)
    t0 = time.time()
    ds = NymeriaJointDataset(
        split="train", num_frames=T, task_weights=T2M,
        bones_text2motion_frac=0.5, cfg_dropout=0.0, train=True, seed=0,
    )
    print(f"[T={T}] BUILD {time.time()-t0:.1f}s  len={len(ds)}", flush=True)
    t0 = time.time()
    for i in range(8):
        ti = time.time()
        it = ds[i]
        m = it.get("motion")
        print(f"[T={T}] getitem[{i}] {time.time()-ti:.2f}s mode={it['mode']} "
              f"motion={None if m is None else tuple(m.shape)} src={it.get('source')}", flush=True)
    print(f"[T={T}] 8 getitems total {time.time()-t0:.1f}s", flush=True)
print("\nDONE", flush=True)
