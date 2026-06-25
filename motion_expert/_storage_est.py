import json
from train_motion_ft import build_text_processor
proc = build_text_processor()
caps = set()
for split in ["pairs_train.jsonl", "pairs_val.jsonl"]:
    for l in open(f"/home/jungbin_cho/cosmos_motion_ft/motion_expert/{split}"):
        caps.add(json.loads(l)["caption"])
caps.add("")
caps = list(caps)
import numpy as np
lens = []
for c in caps[:2000]:
    lens.append(len(proc.tokenize_text(c)) if c else 1)
lens = np.array(lens)
avg = lens.mean()
n = len(caps)
print(f"unique captions (train+val+null): {n}")
print(f"token len over 2000 sample: mean={avg:.1f} p50={np.percentile(lens,50):.0f} p95={np.percentile(lens,95):.0f} max={lens.max()}")
for dt, b in [("float16/bf16", 2), ("float32", 4)]:
    gb = n * avg * 4096 * b / 1e9
    print(f"  H_R cache @ {dt}: {n} x ~{avg:.0f}tok x 4096 x {b}B = ~{gb:.1f} GB")
