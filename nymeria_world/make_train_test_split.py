"""Deterministic random 90:10 train/test split over NymeriaPlus SEQUENCES (not windows).

Per-sequence (hold out whole recordings) to avoid leakage: windows in one recording are
seconds apart (same person/scene/lighting), so a window-level split would put near-identical
clips in both train and test. We split the 728 sequences instead.

Writes /weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json:
  {"seed", "test_ratio", "granularity":"sequence", "train":[uuid...], "test":[uuid...]}
"""
import json
import os
import random

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
CAM_RGB = "/weka/jungbin/nymeriaplus_kimodo_proportional/camera_rgb"
OUT = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"
SEED = 42
TEST_RATIO = 0.10


def main():
    # sequences that actually have preprocessed camera_rgb (usable for training)
    uuids = []
    for line in open(MANIFEST):
        rec = json.loads(line)
        cam = rec.get("camera_path")
        if not cam:
            continue
        rgb = cam.replace("/camera/", "/camera_rgb/")
        if os.path.isfile(rgb):
            uuids.append(rec["uuid"])
    uuids = sorted(set(uuids))
    rng = random.Random(SEED)
    rng.shuffle(uuids)
    n_test = round(len(uuids) * TEST_RATIO)
    test = sorted(uuids[:n_test])
    train = sorted(uuids[n_test:])
    json.dump(
        {"seed": SEED, "test_ratio": TEST_RATIO, "granularity": "sequence",
         "n_total": len(uuids), "n_train": len(train), "n_test": len(test),
         "train": train, "test": test},
        open(OUT, "w"), indent=1,
    )
    print(f"total seqs: {len(uuids)}  -> train {len(train)} / test {len(test)}  (seed {SEED})")
    print(f"wrote {OUT}")
    # per-subject test coverage (sanity)
    from collections import Counter
    c = Counter(u.split("/")[0] for u in test)
    print("test seqs per subject:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
