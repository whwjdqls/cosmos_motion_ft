"""One-off: find calibrated windows with metre-scale |y| that survived the drop filters
(the poison that blew up run 2727 at step 14140 with loss=34.6)."""
import glob
import json
import os
from collections import Counter

import numpy as np

import config
from uniego_layout import JOINT_Y_IDX

cal = json.load(open(config.FLOOR_CALIBRATION_JSON))
deltas = cal["deltas"]
dropped = cal.get("dropped_windows", {})

# locate the uniego npz root the same way the dataset does
probe = glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/*/S01")
uni_root = None
for cand in sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/*")):
    if os.path.isdir(cand) and glob.glob(os.path.join(cand, "S*", "*.npz")):
        # check one file has 'features'
        f0 = glob.glob(os.path.join(cand, "S*", "*.npz"))[0]
        try:
            with np.load(f0) as z:
                if "features" in z:
                    uni_root = cand
                    break
        except Exception:
            continue
print("uniego root:", uni_root)

bombs, n_checked, n_seq = [], 0, 0
with open(config.NYMERIA_MANIFEST) as f:
    for line in f:
        rec = json.loads(line)
        u = rec.get("uuid")
        if u not in deltas:
            continue
        p = os.path.join(uni_root, f"{u}.npz")
        if not os.path.isfile(p):
            continue
        n_seq += 1
        drop = {tuple(x[:2]) for x in dropped.get(u, [])}
        with np.load(p) as z:
            feats = z["features"]
        d = deltas[u]
        for w in rec.get("t2w_windows", []):
            if not w.get("usable", False) or not w.get("caption"):
                continue
            ws, we = int(w["start_frame"]), int(w["end_frame"])
            if (ws, we) in drop:
                continue
            off = w.get("ground_offset_y")
            if off is None:
                continue
            n_checked += 1
            y = feats[ws:min(we, feats.shape[0])][:, JOINT_Y_IDX] - (off + d)
            m = float(np.abs(y).max())
            if m > 2.5:
                bombs.append((u, ws, we, round(m, 1)))

print(f"seqs scanned: {n_seq}, windows checked: {n_checked}")
print(f"SURVIVING BOMBS (|y|>2.5m post-calibration): {len(bombs)} in "
      f"{len(set(b[0] for b in bombs))} seqs")
for u, c in Counter(b[0] for b in bombs).most_common(10):
    mx = max(b[3] for b in bombs if b[0] == u)
    print(f"   {u}: {c} windows, max|y|={mx}m")
out = "/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bomb_windows.json"
json.dump([{"uuid": u, "ws": int(a), "we": int(b), "maxy": m} for u, a, b, m in bombs],
          open(out, "w"), indent=1)
print("saved", out)
