"""Seek-based scan: replays the REAL decode_window_pyav (seek to keyframe + windowed decode) for every
training window, grouped per-sequence, each sequence in a SUBPROCESS with a hard-kill timeout.
Catches seek-level hangs the full-decode scan misses. Flags hanging/erroring sequences."""
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
PY = "/home/jungbin_cho/miniforge3/envs/cosmos/bin/python"
HERE = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
TRAIN = set(json.load(open("/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"))["train"])
T = int(os.environ.get("SCAN_T", "97")); FPS = 20.0
TIMEOUT = int(os.environ.get("SCAN_TIMEOUT", "240")); WORKERS = int(os.environ.get("SCAN_WORKERS", "24"))

# group train windows by sequence (usable, valid for T)
seqs = []
for line in open(MANIFEST):
    r = json.loads(line)
    if r.get("uuid") not in TRAIN or not r.get("vision_path"): continue
    nb = int(r.get("nb_frames", 0)); vis = r["vision_path"]
    starts = [int(w["start_frame"]) for w in r.get("t2w_windows", [])
              if w.get("usable", False) and w.get("caption") and int(w["start_frame"]) + T <= nb]
    if starts: seqs.append((r["uuid"], vis, starts))
print(f"seek-scanning {len(seqs)} train sequences, {sum(len(s[2]) for s in seqs)} windows (T={T}, timeout {TIMEOUT}s/seq, {WORKERS} workers)", flush=True)

# subprocess decodes EVERY window of one sequence via the real decode_window_pyav
CODE = f"""
import sys, json
sys.path.insert(0, "{HERE}")
from nymeria_camera_dataset import decode_window_pyav
vis = sys.argv[1]; starts = json.loads(sys.argv[2])
for s in starts:
    decode_window_pyav(vis, s, {T}, {FPS})
sys.stdout.write("OK")
"""

def test(item):
    u, vis, starts = item
    if not os.path.isfile(vis): return (u, vis, "MISSING", "")
    try:
        r = subprocess.run([PY, "-c", CODE, vis, json.dumps(starts)], timeout=TIMEOUT, capture_output=True)
        if r.returncode != 0: return (u, vis, "ERROR", r.stderr.decode()[-200:].replace("\n", " "))
        return (u, vis, "OK", f"{len(starts)} windows")
    except subprocess.TimeoutExpired:
        return (u, vis, "HANG", f">{TIMEOUT}s ({len(starts)} windows)")

bad, done = [], 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(test, it) for it in seqs]
    for fu in as_completed(futs):
        u, vis, status, info = fu.result(); done += 1
        if status != "OK":
            bad.append((u, vis, status, info)); print(f"  [{status}] {u}  {info}", flush=True)
        if done % 50 == 0: print(f"  ...{done}/{len(seqs)}", flush=True)
print(f"\n=== DONE: {len(bad)} bad / {len(seqs)} sequences ===")
json.dump([{"uuid": u, "vision_path": v, "status": s, "info": i} for u, v, s, i in bad],
          open("/weka/jungbin/cosmos_motion_ft_runs/bad_seek_seqs.json", "w"), indent=2)
for u, v, s, i in bad: print(f"  {s}  {u}")
