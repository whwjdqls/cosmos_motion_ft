"""Per-WINDOW re-test of the candidate sequences flagged by the per-sequence scan.
Each window decoded individually in a subprocess with a SHORT hard-kill timeout, so a real seek-hang
(infinite C call) is unambiguously distinguished from 'slow/many windows'. A normal 97-frame window
decodes in <2s; >PER_WIN_TIMEOUT = genuinely hung -> the corrupt window."""
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
PY = "/home/jungbin_cho/miniforge3/envs/cosmos/bin/python"
HERE = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
CAND = json.load(open("/weka/jungbin/cosmos_motion_ft_runs/bad_seek_seqs.json"))
cand_uuids = {c["uuid"] for c in CAND}
T = 97; FPS = 20.0
PER = int(os.environ.get("PER_WIN_TIMEOUT", "60")); WORKERS = int(os.environ.get("SCAN_WORKERS", "24"))

# collect every (uuid, vis, start_frame) window of the candidate sequences
wins = []
for line in open(MANIFEST):
    r = json.loads(line)
    if r.get("uuid") not in cand_uuids or not r.get("vision_path"): continue
    nb = int(r.get("nb_frames", 0)); vis = r["vision_path"]
    for w in r.get("t2w_windows", []):
        s = int(w["start_frame"])
        if w.get("usable", False) and w.get("caption") and s + T <= nb:
            wins.append((r["uuid"], vis, s))
print(f"per-window re-test: {len(wins)} windows across {len(cand_uuids)} candidate sequences (timeout {PER}s/window)", flush=True)

CODE = f"""
import sys
sys.path.insert(0, "{HERE}")
from nymeria_camera_dataset import decode_window_pyav
decode_window_pyav(sys.argv[1], int(sys.argv[2]), {T}, {FPS})
sys.stdout.write("OK")
"""

def test(item):
    u, vis, s = item
    try:
        r = subprocess.run([PY, "-c", CODE, vis, str(s)], timeout=PER, capture_output=True)
        if r.returncode != 0: return (u, s, "ERROR", r.stderr.decode()[-160:].replace("\n", " "))
        return (u, s, "OK", "")
    except subprocess.TimeoutExpired:
        return (u, s, "HANG", f">{PER}s")

hangs, done = [], 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(test, it) for it in wins]
    for fu in as_completed(futs):
        u, s, st, info = fu.result(); done += 1
        if st != "OK":
            hangs.append((u, s, st, info)); print(f"  [{st}] {u} @start={s}  {info}", flush=True)
        if done % 500 == 0: print(f"  ...{done}/{len(wins)}", flush=True)

print(f"\n=== DONE: {len(hangs)} truly-hanging/erroring windows / {len(wins)} ===")
from collections import Counter
byseq = Counter(u for u, s, st, i in hangs if st == "HANG")
print("REAL-HANG sequences (window count):")
for u, n in byseq.most_common(): print(f"  {u}: {n} hung windows")
json.dump([{"uuid": u, "start_frame": s, "status": st, "info": i} for u, s, st, i in hangs],
          open("/weka/jungbin/cosmos_motion_ft_runs/real_hang_windows.json", "w"), indent=2)
json.dump(sorted(byseq.keys()), open("/weka/jungbin/cosmos_motion_ft_runs/real_hang_uuids.json", "w"), indent=2)
print("wrote real_hang_windows.json + real_hang_uuids.json")
