"""Decode-test every video in the manifest in a SUBPROCESS with a hard timeout (SIGKILL on hang).
A C-level PyAV hang that SIGALRM can't interrupt IS killable as a subprocess. Reports HANG/ERROR videos."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
PY = "/home/jungbin_cho/miniforge3/envs/cosmos/bin/python"
SPLIT = json.load(open("/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"))
TRAIN = set(SPLIT["train"]); TEST = set(SPLIT["test"])
TIMEOUT = int(os.environ.get("SCAN_TIMEOUT", "90"))
WORKERS = int(os.environ.get("SCAN_WORKERS", "24"))

# one (uuid, vision_path) per sequence (train+test); dedup by path
seen, vids = set(), []
for line in open(MANIFEST):
    r = json.loads(line); v = r.get("vision_path"); u = r.get("uuid")
    if not v or v in seen: continue
    seen.add(v); split = "train" if u in TRAIN else ("test" if u in TEST else "?")
    vids.append((u, v, split))
print(f"scanning {len(vids)} unique videos (timeout {TIMEOUT}s, {WORKERS} workers)", flush=True)

# decode ALL frames sequentially in a subprocess (catches frame-level corruption + hangs)
CODE = "import av,sys\nc=av.open(sys.argv[1]); n=0\nfor f in c.decode(video=0): n+=1\nsys.stdout.write(str(n))"

def test(item):
    u, v, split = item
    if not os.path.isfile(v): return (u, v, split, "MISSING", "")
    try:
        r = subprocess.run([PY, "-c", CODE, v], timeout=TIMEOUT, capture_output=True)
        if r.returncode != 0: return (u, v, split, "ERROR", r.stderr.decode()[-180:].replace("\n"," "))
        return (u, v, split, "OK", r.stdout.decode().strip())
    except subprocess.TimeoutExpired:
        return (u, v, split, "HANG", f">{TIMEOUT}s")

bad = []; done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(test, it) for it in vids]
    for fu in as_completed(futs):
        u, v, split, status, info = fu.result(); done += 1
        if status != "OK":
            bad.append((u, v, split, status, info))
            print(f"  [{status}] ({split}) {u}  {info}", flush=True)
        if done % 100 == 0: print(f"  ...{done}/{len(vids)}", flush=True)

print(f"\n=== DONE: {len(bad)} bad / {len(vids)} ===")
out = "/weka/jungbin/cosmos_motion_ft_runs/bad_videos.json"
json.dump([{"uuid":u,"vision_path":v,"split":s,"status":st,"info":i} for u,v,s,st,i in bad], open(out,"w"), indent=2)
print("wrote", out)
for u,v,s,st,i in bad: print(f"  {st} ({s}) {u}")
