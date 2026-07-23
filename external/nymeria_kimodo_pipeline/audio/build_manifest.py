"""Stage 0 - enumerate NymeriaPlus head recordings that carry audio.

Every head recording stores its 7-mic, 48 kHz Aria audio inside the main
`recording_head/data/data.vrs` (stream `231-1/mic`).  The dataset also ships a
standalone `audio.vrs`, but the high-level projectaria provider refuses to open an
audio-only VRS ("No stream activated"), so we always read the mic stream out of
`data.vrs` instead (random access -> only the ~1 GB of audio records are touched,
not the 16 GB of images).

Output: <OUT>/manifest.jsonl  - one row per head recording:
    {subject, seq, data_vrs, has_data_vrs}

Env: any python with stdlib (no deps).  Run on a login node.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

NROOT = Path("/weka/jungbin/nymeriaplus")
OUT = Path("/weka/jungbin/nymeriaplus_audio")


def iter_recordings(nroot: Path):
    for subj in sorted(p.name for p in nroot.iterdir() if p.is_dir() and p.name.startswith("S")):
        sdir = nroot / subj
        for seq in sorted(p.name for p in sdir.iterdir() if p.is_dir()):
            yield subj, seq, sdir / seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nroot", default=str(NROOT))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    nroot = Path(args.nroot); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, n_audio = [], 0
    for subj, seq, rdir in iter_recordings(nroot):
        data_vrs = rdir / "recording_head" / "data" / "data.vrs"
        has = data_vrs.is_file()
        n_audio += int(has)
        rows.append({"subject": subj, "seq": seq,
                     "data_vrs": str(data_vrs), "has_data_vrs": has})
    mpath = out / "manifest.jsonl"
    with open(mpath, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {mpath}: {len(rows)} recordings, {n_audio} with head data.vrs")


if __name__ == "__main__":
    main()
