"""Stage 1 - extract egocentric (head) audio from the Aria VRS to a 16 kHz mono WAV.

Source : `recording_head/data/data.vrs`, stream `231-1/mic`
         - 7 channels, 48 kHz, 32-bit PCM (stored as int64; full-scale = 2**31).
         - audio arrives in blocks of 2048 frames; ~25k blocks ~= 18 min.
Target : <OUT>/raw/{subject}/{seq}.wav  - mono, 16 kHz, 16-bit PCM.

Why mono ch0 (not a 7-mic sum): the Aria mics are spatially distributed, so summing
them comb-filters off-axis sources.  A single channel is the clean default for a
downstream VAD / speech-separation front end.  Use --downmix mean to average instead.

Why 16 kHz: it is the native rate for the VAD (FireRedVAD / Silero) and most speech
separators; resampling once here keeps the rest of the pipeline rate-consistent.

Resumable: skips a sequence whose WAV already exists (unless --overwrite).
Shardable: --shard i/N processes manifest rows where (row_index % N == i) for slurm arrays.

Env: `nymeria_plus` (projectaria_tools + scipy + numpy).  CPU only.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
import argparse, json, os, wave, time
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

OUT = Path("/weka/jungbin/nymeriaplus_audio")
MIC = "231-1"
SRC_SR = 48000
DST_SR = 16000
FULL_SCALE = float(2 ** 31)        # Aria mic is 32-bit PCM


def read_mic(data_vrs: str):
    """Return (samples_float[N,7], src_sr).  Concatenates every mic block."""
    dp = data_provider.create_vrs_data_provider(data_vrs)
    sid = StreamId(MIC)
    n = dp.get_num_data(sid)
    if n == 0:
        raise RuntimeError("no mic records")
    cfg = dp.get_audio_configuration(sid)
    ch = int(cfg.num_channels)
    blocks = []
    for i in range(n):
        ad = dp.get_audio_data_by_index(sid, i)
        a = np.asarray(ad[0].data)                 # interleaved (frames*ch,)
        blocks.append(a.reshape(-1, ch))
    x = np.concatenate(blocks, axis=0).astype(np.float64) / FULL_SCALE
    return x, int(cfg.sample_rate)


def to_mono(x: np.ndarray, mode: str, channel: int) -> np.ndarray:
    if x.shape[1] == 1:
        return x[:, 0]
    return x.mean(axis=1) if mode == "mean" else x[:, channel]


def write_wav16(path: Path, mono: np.ndarray, sr: int):
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def extract_one(row: dict, out: Path, downmix: str, channel: int, overwrite: bool) -> dict:
    subj, seq = row["subject"], row["seq"]
    dst = out / "raw" / subj / f"{seq}.wav"
    if dst.is_file() and not overwrite:
        return {"subject": subj, "seq": seq, "status": "skip_exists", "wav": str(dst)}
    if not row.get("has_data_vrs"):
        return {"subject": subj, "seq": seq, "status": "no_data_vrs"}
    t0 = time.time()
    try:
        x, src_sr = read_mic(row["data_vrs"])
    except Exception as e:
        return {"subject": subj, "seq": seq, "status": "read_error", "err": repr(e)[:200]}
    mono = to_mono(x, downmix, channel)
    if src_sr != DST_SR:
        # 48000 -> 16000 is exactly /3; resample_poly handles the general case too.
        from math import gcd
        g = gcd(DST_SR, src_sr)
        mono = resample_poly(mono, DST_SR // g, src_sr // g)
    write_wav16(dst, mono, DST_SR)
    dur = len(mono) / DST_SR
    return {"subject": subj, "seq": seq, "status": "ok", "wav": str(dst),
            "src_sr": src_sr, "src_channels": int(x.shape[1]), "downmix": downmix,
            "channel": (channel if downmix == "ch" else None),
            "duration_s": round(dur, 2), "peak": round(float(np.abs(mono).max()), 4),
            "rms": round(float(np.sqrt(np.mean(mono ** 2))), 5),
            "sec_to_extract": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--manifest", default=None, help="defaults to <out>/manifest.jsonl")
    ap.add_argument("--downmix", choices=["ch", "mean"], default="ch")
    ap.add_argument("--channel", type=int, default=0, help="mic channel used when --downmix ch")
    ap.add_argument("--shard", default="0/1", help="i/N for slurm arrays")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subj:seq filters")
    args = ap.parse_args()
    out = Path(args.out)
    manifest = Path(args.manifest) if args.manifest else out / "manifest.jsonl"
    rows = [json.loads(l) for l in open(manifest)]
    if args.only:
        keep = set(args.only)
        rows = [r for r in rows if f"{r['subject']}:{r['seq']}" in keep]
    i, N = (int(x) for x in args.shard.split("/"))
    rows = [r for k, r in enumerate(rows) if k % N == i]
    if args.limit:
        rows = rows[:args.limit]
    log_dir = out / "logs" / "extract"; log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard_{i}_of_{N}.jsonl"
    n_ok = 0
    with open(log_path, "w") as lf:
        for r in rows:
            res = extract_one(r, out, args.downmix, args.channel, args.overwrite)
            n_ok += int(res["status"] in ("ok", "skip_exists"))
            lf.write(json.dumps(res) + "\n"); lf.flush()
            print(json.dumps(res))
    print(f"[shard {i}/{N}] {n_ok}/{len(rows)} ok, log -> {log_path}")


if __name__ == "__main__":
    main()
