"""Stage 2 - voice-activity detection over the extracted 16 kHz mono WAVs.

Decides, per recording, whether speech is present and where (segment timestamps).
Downstream `separate.py` only bothers running speech separation on recordings that
actually contain speech.

Backends (`--backend`):
  firered : FireRedVAD (default, SOTA DFSMN VAD/AED).  pip install fireredvad ;
            weights `FireRedTeam/FireRedVAD` (see README "Model setup").  Wants
            16 kHz 16-bit mono PCM - exactly what extract_audio.py writes.
            API: FireRedVad.from_pretrained(dir, FireRedVadConfig(...)).detect(wav)
                 -> ({'dur','timestamps':[(s,e),...],'wav_path'}, probs)
  silero  : Silero VAD via torch.hub - CPU-friendly fallback so the pipeline runs
            before FireRedVAD weights are in place.  Needs torch + torchaudio.

Output: <OUT>/vad/{subject}/{seq}.json
    {subject, seq, backend, sr, duration_s, has_speech, n_segments,
     total_speech_s, speech_ratio, segments:[[start_s,end_s],...]}

Resumable (skips existing json) and shardable (--shard i/N) for slurm arrays.
Env: `audio` (GPU recommended for firered).
"""
from __future__ import annotations
import argparse, json, os, wave
from pathlib import Path
import numpy as np

OUT = Path("/weka/jungbin/nymeriaplus_audio")
FIRERED_DIR = os.environ.get("FIRERED_VAD_DIR",
                             "/home/jungbin_cho/audio_models/FireRedVAD/VAD")


def read_wav_mono16k(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sr


def segments_summary(subject, seq, backend, sr, dur, segs):
    total = float(sum(e - s for s, e in segs))
    return {"subject": subject, "seq": seq, "backend": backend, "sr": sr,
            "duration_s": round(dur, 2), "has_speech": len(segs) > 0,
            "n_segments": len(segs), "total_speech_s": round(total, 2),
            "speech_ratio": round(total / dur, 4) if dur > 0 else 0.0,
            "segments": [[round(float(s), 3), round(float(e), 3)] for s, e in segs]}


# ----------------------------------------------------------------------------- backends
class FireRedBackend:
    name = "firered"

    def __init__(self, threshold=0.4, use_gpu=True):
        from fireredvad import FireRedVad, FireRedVadConfig
        cfg = FireRedVadConfig(use_gpu=use_gpu, smooth_window_size=5,
                               speech_threshold=threshold, min_speech_frame=20,
                               max_speech_frame=2000, min_silence_frame=20,
                               merge_silence_frame=0, extend_speech_frame=0,
                               chunk_max_frame=30000)
        self.vad = FireRedVad.from_pretrained(FIRERED_DIR, cfg)

    def detect(self, wav_path):
        result, _ = self.vad.detect(wav_path)
        segs = [(float(s), float(e)) for s, e in result["timestamps"]]
        return float(result["dur"]), segs


class SileroBackend:
    name = "silero"

    def __init__(self, threshold=0.5, use_gpu=False):
        import torch
        self.torch = torch
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True, onnx=False)
        self.model = model
        self.get_ts = utils[0]                      # get_speech_timestamps
        self.threshold = threshold

    def detect(self, wav_path):
        pcm, sr = read_wav_mono16k(wav_path)
        t = self.torch.from_numpy(pcm)
        ts = self.get_ts(t, self.model, sampling_rate=sr, threshold=self.threshold)
        segs = [(d["start"] / sr, d["end"] / sr) for d in ts]
        return len(pcm) / sr, segs


def make_backend(name, threshold, use_gpu):
    if name == "firered":
        return FireRedBackend(threshold=threshold if threshold is not None else 0.4, use_gpu=use_gpu)
    if name == "silero":
        return SileroBackend(threshold=threshold if threshold is not None else 0.5)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--backend", choices=["firered", "silero"], default="firered")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", help="force CPU (firered)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    manifest = Path(args.manifest) if args.manifest else out / "manifest.jsonl"
    rows = [json.loads(l) for l in open(manifest)]
    if args.only:
        keep = set(args.only); rows = [r for r in rows if f"{r['subject']}:{r['seq']}" in keep]
    i, N = (int(x) for x in args.shard.split("/"))
    rows = [r for k, r in enumerate(rows) if k % N == i]
    if args.limit:
        rows = rows[:args.limit]

    backend = make_backend(args.backend, args.threshold, use_gpu=not args.cpu)
    n_ok = 0
    for r in rows:
        subj, seq = r["subject"], r["seq"]
        wav = out / "raw" / subj / f"{seq}.wav"
        dst = out / "vad" / subj / f"{seq}.json"
        if dst.is_file() and not args.overwrite:
            n_ok += 1; continue
        if not wav.is_file():
            print(json.dumps({"subject": subj, "seq": seq, "status": "no_wav"})); continue
        dur, segs = backend.detect(str(wav))
        rec = segments_summary(subj, seq, backend.name, 16000, dur, segs)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rec))
        n_ok += 1
        print(json.dumps({k: rec[k] for k in ("subject", "seq", "has_speech",
                                              "n_segments", "speech_ratio")}))
    print(f"[vad {args.backend} shard {i}/{N}] {n_ok}/{len(rows)} done")


if __name__ == "__main__":
    main()
