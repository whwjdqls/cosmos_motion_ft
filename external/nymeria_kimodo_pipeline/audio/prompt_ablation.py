"""Ablate SAM-Audio speech-removal prompts / reranking on a single recording.

Separates the raw WAV with a given --desc (and --rerank), writes the residual to a
scratch dir, runs FireRedVAD on it, and reports how much speech remains and how much
non-speech (ambient) is preserved vs the raw. Lower residual_speech + high outside-RMS
ratio = better. Run several configs in parallel on different GPUs to compare.

Env: `audio`, GPU (CUDA_VISIBLE_DEVICES picks the device).
"""
from __future__ import annotations
import argparse, json, wave
from pathlib import Path
import numpy as np
from separate import SamAudioSeparator, write_wav16
from vad import FireRedBackend

OUT = Path("/weka/jungbin/nymeriaplus_audio")
SCRATCH = OUT / "ablation"


def load(p):
    w = wave.open(str(p)); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return x, sr


def rms_in(x, segs, sr, inside=True):
    mask = np.zeros(len(x), bool)
    for a, b in segs:
        mask[int(a * sr):int(b * sr)] = True
    sel = x[mask] if inside else x[~mask]
    return float(np.sqrt((sel ** 2).mean())) if sel.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--rerank", type=int, default=1)
    ap.add_argument("--predict-spans", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.desc.replace(' ', '_')}_r{args.rerank}{'_ps' if args.predict_spans else ''}"
    raw = OUT / "raw" / args.subject / f"{args.seq}.wav"
    vadj = json.load(open(OUT / "vad" / args.subject / f"{args.seq}.json"))
    segs = vadj["segments"]

    sep = SamAudioSeparator(description=args.desc, reranking_candidates=args.rerank,
                            predict_spans=args.predict_spans)
    res16, n_sep, n_pass = sep.residual(str(raw), segs=segs, vad_skip=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    outp = SCRATCH / f"{args.seq}__{tag}.wav"
    write_wav16(outp, res16, 16000)

    raw_x, sr = load(raw); n = min(len(raw_x), len(res16)); raw_x = raw_x[:n]; sup = res16[:n]
    vad = FireRedBackend(use_gpu=True)
    dur, ssegs = vad.detect(str(outp))
    resid_ratio = sum(e - s for s, e in ssegs) / dur if dur else 0.0
    rep = {"tag": tag, "desc": args.desc, "rerank": args.rerank,
           "orig_speech_ratio": vadj["speech_ratio"],
           "residual_speech_ratio": round(resid_ratio, 4),
           "speech_dur_reduction": round(1 - resid_ratio / max(vadj["speech_ratio"], 1e-9), 3),
           "inside_rms_ratio": round(rms_in(sup, segs, sr) / max(rms_in(raw_x, segs, sr), 1e-9), 3),
           "outside_rms_ratio": round(rms_in(sup, segs, sr, False) / max(rms_in(raw_x, segs, sr, False), 1e-9), 3),
           "n_sep": n_sep, "n_pass": n_pass, "wav": str(outp)}
    print("ABLATION " + json.dumps(rep))


if __name__ == "__main__":
    main()
