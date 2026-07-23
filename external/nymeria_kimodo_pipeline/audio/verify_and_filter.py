"""Stage 4 - verify speech was removed + drop silent/broken outputs -> cleaned set.

Runs over every <OUT>/speech_suppressed/{subj}/{seq}.wav and decides keep/drop:

  1. Second VAD pass (same backend as stage 2) on the speech-SUPPRESSED audio.
     residual_speech_ratio = speech_time / duration.  If it exceeds --max-residual-speech
     the suppression failed (speech leaked through) -> drop, reason "residual_speech".
     (Passthrough files - no speech to begin with - are expected to stay ~0.)
  2. Silent: full-clip RMS < --min-rms (dB-ish floor) -> drop, reason "silent".
  3. Broken: NaN/Inf, zero-length, shorter than --min-dur, or clipping fraction
     (|x|>0.999) above --max-clip-frac -> drop, reason "broken:*".

Output: <OUT>/cleaned.jsonl  - one row per recording:
    {subject, seq, wav, keep, reasons[], duration_s, rms,
     residual_speech_ratio, had_speech, status}
and <OUT>/cleaned_summary.json with counts.  Only `keep==true` rows feed training.

Env: `audio` (GPU recommended for the firered second pass).
"""
from __future__ import annotations
import argparse, json, wave
from pathlib import Path
import numpy as np
from vad import make_backend, read_wav_mono16k

OUT = Path("/weka/jungbin/nymeriaplus_audio")


def wav_stats(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    dur = len(x) / sr if sr else 0.0
    finite = np.isfinite(x).all()
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size and finite else 0.0
    clip_frac = float(np.mean(np.abs(x) > 0.999)) if x.size else 1.0
    return x, sr, dur, finite, rms, clip_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--backend", choices=["firered", "silero"], default="firered")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-residual-speech", type=float, default=0.25,
                    help="max speech fraction allowed to remain in suppressed audio "
                         "(empirically SAM-Audio leaves ~0.10-0.15 on speech-heavy clips; "
                         "tune from the cleaned_summary distribution)")
    ap.add_argument("--min-rms", type=float, default=1e-4)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-clip-frac", type=float, default=0.01)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)

    sep_dir = out / "separate"
    logs = [json.loads(p.read_text()) for p in sorted(sep_dir.rglob("*.json"))]
    if args.only:
        keep = set(args.only); logs = [r for r in logs if f"{r['subject']}:{r['seq']}" in keep]
    i, N = (int(x) for x in args.shard.split("/"))
    logs = [r for k, r in enumerate(logs) if k % N == i]

    backend = make_backend(args.backend, args.threshold, use_gpu=not args.cpu)
    results, kept = [], 0
    for r in logs:
        subj, seq = r["subject"], r["seq"]
        wav = out / "speech_suppressed" / subj / f"{seq}.wav"
        reasons = []
        if not wav.is_file():
            results.append({"subject": subj, "seq": seq, "wav": str(wav),
                            "keep": False, "reasons": ["missing"]}); continue
        x, sr, dur, finite, rms, clip_frac = wav_stats(str(wav))
        if not finite:
            reasons.append("broken:nonfinite")
        if dur < args.min_dur:
            reasons.append("broken:too_short")
        if clip_frac > args.max_clip_frac:
            reasons.append(f"broken:clipping_{clip_frac:.3f}")
        if rms < args.min_rms:
            reasons.append("silent")
        # second VAD pass (skip if already broken to save compute)
        resid_ratio = None
        if not reasons:
            d, segs = backend.detect(str(wav))
            resid_ratio = round(float(sum(e - s for s, e in segs)) / d, 4) if d > 0 else 0.0
            if resid_ratio > args.max_residual_speech:
                reasons.append(f"residual_speech_{resid_ratio:.3f}")
        keep = len(reasons) == 0
        kept += int(keep)
        results.append({"subject": subj, "seq": seq, "wav": str(wav), "keep": keep,
                        "reasons": reasons, "duration_s": round(dur, 2),
                        "rms": round(rms, 6), "clip_frac": round(clip_frac, 5),
                        "residual_speech_ratio": resid_ratio,
                        "had_speech": r.get("status") == "separated",
                        "status": r.get("status")})
        print(json.dumps({"subject": subj, "seq": seq, "keep": keep, "reasons": reasons}))

    cleaned = out / (f"cleaned.jsonl" if N == 1 else f"cleaned.shard{i}_of_{N}.jsonl")
    with open(cleaned, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    if N == 1:
        from collections import Counter
        rc = Counter(x for r in results for x in (r["reasons"] or ["kept"]))
        summary = {"total": len(results), "kept": kept,
                   "dropped": len(results) - kept, "reason_counts": dict(rc)}
        (out / "cleaned_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    print(f"[verify shard {i}/{N}] kept {kept}/{len(results)} -> {cleaned}")


if __name__ == "__main__":
    main()
