"""Stage 3 - speech separation; keep the RESIDUAL as the speech-suppressed track.

Goal: training audio that captures activity / environment sound with the wearer's
(and bystanders') speech removed - both for privacy and because speech is not
correlated with body motion.

For each recording that VAD flagged `has_speech`:
  - run SAM-Audio (Meta "Segment Anything in Audio") with a speech text prompt,
  - `result.target[0]`   = the isolated speech (discarded, target prompt = "speech"),
  - `result.residual[0]` = mixture minus speech = the speech-SUPPRESSED audio we keep.
Recordings with no detected speech are passed through (the raw WAV is already clean).

WINDOWED inference: SAM-Audio resamples to its native **48 kHz** internally, so a whole
18-min clip is ~52 M samples and OOMs (one forward tried to alloc 87 GiB). We therefore
process the clip in `--chunk-sec` windows (default 20 s) with `--overlap-sec` overlap and
a linear crossfade overlap-add. Windows that don't overlap any VAD speech span are passed
through unchanged (no speech to remove) unless `--no-vad-skip`.

Bandwidth note: our `raw/` WAVs are 16 kHz (the VAD front-end rate), so SAM-Audio sees
16 kHz upsampled to 48 kHz - there is no real >8 kHz content, and we store the residual
back at 16 kHz for consistency with the rest of the pipeline. For a true full-band result,
re-extract at 48 kHz (extract_audio.py with DST_SR=48000) and feed that here.

SAM-Audio API (facebookresearch/sam-audio; gated weights facebook/sam-audio-large):
    from sam_audio import SAMAudio, SAMAudioProcessor
    model = SAMAudio.from_pretrained(ID).eval().cuda(); proc = SAMAudioProcessor.from_pretrained(ID)
    batch = proc(audios=[wav_tensor_or_path], descriptions=["speech"]).to("cuda")
    result = model.separate(batch, predict_spans=False, reranking_candidates=1)
    sr = proc.audio_sampling_rate            # 48000 ; result.residual[0] is a 1-D wav

Output: <OUT>/speech_suppressed/{subject}/{seq}.wav   (mono 16 kHz)
        <OUT>/separate/{subject}/{seq}.json            (status + provenance)
Resumable / shardable.  Env: `audio`, GPU required.
"""
from __future__ import annotations
import argparse, json, os, wave, shutil
from pathlib import Path
import numpy as np

OUT = Path("/weka/jungbin/nymeriaplus_audio")
SAM_ID = os.environ.get("SAM_AUDIO_ID", "facebook/sam-audio-large")
STORE_SR = 16000


def read_wav_mono(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def write_wav16(path: Path, mono: np.ndarray, sr: int):
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def _window_has_speech(start, end, segs, sr):
    """segs are [start_s,end_s] in seconds; start/end are sample indices at sr."""
    s0, e0 = start / sr, end / sr
    for a, b in segs:
        if b > s0 and a < e0:
            return True
    return False


class SamAudioSeparator:
    def __init__(self, model_id=SAM_ID, description="speech", predict_spans=False,
                 reranking_candidates=1, chunk_sec=20.0, overlap_sec=1.0):
        import torch, torchaudio
        from sam_audio import SAMAudio, SAMAudioProcessor
        self.torch = torch; self.ta = torchaudio
        self.model = SAMAudio.from_pretrained(model_id).eval().cuda()
        self.proc = SAMAudioProcessor.from_pretrained(model_id)
        self.sr = int(self.proc.audio_sampling_rate)          # 48000
        self.description = description
        self.predict_spans = predict_spans
        self.reranking_candidates = reranking_candidates
        self.chunk = int(chunk_sec * self.sr)
        self.overlap = int(overlap_sec * self.sr)
        self.hop = max(1, self.chunk - self.overlap)

    def _sep_chunk(self, seg):                                # seg: (1,L) tensor @48k
        batch = self.proc(audios=[seg], descriptions=[self.description]).to("cuda")
        with self.torch.inference_mode():
            res = self.model.separate(batch, predict_spans=self.predict_spans,
                                      reranking_candidates=self.reranking_candidates)
        return res.residual[0].detach().float().cpu()         # (L',)

    def residual(self, wav_path, segs=None, vad_skip=True):
        torch = self.torch
        x, sr0 = read_wav_mono(wav_path)
        wav = torch.from_numpy(x)[None]                       # (1,T0)
        wav48 = self.ta.functional.resample(wav, sr0, self.sr) if sr0 != self.sr else wav
        T = wav48.shape[1]
        out = torch.zeros(T); wsum = torch.zeros(T)
        ramp = torch.linspace(0, 1, self.overlap) if self.overlap > 0 else None
        n_sep = n_pass = 0
        start = 0
        while start < T:
            end = min(start + self.chunk, T)
            seg = wav48[:, start:end]
            L = end - start
            if vad_skip and segs is not None and not _window_has_speech(start, end, segs, self.sr):
                res = seg[0].clone(); n_pass += 1
            else:
                r = self._sep_chunk(seg); n_sep += 1
                if r.shape[0] >= L:                           # align to window length
                    res = r[:L]
                else:
                    res = torch.zeros(L); res[:r.shape[0]] = r
            win = torch.ones(L)
            if ramp is not None:
                k = min(self.overlap, L)
                win[:k] *= ramp[:k]; win[L - k:] *= ramp[:k].flip(0)
            out[start:end] += res * win; wsum[start:end] += win
            if end >= T:
                break
            start += self.hop
        out = out / wsum.clamp_min(1e-6)
        out16 = self.ta.functional.resample(out[None], self.sr, STORE_SR)[0].numpy()
        return out16, n_sep, n_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--description", default="speech")
    ap.add_argument("--predict-spans", action="store_true")
    ap.add_argument("--reranking", type=int, default=1)
    ap.add_argument("--chunk-sec", type=float, default=20.0)
    ap.add_argument("--overlap-sec", type=float, default=1.0)
    ap.add_argument("--no-vad-skip", action="store_true",
                    help="separate every window even if VAD found no speech in it")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)

    rows = [json.loads(p.read_text()) for p in sorted((out / "vad").rglob("*.json"))]
    if args.only:
        keep = set(args.only); rows = [r for r in rows if f"{r['subject']}:{r['seq']}" in keep]
    i, N = (int(x) for x in args.shard.split("/"))
    rows = [r for k, r in enumerate(rows) if k % N == i]
    if args.limit:
        rows = rows[:args.limit]

    sep = None
    n_done = 0
    for r in rows:
        subj, seq = r["subject"], r["seq"]
        raw = out / "raw" / subj / f"{seq}.wav"
        dst = out / "speech_suppressed" / subj / f"{seq}.wav"
        logp = out / "separate" / subj / f"{seq}.json"
        if dst.is_file() and not args.overwrite:
            n_done += 1; continue
        logp.parent.mkdir(parents=True, exist_ok=True)
        if not r.get("has_speech"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw, dst)
            log = {"subject": subj, "seq": seq, "status": "passthrough", "model": None,
                   "out_sr": STORE_SR, "out": str(dst)}
        else:
            if sep is None:
                sep = SamAudioSeparator(description=args.description,
                                        predict_spans=args.predict_spans,
                                        reranking_candidates=args.reranking,
                                        chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec)
            res16, n_sep, n_pass = sep.residual(str(raw), segs=r.get("segments"),
                                                vad_skip=not args.no_vad_skip)
            write_wav16(dst, res16, STORE_SR)
            log = {"subject": subj, "seq": seq, "status": "separated", "model": SAM_ID,
                   "description": args.description, "reranking": args.reranking,
                   "out_sr": STORE_SR, "out": str(dst),
                   "n_windows_separated": n_sep, "n_windows_passthrough": n_pass,
                   "chunk_sec": args.chunk_sec}
        logp.write_text(json.dumps(log))
        n_done += 1
        print(json.dumps({k: log.get(k) for k in ("subject", "seq", "status",
                                                  "n_windows_separated", "n_windows_passthrough")}))
    print(f"[separate shard {i}/{N}] {n_done}/{len(rows)} done")


if __name__ == "__main__":
    main()
