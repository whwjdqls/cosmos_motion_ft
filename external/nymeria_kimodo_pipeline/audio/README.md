# NymeriaPlus audio preprocessing

Turn the raw egocentric Aria audio in NymeriaPlus into **speech-suppressed** training
audio: keep the activity / environment sound, remove speech (privacy + speech is not
correlated with body motion).

```
Aria VRS mic (231-1, 7ch @ 48 kHz)
  └─ Stage 1 extract ────────► raw/{Sxx}/{seq}.wav            (mono 16 kHz 16-bit)
        └─ Stage 2 VAD (FireRedVAD) ─► vad/{Sxx}/{seq}.json   (has_speech, segments)
              └─ Stage 3 SAM-Audio separate ─► speech_suppressed/{Sxx}/{seq}.wav
                    │   has_speech → keep SAM-Audio RESIDUAL (mix − speech)
                    │   no speech  → passthrough (raw is already clean)
                    └─ Stage 4 verify + filter ─► cleaned.jsonl  (keep/drop + reasons)
                          2nd VAD pass (speech leaked?) + silent/broken removal
```

All paths live under **`/weka/jungbin/nymeriaplus_audio/`** (output root, created by stage 0).
Code lives here in `/home/jungbin_cho/nymeria_kimodo_pipeline/audio/`.

## Data facts (verified 2026-06-16)
- **760** head recordings enumerated, **736** have `recording_head/data/data.vrs`.
- Audio is the Aria **`231-1/mic`** stream inside `data.vrs`: **7 channels, 48 kHz, 32-bit
  PCM** (stored as int64, full-scale `2**31`), delivered in 2048-frame blocks.
- The standalone `recording_head/data/audio.vrs` is **not** used: the high-level
  projectaria provider refuses to open an audio-only VRS (`No stream activated`). We read
  the mic stream out of `data.vrs` by random access, so only the ~1 GB of audio records
  are touched, not the 16 GB of images.
- kirk_flowers reference clip: 1090.7 s → 17,451,003 samples @ 16 kHz, peak 0.51, RMS 0.012.

## Environments
- **Stage 1** → existing **`nymeria_plus`** env (projectaria_tools + scipy + numpy). CPU.
- **Stages 2–4** → new **`audio`** env. `bash setup_env.sh` creates it (python 3.11,
  torch+torchaudio cu128, soundfile, `fireredvad`, `sam-audio`, huggingface_hub).
- Cluster: partition **`a3ultra`** (4 nodes × 8 H200). Stage 1 is CPU-only but the only
  partition is GPU; either let it idle a GPU or run stage 1 locally (see below).

## Model setup
- **FireRedVAD** (`github.com/FireRedTeam/FireRedVAD`, HF `FireRedTeam/FireRedVAD`):
  `pip install fireredvad` + `huggingface-cli download FireRedTeam/FireRedVAD`.
  Wants 16 kHz 16-bit mono PCM — exactly what stage 1 writes. `vad.py` reads weights from
  `$FIRERED_VAD_DIR` (default `/home/jungbin_cho/audio_models/FireRedVAD/VAD`).
- **SAM-Audio** (`github.com/facebookresearch/sam-audio`, HF `facebook/sam-audio-large`):
  **gated** — request access on the HF page, then `hf auth login`. `pip install -e` the repo.
  We prompt the separator with the text description **`"speech"`** and keep `result.residual`.
- **Silero VAD** fallback (`vad.py --backend silero`) needs only torch + torchaudio
  (its torch.hub code imports torchaudio), so it also runs in the `audio` env — use it to
  exercise the pipeline before FireRedVAD weights are downloaded.

## Run order (32-way slurm arrays)
```bash
# stage 0 — manifest (login node, no deps)
python build_manifest.py

# stage 1 — extract (nymeria_plus). ~122 s/recording; 32 shards ≈ 45 min.
sbatch --array=0-31 --export=ALL,STAGE=extract,N=32 pipeline.sbatch
#   or locally without slurm:
#   for i in $(seq 0 31); do python extract_audio.py --shard $i/32 & done; wait

# stage 2 — VAD (audio env, FireRedVAD)
sbatch --array=0-31 --export=ALL,STAGE=vad,N=32 pipeline.sbatch

# stage 3 — SAM-Audio separation (audio env, gated weights + hf auth login)
sbatch --array=0-31 --export=ALL,STAGE=separate,N=32 pipeline.sbatch

# stage 4 — verify + filter. Run N=1 so it writes the merged cleaned.jsonl + summary.
sbatch --array=0-0 --export=ALL,STAGE=verify,N=1 pipeline.sbatch
```
Every stage is **resumable** (skips existing outputs) and **shardable**
(`--shard i/N`, row/file index mod N). Quick smoke test on one recording:
`--only S02:20231006_s1_kirk_flowers_act0_hfjvo9`.

## Outputs / schema
- `manifest.jsonl` — `{subject, seq, data_vrs, has_data_vrs}`
- `raw/{Sxx}/{seq}.wav` — mono 16 kHz 16-bit PCM. Default mono = mic **channel 0**
  (`extract_audio.py --downmix mean` averages the 7 mics; channel 0 is the default to
  avoid comb-filtering off-axis sources when summing spatially-separated mics).
- `vad/{Sxx}/{seq}.json` — `{has_speech, n_segments, total_speech_s, speech_ratio,
  segments:[[start_s,end_s],…], backend, duration_s}`
- `speech_suppressed/{Sxx}/{seq}.wav` — the kept track (SAM-Audio residual, or raw
  passthrough when no speech). `separate/{Sxx}/{seq}.json` records provenance
  (`status` = `separated` | `passthrough`, model id, output sample rate).
- `cleaned.jsonl` — final gate, one row/recording:
  `{subject, seq, wav, keep, reasons[], duration_s, rms, clip_frac,
    residual_speech_ratio, had_speech, status}`. `cleaned_summary.json` has counts.
  **Training consumes only `keep==true` rows.**

### Filter thresholds (stage 4, all tunable)
- `--max-residual-speech 0.02` — drop if >2 % of the suppressed audio still reads as
  speech in the 2nd VAD pass (suppression failed / speech leaked).
- `--min-rms 1e-4` — drop near-silent clips.
- `--min-dur 1.0` — drop clips shorter than 1 s.
- `--max-clip-frac 0.01` — drop clips with >1 % samples at full scale (broken/clipped).
- NaN/Inf → `broken:nonfinite`.

## Status (2026-06-16)
- **Stage 0 manifest:** ✅ 760 recordings, 736 with head `data.vrs`.
- **Stage 1 extract:** ✅ **736/736** WAVs, 0 read errors, 225.9 h total, 25 GB.
- **Stage 2 VAD (FireRedVAD):** ✅ **736/736** on node 1 (8 GPUs, ~3 min).
  **735 have speech** (99.9 %; mean speech_ratio 0.189, ~138 segments/rec); 1 no-speech
  (`S06/…barbara_norman_act2_h2bvu0`). So stage 3 must run on ~735 recordings.
- **Stage 3 SAM-Audio separate:** ✅ **736/736** (735 separated + 1 passthrough) on node 1
  (8 GPUs) using `facebook/sam-audio-large`, **prompt `"speech"`, `reranking_candidates=3`**.
  Windowed inference (20 s chunks, 1 s overlap-add) avoids the full-clip OOM; VAD-guided
  window skipping made ~34 % of windows passthrough. Residual stored at 16 kHz mono.
  (A prompt/reranking ablation — `prompt_ablation.py` — found `"speech"` beats
  `person speaking`/`voice`/`human voice`, and rerank 3 is the main quality lever; SAM-Audio
  is stochastic per run, so reranking picks the best of N candidates.)
- **Stage 4 verify+filter:** ✅ **736/736 kept** → `cleaned.jsonl` + `cleaned_summary.json`.
  Second FireRedVAD pass: residual_speech_ratio mean **0.035** / median 0.026 / p99 0.154 /
  max 0.213 (~**78–81 %** of detected speech duration removed). 0 dropped at the
  `--max-residual-speech 0.25` gate. (rerank-1 was mean 0.055 / median 0.045 / max 0.253.)

**PIPELINE COMPLETE (rerank-3, 2026-06-18):** training set = the **736** `keep==true` rows in
`cleaned.jsonl` → `speech_suppressed/{Sxx}/{seq}.wav` (16 kHz mono, speech removed).

### Env build notes (folded into setup_env.sh)
- `audio` env: torch/torchaudio **2.11.0+cu128**, fireredvad, sam_audio 0.1.0, hf_hub 1.19.
- Two repairs were needed after `pip install -e sam-audio` (now scripted):
  (1) reinstall **torchvision==0.26.0+cu128** (its deps pulled a cu130 build → CUDA-major
  mismatch); (2) patch `sam_audio/model/base.py` for huggingface_hub 1.x (make
  `proxies`/`resume_download` optional, drop the removed `resume_download` snapshot arg).
- FireRedVAD weights → `/home/jungbin_cho/audio_models/FireRedVAD/VAD` (`model.pth.tar`+`cmvn.ark`).
