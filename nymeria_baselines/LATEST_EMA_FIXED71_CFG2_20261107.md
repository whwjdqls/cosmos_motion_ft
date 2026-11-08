# sep-2 latest EMA checkpoints: fixed71 CFG2

Completed all426 predictions:71 V2M and71 M2V for each of three caption-free40/40/20 MMDiT models. The exact original held-out71 windows, case order, hashes and seeds are retained; no new windows or replacement samples.

[Full quantitative tables,12 comparison videos and evidence ZIP](https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b). The latest-results toggle preserves the earlier10k report. All28 published table rows match the local report; all13 uploaded files were downloaded and SHA256-verified.

| Model | Step | EMA updates | M2V PSNR dB | M2V SSIM | LPIPS | DreamSim | I3D-FVD | CD-FVD | V2M MPJPE cm | TMR FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bidirectional | 53491 | 3474 | 18.561240 | 0.566123 | 0.267132 | 0.158750 | 336.130361 | 316.938463 | 27.219767 | 0.351295 |
| directional | 60000 | 9181 | 18.603965 | 0.567953 | 0.270084 | 0.159398 | 330.206098 | 327.177524 | 25.014273 | 0.332573 |
| directional_video1 | 20000 | 20000 | 18.037092 | 0.546211 | 0.286802 | 0.163239 | 353.949858 | 318.780714 | 23.425625 | 0.289277 |

These are latest available snapshots at selection, with different training ages. Do not interpret these point estimates as a matched-step architecture or EMA ablation. Full physical/reconstruction/jerk/retrieval metrics, measured GT rows and paired5000-bootstrap intervals are in the report/archive.

All sampling uses EMA bridge weights, the same frozen Phase1 video-I/O EMA200k and Phase2 200k specialists, empty captions, provided neutral shape, native35-step UniPC, motion shift3/video shift10, T97/20fps/256. M2V keeps its one initial image latent clamped and conditions on clean motion; V2M conditions on full clean video. At CFG2 the reference disables all seven bridges, including within-modality updates. This is whole-bridge extrapolation, distinct from sep-1 edge-only references.

Native CFG1 single-case sampling parity passed at bit precision for all six model/task combinations before CFG2. Benchmarks16/32/64 selected32 for B V2M and64 for every other task. Per-case seeds remain independent of batching. Bit-exactness is not claimed between different batch shapes.

Video scores exclude frame0 and use lossless decoded future frames1–96. Pooled71 FVD uses population covariance and the stable low-rank nuclear-norm trace. This differs from periodic426-descriptor monitoring. I3D features use canonical case order/chunks4/pairs2 for every model; both GT I3D and VideoMAE features match exactly. B I3D was re-extracted from unchanged pixels after shard grouping produced small roundoff differences; no samples were regenerated. TMR evaluator91f9d8adec6bcaa4… and all eight motion-statistics files reproduce saved GT embeddings exactly. Reference text scores retrieval only.

Six qualitative cases0/14/28/42/56/70 were fixed before viewing outputs. Each M2V compares GT+B+D10+D1; V2M adds observed RGB and uses the same GT virtual camera/scale with raw motion, without extra alignment, smoothing or grounding. Every video decodes97frames at20fps. The ZIP includes all213 normalized generated motion arrays, per-case metrics, embeddings/ranks, FVD features, source identities and evaluator code.

User explicitly authorized temporary sharing of owned GPUs1/3 and a brief bidirectional save/restart to obtain current EMA. B saved53491 and strictly resumed with unchanged native fingerprint, model/EMA/98 Adam states, scheduler, two-rank RNG and cursor; its original W&B ID/display name remain. All three trainers continued and all evaluation GPU processes have exited. Current B manager is this operation’s checkpoint_bidirectional.py; read restart_bidirectional/status.json and run/launch/status.json for the live PID.

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_latest_ema_fixed71_cfg2_20261107T014607Z`.

Archive: latest_ema_fixed71_cfg2_metrics.zip,16519185bytes, SHA256 `00bdc4d1de87bcbdabd4f61e561ca5c34fc49d7456d0e540377751966d12bf62`.

Read COMPLETE.json, ARTIFACTS_VALIDATED.json, I3D_CANONICAL_PAIRING_VERIFIED.json, TRAINING_CONTINUES_VERIFIED.json and notion_publication.json for evidence. Original launch/scoring failures and pre-canonical features remain under attempts/; no source fingerprint or metric guard was relaxed.
