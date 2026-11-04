# Camera prediction with video loss weight one

Fresh controlled variant of the existing bridges-only camera-prediction run.
Only `video_weight` changes from10 to1; the experiment identity is distinct.
The factor applies to native video velocity loss in both M2V and joint samples.
All motion reconstruction/physical weights, including foot height10, are retained.
Camera loss remains absent.

The original packages are reused without modifying their files. Explicit,
source-guarded adapters change the two native video multipliers, the trainer's
reported weighted-video metric and the acceptance label. Contracts include the
original source fingerprints plus this package and prevent resuming weight10
checkpoints into this experiment.

The launcher compares the complete original camera recipe/source contract,
allowing only experiment identity and video weight to differ. Use its existing
fresh two-GPU/microbatch16 native parity/freeze/backward acceptance, disposable
four-update/save/reload/two-update probes, then fresh production at step0.
Checkpoint/probe identities and the200k schedule remain strict. Do not initialize
production from disposable probes.

Same Phase1 video-I/O EMA200k and selected Phase2 200k, same local112503-window
cohort/camera cache,40/40/20 tasks, camera noising/attention, captions, global32,
AdamW2e-4/betas0.9,0.99, clip1,1000-step warmup then cosine through200k, seed0,
BF16 model/FP32 motion I/O and bridges, TF32 off,10k saves and W&B streaming.

Entrypoints: `run_foundary.sh`, `train.py`, and `sample.py`. Launch with
`--reference-contract` pointing to the original weight10 camera run's contract.
All data, checkpoints and caches must be node-local. The normal GPU locks and
empty-GPU checks remain in force. User requested the interim40k evaluation and
Notion publication complete before this experiment starts; consult the local
operation and live receipts for launch status.
