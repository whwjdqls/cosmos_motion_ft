# W&B display-name convention

Observed and matched to the current sep-1/sep-2 runs in
`jungbinc-upenn/cosmos-motion-ft` on2026-11-07:

```text
[server] phase architecture/direction | task mix | video weight | EMA | text | attention/precision | frames/global batch
```

Fields describe the actual run, rather than the current defaults in code.
For Phase3, `V2M40/M2V40/J20` means40/40/20 task sampling; `videoW1` is the
native video-loss weight. Use `EMAoff` when disabled; enabled experiments
include the decay and initialization step, distinguishing fresh EMA from
EMA introduced later. `NoText` means every task has empty dataset captions;
`M2VText` means only M2V retains its caption/dropout policy. Phase2 uses its
own task/data/shape fields instead of video weight/local bridge attention.

## Yonsei names verified on the server

### Current caption-free Phase3

`[yonsei] P3 MMDiT bidirectional | V2M40/M2V40/J20 | videoW1 | EMAoff | NoText | LocalAttn BF16 | T97 GBS32`

Run [p3-d9fc2424b97b1163e3b2](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d9fc2424b97b1163e3b2).
Operation: `phase3_mmdit_bidir_joint_wv1_notext_yonsei_20261106`.

### Stopped M2V-captioned Phase3

`[yonsei] P3 MMDiT bidirectional | V2M40/M2V40/J20 | videoW1 | EMAoff | M2VText | LocalAttn BF16 | T97 GBS32`

Run [p3-eed7a56cdcf72f3f4569](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-eed7a56cdcf72f3f4569).
Operation: `phase3_mmdit_bidir_joint_wv1_yonsei_20261105`. Stopped at17114;
last logged scalar step17110. Do not resume it automatically.

### Completed no-shape Phase2

`[yonsei] P2 SelfCross final-hidden KNorm | T2M75/TI2M25 | BONES50-in-T2M | NoShape | EMAoff | CaptionOnly | BF16 | T200/TI97 GBS256`

Run [cfj632ue](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/cfj632ue).
Operation: `phase2_no_shape_pro2_20261105`. W&B reports finished at200000.
BONES is50% within T2M; `T200` is the maximum variable T2M length, and TI2M
uses97 aligned frames.

## Resume and provenance

Only W&B `displayName` was updated, using a minimal metadata mutation.
Readback verified the names and unchanged run configuration, group, tags,
notes and state. Run IDs, metric history, output directories, checkpoint
paths, frozen source snapshots and GPU trainers are retained.

Each operation under `/scratch2/whwjdqls99/cosmos/operations/` now contains
`wandb_display_name.txt`. Use that exact value for the logging sidecar's
`--name` on future resumes, rather than the historical directory basename.
The running immutable sidecar has already been renamed on W&B and needs no
restart. Its frozen launch default still uses the old directory name, so
explicitly preserve the documented display name when starting a new logger.

Full before/after evidence, reference run names and verified metadata hashes:
`operations/wandb_run_naming_20261107/VERIFIED.json` on Scratch2.
