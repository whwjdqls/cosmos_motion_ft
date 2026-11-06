# Yonsei Phase3: matched video-weight-one run without M2V captions

User-requested fresh ablation, 2026-11-06. The captioned run stopped cleanly
at step17114 and saved `checkpoints/recovery_latest.pt`; its files and
immutable source snapshot are preserved. Allocation2616204/node58, two
PRO6000s, is retained for this replacement. Phase2 no-shape remains separate.

Only the experiment identity and caption policy differ: M2V receives empty
text instead of dataset captions/dropout0.1. Caption dropout becomes0 because
all captions are empty. V2M/joint already used empty captions. Both frozen
specialists receive empty text for every task, enforced in data, noise and
model packing, with a real two-rank poisoned-caption acceptance check.

Unchanged: the same video-I/O Phase1EMA200k and supplied-shape Phase2
Self-cross caption-onlyB256200k, all112503 Nymeria cached windows, camera/head
statistics, seven bidirectional MMDiT bridges, groupedBF16, shape excluded
from bridges, independent clocks, T97/20fps/256px, task40/40/20, video weight1,
motion/contact losses1/1/5 and0.05/1/10, fresh200k, global32=2x16accum1,
AdamW2e-4/warmup1000/cosine floor0/seed0, no extra LoRA/EMA, save10k with
latest+100k retention, fixed71 validation every10k/UniPC35/CFG1. Validation
uses the same exact checksum-pinned configuration and GT references.

The launcher performs two-rank native acceptance, fresh4/reload2 disposable
probes, then starts fresh production automatically. Probe weights never
initialize production. All source/evaluator code is snapshotted. New checkout
validation uses the tested all-rank VAE initialization fix to avoid the old
rank-zero constructor collective mismatch.

- Run name: `edge_p3_mmdit_bidir_joint402020_groupedbf16_wv1_notext_T97_gbs32_yonsei_20261106`.
- Root: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/`.
- Operation: `/scratch2/whwjdqls99/cosmos/operations/phase3_mmdit_bidir_joint_wv1_notext_yonsei_20261106`.
- Tmux: `phase3_mmdit_wv1_notext_yonsei_20261106`.
- W&B entity/project: `jungbinc-upenn/cosmos-motion-ft`.

Read operation `RECIPE_PARITY.json`, run `launch/status.json`,
`launch/training.log`, `metrics.jsonl` and the production verification receipt
for actual progress. Preflight success alone is not production. Resume this
run only with its own immutable source snapshot, acceptance, data and
checkpoint; never reuse the captioned run's bridges or optimizer.

## Verified production startup

Around09:45KST on2026-11-06, all30 focused tests passed, followed by native
two-rank acceptance (including poisoned-caption invariance for every task at
both specialists, gradients, frozen specialists and exact save/reload),
fresh4 and resume4->6 probes. Actual production started at1 and reached20
with finite loss0.441415 and nonzero gradient0.638194. Observed update time
was~2.27seconds, peak allocated memory17.80GiB/GPU. These are early readings.

The real preflight/production contracts match the stopped run's frozen
specialists, dataset fingerprint and shared bridge implementation. The plan
comparison changes only experiment identity, M2V caption policy and dropout.
`PRODUCTION_RUNNING.json` records this verification and the live run:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d9fc2424b97b1163e3b2

The retained allocation has about50hours left at launch; the full200k run
will require a later resume. Preserve global32, the source snapshot, this
run's acceptance and full optimizer/RNG state when moving allocations.

## Optional EMA for new runs

The existing run remains EMA-off. Future launches may append
`--ema-decay 0.999 --validation-weights ema`, then use `--weights ema` for
offline sampling. Raw and EMA checkpoints are saved/restored together with
strict counters and recipe checks. See
[optional EMA](../motion_expert_phase3_edge_joint/OPTIONAL_EMA.md).
