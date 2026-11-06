# Phase3 first-validation VAE broadcast stall and recovery

Status inspection on2026-11-06 found Phase2 no-shape healthy near45k, but the
new bidirectional joint/video1 Phase3 stopped advancing at its10k validation.
It had saved a complete checkpoint and generated all142 V2M/M2V predictions.
Rank0 then entered the native VAE loader's `sync_model_states` broadcast,
while rank1 was waiting for the final validation-result collective. This
collective mismatch spun the GPUs without making progress. It was not a
training-loss failure or a missing checkpoint.

## Fix and verification

The original runtime `validation_metrics.load_scoring_vae` constructs the fully loaded scoring
VAE on rank0 while suppressing only the tokenizer constructor's world
broadcast. The native loader already reads every VAE checkpoint tensor on
rank0. The helper rejects other ranks and restores the native synchronization
function even if construction fails. Training/DDP collectives are unchanged.

On owned2673623/node32, a real two-rank NCCL test loaded/decoded the same
T97 latent first without distributed initialization, then with rank1 waiting
for scoring results. Outputs matched **bit-for-bit**, and the result collective
completed. All12 validation regressions, including exception restoration and
rank rejection, passed on owned2675047 compute CPUs.

Evidence under
`/scratch2/whwjdqls99/cosmos/operations/phase3_validation_vae_sync_fix_20261106`:
`VAE_DDP_PARITY.json`, `probe.log`, `tests.log`.

## Existing run recovery

- Run: `edge_p3_mmdit_bidir_joint402020_groupedbf16_wv1_T97_gbs32_yonsei_20261105`.
- Retained allocation2616204/node58, two PRO6000s. Only stalled work
  step2616204.13 was killed; parent allocation and all other jobs retained.
- Checkpoint `step_000010000.pt`: step10000, next global draw320000,
  both rank RNG states,98 optimizer states/98 bridge tensors verified.
  SHA256 `7df1f378e6007bc033b175337f68c2d7063e2d3ef745a38c6fb2facdcd6d49d7`.
- Original training source snapshot, checkpoint, recipe, acceptance fingerprint
  `ac191642a3c995e566db01be0f8945f2df9781f8d6ec7179f5ea8938018a96b0`,
  global32, optimizer/schedule and normalization remain intact.
- For this already accepted run, `resume_bootstrap.py` loads the unchanged
  snapshot and applies only the tested scorer-constructor wrapper. This is an
  explicitly recorded runtime evaluation fix; it is **not** covered by the old
  source fingerprint. `RUNTIME_VALIDATION_FIX.json` pins the copied helper's
  SHA256 and parity evidence. Future checkpoints add this provenance under
  `runtime_validation_fix`; the existing10k checkpoint is not rewritten.
  No fingerprint check is disabled or forced to match a changed training source.
- Resume replays pending10k validation using existing predictions. No new
  sampling seeds, guidance, generation settings, losses or trained weights are
  introduced by the fix. Training RNG restoration remains in the inherited
  validation guard. After merging the independent upstream fix, new checkout
  launches initialize the VAE on all ranks before rank-zero scoring. The
  explicit legacy helper is now named `load_rank_zero_scoring_vae`; the
  live run keeps its original pinned runtime copy.
- New tmux `phase3_mmdit_joint_wv1_resume3_20261106`; active trainer log is
  `training.log` in this operation, **not** the old run's `launch/training.log`.
  `resume_worker.sh` restores the original snapshot-first PYTHONPATH and exact
  full recipe flags. Use this bootstrap for subsequent resumes of this run.
- The W&B sidecar resumes the same ID
  https://wandb.ai/scene_motion/cosmos-motion-ft/runs/p3-eed7a56cdcf72f3f4569
  and same append-only metrics history; completed validation is idempotent.

The first recovery wrapper lacked a `__main__` guard, causing spawned data
workers to re-enter training and contend for its writer lock. It exited before
any new optimizer update. The corrected wrapper passed an actual
`runpy(..., run_name='__mp_main__')` import check on compute. Attempt1 logs
are archived with `.attempt1`; the second attempt uses the same intact10k
checkpoint and adds no training changes.

The next attempt correctly restored10k and reached scoring, where the LPIPS
guard rejected a Torch Hub directory differing from the pinned configuration.
The original launcher had re-sourced `restored_env.sh`, selecting the Lustre
Torch cache; the manual resume wrapper had retained the operation's cache.
The wrapper now explicitly restores `$PROJECT_ROOT/.cache/torch` and checks
the exact pinned AlexNet path before model loading. No checksum/path guard
was disabled and no weights were downloaded or changed. Attempt2 logs are
preserved separately; attempt3 is the active recovery.

Consult the operation's live logs, the run's `metrics.jsonl` and validation
`COMPLETE.json`, and `RESUMED_RUNNING.json` for actual recovery progress.
Reload initiation or the parity test alone is not a claim that production
updates after10000 have resumed.

Recovery completed around2026-11-06 04:53KST: all71V2M and71M2V cases were
scored, the single10k validation event was appended, and actual optimizer
updates10001,10010 and10020 completed with finite losses/gradients at~2.3s
per update. Video weight1, global32, task mix and LR schedule are retained.
The operation's `RESUMED_RUNNING.json` checks full validation coverage,
idempotent event identity, real optimizer progress and the same remote W&B
run's training/validation axes. The original10k checkpoint remained intact.

For the current W&B destination and future-resume wrapper, see
[migration/WANDB_DESTINATION_20261106.md](../migration/WANDB_DESTINATION_20261106.md).
Earlier entity/worker references above describe the initial recovery.
