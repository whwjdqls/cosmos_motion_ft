# sep-2 joint MMDiT validation recovery

Both 40/40/20 MMDiT experiments resumed from full step10,000 checkpoints on
2026-11-05. Bidirectional uses GPUs0/3; directional uses GPUs1/7. At19:28UTC
they reached10120/10130 with finite losses and gradients, about1.34–1.36s/update.
This is a historical verification snapshot; inspect live run metrics for status.

The first validation sampled all71 V2M and71 M2V predictions per arm, then
failed because rank0 loaded a Wan VAE that performs distributed broadcasts
while rank1 waited at the scoring-result collective. The reviewed fix in
master416b2ea initializes the VAE on both ranks before rank0 scores. That
native two-GPU path passed full cached validation for both arms before training
was resumed. No cached predictions were regenerated; all284 files retained
their SHA256 values.

Recovery exports preserve347 of349 original Python files byte-for-byte and
replace only core `validation.py` and `validation_metrics.py`. Original sources
and full checkpoints remain untouched. The documented checkpoint migration
changes the two implementation hashes and adds provenance; recursive equality
checks verified all98 bridge tensors,98 Adam states, two rank RNG states,
scheduler and data cursor. Native strict checkpoint fingerprint checks remain
enabled and passed. Both runs advanced past10,100, and their original W&B IDs
contain the completed10k validation and subsequent optimizer steps.

The recipe remains global32=2×16, accumulation1, 40% V2M/40% M2V/20% joint,
frozen Phase1 video-I/O EMA200k and Phase2 200k, seven grouped-BF16 MMDiT
bridges, clean E(0), video weight10, no additional Phase3 LoRA/EMA. Validation
remains fixed71 every10k, native UniPC35/CFG1, sampling32 and decode4.

Operation root on sep-2 local NVMe:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_mmdit_resume_validationfix_20261105`.
`plan.json` records exact original run paths, source snapshots, derived resume
checkpoints and source fingerprints. Per-arm `MIGRATION.json`,
`GPU_VALIDATION_COMPLETE.json` and `STRICT_RESUME_VERIFIED.json` record the
native checks. Root `RUNNING_VERIFIED.json` and `REMOTE_WANDB_VERIFIED.json`
record subsequent live progress. `COMPLETE.json` means recovery completed,
not that200k training finished. Detached `run_arm.py` controllers keep GPU
leases and supervise the original W&B loggers.

Do not rerun the initial recovery helper blindly: validation identities have
already migrated. Future native resumes must use the latest full checkpoint
with its matching source snapshot and contract. The older50/50 experiments
remain stopped with their preserved recovery states.

[Notion report](https://app.notion.com/p/40fd2a44f44b814fb925c3d86c250284)
contains both original W&B links and the recovery status.
