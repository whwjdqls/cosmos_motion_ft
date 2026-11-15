# Separate100k Phase3 checkpoint preservation, sep-1

User explicitly asks to preserve100k files separately because rolling training
checkpoints may be deleted. Protected node-local NVMe archive:

`/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108`.

All three saved: `sep1_dir5050_w10_lateEMA/step_000100000.pt`,
`sep1_bidir5050_w10_lateEMA/step_000100000.pt` and
`sep1_dir5050_w1_EMA0/step_000100000.pt`. These are independent copied
files, not hardlinks. Different source/destination inodes and linkcount1 were
checked; SHA256 matches the original. Per-model PRESERVED.json records hashes,
full100k state, source/model identities and copied immutable source snapshots.
Raw trainables, EMA, fullAdamW/scheduler state, two rank RNGs and draw3200000
are present. Common frozenPhase1/2 weights remain local at recorded paths.

W1 reached100k and its exact native checkpoint was independently copied and
verified at2026-11-08 05:42UTC. RootCOMPLETE, STATUS and all3 PRESERVED.json
receipts confirm completion. The watcher and detached tmux session exited
normally. WATCHER.json records historical process information, not a live job.
Copying used a new temporary file, checksum verification/fsync and atomic rename.

The completed watcher had no GPU visibility and did not modify training.
No shared storage has been written. Preserve this directory during cleanup;
it is outside all trainer checkpoint rotation folders.

## All retained sep-1 variants: additional camera run armed (2026-11-08)

User reiterates preserving exact100k for ALL runs. The original three files
above remain independent copies; the new CPU-only watcher recomputed their
full-file SHA256 and checked linkcount1/size against PRESERVED receipts.

The new fourth retained run, `sep1_bidir5050_w1_camera_nocamloss_EMA0`, has not
reached100k. Its immutable live trainer already saves every10k and explicitly
exempts `step_000100000.pt` from rotation. An independent verified copy is now
armed for:

`/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108/sep1_bidir5050_w1_camera_nocamloss_EMA0/step_000100000.pt`.

Controller operation:
`/mnt/local/jinhyung.kim/cosmos/operations/preserve_all_phase3_100k_sep1_20261108T234158Z`.
Tmux session `phase3_100k_sep1_all_20261108`, CUDA_VISIBLE_DEVICES empty,
threads1, polling only the four registered local milestone paths every30s.
STATUS.json records3 reverified saved/1waiting; COMPLETE.json is produced only
when all4 are preserved. The archive's older COMPLETE.json covers only the
original three; ACTIVE_WATCHER.json points to the new all-run controller.

The watcher reuses the proven exact100k preservation routine, with a separate
controller status directory, exclusive archive lock, source/copy SHA comparison,
fsync/atomic rename, full recovery-state/fingerprint checks and copied immutable
source/contract metadata. It never restarts, signals or modifies a trainer.

Sep-2 is managed by its own agent. Final Variations was updated there at23:36UTC:
old joint baselines C4–C6 have protected100k; active directional temporal-local,
directional global, and cross-only local camera runs each have armed CPU
preservation watchers plus native100k rotation exemptions. These are remote
agent reports, not direct sep-1 filesystem verification. The deleted sep-2
bidirectional camera duplicate is outside retained runs and must not resume.
