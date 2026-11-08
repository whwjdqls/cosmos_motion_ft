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
