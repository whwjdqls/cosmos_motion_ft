# Preserved Phase3 100k checkpoints on sep-2

All three exact100,000-step checkpoints are preserved and verified. This is sep-2 only; the separate sep-1 preservation record is migration/FOUNDRY_PHASE3_100K_PRESERVE_20261108.md after merging master.

Node-local archive: `/mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_100k_20261108`. These are retained hardlinks outside the trainers’ checkpoint directories, so unlink-based checkpoint rotation cannot remove them. They are not independent copied files or shared/off-node backups. No shared-storage transfer was performed.

| Model / W&B run ID | EMA start | EMA updates at100k | SHA-256 |
| --- | ---: | ---: | --- |
| Bidirectional / videoW10 / `p3-d47749076808b4bfecc2` | 50017 | 49983 | `6216ae0679ea5e1a8f829c3559d6dafd274512d996ffc693e584fbe8a5564d1e` |
| Directional / videoW10 / `p3-79f9ba1f501245ec7b18` | 50819 | 49181 | `93570240c2043a0a79e9623218fed6bf07b4c9e8a2b98e4a808415f059f74027` |
| Directional / videoW1 / `p3-aa6c658167d6bc2eec5f` | 0 | 100000 | `826c7c917d651dc564a020019d39628796c7b2d75959ef8ea39b22c4054116b4` |

Each model subdirectory contains `step_000100000.pt` and `VERIFIED.json`. Root `COMPLETE.json` confirms all three, with exact run identity, size and full-file SHA-256. Each file is approximately3.76GB and contains raw trainables, EMA, optimizer/scheduler state, both rank RNG states and data cursor3,200,000. Common frozen Phase1/Phase2 checkpoints and immutable source snapshots remain at the paths recorded in registry.json.

The CPU-only preserve_100k.py watcher finished after all three receipts were verified; its historical PID is not an active GPU job. No trainer was restarted or changed. The existing trainer policy also exempts100k from pruning.

The older bidirectional/videoW10 and directional/videoW1 exact90k checkpoints are retained separately in `/mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_90k_20261108`. Directional/videoW10 exact80k and90k were already pruned before this preservation request; do not claim the100k file reconstructs or replaces those milestones.
