# Foundary sep-1 local cleanup, 2026-11-01

The user explicitly requested removal of excessive local files and old
checkpoints. Cleanup was restricted to owned files under
`/mnt/local/jinhyung.kim/cosmos` on `lg-gwangmo-4xh200-sep-1`.

Removed78 superseded native Phase1/EgoControl checkpoint directories, four
disposable FiLM save/reload probe weights, and four redundant download archives.
Approximately2.00TiB was reclaimed; available local capacity increased from
252.67GiB to2296.36GiB (2.24TiB). Node disk usage fell from99% to85%.

## Retained state

- Original frozen Phase1 EMA200k and Phase2 selected200k, plus the newly staged
  two-task Phase1 final200k, are unchanged.
- Stopped action-only174064 and action+video125764 final checkpoints remain both
  in their original runs and in the hardlinked `phase1_preserved` tree.
- Each old native experiment retains its latest checkpoint. The two main
  projection ablations also retain100k and the120k snapshots used by the
  published CFG1/1.5/2 qualitative comparisons.
- EgoControl retains100k, its latest fixed-step milestone175k, and the recent
  rolling recovery window. The separate unique original91345 checkpoint in
  the old gate-output tree is retained. Subsequent native retention may prune
  older rolling saves normally; cleanup did not change that policy.
- Both active FiLM runs, their original recovery state,100k milestones/latest
  state, and pinned evaluation snapshots (hidden150k/trajectory130k) remain.
- Datasets, training caches, environments, model assets, logs, published
  predictions, metrics, videos and quantitative packages are unchanged.

No shared-storage files or other users' files were modified. No trainer was
stopped, restarted or reconfigured. This was a one-time cleanup; future normal
checkpoint saves retain their existing settings.

## Evidence and checks

Operation:
`/mnt/local/jinhyung.kim/cosmos/operations/cleanup_local_20261101`.

`deletion_plan.json` records every candidate/file size, inode, link count and
mtime, protected paths/reasons, original checkpoint pointers, and18 retained
DCP-path completeness checks. Component metadata and referenced shard lengths
were checked for model/optimizer/scheduler/trainer without reading full model
tensors. Live references, open files, recent saves and all current pointers
were excluded before deletion. `deletions.jsonl` records each deletion.

`archive_deletion_plan.json` records three previously verified resume archives
whose originals and matching receipts remain on shared storage. Extracted
initial checkpoint/contract bytes were rechecked; the only allowed source-JSON
differences are the two documented site-path translations, with byte-exact
historical metadata retained in the handoff snapshot. The evaluator tar was
removed only after every extracted member matched byte-for-byte.

`COMPLETE.json`, `archive_cleanup.json` and `post_cleanup_health.json` establish
completion, reclaimed capacity and continuing trainer progress. These records
supersede earlier disk-capacity snapshots. Probe weights and redundant local
archives referenced by historical acceptance records are intentionally pruned;
their recorded results, sources, hashes, logs and required production state remain.
