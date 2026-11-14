# Three camera-run exact100k milestones: protection armed

Update (2026-11-08): the user cancelled the duplicate sep-2 bidirectional run
and requested deletion of its W&B run and checkpoints. Its registry entry has
been removed. Only the two directional runs remain protected; the restarted
watcher PID is1360307 (verify current WATCHER.json). The original three-run
setup below is historical and must not be used to restart the duplicate.

Node: sep-2. Protected node-local root:
`/mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_camera_nojoint_100k_20261108`.

The three fresh camera runs have not reached100k yet. Their exact100k native
checkpoints are explicitly exempt from training checkpoint rotation. A separate
CPU-only watcher also waits for each exact `step_000100000.pt` and preserves
an independent copied file outside the run folders.

| Registry key | Model | GPUs |
| --- | --- | --- |
| directional_local | Directional temporal-local, videoW1 | 3,5 |
| directional_global | Directional global attention, videoW1 | 1,7 |
| bidirectional_local | Bidirectional temporal-local, videoW1 | 0,6 |

All three: V2M50/M2V50/J0, EMA0.999 from step0, no captions, camera prediction
with zero camera loss, global32=2x16/accum1, fresh200k. The exact output paths
and expected recipes are in root `registry.json`. The new bidirectional run
was GPU/W&B verified after the old joint/videoW10 run stopped104482. Its W&B:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-48abfc4c04cf999c28aa .

Watcher: `scripts/preserve_phase3_milestones.py`. It checks the exact step,
training fingerprint/recipe, raw bridges, matching finite EMA tensors with
100000 updates, full optimizer/scheduler, two rank RNG records and draw cursor
3200000. Copies use a new temporary file, fsync, source/destination SHA equality,
source stability checks, independent inode/linkcount1 and atomic promotion.
Immutable source snapshots, manifests and contracts are copied alongside.
No frozen specialist weight duplication or shared-storage transfer occurs.

`WATCHER.json` records PID/start identity; `status.json` records current waiting
or verified state. `VERIFIED.json` exists per model only after its100k copy and
full-state checks pass. Root `COMPLETE.json` is written only once all three
are verified. Pending status does not imply the milestones already exist.
`FAILED.json` indicates a watcher error and must be investigated.

The initial watcher PID is1265412. Verify its start identity before using it.
It survives terminal hangups, consumes no GPU, and exits after all three copies.
After a node reboot it can be restarted with the same registry/root; native
trainers retain100k even while the watcher is down. Existing protected receipts
are checksum-verified on watcher restart.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/preserve_phase3_milestones.py \
  --registry /mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_camera_nojoint_100k_20261108/registry.json \
  --root /mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_camera_nojoint_100k_20261108
```

Use the node-local project Python environment and caches. Never start a second
watcher against a live one: an exclusive watcher lock rejects duplicates.
Two focused preservation tests passed (independent copy survives source unlink;
wrong-step checkpoint cannot be promoted), alongside nine camera policy/mask
tests and actual two-rank native/camera/save-reload production gates.

Operation receipts and live verification:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_camera_bidir_nojoint_replace_20261108`.
