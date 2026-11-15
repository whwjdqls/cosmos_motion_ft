# Bidirectional camera, temporal-local, no joint task

This is the bidirectional partner of the current directional temporal-local
camera run. Only experiment identity and conditioning attention differ in the
recipe. It starts fresh from the same standard Phase1 video-I/O EMA200k and
Phase2 200k, with V2M50/M2V50/J0, video weight1, EMA0.999 from0, no captions,
global32=2x16/accum1 and a fresh200k schedule. Motion/camera representations,
native objectives and noise schedules are unchanged.

All seven MMDiT bridges retain causal Wan temporal grouping. Both modalities
may read the other modality within their matching temporal group, including
clean queries. Shape bypasses the bridge, and no direct camera-motion edges
are introduced. Camera tokens participate in native generator attention:
noisy GT camera during training, detached camera readout and zero camera loss.
Validation starts camera from independent noise, never test GT camera.

The isolated wrapper reuses the current verified native camera trainer and
its preflight,4→6 strict reload and camera-aware fixed71 EMA validation every10k.
It freezes all base and wrapper sources. Existing directional trainers continue
using their own immutable snapshots and are not modified or restarted.

Launch module: `motion_expert_phase3_edge_camera_bidir_nojoint.launch`.
It accepts the same flags as the directional package. Explicitly coordinated
owned PID sharing uses a separate handover lease because the current temporal
directional run already holds its original lease; all UID/start checks and
the64GiB free-memory launch gate remain. It cannot share foreign resident PIDs.

Node operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_camera_bidir_nojoint_replace_20261108`.
Consult COMPLETE.json for actual cutover. User initially requested overlap,
then authorized checkpoint-stop-first because GPU6 lacked memory. Original
bidirectional joint/videoW10 stopped104482 with full recovery state preserved.
After GPU6 became available again, this replacement passed production/W&B
checks on0/6 at2026-11-08 13:23UTC. Both temporary reservation processes exited.
The two directional camera jobs were left unchanged.

Exact100k checkpoints for this run and the two current directional camera
runs are retained by the native trainer and independently copied/verified by
`scripts/preserve_phase3_milestones.py`. Protected node-local registry/archive:
`/mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_camera_nojoint_100k_20261108`.
The watcher preserves full raw/EMA/optimizer/scheduler/RNG/cursor state and
immutable source files outside the rotating training directories. Pending
watcher status is not evidence that these fresh runs have reached100k.
