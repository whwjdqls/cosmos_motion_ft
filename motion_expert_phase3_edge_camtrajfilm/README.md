# Floor-aware trajectory + motion-noise FiLM

This isolated fresh ablation uses four caption-empty tasks: 50% motion-to-video,
with the remaining 50% split equally among video-to-motion, camera-to-motion,
and camera+video-to-motion. It retains frozen Phase 1 EMA 200k and Phase 2 self/cross K-normalized caption-only
B256 200k specialists, seven attention bridges, losses, data cohort, and global
batch 32 of `motion_expert_phase3_edge_camfilm`. It does not resume that run's
learned bridges. The original package and all existing runs remain unchanged.

After each bridge, camera-conditioned tasks receive framewise FiLM using the
current 2048D motion hidden state, direct 18D camera features, and the frozen
motion timestep embedder evaluated at normalized motion sigma. The two new
context projections produce 256D each. Concatenated context is 2560D, followed
by Linear(2560,256), SiLU, Linear(256,4096), yielding 2048D scale and shift.
All FiLM computations and non-affine LayerNorms use FP32. The final linear is
zero-initialized. There is no additional FiLM gate. Shape tokens, generator
tokens, and non-camera tasks are not modulated. Existing bridge gates remain.

The camera representation reuses `nymeria_baselines.prepare.camera_trajectory18`:
yaw-removed rotation 6D, local position 3D retaining floor-relative height,
incremental heading rotation 6D, and incremental horizontal translation 3D.
Upright source poses are converted from Z-up to Y-up; the calibrated TOTAL
floor offset from the original Nymeria dataset index is subtracted, matching
motion targets. First-frame delta is identity/zero, without removing height.
97 camera rows align one-to-one with 97 motion frames. Per-channel mean/std
are fitted on exactly the existing Phase 3 training cache, not test data.
The native generator still receives its original 96 raw9/pad64 camera actions.

This changes camera information, explicit noise conditioning, and the sampling
balance relative to the active equal-four-task FiLM reference. It is not an
information-, parameter-, or task-mixture-matched source-only ablation.

The floor lookup includes recording, start frame, and the cached humanized
caption. Recording plus start alone is not unique because overlapping caption
annotations can carry different floor estimates. The approved read-only audit
found unique matches for all 112,503 cached windows with this complete key.

Yonsei operation directory:
`/scratch2/whwjdqls99/cosmos/operations/phase3_camtrajfilm_20261022`.
The retained allocation is 2599826, node34, 2x4090. Cache preparation, preflight,
and training run on that compute node, never on a login node. CLI path options
and `COSMOS_SITE` retain Grasp portability. Shared AGENTS documentation is not
edited because it contains concurrent work from another session.

The approved preflight checks source/recipe parity, floor/height behavior,
frame alignment, zero-initialization, original frozen predictions, task masks,
trajectory and timestep influence, absence of forbidden-condition leakage,
four-task DDP backward/overfit, frozen weights, and checkpoint/optimizer/RNG
reload. Production reloads fresh specialist checkpoints, not preflight weights.
Checkpoint retention is latest plus 100k. The monitor's 100-step milestone is
not a training stop; the configured target remains 200k.

See `WANDB.md` for automatic, resume-safe logging.
`OPERATIONS.md` records launch fixes, preflight evidence, and the completed
100-step monitoring audit. The launcher must set
`PHASE3_RUN_DIR` to its exact training output directory. Production requires a
successful matching `acceptance.json`; creating a READY marker is not approval
to skip that gate.
