# Temporally local joint attention in Phase3

Initial change: only bridge attention masking changed on sep-1 on 2026-11-05. Normalization,
Q/K/V/O parameters, residual calculations, zero gates, frozen specialists,
objectives and training configuration are retained. No training job was started,
stopped or restarted.

## Temporal groups

Both same-modality and cross-modal keys are restricted to one causal Wan group:

| Group | Video keys/queries | Motion keys/queries |
|---|---|---|
| 0 | 64 tokens of latent0 | frame0 |
| 1 | 64 tokens of latent1 | frames1–4 |
| 2 | 64 tokens of latent2 | frames5–8 |
| … | … | … |
| 24 | 64 tokens of latent24 | frames93–96 |

Motion frame f belongs to group `ceil(f/4)`, including group0 for frame0.
No temporal query can read a temporal key from another group. This applies
to both conditioning policies and all bridge parameterizations. Same-stream
attention is local as well: motion1–4 share a block, but motion4 cannot read
motion5; video latent1 cannot read latent2. Unidentified non-video GEN rows
have no fallback to global temporal attention.

The forward implementation computes a **single joint softmax** over
the permitted video and temporal-motion keys. Shape is excluded from bridges.
There are no separate self-attention/cross-attention normalization passes.

## Directional and bidirectional conditioning

The existing `--conditioning-attention` switch is unchanged:

- `directional`: only noisy target queries may read the opposite modality's
  aligned frame keys. Clean conditioning queries have no cross-modal edges.
  Both noisy targets in a joint task may read each other within their group.
- `bidirectional`: either modality may read aligned keys of the other modality,
  including clean queries. Temporal-frame edges are symmetric.

No policy relaxes temporal locality. Prefix1, task ratios, noise schedules,
specialist losses, LR, weight decay, EMA and additional-LoRA settings are unchanged.
No new configuration flag or checkpoint tensors are introduced.

## Shape excluded from bridge attention (subsequent user request)

The non-temporal body-shape token now bypasses the bridge entirely. Before
normalization or Q/K/V projection, all non-frame motion rows are removed;
only video and temporal-motion rows enter attention. Removed rows are restored
bit-for-bit from the input, with no residual update. This supersedes the
earlier self-only shape query/global shape key design. There is no bridge
LayerNorm or K/V projection for shape. See [shape exclusion](SHAPE_EXCLUSION.md).

When the standalone mask helper is called with unfiltered metadata, non-frame
rows have self-only numerical placeholders and no temporal edges. Actual
bridge execution removes these rows before invoking attention. Grouped plans
never duplicate shape keys into temporal groups.

This locality guarantee applies to the modality bridge. The frozen generator
and motion-specialist attention remains unchanged and can carry temporal
context into bridge inputs, including shape. This change does not claim to
make the entire Phase3 network temporally disconnected.

## Implementation and compatibility

`temporally_local_attention_mask` in
`motion_expert_joint_attention/modality_bridge.py` owns the mask. The normal
bridge and `scripts/phase3_performance.py` cached-bias path use this same
implementation. The cache key is versioned separately from the former global
same-modality mask, so an old bias on the same metadata object is ignored.

Parameter initialization/order and state-dict keys/shapes are unchanged. For
the initial masking-only edit, an AST comparison confirmed that
`_attention_mask` was the only changed bridge method;
`attention_bias` was the only changed execution-policy function. The subsequent
shape exclusion also adds a shared frame-only forward wrapper to the dense,
cached and grouped implementations. GPU preflight
diagnostics check temporal-block symmetry and the absence of all temporal
edges to/from the excluded shape rows.

Historical checkpoint weights still have compatible tensor shapes, but their
nonzero-gate outputs can change under the new mask. Use their original source
snapshots to reproduce historical evaluations/resume fingerprints. Existing
source/acceptance gates remain in force.

## Verification

Nine new CPU tests cover the complete 1698×1698 mask against explicit groups
for all three tasks/both policies; first/last groups and frame boundaries;
single joint-softmax normalization; cache invalidation; shape self-only rows;
zero gradient to blocked inputs; and repeated three-bridge perturbation tests
across all four parameterizations proving no bridge cross-group/shape relay.

All 25 focused mask/conditioning/MMDiT tests passed, including cached/plain
forward/backward parity and real two-process CPU DDP. Another 31 regression
tests passed for LoRA/EMA recovery, camera and scratch contracts, video-loss
weighting, native objectives/noise schedules, task sampling and spawned data
workers. The actual preflight mask assertions passed on CPU for all eight
conditioning/parameterization combinations; the camera trainer's source
overlays/argument parsing also passed.

Native GPU/data acceptance has not been run. The implementation used no GPU
sampling or training.
