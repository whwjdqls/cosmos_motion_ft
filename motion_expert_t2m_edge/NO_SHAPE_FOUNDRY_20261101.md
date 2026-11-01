# Phase2 without an external shape condition

`EDGE_PHASE2_SHAPE_CONDITIONING=none` removes the neutral-joint shape token from
the packed motion sequence and freezes the unused shape encoder/type embedding.
It retains each motion frame's original mRoPE coordinates and the original
motion heads, timestep embedding, targets, normalization and physical losses.
Neutral joints may remain in stored samples for GT/reference rendering, but
the model does not read them in this mode and also accepts `None`.

The default `neutral_joints` mode preserves the previous architecture and
checkpoint contract. Shape-free checkpoints record their mode in
`contract.phase2_ablation.shape_conditioning`. Inference resolves it before
constructing the model; an explicit conflicting mode fails. This ablation
starts fresh, rather than reinterpreting an already trained shape-conditioned
expert or resuming one with a changed input contract.

## Foundary conditional launch

The user selected the original200k Phase2 recipe used by the ongoing FiLM runs:
Self→cross final-hidden with cross-key RMSNorm, caption-only packing, seven
motion layers, UniEgo283 camera/head statistics, FP32 heads, T200 (TI2M97),
T2M/TI2M75/25, BONES50% within T2M, historical caption dropout10%, and global256
(two H200s×128, accumulation1). Optimizer/loss/schedule remain AdamW beta2.99,
weight decay0, clip1, LR2e-4, warmup1000, cosine to0 at200k, Nano per-task losses
1/1/5 and contact/foot-velocity/foot-height.05/1/10, seed0 and TF32 off.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase2_no_shape_after_ego_20261101`.
Its independent monitor polls EgoControl every60seconds. It requires successful
completion and a full native200k checkpoint, checks model/optimizer/scheduler/
trainer metadata/shard lengths, and waits for GPUs0/4 and their existing owned
locks to be released. It never stops EgoControl or another user's process.

The conditional pipeline waits for original-recipe data staging/checks, then
runs a disposable full-global256 two-rank update/save/reload/update gate with
finite nonzero gradients, changed motion-head weights, no frozen gradients,
shape-input invariance and finite T2M/TI2M/BONES previews. Production starts
fresh with online W&B only if that gate passes. The no-shape model is not
declared GPU-accepted or running by this preparation note; trust the operation's
`GATE_PASSED.json`, `PRODUCTION_RUNNING.json`, logs and monitor status.

The prior three trainers use separate historical source checkouts. They are
not changed to adopt this ablation. BONES is required: a missing source must
fail instead of silently falling back to Nymeria-only. CPU checks pass32 tests
under the explicit Yonsei artifact profile, including new shape invariance,
shape trainability and checkpoint-mode tests.

The verified preparation report is
https://app.notion.com/p/40bd2a44f44b8187850ac41dd36ef75c . After verified
production starts, a separate idle-GPU monitor launches the final EgoControl
EMA200k evaluation on every eligible native test window; see
`egocontrol_edge/FULL_TEST_FOUNDRY_20261101.md`. Its readiness is not a claim
that either production workload has started.

The actual launcher environment/artifact checks pass on CPU. Its parsed
production CLI matches the selected original `step_000200000.pt` scientific
arguments, and architecture contracts differ only in the shape-conditioning
contract and relocated paths. Evidence is `CONFIG_CPU_ACCEPTANCE.json` and
`RECIPE_PARITY.json` in the operation directory. Tracking/output identity and
fresh-versus-resume launch are operational differences; the original200k
checkpoint is a reference, not an initialization. GPU acceptance is pending.
