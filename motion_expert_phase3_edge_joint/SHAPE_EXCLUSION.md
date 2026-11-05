# Shape bypasses Phase3 modality bridges

Implemented on sep-1 on 2026-11-05 after the user superseded the proposed
shape-query residual freeze with complete removal from bridge attention.

`LocalModalityBridge.forward` removes motion rows whose `motion_frame < 0`
before LayerNorm and Q/K/V projection. The bridge receives only temporal body
motion and generator/video tokens. Removed rows are restored from the input
with `index_copy`, so shape output is bit-for-bit identical to its input,
including signed zeros and BF16 values. Row positions are determined by
metadata, not an assumption that shape occupies row zero. Shape-only inputs
bypass the bridge entirely.

This applies to `shared`, `separate_norm`, `separate_norm_out`, and `mmdit`,
both conditioning policies, all tasks, and dense/cached/grouped execution.
Immutable frame indices and filtered metadata are cached only within the
current forward's metadata, enabling reuse across seven layers and checkpoint
recomputation. No activations or new model parameters are cached/registered.
Mask and grouped-plan caches have new version keys.

## Normalization and projections

Previously shape used the motion stream's `norm_m`, `k_m`, `v_m` in MM-DiT
(or shared `norm`, `k`, `v` in the legacy shared parameterization). It now
uses **none of these bridge layers**. Body-motion frames still use their
existing normalization/projections; generator/video routing is unchanged.
Shape encoding and attention inside the motion expert remain unchanged.
This does not remove body-shape conditioning from the model as a whole.

## Compatibility and verification

Parameter counts, initialization order, state-dict keys/shapes and temporal
group definitions are unchanged. Nonzero-gate outputs deliberately change
because shape no longer contributes to bridge softmax/values or receives a
bridge residual. Historical exact resume/evaluation must continue using the
original source snapshot and matching acceptance fingerprint. The previously
uploaded scratch46338 archive is immutable and was not replaced by this edit.
No existing training process was started, stopped, or restarted.

Focused CPU tests cover bit-exact shape bypass with opened gates, all four
parameterizations, all tasks/both policies, FP32/BF16, arbitrary shape-row
positions and NaN perturbations of shape; absence from norm/QKV/attention;
identity-only shape gradients; no shape gradient from temporal outputs;
no-shape/shape-only inputs; dense/cached/grouped parity; and two-process CPU
DDP, recomputation and complete optimizer/EMA recovery. GPU acceptance for
new training remains pending. All 38 focused shape/conditioning/locality/
MM-DiT/grouped-attention CPU tests passed, including both two-process Gloo
DDP tests. `git diff --check` also passed. Earlier grouped GPU benchmarks
precede shape exclusion and do not establish its updated performance.
