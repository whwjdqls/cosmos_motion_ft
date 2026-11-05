# Grouped BF16 bridge experiments

Implemented and benchmarked on sep-1 on 2026-11-05. The user's two requested
choices were interpreted as the first two lower-precision options in the table:
BF16 attention only, and BF16 projections plus attention. Both were compared
with a freshly measured grouped FP32 reference.

All three use the current **shape-excluded** bridge. Earlier grouped GPU results
predated shape exclusion and are not a matched reference for this experiment.
No existing trainer was stopped, restarted, resumed or converted to BF16.

## Compute modes

| Mode | Q/K/V/O matmuls | Attention QK/AV matmuls | Norms/softmax | Parameter/gate/optimizer storage |
|---|---|---|---|---|
| `fp32` | FP32 | FP32 | FP32 | FP32 |
| `bf16_attention` | FP32 | BF16 | FP32 | FP32 |
| `bf16_bridge` | BF16 | BF16 | FP32 | FP32 |

The grouped kernel casts Q/K/V to the selected compute type, converts scores
to FP32 before scaling/masking/softmax, and casts probabilities to the selected
type for the attention-value matmul. It returns FP32 attention features before
output projection. `bf16_bridge` uses autocast only inside the compact frame
bridge, covering linear matmuls while FP32-input LayerNorm stays FP32. Shape
bypass and frozen specialist code are outside that autocast scope.

These are manual grouped kernels; SDPA/FlashAttention was not introduced in
this experiment. Masks, one joint softmax, residuals/gates, parameter identities,
state-dict keys, losses and scientific defaults are retained. No FP16 or gradient
scaler is used. GPU hooks verified norm and projection output dtypes; master
parameters, parameter gradients and AdamW tensor states stayed FP32.

The modes are opt-in:

```python
with execution_policy(grouped_attention=True, precision="bf16_attention"):
    # Keep this context active through backward/checkpoint recomputation.
    output = model(**inputs)
    loss.backward()
```

Use `precision="bf16_bridge"` for the second choice, or `"fp32"` for the
reference. Omitted precision inherits an outer policy, otherwise defaults to
FP32. Nested existing cached policies preserve the selected precision and
avoid constructing dense masks. BF16 modes require grouped execution.

## H200 timing results

GPUs5/6 remained occupied by another researcher. The benchmark used previously
authorized sharing of the user's GPU0 with trainer3157418. The same owned
trainer was present before and after; no foreign GPU was used. These are
**contention-affected timings**, not isolated throughput.
GPU5 briefly became free, but another job occupied it before an attempted
isolated repeat could start. The ownership guard rejected that attempt before
CUDA initialization; no isolated result is claimed and no foreign job was shared.

The workload is seven independently parameterized MMDiT bridges,1600 video
tokens plus97 temporal motion tokens, hidden2048,16heads×128, BF16 streams,
nonzero gates0.1, activation checkpointing and fresh metadata per stack. TF32
and any outer-model autocast are disabled. Models, inputs and weights match
across modes. Four alternating-order trials ×six iterations give24 samples
per mode and phase, after warmup.

| Seven-bridge workload | Grouped FP32 | BF16 attention only | BF16 projections + attention |
|---|---:|---:|---:|
| Median forward | 29.418ms | 29.366ms | 11.144ms |
| Median forward + backward | 103.368ms | 103.105ms | 36.146ms |
| Observed forward/backward speedup | 1.000× | 1.003× | 2.860× |
| Minimum forward/backward | 54.405ms | 53.936ms | 33.122ms |
| Incremental peak forward/backward | 990.5MiB | 975.4MiB | 997.4MiB |

The FP32 timings varied substantially under contention, as the median/minimum
gap shows. The observed2.86× ratio must not be presented as an isolated or
whole-training speedup. Attention-only conversion showed no material benefit;
converting the projection matmuls as well is the stronger throughput candidate.
Its additional peak memory was slightly higher, not lower, in this checkpointed
test because FP32 master storage remains and autocast needs temporary casts.

This is a synthetic bridge-stack benchmark. Frozen generator/motion blocks,
DataLoader, distributed communication and optimizer work are excluded from the
timed sections. No end-to-end Phase3 update or generated-video/motion quality
comparison was run.

## Numerical and optimizer checks

| Against grouped FP32 | BF16 attention only | BF16 projections + attention |
|---|---:|---:|
| Combined output relative L2 | 0.0557% | 0.0668% |
| Combined parameter/input-gradient relative L2 | 0.2662% | 0.3741% |
| Synthetic loss relative difference | 5.95e-7 | 7.14e-7 |
| Maximum output absolute difference | 0.0390625 | 0.03125 |

All outputs and gradients were finite, and shape bypass stayed bit-exact.
Each mode also completed three disposable AdamW updates with LR2e-4,
betas0.9/0.99,eps1e-8,WD0,gradient clip1. Fixed-batch loss decreased from
approximately2.00358 to1.98363 for all three modes. Optimizer states stayed
FP32. These checks demonstrate numerical/trainability sanity on this synthetic
batch, not convergence or quality parity on Nymeria.

All74 focused CPU tests passed, including five new precision tests for actual
matmul/projection/norm dtypes, nested-context restoration, bit-exact shape bypass,
finite gradients across all variants/tasks/policies, FP32 Adam states, and
two-process Gloo DDP with activation-checkpoint recomputation.

## Reproduction

Runner: `scripts/benchmark_bridge_precision.py`. Use the local Cosmos Python,
site configuration and repository/framework PYTHONPATH:

```bash
python -m scripts.benchmark_bridge_precision --gpu 0 --share-owned-gpu \
  --out /mnt/local/jinhyung.kim/cosmos/operations/benchmark_bridge_bf16_20261105/gpu0_shared_seven_bridge.json
```

The output must be new. The runner checks hostname, local paths, GPU ownership
and free memory, and refuses other users' GPU processes. Raw timing samples,
operator dtypes, numerical differences and disposable optimizer losses are
saved in that JSON. Source copies/manifests and a completion receipt are stored
alongside it. No production checkpoint or training configuration was changed.
