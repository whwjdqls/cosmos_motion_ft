# Grouped FP32 bridge attention: implementation and H200 benchmark

Implemented and benchmarked on Foundary sep-1 on 2026-11-05. The user authorized
stopping the no-joint camera run on GPUs5/6 to obtain timings without another
process on the benchmark GPU. That run stopped gracefully at step60,288.

## Result

The primary benchmark uses seven independently parameterized MMDiT bridges,
BF16 input/output streams, FP32 norms/projections/attention parameters and
arithmetic, nonzero residual gates, activation checkpointing, and fresh metadata
once per stack. It includes packing/first-use metadata overhead. TF32 and
autocast are off. Shapes are video1600 plus motion98, hidden2048,16heads×128.

| Seven-bridge workload | Dense local mask | Grouped FP32 | Speedup |
|---|---:|---:|---:|
| Forward | 17.895ms | 12.173ms | 1.470× |
| Forward + backward, checkpointed | 66.470ms | 46.610ms | 1.426× |

This is about30% less bridge forward/backward time. It is a synthetic bridge
stack, **not an end-to-end Phase3 update**: frozen native/motion blocks,
DataLoader work, optimizer updates and distributed communication are excluded.
No claim is made that whole-model training becomes1.43× faster.

Additional PyTorch-allocated peak memory during the checkpointed forward/backward
fell from1597.0MiB to989.8MiB, about38% lower. These increments exclude already
allocated parameter/cache storage and are not whole-model peak memory.

An isolated one-bridge test on GPU6 with BF16 streams and FP32 compute measured
7.089→5.016ms forward/backward (1.413×). Attention alone on GPU5 measured
1.036→0.252ms forward (4.108×). Four alternating dense/grouped trials with
warmups were used; the stack has24 timed samples per variant/phase, the single
bridge48. Raw per-iteration CUDA-event and wall-clock timings are retained.

## Numerical agreement

The GPU FP32-stream MMDiT single-bridge comparison measured maximum output difference
2.38e-7 and maximum relative gradient-L2 difference4.55e-7. The seven-bridge
BF16-stream stack measured output difference0.0078125 and relative gradient-L2
difference5.83e-5, within the BF16 comparison tolerance. Grouping changes
reduction order, so bitwise equality is not claimed. CPU tests independently
check exact mask reconstruction and close output/parameter/input-gradient parity.

## Implementation

`grouped_joint_attention` in `motion_expert_joint_attention/modality_bridge.py`
packs the25 Wan temporal groups as a batch. Group0 has65 temporal queries;
later groups have68. Each temporal group's keys also include shape. Shape
queries get self-only singleton groups. Small groups are padded, padding keys
are masked, and padded query outputs are discarded before restoring the original
packed order. Q/K/V are still jointly normalized by one softmax per query.

The normal layout therefore uses26×68×69 attention slots per head, instead of
1698×1698. Grouping preserves the existing directional/bidirectional roles and
shape-key/query restrictions. Arbitrary token ordering, incomplete groups,
absent shape and unidentified GEN rows are handled by the same layout logic.

Plans are cached on immutable per-forward metadata and in a bounded32-entry
content-keyed layout cache. Metadata is copied to CPU once per new metadata
object for grouping; subsequent bridges/checkpoint recomputations reuse the
plan. No dense1698×1698 mask is built in grouped execution. Attention arithmetic
explicitly disables autocast and runs FP32. Norm/projection weights, gates,
residuals, optimizer state, losses and scientific recipes are unchanged.

The kernel is opt-in through the existing execution-policy context:

```python
from scripts.phase3_performance import execution_policy

with execution_policy(grouped_attention=True):
    output = model(**inputs)
```

Nested cached execution policies also avoid dense-bias construction. The dense
reference remains the default and can be used for comparisons. Grouped mode
supports the canonical temporally-local mask; an explicit additional attention
bias is rejected rather than silently ignored. Additional custom inference-mask
interventions should use the dense path until supported explicitly.

## Reproduction and evidence

Runner: `scripts/benchmark_grouped_bridge.py`. The script checks hostname,
ownership of all GPU processes, free GPU memory and local output paths. It
never stops jobs. `--share-owned-gpu` is required for occupied own-user GPUs;
another user's GPU process is always rejected.

The final isolated stack run was:

```bash
python -m scripts.benchmark_grouped_bridge --gpu 5 \
  --stream-dtype bfloat16 --parameterizations mmdit --layers 7 \
  --checkpoint-bridges --fresh-metadata --iterations 6 --repeats 4 \
  --out /mnt/local/jinhyung.kim/cosmos/operations/benchmark_grouped_fp32_bridge_20261105/isolated_gpu5_seven_bridge_checkpointed.json
```

Use the node's `.site.local.sh`, local Cosmos Python and standard repo/framework
PYTHONPATH. The output file must be new. GPU5 had no other GPU process before or
after this measurement. GPU6 was also free of other jobs for its isolated run.

All reports are under local operation
`/mnt/local/jinhyung.kim/cosmos/operations/benchmark_grouped_fp32_bridge_20261105`:
`isolated_gpu5_seven_bridge_checkpointed.json` is primary;
`isolated_gpu5_fp32.json` and `isolated_gpu6_bf16_stream_fp32_compute.json`
contain the single-bridge controls. Earlier `fp32.json` and
`bf16_stream_fp32_compute.json` shared GPU0 with training and are **not** the
primary results. Early runner reports used a generic contention label;
the recorded initial PID lists distinguish shared versus isolated runs.

Stopped run recovery checkpoint:

```text
/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_camera_bidir5050_nojoint_nocamloss_wv1_T97_gbs32_sep1_20261103/checkpoints/recovery_latest.pt
```

Verified step60,288, next global draw1,929,216, both rank RNG states,56 optimizer
parameter states and56 bridge tensors. The run remains stopped. GPUs5/6 were
released after benchmarking; another researcher subsequently started jobs there.
Do not treat them as idle or terminate/share those processes. The other three
trainers continue.

All65 focused CPU tests passed:9 grouped-execution tests plus56 existing
mask/MMDiT/conditioning/LoRA/EMA/camera/loss/objective/data-worker checks.
They include two-process Gloo DDP, checkpoint recomputation, strict mask
reconstruction, output/parameter/input-gradient parity and context restoration.
