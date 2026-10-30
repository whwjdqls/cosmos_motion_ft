# Phase2 training memory and speed: Yonsei RTX4090

Measured 2026-10-31 on node33's idle GPU247c7e54, retained allocation2608785.
The other allocated GPU serves the application and was untouched.

Reference: Self→cross final-hidden/K-norm/caption-only B256 200k architecture,
seven layers, T200/TI97, Nymeria+BONES50 within T2M, task ratio75:25, original
camera/head statistics and losses1/1/5 + contact0.05/velocity1/height10.
The isolated run's architecture contract matches the reference exactly.
TF32 disabled, FP32 motion I/O, eight loader workers and existing caches.

| Microbatch | Accumulation | One-GPU global batch | Mean s/update | Median s/update | Peak allocated GiB | Peak reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 128 | 1.7449 | 0.9589 | 20.804 | 21.270 |
| 128 | 2 | 256 | 3.3380 | 2.4613 | 21.275 | 22.012 |

Device memory sampled with nvidia-smi once per second peaked at23024MiB
(22.484GiB), including driver/context overhead. Batch128 fits in this40-update
test on a24GB4090; this is not an exhaustive test of every sequence/token length.
It does not test microbatch256 or two-GPU DDP speed.

Twenty actual AdamW updates per phase, first five excluded from each timing
aggregate. CUDA-synchronized timing covers loader wait, conditioning, forward,
decoded loss, backward, clipping and optimizer. Startup, checkpointing,
visualization and post-update log overhead are excluded. All40updates completed,
with zero guarded batches/skipped optimizer updates in the trainer log.

Mean loader waits were0.783s and1.460s/update. Subtracting those measured waits
leaves0.962s and1.878s for conditioning/training; those are not isolated GPU
kernel timings. Some real bad-window retries/storage stalls were observed.
The mean includes these slow batches; the median alone overstates sustained
throughput. Fresh isolated model/optimizer updates were used; the second batch
setting continued the optimizer/cache state initialized during the first setting.
Neither production weights nor its checkpoints were modified. No test
checkpoints or visualization media were saved. The benchmark process exited0.

Reproduction evidence and wrapper:
`/scratch2/whwjdqls99/cosmos/operations/phase2_training_benchmark_20261031`
contains `benchmark.py`, `run.sh`, `protocol.json`, real per-update
`steps.jsonl`, `gpu_memory.csv`, `summary.json`, `memory_audit.json` and logs.

Notion report:
https://app.notion.com/p/409d2a44f44b81e5b4c7d2266be3e9c3
