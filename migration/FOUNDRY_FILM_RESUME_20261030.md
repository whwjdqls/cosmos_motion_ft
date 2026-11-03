# Yonsei FiLM continuation on Foundary sep-1

The user requested stopping both local projection-only Phase1 runs once the
two Phase3 FiLM continuations were ready. This supersedes waiting for their
200k targets. The action-only full checkpoint174064 and action+video projection
full checkpoint125764 were validated and pinned with hard links before signaling
only their owned torchrun processes. Both Phase1 trainers are stopped; neither
is claimed to have completed200k.

| Continued experiment | Original checkpoint step | GPUs | W&B ID |
| --- | ---: | --- | --- |
| Hidden-state context camera FiLM | 92879 | 6,7 | p3-eff9fe58905591a19d96 |
| Floor-aware trajectory FiLM | 63864 | 3,5 | p3-e75dd21f3b540efbe1c8 |

Both production launches restored the original training states and advanced.
Live logs establish current progress. No four-task control or EgoControl trainer
was started by this operation.

## Preserved scientific settings

Use each archived contract as authoritative. Both retain T97/20fps/256,
global32=two ranks×microbatch16×accumulation1, four DataLoader workers per rank,
frozen original Phase1 EMA200k and Phase2 Self→cross/K-norm/caption-only B256
200k, the same Nymeria cohort/statistics, losses, gradient clipping, AdamW state,
two-rank RNG state, global sampler cursor and200k cosine horizon. No optimizer
reset or new warmup occurs. These frozen specialists are not the stopped local
projection ablations.

Hidden FiLM retains equal25% sampling for M2V/V2M/camera2motion/camera+video2motion.
Trajectory FiLM retains50% M2V and one-sixth for each motion-target task,
the original floor-aware18D trajectory normalization and explicit motion-noise
path. Both preserve no-caption conditioning. Native activation checkpointing
and the validated FP32 forward-local bridge-mask cache remain enabled; TF32
is disabled. Checkpoint policy and loss implementations are unchanged.

## Isolated execution adaptation

The dedicated detached checkout is
`/mnt/local/jinhyung.kim/cosmos/repos/phase3_resume_20261030`, based on268fe8d.
Its sole tracked source difference is the verified historical shared
`motion_expert_phase3_edge/data.py` overlay with SHA256
`c21a7bb57e2276f668ea7fed5bfba35ea46756f3e9443003c6efce82ac985262`.
All source hashes recorded by both scientific contracts are checked exactly.
The regular working copy and Phase1 implementations were not changed.

Execution adapters live outside the hashed scientific packages under
`/mnt/local/jinhyung.kim/cosmos/operations/resume_film_20261030`:

- `runtime.py` resolves archived filesystem reads to NVMe, retains the historical
  trajectory-cache path as its logical identity only after checking the exact
  physical cache fingerprint/source/statistics, and compares the entire resulting
  live contract with the original. Cache indices, saved checkpoints and scientific
  fingerprints are not rewritten. Host admission validates an owned live launcher
  ancestor and the exact visible GPUs; no Slurm job ID is fabricated.
- `probe.py` restores full bridge/FiLM, optimizer, schedule and RNG states,
  checks all four modes, performs two updates with save/reload between them,
  checks replica equality and frozen parameters, and writes rank reports.
  Single-GPU rehearsals and two-rank16-per-GPU gates passed for both variants.
  Rehearsal updates are disposable; production starts from the original checkpoint.
- `launch_variant.py` owns the GPU locks, enforces destination gates and starts
  the existing trainer through the original performance wrapper. `stream_wandb.py`
  is the captured logger with explicit archived run ID and `resume=must`, so the
  relocated output path cannot silently select a new W&B run.
- `plan.json`, `execution_source_manifest.json` and `execution_source.tar.gz`
  preserve exact paths, launcher source and hashes. `phase1_preserved/` contains
  the full pinned Phase1 states and configurations. The original shared archives
  remain available separately.

## Verification and live evidence

`hidden_film/single_gpu_gate/rank0.json` and the corresponding trajectory report
record CPU/source and full-state GPU rehearsal. Each `ddp_gate/rank*.json`
records the successful two-rank save/reload/update gate; destination acceptance
records retain the original contract fingerprint. Production restored92879 and
63864 and first advanced to92880 and63865. The first production losses match
the respective full-batch rehearsal losses within FP32 tolerance. Backward
gradients are finite, but bitwise gradient identity across independent CUDA
executions is not claimed.

For each variant, `production.log`, `status.json`, `wandb.log`,
`wandb_stream/status.json`, `destination_acceptance.json` and the actual run's
`foundary_runtime/identity_migration.json` establish execution state. W&B resumes
the original project `jungbinc-upenn/cosmos-motion-ft` using preserved JSONL
history and source-record numbering. No prior history was truncated.
