# Cosmos-3 Edge Phase 2: T2M + TI2M motion expert

This directory is the isolated Cosmos-3 Edge Phase-2 implementation.  It trains a
new motion expert from scratch while keeping the Edge Nemotron reasoner frozen.
It does **not** pack video-generator tokens and it does not load Phase-1 LoRA or
action-head weights.

Depending on the sampled task, the packed sequence is:

```text
[ frozen reasoner text                         | trainable motion shape + frame tokens ]  T2M
[ frozen reasoner frame-0 image + caption text | trainable motion shape + frame tokens ]  TI2M
```

The information-flow contract is asymmetric:

- reasoner queries attend causally to reasoner tokens only;
- motion queries attend fully to reasoner and motion tokens;
- the reasoner never reads motion tokens.

Edge requires a separate normalized copy of reasoner keys for the motion-to-text
attention edge.  The raw keys remain in the causal reasoner self-attention path.
This is implemented in `attention.py` and `layer.py`.

The default training contract mirrors the corrected Nano Phase-2 recipe:

- tasks: T2M + reasoner-image TI2M, weighted `0.75/0.25`;
- representation: `camera_head_recanonicalization_v1`;
- source: Nymeria only by default (`bones_frac=0`);
- effective mass: 75% Nymeria T2M and 25% Nymeria TI2M;
- output capacity: 200 frames at 20 FPS;
- TI2M alignment: 97 valid Nymeria frames, padded and loss-masked to 200;
- TI2M image: synchronized frame 0 at 256x256 through frozen Edge SigLIP2;
- CFG: 10% text dropout; TI2M keeps its image when text is dropped;
- one-GPU batch: 128 clips with gradient accumulation 1 (effective batch 128);
- objective: clean-motion (`x0`) prediction;
- schedule: Cosmos shifted logit-normal, shift 3;
- positions: Cosmos 3D mRoPE;
- motion layers: every fourth Edge layer, exactly seven blocks
  (`[3,7,11,15,19,23,27]`);
- motion FFN: the original Phase-2 three-linear SwiGLU, width 3072.
- caption subject: standalone uppercase `C` becomes sentence-aware `The camera
  wearer` / `the camera wearer`, matching the Phase-1 retraining policy.

The v1 mean/std are computed over all 120,929 retained, floor-filtered Nymeria
train-caption windows (11,888,119 frame occurrences). The contract pins the
summary JSON and its population counts in addition to the mean/std hashes.

The Edge migration changes only the interfaces the shared attention requires:
the residual width is 2048, the projected attention geometry is 16 query heads
and 8 KV heads at 128 dimensions per head, and the wrapper follows Edge's 28
backbone layers.  The motion heads, shape token, timestep conditioning, fresh
motion Q/K/V/O projections, sparse-depth layout, and SwiGLU motion FFN retain
the Nano Phase-2 design.

TI2M does not reintroduce the video generator. Its image tokens are frozen
reasoner tokens, so attention still has only two roles: causal reasoner and
full-attention motion. There are no generator rows, Phase-1 adapters, or
three-way attention.

BONES is an explicit later ablation via `--bones-frac > 0` and remains restricted
to T2M. Its original `/weka/...` paths are remapped at read time on the restored
server. BONES has no synchronized egocamera, so the contract records it as
`legacy_uniego283_motion_only`, not as a camera/head-equivalent `camhead_v1`
source. Shared v1 normalization makes it usable by the one expert but does not
manufacture camera calibration.

Nano Phase-2 `.pt` files are evaluation baselines only.  The Edge checkpoint
loader validates the model family, backbone geometry, motion-layer indices,
representation, normalization hashes, base-DCP identity, and framework commit.

Launch all entry points through `run.sh`, which selects the pinned Edge framework
and the camera-recanonicalized data paths.  A normal GPU verification sequence is:

```bash
bash motion_expert_t2m_edge/run.sh -m unittest discover \
  -s /home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge/tests -v

bash motion_expert_t2m_edge/run.sh \
  /home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge/train.py \
  --smoke --batch-size 1 --T 16
```

Use `sbatch_l40_smoke.sh` for the scheduled one-GPU gate. Production should be
submitted only after that gate records finite forward/backward loss, non-zero
motion gradients, zero frozen-reasoner gradients, checkpoint reload, and a finite
sample. The canonical production launcher is `sbatch_train_1gpu.sh`; checkpoint
resume is supported when the 96-hour allocation ends. Multi-GPU launchers are
optional topology tests, not production prerequisites.

The production launcher also requires online Weights & Biases logging. Scalar
loss, feature/joint/smooth loss components, pre-clip gradient norm, learning
rate, throughput, effective batch, and peak allocated GPU memory are logged
every 20 optimizer steps to project `jungbinc-upenn/cosmos-motion-ft`. A
run-local `wandb_run_id.txt` makes a later Slurm/checkpoint resume continue the
same W&B run. Mutable W&B data/cache are rooted under
`/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/.wandb_runtime`, not the
currently full shared `/home` filesystem. Initialization waits up to 300
seconds per attempt and retries three times with the same persisted run ID.

The one-GPU launcher is preemption-safe. Regular named checkpoints remain every
5,000 steps, while an atomic rolling `checkpoints/recovery_latest.pt` is
overwritten every 250 steps. Each checkpoint includes model, optimizer,
current step, data epoch, and Python/NumPy/Torch CPU/CUDA RNG state. Every
Slurm start uses `--resume auto`, which selects the newest complete regular or
rolling checkpoint; incomplete `.tmp.*` files are ignored. The launcher also
requests Slurm requeue and a USR1 warning 180 seconds before a scheduled
termination so the trainer can write an additional signal-triggered recovery
checkpoint after its current optimizer step.

Checkpoint visualization uses a fixed Nymeria test cohort: five T2M and five
reasoner-image TI2M samples. The exact normalized GT motion, skeleton, TI2M
image, prompt, sample ID, and per-sample inference seed are materialized once
under `<run>/visualizations/fixed_samples/` and strictly reused after resume.
The initial model is visualized at step 0 and the same ten samples are rendered
every 5,000 steps. T2M MP4s show `GT | generated`; TI2M MP4s show
`conditioning image | GT | generated`. Full prompts are recorded in every
local manifest and in W&B media captions/tables. Sampling is fixed-noise
UniPC-35 with CFG 2; rendering uses stride 2 at 10 FPS while saved arrays retain
all generated and GT frames. Production sets both `--require-viz` and
`--require-wandb`, so a required render/upload failure stops only after any
same-step checkpoint has been preserved.

The one-L40S production-shape batch sweep passed batch 128 for both routes:
T2M peaked at 16.22 GiB and TI2M at 17.09 GiB. Batch 128 is therefore the
selected production microbatch with substantial headroom on the 46,068-MiB
L40S. Larger batches also fit, but were not selected.

The current one-L40S schema-v3 T=16 and production-shape T=200/TI2M=97 gates
pass finite forward/backward, nonzero motion gradients, zero frozen gradients,
strict reload, and TI2M sampling. The schema-v3 contract is Nymeria-only and
intentionally rejects the earlier schema-v1 T2M-only and schema-v2
BONES-mixture smoke checkpoints. See `SMOKE_RESULTS.md` for current and
superseded evidence.
A trained checkpoint still requires C45,
foot-skating/floating, trajectory, and diverse SOMA visual comparison against
the Nano baseline.

## W&B/media integration gate

One-L40S job `528071` passed the complete online gate on 2026-09-02. It
uploaded loss and gradient norm for two optimizer steps, one T2M video, one
TI2M image-conditioned video, the standalone TI2M condition image, and the
full-prompt sample table. Both videos are valid H.264; the panels are 1200x600
for T2M and 1800x600 for TI2M. The checkpoint saved and strictly reloaded, and
the W&B API reports the run finished with four media records:

```text
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/5lj0rq5m
```

The repeatable gate is `sbatch_wandb_viz_smoke.sh`.

## First instrumented production attempt

Slurm job `528072` started from scratch on one L40S on 2026-09-02. Its output
directory is:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz
```

The fixed step-0 production cohort contains ten unique IDs/prompts, five per
task. All ten H.264 videos are valid: T2M has 50 rendered frames at 1200x600,
and TI2M has 49 rendered frames at 1800x600. W&B reports 16 media records (ten
videos, five TI2M condition images, and one full-prompt table). The live run is:

```text
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/bupzjaj5
```

The W&B API verified scalar rows at steps 1 and 20. Step 20 remained finite
with loss `1.089049`, pre-clip gradient norm `42.2566`, peak allocation 16.31
GiB, and about 0.270 optimizer steps/s including the initial visualization
time. Both Nymeria T2M and TI2M routes were observed.

This attempt was subsequently preempted twice by Slurm: once after 1h48m57s
and again after 26m17s. The second allocation reached step 480 with finite loss
`0.422049` and gradient norm `114.6186`; neither allocation reached the old
step-5,000 checkpoint interval. Slurm requeued the same job ID, but the old
submitted launcher would have restarted from step 0 again. It was therefore
superseded while pending by the rolling-recovery/auto-resume launcher above.
The output and W&B run remain preserved as preemption history, not as a
checkpointed training run.

## Active preemption-safe submission

Batch job `529851` is the active replacement as of 2026-09-03. It started
immediately on the healthy L40 at `dj-l40-0` through
`sbatch_train_1gpu_batch.sh`, which changes only the Slurm partition/QoS request
and then executes the canonical hardened launcher. Its clean output directory
is:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz_preemptsafe
```

It uses batch 128, rolling recovery every 250 steps, automatic
newest-checkpoint resume, explicit Slurm requeue, and the 180-second USR1
recovery signal. Node `ll-l40-1` remains excluded because its earlier Edge
smoke allocation reported an uncorrectable GPU ECC error. W&B run `isptsmkc`
successfully reused the ID persisted by the failed predecessor. Job `529851`
loaded all 120,929 retained training windows, rendered and uploaded the fixed
five T2M plus five TI2M step-0 comparisons, and completed optimizer step 20
with finite loss `1.089077`, pre-clip gradient norm `49.5103`, and 16.31 GiB
peak allocation. Both Nymeria routes were observed. A manual Slurm-equivalent
USR1 test then wrote a 1.41 GB atomic `recovery_latest.pt` at step 75 on Weka;
training continued through logged step 80, proving the live signal-save path
does not terminate the job. A memory-mapped `torch.load` check confirmed schema
3, step 75, `signal_recovery`, optimizer/data-epoch state, and complete
Python/NumPy/Torch CPU/CUDA RNG state.

Predecessor `liu-compute` job `529512` reached model/data initialization but
stopped before visualization or optimizer step 1 because W&B 0.27.2 did not
publish its local service port within the old 30-second limit. It produced no
checkpoint. The repaired launcher places mutable W&B data/cache on Weka,
extends the per-attempt service wait to 300 seconds, and performs three bounded
attempts using the same run ID. The 13-test contract suite covers the retry
behavior. Repaired `liu-compute` job `529849` remained pending for `Priority`
and was cancelled before allocation after batch job `529851` started, avoiding
two writers to the same output directory. Earlier pending batch job `528385`
was likewise cancelled before it started.

The first `liu-compute` submission, job `528415`, received `ll-l40-0` at
2026-09-02 22:47 EDT but its batch step was cancelled by signal 53 at zero
elapsed time. Top-level accounting records `FAILED (0:53)`, not `PREEMPTED`;
the user batch script never produced its Slurm log, output directory, W&B run,
or checkpoint. Node accounting shows a same-second cluster event: five running
jobs on `ll-l40-0` failed at 22:47:12--14, four newly launched array tasks and
job `528415` then all died with zero-runtime `0:53`, and unrelated jobs started
successfully on that node at 22:47:15. This rules out Phase-2 user code and
supports a transient Slurm/node resource-handoff failure. The exact lower-level
cause (for example prolog, cgroup, GRES, or controller handoff) is not exposed
by user accounting and requires the administrator's `slurmd`/`slurmctld` logs
for that timestamp. It was not a training, model, memory, checkpoint, or
observed CUDA ECC failure.

## Production run history

Slurm job `528052` started the clean one-L40S run on 2026-09-01 at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  edge_7layer_nymeria_t2m_ti2m_v1
```

Its persisted schema-v3 contract was Nymeria-only, batch 128, accumulation 1,
T=200, TI2M-valid-T=97, 0.75/0.25 T2M/TI2M, and 200,000 steps. Step 1 was
finite (`loss=1.152887`, `grad=59.7627`); by step 20 it remained finite
(`loss=1.089056`, `grad=41.7568`) at 16.31 GiB peak allocation and about 0.206
optimizer steps/s. Both condition routes were observed.

This uninstrumented job was intentionally canceled at step 1,000 on 2026-09-02
so it could be replaced by the required W&B/fixed-visualization run. It had not
yet reached the first step-5,000 checkpoint, and its incomplete output remains
preserved for provenance. The instrumented launcher uses the new clean output:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz
```

At that rate, 200,000 steps exceed one 96-hour allocation. Checkpoints are
written every 5,000 steps; resume a later allocation by setting `RESUME` to the
latest `step_XXXXXXXXX.pt` and submitting `sbatch_train_1gpu.sh` again.
