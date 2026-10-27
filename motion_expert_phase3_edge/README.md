# Edge phase 3: frozen specialists, trainable bridges

This package is separate from Nano phase 3 and Edge phase 2. Existing trainers,
checkpoints, optimizers and evaluation jobs are not modified. The only shared
API addition is optional `frozen_edge` injection into `EdgePhase2MotionExpert`;
the existing default loader and checkpoint keys are unchanged.

## Source specialists and architecture

- Phase 1: EMA `iter_000200000` of the filtered Yonsei Edge camera/video run
  dated 20260905; native Edge base plus rank16/alpha32 generator LoRA and action
  heads. All 224 LoRA tensors must load. Action heads are retained but unused.
- Phase 2: `step_000200000.pt` of Self -> cross final hidden + K norm,
  caption-only B256, dated 20260921. Load every motion state tensor strictly.
- Seven zero-gated bridges at layers 3,7,11,15,19,23,27. Width2048, 16 heads
  of128; original `LocalModalityBridge` semantics, FP32 parameters and attention.
- Native generator attention, reasoner K normalization for generation, native
  position IDs, motion self/cross attention and normalization remain unchanged.
  Motion reads final post-norm reasoner memory from raw caption-only text.
  Generator text independently receives its native image2video formatting.
- No gradients update either specialist. Gradients through frozen downstream
  operations are retained. Activation checkpointing bounds intermediate memory.
- Clean cross-modal conditioning cannot read the noisy other modality. Within-
  modality bridge attention retains the Nano reference's full attention.
  Wan alignment is `{0}, {1..4}, ..., {93..96}`, not `4*g..4*g+3`.

## Data and losses

Use all valid train recordings, no actor-name exclusion, no BONES, T97/20fps,
256-resolution cached Wan latents. Camera/head-aligned UniEgo mean/std are
verified against the motion checkpoint. Ground and canonicalize exactly once.
Preparation checks every included latent and motion window on a compute node,
writes read-only normalized motion arrays, and records all rejections. Training
has no raw-video fallback or replacement-window retries. Duplicate caption rows
retain their source sampling weight. Test recordings are never indexed.

The task mixture is 50% video2motion and 50% motimg2video. Motion losses reuse
Phase2 centered feature/joint/displacement weights1/1/5, contact/foot velocity/
foot height0.05/1/10, logit scale2, fps20, and its decoded-nonfinite guard. Video
reuses native velocity loss with weight10, uniform time weighting, conditioning
errors zeroed but conditioning elements retained in the denominator. Task
fractions are separate from objective weights. No camera or head-camera loss.

Motion logit-normal and video Waver training sigmas use shift3. Motion heads
receive normalized sigma directly; native generator timesteps are sigma*1000,
and the native encoder applies its own0.001 scale exactly once.

## Run on Yonsei or Grasp

Set `COSMOS_SITE=yonsei` or `grasp`. The wrapper delegates environment setup to
the established Edge launcher. Set `EDGE_PHASE3_PHASE1` and `EDGE_PHASE3_PHASE2`
to the exact local copies on Grasp. Other paths are explicit CLI arguments.
All Python entrypoints reject login nodes, node37, cs-gpu-01, and missing Slurm
allocations. Yonsei outputs and the processed cache default to Scratch2.

```bash
# Inside an owned compute allocation:
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.data --workers 4
bash motion_expert_phase3_edge/run.sh -m torch.distributed.run --standalone \
  --nproc_per_node=2 -m motion_expert_phase3_edge.train --preflight \
  --microbatch 4 --out /path/to/acceptance
bash motion_expert_phase3_edge/run.sh -m torch.distributed.run --standalone \
  --nproc_per_node=2 -m motion_expert_phase3_edge.train --microbatch 4 \
  --acceptance /path/to/acceptance/acceptance.json
```

Acceptance checks directional masks and causal temporal alignment, bit-exact
zero-bridge specialist parity, native video reduction, finite mixed-task DDP
backward, first-step gate/later core gradients, frozen specialist versions,
disposable overfit, parameter replication, and save/reload/continued update.
Smoke weights never initialize production. The gate binds implementation,
dataset and source identities. A limited preparation cache cannot train.

Production: 200k optimizer steps, global batch32, LR2e-4, warmup1000, cosine
floor0, AdamW betas0.9/0.99, weight decay0, clip1, seed0. Microbatch4 uses
accumulation4 on two GPUs or2 on four. Microbatch2 doubles accumulation.
Save bridge/optimizer/RNG/sampling state every10k; retain latest and100k.
SIGTERM/SIGUSR1 requests recovery at the next optimizer boundary. Resume uses
the same global sample cursor; changing world size reseeds per-rank noise and
is explicitly logged, so it is not a bit-exact continuation.

`launch_when_ready.sh` prepares/tests using `PREP_JOB`, then watches owned
`CANDIDATE_JOBS` and runs the production-node gate before starting training.
`PHASE3_OPS` holds logs/status. It never cancels any allocation. Non-OOM test
failures stop the launcher. A GPU allocation is not evidence of training;
inspect stage logs and optimizer steps before reporting a running model.

## Offline inference and metric policy

`sample.py` provides native UniPC35, guidance1, motion shift3 and generator
shift10. It pins the clean first latent on every video sampling step and never
noises conditioning motion/video. V2M captions remain empty. Fixed cases are a
JSON list of `{id, caption, bundle, seed}`; each NPZ bundle contains normalized
`motion[97,283]`, centered `neutral[30,3]`, `video[48,25,16,16]`, and scalar
`motion_mean_sha256`/`motion_std_sha256`. Use independently prepared held-out
bundles, never the training index. Outputs preserve generated arrays and GT
for the existing offline evaluators/renderers; no training-time rendering.

Report V2M reconstruction/trajectory and contact metrics with corresponding GT.
Video PSNR/SSIM/LPIPS must be computed after VAE decoding against held-out RGB,
excluding conditioned frame0. The sampler does not mislabel latent MSE as a
pixel/video metric. C45 evaluator normalization stays separate from generator
normalization. This package does not claim any phase-3 quality results yet.

### Offline visualizations

`visualize.py` prepares deterministic held-out Nymeria windows using Edge's
`generator` transform and Wan VAE APIs, then calls the existing Phase-3 sampler.
Normalization paths/hashes and frozen specialist paths come from the checkpoint.
It refuses login-node execution and never changes a training run or its models.

Inside an owned GPU allocation, set `COSMOS_SITE=yonsei` (or `grasp`), then run:

```bash
CKPT=/absolute/path/to/pinned/step_000060000.pt
OUT=/absolute/path/to/visualizations/step_000060000
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.visualize prepare \
  --checkpoint "$CKPT" --out "$OUT" --num-cases 4
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.visualize predict \
  --checkpoint "$CKPT" --out "$OUT" --mode video2motion
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.visualize predict \
  --checkpoint "$CKPT" --out "$OUT" --mode motimg2video
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.visualize gallery \
  --checkpoint "$CKPT" --out "$OUT"
```

After preparation the two `predict` commands may run concurrently on separate
GPUs. Each releases its model before rendering. `render` can regenerate movies
from a completed sampling manifest without sampling again. `--nymeria-root`,
`--vae-path` and `--work-dir` provide explicit site/output overrides.

Outputs include raw sampled arrays, GT bundles, comparison MP4s, first-frame
previews and `index.html`. The existing Phase-2 skeleton style preserves initial
head axes and actual floor crossings; generated motions are never re-grounded.
Every comparison includes GT. These small qualitative samples are not a metric
benchmark. Keep the selected checkpoint available until sampling finishes.

### Offline quantitative evaluation

`quantitative.py` scores both bridge directions without training-time callbacks
or rendering every example. First use `visualize prepare --num-cases 71` with a
fresh output directory. Then run one `evaluate` worker per direction, optionally
on separate GPUs:

```bash
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.quantitative evaluate \
  --checkpoint "$CKPT" --input-root "$OUT" --mode video2motion
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.quantitative evaluate \
  --checkpoint "$CKPT" --input-root "$OUT" --mode motimg2video
bash motion_expert_phase3_edge/run.sh -m motion_expert_phase3_edge.quantitative summarize \
  --input-root "$OUT"
```

V2M reports unaligned MPJPE/root errors, whole-clip PA-MPJPE, acceleration,
height-based foot skating, penetration and airborne fraction, with matching GT.
M2V reports PSNR/SSIM/LPIPS on directly decoded RGB frames 1..96, with GT identity
and GT-latent VAE reconstruction baselines. Predictions are never re-grounded.
Per-case arrays, metrics, signatures and `SUMMARY.md` are retained. These are
paired metrics, not a distributional FID/FVD benchmark. The deterministic
first-valid-window cohort can differ from historical `full71_windows.json`;
pass `--reference-windows` to `summarize` to record those differences.
