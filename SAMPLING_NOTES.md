# Text -> Motion Sampling Notes (Cosmos3-Nano finetune)

Pipeline: **caption string -> flow-matching ODE sampler -> normalized 369-d motion
[T,369] -> `motion_decode.decode_features_to_joints` -> world joints [T,30,3] ->
matplotlib 3D stick-figure mp4.**

Files:
- `sample_motion.py` — model load + overlay, sample-time forward, ODE sampler,
  decode + stick-figure render, CLI.
- `run_sample_validation.py` — loads the 14 GB model ONCE and samples several
  prompts (validation driver; avoids re-loading per prompt).
- `motion_decode.py` — pure-torch kimodo decode (unchanged; verified bit-exact).
- artifacts under `samples/`.

---

## 1. Flow-matching / ODE convention (derivation)

Everything is pinned to `train_motion_ft.forward_loss`, which defines the training
target. With per-token scalar time `t in [0,1]`:

```
x_t      = (1 - t) * x0 + t * noise          # forward_loss: x_t = (1-tcol)*x0 + tcol*noise
v_target = noise - x0                          # forward_loss: v_target = noise - x0
```

so `t = 1` is **pure gaussian noise** and `t = 0` is the **clean motion x0**. The
network regresses `v_hat ≈ v_target = noise - x0`.

**ODE.** Differentiate the path w.r.t. t:
```
dx_t/dt = d/dt[(1-t)x0 + t*noise] = noise - x0 = v
```
The velocity field IS `v_hat`. To sample we integrate from `t=1` (noise) down to
`t=0` (clean). With a positive step `dt = 1/N` and moving DOWN in t, the Euler
update is:
```
x_{t-dt} = x_t + (dx/dt) * (-dt) = x_t - dt * v_hat(x_t, t)
```
That is exactly `x = x - dt * v` in `sample_motion.sample` (sign verified: moving
toward smaller t with `dx/dt = v`). The final iterate `x_{t=0}` is the predicted
clean `x0`.

**x0 estimate at any t** (useful for sanity / future correctors): since
`x_t = x0 + t*v`, `x0_hat = x_t - t*v_hat` — the same identity `forward_loss` uses
to reconstruct x0 for the kimodo loss.

**Time grid.** `t = linspace(1, 0, N+1)`; at step i we evaluate `v_hat` at
`t = ts[i]` and step by `dt = ts[i] - ts[i+1]`. Default N = 50 Euler steps.

**Timestep into the net.** `build_pack_from_batch` stores `timesteps = t / TIMESTEP_SCALE`
(0.001) because the network multiplies by `timestep_scale` internally; the
`time_embedder(t_tokens)` call in the forward is fed `t` directly (already in
[0,1]). The sampler mirrors this exactly: `build_sample_pack` stores `t/scale` in
the pack, and `predict_velocity` calls `net.time_embedder(t_tokens)` with raw `t`.

### Key difference vs `build_pack_from_batch` / `forward_loss`
`build_pack_from_batch` samples a RANDOM `t` and `forward_loss` re-noises x0 inside
(`x_t = (1-t)x0 + t*noise`). At sample time we must NOT re-noise — we already hold
`x_t`. So:
- `build_sample_pack` packs **our current `x_t`** as the action payload and **our
  current `t`** (not random).
- `predict_velocity` encodes `action.tokens` (= `x_t`) **directly** through
  `motion2llm + motion_modality_embed + time_embedder(t)`, runs the two-way packed
  MoT forward, and returns `llm2motion(last_hidden)` = `v_hat`. No `torch.randn`,
  no `(1-t)x0 + t*noise`. Otherwise byte-for-byte the same encode as training.

## 2. CFG

Optional classifier-free guidance (`--cfg`, default 2.5; 1.0 disables). Each step
runs a second forward with an **empty caption** (`tokenize([""])`) and combines:
```
v = v_uncond + cfg * (v_cond - v_uncond)
```
This is the standard CFG mix on the velocity field (same role as kimodo viz
`cfg_scale`). The model was NOT trained with caption dropout, so the empty-prompt
branch is only an approximate unconditional — CFG here is a usable knob, not a
guarantee; `--cfg 1.0` gives the pure conditional ODE.

## 3. Model load / overlay

`train_motion_ft.save_checkpoint` stores `payload["model"]` keyed by
`net.named_parameters()` names (the FULL, FSDP-gathered tensors of every
`requires_grad` param = 397 `_moe_gen` tensors + 5 motion-head tensors
[`motion2llm.{weight,bias}`, `llm2motion.{weight,bias}`, `motion_modality_embed`],
plus `time_embedder.*` if trained). `sample_motion.load_model`:
1. builds the net (`build_network`, `materialize`),
2. loads the BASE gen weights (`load_gen_weights`),
3. overlays `payload["model"]` by copying every key present in
   `net.named_parameters()` with a matching shape.

Confirmed at load time (step 2000 ckpt): `loaded=406 skipped=0` (397 `_moe_gen` +
5 motion heads + 4 time_embedder), base load `587/814 remapped, all 397 _moe_gen`.

## 4. How to run

cosmos env, from `cosmos-framework`, one GPU via srun (`--ntasks=1` is required —
without it srun fans out one task per CPU and OOMs a single GPU):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
export LD_LIBRARY_PATH= && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework

# single prompt
srun -p a3ultra --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=0:20:00 \
  bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos && export LD_LIBRARY_PATH= && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd /home/jungbin_cho/cosmos-framework && \
  python /home/jungbin_cho/cosmos_motion_ft/sample_motion.py \
    --ckpt /weka/jungbin/cosmos_motion_ft_runs/full_generator_fsdp8_20260615_214202/ckpt_step002000.pt \
    --prompt "a person walks forward" --frames 120 --steps 50 --cfg 2.5 \
    --out /home/jungbin_cho/cosmos_motion_ft/samples/walk'

# all three validation prompts in one allocation (model loaded once)
srun -p a3ultra --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=0:25:00 \
  bash -lc '... && python /home/jungbin_cho/cosmos_motion_ft/run_sample_validation.py \
    --ckpt .../ckpt_step002000.pt --frames 120 --steps 50 --cfg 2.5'
```

Outputs per prompt: `<out>.npy` (normalized [T,369]), `<out>_joints.npy` ([T,30,3]),
`<out>.mp4` (20 fps stick figure). If ffmpeg is missing the renderer falls back to
`.gif` then per-frame PNGs (ffmpeg IS present in the cosmos env, so mp4 is used).

## 5. Validation results

### Sanity check 1 — decoder + render on REAL motion (CPU, no GPU needed)
Decoded subset sample 0 (caption: *"character goes from drinking from a bottle
while standing to standing"*, source `natural`, T=200) and rendered it:
- `samples/real_sample0_joints.npy` shape **(200, 30, 3)**
- `samples/real_sample0.mp4` — **535,658 bytes**, 20 fps
- joint stats: min -0.303, max 1.779, mean 0.358 (coherent standing human, ~1.78 m
  tall, Y-up) — confirms the decode + FK + render path is correct on ground-truth
  features.

### Sanity check 2 — the trained model GENERATES (GPU)
One H200 via `srun -p a3ultra --ntasks=1 --gres=gpu:1` (job 2428). Model overlay at
load: `loaded=406 skipped=0` (397 `_moe_gen` + 5 motion heads + 4 time_embedder).
Settings: frames=120, steps=50 Euler, cfg=2.5, seed=0. End-to-end in **86.6s** for
all 3 prompts (model resident; ~6s sample + ~20s render each). All x0 finite.

| prompt | x0 shape | mean | std | min | max | finite | joints | mp4 bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| "a person walks forward" | (120,369) | 0.136 | 0.955 | -3.64 | 4.64 | yes | (120,30,3) | 242,504 |
| "a person waves"         | (120,369) | 0.137 | 0.956 | -3.59 | 4.64 | yes | (120,30,3) | 242,256 |
| "a person jumps"         | (120,369) | 0.134 | 0.955 | -3.60 | 4.66 | yes | (120,30,3) | 241,373 |

Artifacts: `samples/gen_{walk,wave,jump}.{npy,_joints.npy,.mp4}`. The generated x0
sits near unit-normalized scale (mean ~0.13, std ~0.96), exactly what a healthy
normalized 369-d sample should look like, and is fully finite — the sampling +
decode + render path runs end-to-end on the trained checkpoint.

## 6. Honest notes / simplifications
- **Motion quality**: the checkpoint is the 4k-subset, 2000-step, velocity-MSE run
  (loss 1.69 -> 0.91, no FK loss). It validates the SAMPLING+DECODE+RENDER path;
  generated motion is rough/under-trained, not production quality. Concretely, the
  decoded generated joints span ~±3.6 (vs ~-0.3..1.8 for the real ground-truth
  sample) and the three prompts produce similar global statistics — i.e. the model
  has learned the normalized feature distribution but is not yet strongly
  text-conditioned at 2000 steps. This is expected and called out honestly; the
  deliverable is the working pipeline, not the motion fidelity.
- **Euler ODE**, fixed uniform grid (no Heun/RK, no x0-clamp, no noise schedule
  warp). Simple and matches the linear rectified-flow path; higher-order solvers
  are an easy follow-on.
- **CFG** uses an empty caption as the unconditional; the model wasn't trained with
  caption dropout, so it's approximate (see §2).
- **Packing borrows the action layout** (action_dim=369) exactly as training does —
  same simplification noted in DESIGN.md; sampling is consistent with training.
- **One sample per forward** in the sampler (B=1). Batching multiple prompts would
  use the same ragged packer but isn't needed for validation.
