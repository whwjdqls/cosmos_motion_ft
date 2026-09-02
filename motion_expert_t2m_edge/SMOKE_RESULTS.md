# Cosmos-3 Edge Phase-2 smoke results

Date: 2026-09-01

> Current scope: schema v3 uses Nymeria only, with 75% T2M and 25%
> reasoner-image TI2M. Standalone uppercase `C` becomes `The camera wearer` or
> `the camera wearer`. BONES is off by default and is available only through an
> explicit ablation flag. Schema-v1 and schema-v2 checkpoints below are retained
> as historical backbone/memory evidence and are intentionally rejected by the
> current loader.

## Schema-v3 Nymeria-only gate

The current contract passed on one L40S in Slurm allocation `527758`, using
`srun --overlap` from `tmux 0`. The T=16 end-to-end output is:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_phase2_schema3_nymeria_tmux0_20260901/
```

Two forced optimizer updates covered the default routes:

- Nymeria T2M: loss `0.577508`, gradient norm `50.2688`;
- Nymeria reasoner-image TI2M: loss `0.749105`, gradient norm `62.5383`.

Both were finite, the frozen reasoner/visual tower received no gradient, peak
allocated memory was 7.50 GiB, and the schema-v3 step-2 checkpoint reloaded
strictly. A two-step TI2M UniPC sample was finite and its manifest records the
expanded prompt beginning `The camera wearer is talking...`.

The production-shape T=200 / TI2M-valid-T=97 output is:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_phase2_schema3_nymeria_T200_tmux0_20260901/
```

Nymeria T2M and TI2M losses were `0.626064` and `0.915822`, with gradient
norms `390.1187` and `203.0914`. Peak allocated memory remained 7.50 GiB and
strict checkpoint reload passed. The persisted contract records schema 3,
`bones_frac=0`, the `0.75/0.25` task weights, the camera-wearer caption policy,
and the exact 120,929-window / 11,888,119-frame-occurrence stats population.

Focused CPU tests passed 9/9 for the Edge Phase-2 code, and both native Phase-1
caption train/eval contract tests passed on the same scheduled node. These are
wiring checks, not learned-quality evidence.

The subsequent one-L40S T=200 batch sweep measured full optimizer updates for
both T2M and reasoner-image TI2M. At the selected microbatch 128, T2M used
16.22 GiB peak allocated memory and TI2M used 17.09 GiB. Gradient accumulation
is 1, so the effective batch is 128. The machine-readable sweep is
`batch_size_sweep_l40s_T200_large_20260901.json` under the Phase-2 run root.

## Schema-v2 T2M + TI2M + BONES gate

The corrected scope was subsequently validated on the healthy L40S in Slurm
allocation `527758`, launched through `srun --overlap` from `tmux 0`. The model
loaded 1.678B frozen reasoner parameters and the pinned 489.3M-parameter frozen
Edge SigLIP2/projector bundle. Trainable motion parameters remain 230.51M.

The T=16 wiring gate is stored at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_phase2_schema2_tmux0_20260901/
```

Three forced optimizer updates covered every production route:

- Nymeria T2M: loss `0.577508`, gradient norm `50.5187`;
- BONES T2M: loss `0.973288`, gradient norm `38.0140`;
- Nymeria reasoner-image TI2M: loss `0.530869`, gradient norm `49.5267`.

All values were finite, frozen reasoner/visual parameters received no gradient,
peak allocated GPU memory was 7.50 GiB, the 1.411 GB schema-v2 motion+optimizer
checkpoint reloaded strictly, and a two-step image-conditioned TI2M UniPC sample
was finite. Its machine-readable manifest is `sample/manifest.json`.

The production-shape T=200 / TI2M-valid-T=97 gate is stored at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_phase2_schema2_T200_tmux0_20260901/
```

It again covered Nymeria T2M, BONES T2M, and Nymeria TI2M, with finite losses
`0.626064`, `0.684850`, and `1.106696`; corresponding nonzero gradient norms
were `392.1181`, `320.0894`, and `135.0565`. TI2M used 97 valid frames padded
and loss-masked to 200. Peak allocated memory remained 7.50 GiB and strict
checkpoint reload passed.

These are wiring and memory checks, not learned-quality evidence. They are
superseded by the schema-v3 data/caption policy; multi-GPU DDP is no longer a
production prerequisite.

## Pinned architecture

- frozen Cosmos3-Edge Nemotron reasoner: 28 layers, hidden 2048, 16 query
  heads, 8 KV heads, head dimension 128;
- trainable motion blocks: exactly seven at `[3,7,11,15,19,23,27]`;
- each motion block retains the original Nano Phase-2 Q/K/V/O + two norms +
  three-linear SwiGLU topology, with private FFN width 3072;
- trainable motion parameters: 230.51M;
- frozen reasoner parameters retained at runtime: 1.678B;
- prototype scope was pure T2M only, using
  `camera_head_recanonicalization_v1`, matching v1 statistics, native caption
  spans, floor calibration, and no BONES samples.

## CPU gates

The following command passed six focused tests for attention directionality,
seven-layer placement, SwiGLU topology, masked x0 noising, shifted scheduling,
and padded loss exclusion:

```bash
bash motion_expert_t2m_edge/run.sh -m unittest discover \
  -s /home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge/tests -v
```

A real dataset probe found 120,929 floor-valid native train-caption windows,
loaded `uniego_rep_camhead_v1`, and confirmed `has_bones=False` and
`modes=['text2motion']`.

## L40S end-to-end gate

GPU work ran through an overlapping `srun` step inside the existing one-GPU
allocation `527758`, issued from `tmux 0`. The healthy device was an NVIDIA
L40S with 46,068 MiB total memory.

The T=16 end-to-end gate is stored at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_tmux0_l40s_v2_20260901/
```

It loaded 254 tensors / 1.678B frozen reasoner parameters from the Edge DCP,
then completed:

- one forward/backward/optimizer update;
- finite total loss 0.468783;
- nonzero motion gradient norm 23.0077;
- no frozen-reasoner gradient;
- 7.50 GiB PyTorch peak allocated memory;
- a 1.411 GB resumable checkpoint containing motion weights and optimizer
  state but no frozen base weights;
- strict checkpoint-contract reload;
- a finite two-step UniPC sample from the saved checkpoint.

The sample contract is recorded in `sample/manifest.json`; raw normalized and
decoded arrays are in `sample/sample.npz`.

The full output-capacity gate (T=200) is stored at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  smoke_T200_tmux0_v2_20260901/
```

It completed one forward/backward/optimizer update with finite loss 1.575684,
nonzero gradient norm 358.1054, strict checkpoint reload, and the same 7.50
GiB peak allocation. Native Nymeria caption spans are ragged and padded to
T=200; padded frames are omitted from attention and loss.

## Failed hardware attempt

The first independent L40 attempt, Slurm job `527953` on
`ll-l40-1.grasp.maas`, failed during `Module.to_empty()` with
`cudaErrorECCUncorrectable` before model weights or a forward pass ran. The
healthy L40S retry above passed. Do not use the failed job as evidence of a
software or memory failure; the node requires cluster-side ECC remediation.

## What this does not establish

The smoke checkpoints contain only one optimizer update and have no qualitative
value. They prove architecture, data, loss, gradient, memory, checkpoint, and
sampler wiring. A production checkpoint must still be evaluated against the
Nano Phase-2 baseline with the existing C45 motion metrics and explicit
foot-contact, planted-foot velocity/skating, foot-height/floating, trajectory,
and diverse SOMA MP4 comparisons before model quality is accepted.

## Online W&B and fixed-visualization gate (schema v3)

One-L40S Slurm job `528071` passed on 2026-09-02 using
`sbatch_wandb_viz_smoke.sh`. It exercised the production logging and media path
with one fixed Nymeria sample per task at T=16:

- online W&B authentication and run creation under
  `jungbinc-upenn/cosmos-motion-ft`;
- total/feature/joint/smooth losses and pre-clip gradient norm at both updates;
- deterministic fixed-sample manifests, exact saved inputs, and task-specific
  fixed inference noise;
- T2M `GT | generated` H.264 at 1200x600;
- TI2M `conditioning image | GT | generated` H.264 at 1800x600 plus the
  standalone conditioning PNG;
- full prompts in the fixed/step manifests and W&B media table/captions;
- finite two-task forward/backward, zero frozen gradients, checkpoint save,
  and strict checkpoint reload.

The W&B API reports a finished run with the two scalar updates and four media
records (two videos, condition image, and prompt table):

```text
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/5lj0rq5m
```

Local smoke artifacts are under:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/
  _smoke_wandb_viz_528071
```
