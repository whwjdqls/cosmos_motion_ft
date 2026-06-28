# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of research scripts that adapt **Cosmos-3 Nano** (NVIDIA's omnimodal world model)
into a **human-motion / egocentric world model**. It is *not* a self-contained package: it imports
an external `cosmos-framework` checkout and runs on a Slurm GPU cluster. The scripts here are the
delta on top of Cosmos — datasets, trainers, samplers, decoders, viz, and launch wrappers.

**This checkout has no runnable environment.** `cosmos-framework`, the conda envs, and the `/weka`
datasets live only on the cluster nodes (`a3ultravis-a3ultranodeset-0/1`, Slurm partition `a3ultra`,
8× H200). You cannot train, sample, or even `import cosmos_framework` from this machine — edit code
and reason about it here; run it on a node. The git root holds all code; the `motion_poc/`
subdirectory (= the primary working dir) is empty.

Read order for orientation: `README.md` (project thesis + task space) → `DESIGN.md` (root experiment
data contract) → the per-experiment `README.md` files. `COSMOS.md` is a trimmed copy of the Cosmos
paper kept for reference (assessment of our recipe vs the paper is in its first section).

## Three independent experiments (do not mix their conventions)

Each has its own motion representation, its own trainer, and its own README. They share only the
"freeze the Cosmos reasoner, adapt the generation pathway" idea.

| Experiment | Dir | Trainer | Motion/action rep | Cross-cutting idea |
| --- | --- | --- | --- | --- |
| **Text→Motion finetune** (root, "Method 2") | repo root | `train_motion_ft.py` (bespoke) | **369-d** kimodo BONES-SEED (`KimodoMotionRep`, SOMA-30) | LoRA/full-FT on `*_moe_gen` + new motion heads; reasoner frozen |
| **MotionExpert POC** ("Method 3" bridge) | `motion_expert/` | `motion_expert/train.py` (bespoke) | **283-d** uniego (`motion_uniego`) | Standalone MotionExpert transformer cross-attends **one-way** to the *frozen* reasoner's cached hidden states `H_R`; Cosmos generator NOT instantiated |
| **Ego camera world model** | `nymeria_world/` | **native** `cosmos_framework.scripts.{train,inference}` (NOT `train_motion_ft.py`) | **9-d** camera pseudo-action (Cosmos native `camera_pose`, domain 2) | Camera = Cosmos's native action modality; reuse Wan2.2-VAE video path + native mid-training recipe |

When editing in one experiment, its README is authoritative — e.g. the root run *borrows* the Cosmos
action packer's layout for the 369-d motion (a known simplification noted in `DESIGN.md`), the camera
run *re-uses pretrained* action heads while the motion run *re-inits* them, and only `nymeria_world`
uses the native Cosmos training stack.

## The two-conda-env split (load-bearing — they cannot share a process)

- **`cosmos`** (Py 3.13, torch 2.10) — owns Cosmos-3 Nano. Used for **all training, sampling,
  inference**, and any `import cosmos_framework`.
- **`kimodo`** (Py 3.10, torch 2.4) — owns `SOMABonesSeedDataset` and the kimodo motion reps. Used
  **only** to **export** (text, motion) pairs and to **decode/render/viz** (numpy + matplotlib).
- **`nymeria_plus`** — only in `nymeria_world/` for `projectaria_tools` / VRS camera extraction.

A typical task crosses envs: export in `kimodo` → train/sample in `cosmos` → decode+render in `kimodo`.
`motion_decode.py` is a pure-torch port of the kimodo FK/decode that is **bit-exact** vs kimodo, so
the `cosmos` env can decode without importing kimodo.

## Running anything (the invariant preamble)

Every launch script repeats the same setup; replicate it exactly when writing new ones:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
export LD_LIBRARY_PATH=                                  # MUST be cleared or torch import fails
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework                    # cwd MUST be the framework root...
```

- **cwd must be `cosmos-framework`** because `build_network` reads a **relative** `QWEN_JSON` config
  path. Scripts in this repo are invoked by absolute path from that cwd (see `motion_expert/run.sh`,
  which encodes this).
- `srun` for a single GPU **requires `--ntasks=1`** — without it srun fans out one task per CPU and
  OOMs the GPU.
- Inference needs **`--no-guardrails`** (nodes have no `uvx`; guardrail download fails — harmless).
- Python stdout is **block-buffered to the log file** — monitor live progress via **TensorBoard**,
  not the `.log`.
- All data and run outputs live on **`/weka/jungbin/...`** (NFS, visible to all nodes);
  runs/checkpoints go to `/weka/jungbin/cosmos_motion_ft_runs/<run_name>/`.

### Root experiment (text→motion, 369-d)

```bash
# 1. Export (kimodo env): sharded (text, motion) pairs -> /weka/jungbin/seed/cosmos_text_motion_full
bash run_shards_node.sh <NUM_SHARDS> <K_START> <K_END>     # wraps export_bones_seed_full.py

# 2a. Train, direct on a node (cosmos env): MODE=lora|full
bash run_train_node.sh <NGPU> <BATCH> <STEPS> <MODE> <LR> <PORT>
# 2b. Train via Slurm
sbatch sbatch_train_8gpu.sh           # FSDP2 x8, full generator
# under the hood:
torchrun --standalone --nproc_per_node=8 /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
  --ddp --lora --loss kimodo --data /weka/jungbin/seed/cosmos_text_motion_full --out <run> \
  --steps 200000 --batch_size 32 --lr 2e-5 --max_frames 200 --lr_schedule constant \
  --save_every 10000 --viz_every 10000 --log_every 20
#   parallelism flags are mutually-exclusive intents: --lora (heads+LoRA) | --fsdp (full-gen shard) | --ddp ;
#   --loss kimodo (per-block smooth-L1 + FK consistency) vs --mse ; --tiny for a 2-layer smoke model.

# 3. Sample one prompt (cosmos env): caption -> ODE -> 369-d -> joints -> mp4
python sample_motion.py --ckpt <run>/ckpt_stepNNNNNN.pt --prompt "a person walks forward" \
  --frames 120 --steps 50 --cfg 2.5 --out samples/walk
#   run_sample_validation.py loads the ~14 GB model ONCE and samples several prompts.

# 4. Milestone eval (waits for a ckpt, samples, measures root/pose jitter)
bash eval_at_ckpt.sh <STEP>            # e.g. 100000
```

`tensorboard --logdir /weka/jungbin/cosmos_motion_ft_runs` for all root-experiment runs.

### MotionExpert POC

`motion_expert/run.sh` sets cwd/env/PYTHONPATH; drive every script through it. Phase order is
`build_pairs.py` → `compute_stats.py` → `precompute_hr.py` (caches frozen reasoner hidden states
`H_R`, ~47 GB, reused across versions) → `train.py` → `sample.py --ablation both` → `viz.py`
(decode+render, **kimodo** env). The POC's whole point is the **cond-vs-null ablation** (does the
frozen reasoner provide useful text semantics). See `motion_expert/README.md`.

### Ego camera world model — uses the native Cosmos stack

Trains/infers with `cosmos_framework.scripts.{train,inference}` via a Hydra experiment
(`world_camera_nymeria_nano`) + TOML, **not** `train_motion_ft.py`. Launch with
`sbatch nymeria_world/sbatch_camera_phase2.sh` (or `launch_camera_phase2.sh` for ssh/manual). LoRA
checkpoints store only the trainable delta, so eval is **merge-then-sample**: `export_merge_lora.py`
→ `prep_test_eval.py` → `run_infer_merged.sh` → `viz_eval_samples.py`. Full details + hard-won
multi-GPU findings (e.g. **`image2video` must be dropped** or distributed collectives desync) in
`nymeria_world/README.md`.

## Architecture: how Cosmos is adapted

Cosmos-3 Nano is an MoT (mixture-of-transformers) with a **reasoner** (understanding/causal pathway,
Qwen3-VL text+vision prior) and a **generator** (`*_moe_gen` weights, the rectified-flow generation
pathway). Across all three experiments the **reasoner is frozen** and supplies conditioning; only the
generation side and the new-modality heads are trained.

- **Text is a condition, never a target** — we do not train a captioner/VLM. The reasoner ingests
  **raw text strings** (tokenized by Cosmos's own processor), *not* precomputed embeddings. (kimodo,
  by contrast, precomputes LLM2Vec 4096-d embeddings — do **not** replicate that here; see `DESIGN.md`.)
- **New continuous modality = per-frame normalized vector, one token per frame.** Added via fresh
  projection heads: `motion2llm` / `llm2motion` / `motion_modality_embed` (root), analogously the
  native `action2llm` / `llm2action` / `action_modality_embed` (camera). The root run *borrows* the
  action packer's token-split / 3D-mRoPE / attention mask for these motion tokens.
- **Rectified-flow objective.** Forward path `x_t = (1-t)·x0 + t·noise`; velocity target
  `v = noise − x0`; network regresses `v_hat`. `t=1` is pure noise, `t=0` is clean motion. Sampling
  integrates the Euler ODE from `t=1→0` (`x ← x − dt·v_hat`); the timestep convention and CFG details
  are derived in `SAMPLING_NOTES.md` and **must stay pinned to `train_motion_ft.forward_loss`** — the
  sampler packs the *current* `x_t` and `t` (it does NOT re-noise the way training does).

### Motion representations (each experiment differs — easy to confuse)

- **369-d (root, BONES-SEED)**: `[0:3] smooth_root_pos · [3:5] heading(cos,sin) · [5:95] joint_pos
  (30×3) · [95:275] rot6d(30×6 global) · [275:365] vel(30×3) · [365:369] foot_contacts(4)`,
  z-scored with `/weka/jungbin/seed/stats/soma_uniform_motions_20fps/`. Decode via `motion_decode.py`.
- **283-d (motion_expert, uniego)**: `[0:270]` per-joint local SE(3) (30×[6D rot ++ 3D trans]) ·
  `[270:279]` canon_delta · `[279:283]` foot contacts. Decode via
  `motion_expert/decode_uniego_torch.py` (no FK). Coordinate convention is **Y-up, +Z-forward**;
  viz remaps to matplotlib's Z-up — plotting `(x,y,z)` directly lays the skeleton on its side.
- **9-d (nymeria_world, camera)**: `[pos(3), rot6d(6)]` relative SE(3) pseudo-action
  `ΔT_t = T_{t-1}^{-1} T_t` via `pose_abs_to_rel(rotation_format="rot6d",
  pose_convention="backward_framewise")`. **No mean/std normalization** (matching Cosmos exactly —
  adding it would diverge).

All three are **20 fps, SOMA-30 skeleton**. fps is fixed (no fps conditioning) in the root run.

## Conventions / gotchas

- **`verify_*.py`** scripts (`verify_export.py`, `verify_full_export.py`, `verify_motion_decode.py`)
  are correctness self-tests for export and the bit-exact decode — run them after touching export or
  `motion_decode.py`.
- A `_`-prefixed script (`_trim_cosmos.py`, `motion_expert/_gt_check.py`, `_profile.py`,
  `_storage_est.py`) is a one-off helper/diagnostic, not part of a pipeline.
- Checkpoints (`sample_motion.load_model`) overlay the trained delta onto freshly loaded base gen
  weights by matching `net.named_parameters()` names — a healthy root overlay is `loaded=406`
  (397 `_moe_gen` + 5 motion heads + 4 `time_embedder`). A drop in that count means a key-name/shape
  mismatch, not a no-op.
- Known root-experiment gaps (from `README.md`): heading augmentation is **OFF** (kimodo applies
  on-the-fly Y-rotation), and fps is fixed at 20.
