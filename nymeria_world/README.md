# Cosmos-3 Nano → egocentric camera-action video world model (NymeriaPlus)

**Goal.** Adapt Cosmos-3 Nano into an **egocentric world model** trained on NymeriaPlus using
**ego-video + text + ego-camera** only (human body motion and audio are out of scope for this
experiment). The ego-camera is used as Cosmos's **native "action" modality** — its pseudo-actions
already *are* delta-SE(3) camera motion — so we use the **native Cosmos training/inference stack**
(`cosmos_framework.scripts.{train,inference}` + Hydra experiment + TOML), **not** the bespoke
`train_motion_ft.py` motion trainer.

Three native modes (Cosmos §2.1.3 / §4.2.2):
- **image2video** — text + image → video (camera-independent)
- **forward_dynamics** — image + (text) + camera → video
- **inverse_dynamics** — video → camera

Note: this is older native-camera work. Current joint-attention context lives in
[`../AGENTS_ALL.md`](../AGENTS_ALL.md).

---

## 1. Data

Source (already built by `nymeria_kimodo_pipeline/`): `/weka/jungbin/nymeriaplus_kimodo_proportional/`
- `video/{Sxx}/{seq}.mp4` — ego RGB, **640², 20 fps**, frame `i` == camera frame `i`. (19 subjects, 728 seqs.)
- `video/manifest_video.jsonl` — per-seq `t2w_windows` (100-frame windows) with rich **captions**, `camera_path`, `nb_frames`, `usable`.
- `camera/{Sxx}/{seq}.npz` — `cam_world_pos (T,3)`, `cam_world_rot (T,3,3)` = `R_world_device`, in the **Aria SLAM world (Z-up), RAW** (no kimodo Z→Y; that is viz-only).

## 2. Action representation — Cosmos-exact, NO extra normalization

Camera = embodiment **`camera_pose`** → **domain_id 2, raw dim 9** = `[pos(3), rot6d(6)]`
(`cosmos_framework/data/vfm/action/domain_utils.py`).

Conversion is the **identical call** Cosmos's DROID dataset uses (`droid_lerobot_dataset.py:406`):
```
pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
# ΔT_t = T_{t-1}^{-1} T_t ; translation_scale=rotation_scale=1.0 (no scaling)
```
**No mean/std/quantile normalization** is applied or needed. Verified:
- `omni_mot_model._normalize_action_databatch` only densifies (no scaling).
- `droid_lerobot_dataset.py:431`: `if action_normalization is None: out_action = action` (camera recipes use `None`).
The 9D pseudo-action **is** the representation. We zero-pad 9→`max_action_dim=64`, masked in loss via
`raw_action_dim=9`. Video stays uint8 `[3,T,H,W]` (model does `/127.5−1` internally).
Round-trip (`pose_abs_to_rel`↔`pose_rel_to_abs`) verified: err ~5e-4, 0 non-finite over all 728 seqs.

## 3. Coordinate frame (IMPORTANT)

Our camera npz is the **raw Aria device frame** (Z-up world). Cosmos expects the **OpenCV optical
frame** (x-right, y-down, z-forward). These differ; we measured the gap:

- Device→RGB extrinsic from VRS: `vrs.get_device_calibration().get_transform_device_sensor("camera-rgb")`
  → **~39° tilt** (device-z vs optical-z) + **1.4 cm lever arm**.
- Our training video was rotated **−90°** from the raw RGB sensor (to make it upright), so the camera
  frame matching the displayed video is **rgb-optical + Rz(−90°)**.

**Chosen camera convention: `rgb-optical + Rz(−90°)` (= OpenCV of the upright video).**
Validated against the pretrained model's predicted camera direction (mean cosine, no alignment):

| frame | cosine(pred, GT) |
|---|---|
| raw device (initial) | +0.51 |
| rgb optical (raw sensor) | +0.45 |
| **rgb optical + Rz(−90°)** | **+0.70** |

Per-sample no-align dir-cosine (pred · GT): corrected **0.81 / 0.91 / 0.38** vs device 0.69 / 0.71 / 0.12.

> Note: for **finetuning** the frame is a *consistent* transform and the action heads are re-initialized,
> so any consistent frame is learnable — it won't break training. We adopt OpenCV for semantic
> alignment with Cosmos + fair zero-shot comparison. Scale (below) is **coordinate-invariant**
> (frame change moves `‖Δt‖` by only ~0.1 mm, the lever arm).

## 4. The camera "scale gap" — corrected diagnosis: action TEMPORAL-STEP mismatch, not metric scale

Zero-shot **inverse_dynamics** raw output is **~7–17×** our metric GT per-frame `‖Δt‖`:

| sample | GT `‖Δt‖` (m/frame) | pred `‖Δt‖` | ratio | trans-corr | rot-corr |
|---|---|---|---|---|---|
| S01 | 0.020 | 0.137 | ~7× | 0.80 | 0.54 |
| S02 | 0.028 | 0.192 | ~7× | 0.62 | 0.54 |
| S03 | 0.013 | 0.226 | ~17× | 0.60 | 0.74 |

**What it is NOT:**
- NOT coordinates: `‖Δt‖` is rotation/frame-invariant (verified Δ≈1e-16).
- NOT fps: swept conditioning fps 10–160; pred does NOT follow 1/fps (S01 flat ~0.13; S03 *rises*) and
  never approaches GT — so an fps tweak can't explain or fix it. (`invdyn_fps_out/fps_sweep.png`)
- NOT a de-normalization bug: inference (`inference.py:1546-1554`) saves the **raw** action-head output
  (only sliced to `raw_action_dim=9`); no de-norm. No camera_pose normalization stats exist in the repo.

**What it IS — action temporal-step / definition mismatch (k-sweep smoking gun):** the model's per-action
magnitude ≈ our GT displacement over **~7 frames**, not 1:

| sample | model per-action | GT k=1 | GT k=6 | GT k=7 | GT k=8 |
|---|---|---|---|---|---|
| S01 | 0.137 | 0.020 | 0.117 | **0.135** | 0.154 |
| S02 | 0.192 | 0.028 | 0.169 | **0.198** | 0.226 |

The inference loader reads `action_chunk_size+1` **consecutive** frames, so *our* actions are true 1-frame
deltas — yet the model emits ~7-frame-magnitude steps. NVIDIA's own `camera_action_44.json` is ~0.88/step
(≈ our GT over ~30–44 frames) → the `camera_pose` training distribution is **faster cameras / coarser
steps** than 20 fps head motion. So the ~7× is the model's learned action-step ≠ our 1-frame step, plus a
faster-camera prior — **NOT** validated as "Cosmos cameras move 7× faster in meters."

**Implication:** Phase-2 defines the action as our own per-frame metric delta and **fine-tunes the
PRETRAINED `camera_pose` action heads** (domain 2 — NOT re-initialized; they already exist and the zero-shot
inference used them), so the model adapts its convention/scale/step to ours → the mismatch **self-resolves
under finetuning**. *(Re-init was only correct for the separate MOTION experiment, where the motion heads were
genuinely new.)* *Design note:* at 20 fps, 1-frame deltas are tiny (~0.02 m, low rot6d SNR) — consider defining
the action over a coarser stride (2–4 frames) in the dataset.

## 5. Scripts (`nymeria_world/`)

| file | env | purpose |
|---|---|---|
| `camera_to_action.py` | cosmos | SE(3)→9D camera pseudo-action (Cosmos-exact) + round-trip self-test |
| `nymeria_camera_dataset.py` | cosmos | `NymeriaPlusCameraActionDataset` (PyAV windowed decode) → Cosmos action contract; no-GPU smoke through real `ActionTransformPipeline` |
| `extract_camera_opencv.py` | nymeria_plus | device→RGB extrinsic from VRS → `gt_camera_opencv.npz` (rgb optical) |
| `prep_baseline_inputs.py` | cosmos | pick windows; emit first-frame, frame-exact GT clip, GT camera, i2v/invdyn `.jsonl` |
| `prep_forward_dynamics.py` | cosmos | fd inputs (device frame) + camera_action JSON (+×8); patch i2v aspect→1,1 |
| `prep_fd_cosmos_frame.py` | cosmos | corrected-frame (`rgb + Rz(−90)`) → `gt_camera_cosmos.npz`, fd inputs |
| `viz_invdyn_camera.py` | kimodo | static invdyn plot: Umeyama 3D path + per-step `‖Δt‖`/angle + dir-cosine (`--gt_name`) |
| `viz_invdyn_video.py` | kimodo | animated invdyn 3D path, **no rotation-align**: pred vs GT-corrected vs GT-device |
| `test_camera_frame.py` | kimodo | frame discriminator (cosine vs model over device/rgb/rgb+Rz) + invariance checks |
| `viz_vggt_compare.py` | kimodo | GT vs Cosmos-invdyn vs VGGT-Omega trajectories with camera frusta (Umeyama-aligned) |
| `run_vggt.py` | cosmos | VGGT-Omega camera-pose prediction on clips → `vggt_cameras.npz` |
| `nymeria_camera_rgb_dataset.py` | cosmos | **Phase-2 dataset** `get_nymeria_camera_sft_dataset` — preprocessed upright-RGB poses, 4-task mixture, 10% text-drop |
| `launch_camera_phase2.sh` | cosmos | ssh/manual launcher (cd+env+torchrun); args `NPROC MAX_ITER LOG [opts]` |
| `sbatch_camera_phase2.sh` | cosmos | **sbatch** launcher (a3ultra, 8 GPU); env MAX_ITER/SAVE_ITER/BATCH/NPROC/GRAD_ACCUM/RUN_NAME/NYMERIA_NUM_FRAMES/NYMERIA_DROP_MODES/NYMERIA_MODE/EXTRA_OPTS; sets TB_LOG_DIR |
| `make_train_test_split.py` | cosmos | deterministic per-sequence 90:10 split → `train_test_split.json` (642 train / 71 test) |
| **`export_merge_lora.py`** | cosmos | merge a trainable-only delta (LoRA `W+=2·B·A` + action heads, + net_ema mirror) into base → full DCP for inference |
| **`prep_test_eval.py`** | cosmos | 3-task inference inputs (invdyn/fd/policy) from HELD-OUT test seqs, upright-RGB frame |
| **`run_infer_merged.sh`** | cosmos | inference launcher (cd+env): native inference on a merged DCP via the experiment (`lora_enabled=false`, local VAE) |
| **`sbatch_infer_3tasks.sh`** | cosmos | dedicated-node 3-task sampling from a merged DCP |
| **`viz_eval_samples.py`** | kimodo | **eval viz**: invdyn camera **frusta** (orientation) pred-vs-GT (Umeyama) → `invdyn_camera.png`; fd/policy GT‖gen video stacks |
| (pipeline) `../../nymeria_kimodo_pipeline/camera/preprocess_camera_rgb.py` | nymeria_plus | T_world_device → upright-RGB poses + 9D actions (all 728 seqs) |

Envs: `cosmos` (training/inference + framework imports), `kimodo` (numpy/matplotlib viz),
`nymeria_plus` (projectaria_tools / VRS). Run framework code with
`export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world; unset LD_LIBRARY_PATH`.

## 6. Reproduce — zero-shot baseline

Checkpoint (local diffusers, ships trained action heads):
`/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/03c14e74a6ddb51985d614b75d70f2443efc6a05`

Inference must use **`--no-guardrails`** (node has no `uvx`; guardrail download fails — harmless for our data).
Input files must be **`.jsonl`** (one record/line; `.json` = a single record). Single-GPU = plain `python -m` (no torchrun).
```bash
# on a GPU node, from cosmos-framework, env: cosmos, PYTHONPATH=., unset LD_LIBRARY_PATH
python -m cosmos_framework.scripts.inference -i <input>.jsonl -o <out> --no-guardrails \
  --checkpoint-path <CKPT>
```
- image2video → `<out>/<name>/vision.mp4` (set `aspect_ratio:"1,1"` for square ego framing).
- forward_dynamics → needs `action_path` JSON `[T,9]`; output `vision.mp4`.
- inverse_dynamics → needs only the video; predicted `[T,9]` in `<out>/<name>/sample_outputs.json` → `outputs[0].content.action`.

Artifacts: `/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline/`
(`i2v_sq_out/`, `fd_cosmos_out/` triples, `invdyn_out/` plots+`invdyn_traj.mp4`, `invdyn_fps_out/fps_sweep.png`).

## 6b. Phase 2 — camera-only world-model training (LoRA, 4-task mixture) — RUNNING

Native Cosmos training (`cosmos_framework.scripts.train`). **4-task mixture** per sample
(`MODE_WEIGHTS = {forward_dynamics .40, inverse_dynamics .25, policy .20, image2video .15}`);
text dropped 10% → CFG-null; inverse_dynamics has **no** text:

| task | mode | conditions | targets |
|---|---|---|---|
| video→camera | `inverse_dynamics` | video (no text) | camera action |
| (img+action[+text])→video | `forward_dynamics` | first frame + camera action [+ caption] | video |
| img→(action+video[+text]) | `policy` | first frame [+ caption] | camera action + video |
| img→video[+text] | `image2video` | first frame [+ caption] | video |

### What trains — two modes (env `NYMERIA_FULL_FT`)
- **LoRA (default).** Reasoner FROZEN; **LoRA on the generator** attention (`q/k/v/o_proj_moe_gen`, rank 16 /
  alpha 32 → 15.3M) + **fine-tuned PRETRAINED camera action heads** (`action2llm`/`llm2action`/`action_modality_embed`,
  domain `camera_pose`, 16.9M — loaded & adapted, **not** re-initialized). Total trainable **32.2M**. lr 2e-4.
- **Full-gen (`NYMERIA_FULL_FT=1`).** Reasoner FROZEN; **full-parameter finetune of the whole generation pathway**
  `keys_to_select=[moe_gen, time_embedder, vae2llm, llm2vae, action2llm, llm2action, action_modality_embed]`.
  Total trainable **~7B** (`lora_enabled=False`). lr 1e-4 (lower, to protect the pretrained generator).
  **MUST use FSDP sharding** (`dp_shard=8`) — see Parallelism below; replicate OOMs.

### Parallelism — LoRA: replicate (`dp_shard=1`); full-gen: sharded (`dp_shard=8`)
The native stack always runs FSDP2 `fully_shard` on multi-GPU (`dp_enabled` is forced True when world_size>1;
`distributed_parallelism="ddp"` would wrap torch-DDP on the FSDP mesh → crash). Two regimes:
- **LoRA → `dp_shard=1`** (TOML default): `fully_shard` in **pure-replicate mode — NO param sharding** (full 16B
  per GPU, grad all-reduce = functionally DDP). Best for LoRA (tiny optimizer; avoids redundant base all-gather).
  ~100 GB/GPU.
- **Full-gen → `dp_shard=8`** (override `model.config.parallelism.data_parallel_shard_degree=8` via `EXTRA_OPTS`):
  **MEASURED**: replicate **OOMs** — the ~7B trainable params' fp32 Adam state is **~84 GB**, and replicated
  per-GPU that + the 16B model + video activations hit 143 GB → OOM at the optimizer step (pre-step it sat at
  ~96 GB, deceptively fine). Sharding spreads the 84 GB optimizer + the model across 8 GPUs → **~40 GB/GPU**,
  fast (~1.5 s/iter at 97f). So full-gen REQUIRES sharding; LoRA does not.

### Recipe
FusedAdamW **lr 2e-4** (LoRA needs the higher rate; uniform, no per-head multiplier), wd 0.05, grad-clip 1.0,
LambdaLinear(start 0.4, cycle 100k, warmup 500), **action_loss_weight 10**, **256p shift 3**, batch 16 (×8 GPU).

### Pieces / framework changes
- Dataset: `nymeria_camera_rgb_dataset.py` (`get_nymeria_camera_sft_dataset`) — reads preprocessed
  `camera_rgb` upright poses, computes per-frame 9D actions, emits the mixture (inverse_dynamics → empty caption).
- Experiment config: `cosmos-framework/.../action/posttrain_config/world_camera_nymeria_nano.py`
  (imported in `configs/base/config.py`); TOML `examples/toml/sft_config/world_camera_nymeria_repro.toml`.
- **Framework edits (for LoRA + extra-trainable heads):**
  - `configs/base/defaults/model_config.py` — added `lora_keep_trainable_modules: str = ""` field.
  - `model/vfm/omni_mot_model.py` `build_net` — after LoRA injection, re-enable params matching
    `lora_keep_trainable_modules` (LoRA otherwise freezes ALL non-LoRA, and `build_optimizer` can only narrow
    already-trainable params, so `keys_to_select` alone can't re-enable the heads). Set to the action heads.
  - `callbacks/tensorboard_log.py` (new) + registered in `configs/base/defaults/callbacks.py` `BASIC_CALLBACKS`.
- Prereqs (one-time, done): Nano diffusers→DCP `python -m cosmos_framework.scripts.convert_model_to_dcp
  --checkpoint-path Cosmos3-Nano -o /weka/jungbin/cosmos3_nano_dcp`; Wan VAE (HF `Wan-AI/Wan2.2-TI2V-5B`,
  `Wan2.2_VAE.pth`) → `/weka/jungbin/wan22_vae/Wan2.2_VAE.pth`. **Training needs DCP — it cannot load the
  diffusers snapshot (only inference can).**
- **1-step smoke PASSED** (final config): DCP load, LoRA inject (144 modules), action heads kept trainable
  (32.2M), Wan-VAE encode, MoT forward, RF loss (video + action×10), FSDP-replicate backward, ckpt save.

### Hard-won multi-GPU findings (load-bearing)
- **image2video is DROPPED** (`NYMERIA_DROP_MODES=image2video`). Root cause (isolated via single-GPU works →
  single-task works → no-i2v works): i2v has **no action**, so on an i2v step no rank uses the trainable action
  heads → those params get no gradient → **distributed collective desync / hang**. Random mixture limped to
  iter 5–14 (some ranks had action tasks); long 97-frame sequences made it worse. Not compile/batch/fps/memory.
  We keep the **3 camera tasks** (forward/inverse/policy) — exactly the camera-world tasks; i2v (img→video, no
  camera) is least relevant anyway.
- **Synchronized task-per-step (kept):** one stream per task via the JointDataLoader (it selects the same stream
  for all ranks each step via `seed+global_id`) — `_DATASETS` built per task in the experiment config. Robust;
  but it does **not** save i2v (an all-i2v step still has zero action grads → must drop i2v regardless).
- **No torch.compile for 97-frame** (`model.config.compile.enabled=false`) — long-seq + mixture recompiles can
  deadlock; 33-frame is fine with compile on.
- **Per-frame batch:** 33f batch 16; 97f batch **4–8** (97 frames → 25 VAE latents, ~2.7× tokens). Memory floor
  is the resident 16B (full activation-checkpointing), so batch barely moves it; use `GRAD_ACCUM` for larger
  effective batch without OOM.

### Train/test split (`make_train_test_split.py` → `train_test_split.json`)
Random **per-sequence** 90:10 (hold out whole recordings — a window-level split would leak, windows in one
recording are near-identical). **642 train / 71 test** seqs (127,407 / 13,406 windows). Dataset filters by
`split` (default `"train"`); test is held out for eval.

### Checkpoints — LoRA + action heads only (~284–437 MB)
`SAVE_TRAINABLE_ONLY=1` saves only trainable params (no 16B base/EMA). **BUG FIXED:** the original filter matched
`named_parameters()` FQNs vs `get_model_state_dict` keys — under FSDP the deep LoRA params mismatched and were
**silently dropped** (only action heads saved). Fixed to substring-match the DCP keys (`lora_A`/`lora_B` +
`SAVE_TRAINABLE_KEYS`) in `checkpoint/dcp.py` `trainable_state_dict`. Verified: a checkpoint now has 576 LoRA +
10 head keys (was 0 LoRA). Reconstruct full model at eval by merging onto the base (see §6c).

## 6c. Sampling / eval (LoRA checkpoint → 3-task inference vs GT)

Our checkpoints are LoRA+head **deltas on top of the base** — native inference has **no LoRA-at-inference and no
two-checkpoint overlay**, so we **merge then sample**:
1. **Merge** (`export_merge_lora.py`, state-dict level, no 16B build): `W_gen += (α/r)·B·A` for each LoRA layer,
   overwrite the fine-tuned action heads, mirror `net.*→net_ema.*`, copy `config.json` to the dir top →
   one **full DCP** (~59 GB). Reuses a cached `base.pt`.
2. **Prep** (`prep_test_eval.py`): invdyn/fd/policy `.jsonl` + GT clips/cameras from **held-out test** seqs
   (97-frame → `nymeria_eval/`, 33-frame → `nymeria_eval_33/`).
3. **Sample** (`run_infer_merged.sh` or `sbatch_infer_3tasks.sh`): native inference via
   `--experiment world_camera_nymeria_nano --experiment-overrides model.config.lora_enabled=false
   model.config.tokenizer.vae_path=<local Wan VAE>`. invdyn → predicted `[T,9]` action JSON; fd/policy → `vision.mp4`.
   Outputs saved **into the checkpoint dir** `…/iter_NNNNNN/samples/{invdyn,fd,policy}_out/`.
4. **Viz** (`viz_eval_samples.py`, kimodo): invdyn **camera frusta** (pred vs GT, Umeyama-aligned, orientation
   shown) → `samples/viz/invdyn_camera.png`; fd/policy GT‖generated stacks → `samples/viz/<subj>_<task>.mp4`.

**Gotchas:** inference needs `cd cosmos-framework` (relative config path), `--no-guardrails` (no `uvx`), the local
VAE override, and `lora_enabled=false` (merged weights have no LoRA modules); co-locate on a node with >~40 GB
free GPU or cuSOLVER (UniPC solver) fails. **iter-2500 first result:** invdyn predicted `|Δt|` ≈ 0.05 m (zero-shot
was ~0.14–0.23, GT ~0.02) — scale gap already collapsing from ~8× → ~2.5× as the action heads adapt.

### Launch
- **sbatch** (preferred): `sbatch nymeria_world/sbatch_camera_phase2.sh` (a3ultra, 1 node, 8 GPU; env
  `MAX_ITER`/`SAVE_ITER`/`BATCH`). Sets `TB_LOG_DIR`. Run-log path written to
  `/weka/jungbin/cosmos_motion_ft_runs/world_camera_LATEST.txt`; slurm out `slurm-camworld-<jobid>.out`.
- manual: `bash nymeria_world/launch_camera_phase2.sh <NPROC> <MAX_ITER> <LOG> [opts]` (ssh into a node).
- **Gotchas:** must `cd cosmos-framework` (hardcoded relative config paths); use the conda env's full
  `torchrun` path; `opts` are POSITIONAL (not `--opts`); `--no-guardrails` for any inference.

### TensorBoard
`callbacks/tensorboard_log.py` logs **`train/loss`** + `train/loss_ema` (rank-0, every `logging_iter`) to
`$TB_LOG_DIR` = `/weka/jungbin/cosmos_motion_ft_runs/tensorboard/world_camera_nymeria`.
View: `tensorboard --logdir /weka/jungbin/cosmos_motion_ft_runs/tensorboard`.

## 7. Status & next

- **Dataset + preprocessing DONE**: 728/728 seqs → `camera_rgb/` upright poses + 9D actions; 141k 33-frame windows.
- **Zero-shot baseline DONE** (15 clips): camera frame fix `rgb+Rz(−90)`; ~7–17× "scale gap" = action
  temporal-step mismatch (NOT coords/fps); VGGT-Omega aux reference (Umeyama ATE ~0.05 m).
- **Phase 2 training RUNNING** (sbatch on a3ultra, synchronized 3-task streams, i2v dropped, checkpoints every 1k):
  - `world_camera_nymeria_97f` — **LoRA**, 97-frame, batch 8, compile off, FSDP-replicate, lr 2e-4 (caption-aligned).
  - `world_camera_nymeria_33f` — **LoRA**, 33-frame, batch 16, compile on, FSDP-replicate, lr 2e-4 (more windows).
  - `world_camera_nymeria_fullft_97f` — **FULL-GEN** (~7B), 97-frame, batch 16, compile off, **FSDP-sharded ×8**,
    lr 1e-4 (`NYMERIA_FULL_FT=1` + `EXTRA_OPTS=...data_parallel_shard_degree=8`). For LoRA-vs-full-gen comparison.
- **Sampling/eval DONE (first pass)**: 3-task samples on held-out test seqs at 97f iter-2500 & 33f iter-3000,
  saved in each ckpt's `samples/` (+ `viz/`). invdyn camera-scale already ~2.5× GT (from ~8× zero-shot).
- **Next**: let runs train up; re-merge+sample at later milestones (e.g. 10k) with `export_merge_lora.py` +
  `run_infer_merged.sh` + `viz_eval_samples.py`; compare 97f vs 33f; consider coarser action stride / 480p.
