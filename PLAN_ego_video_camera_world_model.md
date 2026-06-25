# Plan: Mid-train Cosmos-3 Nano as an egocentric camera-action video world model (NymeriaPlus)

> **As-built documentation:** [`nymeria_world/README.md`](nymeria_world/README.md) — dataset, action
> representation, the OpenCV camera-frame finding, zero-shot baselines, and the scale-gap/fps analysis.

**Date:** 2026-06-19
**Scope (user re-scope):** Ignore human body motion and audio. Use only **egocentric video + text + ego-camera**,
with the ego-camera treated **exactly as Cosmos-3's native "action" modality**, following Cosmos-3's
**mid-training** recipe (paper §4.2.2). This is the world-model direction, native to Cosmos — we reuse the
existing Wan2.2-VAE video path + `action2llm`/`llm2action` heads + frozen Qwen3-VL text reasoner. **We do NOT
use `train_motion_ft.py`** (that was the bespoke 369-d motion trainer); we use Cosmos's native training stack.

---

## 0. What Cosmos's mid-training does with these 3 modalities (the recipe we copy)

From COSMOS.md §2.1.3 (Action), §4.2.2 (Mid-Training):
- **Action = relative-pose pseudo-action.** For consecutive SE(3) poses, `ΔT_t = T_{t-1}^{-1} T_t`.
  Camera motion = **9D ego-pose** = 3D translation + 6D rotation (Zhou et al. 2019, OpenCV convention).
  Domain-aware in/out projections (`DomainAwareLinear`) per embodiment; decode 6D→SO(3) via SVD.
- **Action stream is text–video–action, trained in 3 modes:** forward dynamics (action→future video),
  inverse dynamics (video→action), and policy (joint action+video). Clean-prefix / noisy-target formulation
  shared with T2V/I2V/V2V so visual priors stay active.
- **Objective:** rectified-flow velocity MSE per modality (`v = ε − x0`); **action loss ×10** (small per-element
  MSE of normalized action vectors); action inherits the vision noise schedule.
- **Optimizer:** FusedAdamW, **lr 1e-4** (mid-training; DROID post-train recipe uses 2e-4), wd 0.05,
  grad-clip 1.0, **loss-scale 10**, LambdaLinear schedule (start factor 0.4, cycle 100k).
- **Multi-resolution** 256p/480p/720p with rectified-flow **shift 3 / 5 / 10**. We start at **256p (shift 3)**.

## 1. Existing Cosmos machinery we reuse (verified, file:line)

| Need | Where it already exists |
| --- | --- |
| Camera action domain | `data/vfm/action/domain_utils.py` → `camera_pose` = **domain id 2, raw dim 9** |
| 9D spec (3D pos + 6D rot) | `data/vfm/action/action_spec.py:19` `build_action_spec(Pos(), Rot("rot6d"))` |
| ΔT=T⁻¹T pseudo-action | `data/vfm/action/pose_utils.py:439` `pose_abs_to_rel(rotation_format="rot6d", pose_convention="backward_framewise")`; inverse `pose_rel_to_abs` |
| Action heads | `model/vfm/mot/cosmos3_vfm_network.py:185-211` `action2llm`/`llm2action` (`DomainAwareLinear`) + `action_modality_embed`; pad to `max_action_dim=64`, mask via `raw_action_dim` |
| Video (Wan2.2 VAE) | `model/vfm/tokenizers/wan2pt2_vae_4x16x16.py`; on-the-fly encode in `omni_mot_model.get_data_and_condition`; `_normalize_video_databatch_inplace` ([0,255]→[-1,1]) |
| Action↔video interleave | temporal-causal supertokens `[a0,v0,a1,v1,…]`, `action_start_frame_offset=1` (`sequence_packing.py`, `transforms.py:240` `build_sequence_plan_from_mode`) |
| RF loss + action×10 | `omni_mot_model.py:_compute_losses` (`action_loss_weight=10.0`) |
| Training entry to clone | `configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py` + `examples/toml/sft_config/action_policy_droid_repro.toml`; launch `torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train --sft-toml <toml>` |
| Sampling | `inference/defaults/forward_dynamics/sample_args.json` (num_steps 30, guidance 1.0, shift 10, 256p); `inference/action.py`, `scripts/inference.py` |

**Model config base:** `NANO_MODEL_CONFIG` (`configs/base/experiment/sft/models/nano_model_config.py`) — already sets
`action_gen=True`, `vision_gen=True`, `max_action_dim=64`, `num_embodiment_domains=32`, `sound_gen=False`.
**Trainable set (selective full-finetune, reasoner frozen):**
`keys_to_select=["moe_gen","time_embedder","vae2llm","llm2vae","action2llm","llm2action","action_modality_embed"]`;
**re-init action heads** via `keys_to_skip_loading` (they're not camera-trained in the base ckpt);
`lr_multipliers` 5.0 on the action heads.

## 2. Data status (verified on disk)

`/weka/jungbin/nymeriaplus_kimodo_proportional/`
- `video/{Sxx}/{seq}.mp4` — **done**: 640², 20fps, frame i == camera frame i. `video/manifest_video.jsonl` (64 MB)
  has per-seq `t2w_windows` with `start/end_frame` (100-frame / 5s windows) + **caption** (rich narration) +
  `camera_path` + `motion_path` + floor/foot-skating flags + `usable`.
- `camera/{Sxx}/{seq}.npz` — **done** (19 subjects): `cam_world_pos (T,3)`, `cam_world_rot (T,3,3)` (R_world_device,
  Aria SLAM world), `timestamps_us`. Frame-aligned to video.
- **Captions** present and dense (atomic-action descriptions). Audio ignored.

**Remaining data work:** convert camera SE(3) → 9D camera pseudo-actions, write a dataset class, compute action
normalization stats, and re-time windows to the VAE's `T = 4N+1` requirement.

## 3. New code to write (small; mostly glue around existing utils)

1. **`nymeria_camera_action.py`** — pose→action util.
   `T_world_device[i] = [[cam_world_rot[i], cam_world_pos[i]],[0,1]]`, then
   `pose_abs_to_rel(T, rotation_format="rot6d", pose_convention="backward_framewise")` → `(T-1, 9)`.
   (World-frame choice cancels in `T⁻¹T`, so Aria Z-up needs no remap; optional device→OpenCV axis map only for
   interpretability since action heads are re-init from scratch.) Pad 9→64, `domain_id=2`.
2. **`NymeriaPlusActionSFTDataset`** — modeled on `DROIDLeRobotDataset`/`ActionSFTDataset`, emitting the Cosmos
   sample contract per window: `video [3,T,H,W] uint8`, `action [T,64]` (+`raw_action_dim=9`), `ai_caption`,
   `domain_id=2`, `mode` (forward_dynamics / inverse_dynamics / policy), `fps`. Reads `manifest_video.jsonl`,
   decodes the mp4 window, slices the camera npz, runs util #1. Filter `usable==true`.
3. ~~Action normalization JSON~~ **NOT NEEDED — verified.** Cosmos applies **no** mean/std action
   normalization for camera: `omni_mot_model._normalize_action_databatch` only densifies (no scaling), and
   `droid_lerobot_dataset.py:431` uses the raw action when `action_normalization is None` (the camera/pose
   recipes use None). The 9D pseudo-action (`pose_abs_to_rel(rot6d, backward_framewise)`, scales=1.0) **is** the
   representation. Adding mean/std would *diverge* from Cosmos. → `action_normalization=None`.
4. **Experiment config** `world_camera_nymeria_nano.py` (clone of `action_policy_droid_nano.py`) + a run **TOML**:
   point dataset at our manifest, embodiment `camera_pose`, mode mixture, 256p, shift 3, FusedAdamW lr 1e-4,
   wd 0.05, clip 1.0, loss-scale 10, LambdaLinear(0.4, 100k), keys_to_select/skip/lr_mult as above.
5. **Eval scripts** — forward_dynamics (text+action→video, decode Wan VAE→mp4) and inverse_dynamics
   (video→action, report ΔT pose error).

## STATUS (2026-06-19)

- **Phase 1 core DONE & verified** (`/home/jungbin_cho/cosmos_motion_ft/nymeria_world/`):
  - `camera_to_action.py` — SE(3)→9D camera pseudo-action; round-trip err ~5e-4 across all 728 seqs, 0 non-finite;
    matches Cosmos exactly (`pose_abs_to_rel(rot6d, backward_framewise)`, no scales, no mean/std).
  - `nymeria_camera_dataset.py` — `NymeriaPlusCameraActionDataset` (PyAV windowed decode) emitting the exact Cosmos
    action contract; **141,391 usable 33-frame windows**; verified through the real `ActionTransformPipeline`
    (video `[3,33,256,256]`, action `[32,64]` + `raw_action_dim=9`, domain 2) with correct per-mode sequence plans.
- **Data-quality note:** ≥1 seq has a ~0.94 m single-frame camera-translation jump (tracking discontinuity) — add an
  outlier filter/clamp before full training.
- **Remaining for Phase 1:** GPU 1-step training smoke (needs the Phase-2 experiment config to build the Nano model).

## 4. Phases

**Phase 0 — Data verification & action stats (no training).**
Verify video/camera/caption frame-alignment & window count; build 9D camera actions for all `usable` windows;
**round-trip test** `pose_abs_to_rel`→`pose_rel_to_abs` (recover absolute trajectory within tol); compute action
normalization JSON; confirm window re-timing to `T=4N+1` (e.g. 100-frame window → 97 = 4·24+1, or use 33-frame
sub-windows) and target fps. Tally usable clip count (drives compute estimate).

**Phase 1 — Dataset class + 1-step smoke.**
`NymeriaPlusActionSFTDataset` emits the contract; inspect one batch (shapes, action range after norm, domain_id,
text ids). Run a **single training step** (`forward_dynamics`) on 2–4 clips at 256p: confirm RF losses finite
(vision FM + action FM×10), memory fits on one H200, action heads receiving grad.

**Phase 2 — Experiment config wired.**
Finalize `world_camera_nymeria_nano.py` + TOML; confirm startup log shows reasoner frozen, gen+heads trainable,
action heads re-initialized, optimizer/schedule per recipe, mode mixture loaded.

**Phase 3 — Train (mid-training recipe).**
Launch `torchrun -m cosmos_framework.scripts.train --sft-toml …` on a free node. Mode mixture (recommended
**50% forward / 25% inverse / 25% policy**). Start **256p (shift 3)**, 33–97-frame windows. Monitor per-modality
losses; checkpoint + sample every N steps. Optionally ramp to 480p (shift 5) later.

**Phase 4 — Eval.**
- *Forward dynamics:* given text + a camera-action sequence (+1 clean start frame), generate video
  (num_steps 30, shift 10, guidance 1.0), decode Wan VAE → mp4; check the video follows the commanded camera path.
- *Inverse dynamics:* given a real video, predict the camera-action; report per-frame ΔT translation/rotation error
  vs the GT camera trajectory.
- Qualitative side-by-sides; send clips.

## 5. Scope / risk callouts

- **Heavier than the motion run.** Video generation encodes Wan-VAE latents every step → larger activations/memory.
  Start 256p + short windows; expect lower throughput than the 369-d motion trainer.
- **Action heads are init-from-scratch** (camera not in base ckpt) → need warmup; hence the 5× head LR and ×10
  action loss (both already in the recipe).
- **Single domain (19 subjects, egocentric daily activity)** → domain-specialization finetune. The clean-prefix
  formulation preserves the visual prior; optionally mix a little generic T2V/I2V to further guard against
  forgetting (no such data staged yet — accept some specialization for v1).
- **Reasoner stays frozen** — text understanding reuses the pretrained Qwen3-VL prior; no new understanding params.

## 6. Small defaults chosen (tunable, not blocking)

- LR **1e-4** (paper mid-training) — switch to 2e-4 if action heads converge slowly.
- Window **97 frames** (≈4.85 s, =4·24+1) from the existing 100-frame windows, or 33-frame sub-windows for cheaper
  early iters. Resolution **256p** to start. Mode mixture **50/25/25**.
