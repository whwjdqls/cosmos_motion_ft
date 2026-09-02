# Restored Data, Models, and Fixture Path Reference

**Host Base Path**: `/mnt/projects/ll/jungbinc`  
**Root Environment Variables**:
```bash
export WEKA_ROOT="/mnt/projects/ll/jungbinc/weka"
export RUN_ROOT="${WEKA_ROOT}/cosmos_motion_ft_runs"
export TORCH_HOME="/mnt/projects/ll/jungbinc/.cache/torch"
export REPO_ROOT="/home/jungbinc/cosmos_motion_ft"
export COSMOS_FRAMEWORK_ROOT="/mnt/projects/ll/jungbinc/cosmos-framework"
```

---

## 1. Derived Datasets (`$WEKA_ROOT/`)

| Dataset / Component | Exact Local Path | Description | Size |
|---|---|---|---|
| **Nymeria Proportional Motion** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/{S01..S20}/*.npz` | Shape-aware SMPL→SOMA proportional motion representations (`local_rot_mats (T,77,3,3)`, `neutral_joints (77,3)`). | ~30 GB |
| **Nymeria UniEgo Representation** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep/{S01..S20}/*.npz` | 283-D UniEgo motion format for Phase-2/3 models (30 joints rot/pos + canonical delta + foot contacts). | 10 GB |
| **Nymeria Camera-aligned UniEgo V1** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1/{S01..S20}/*.npz` | Versioned 283-D corpus with Head world rotation aligned to upright RGB-camera orientation and full re-canonicalization; original corpus is unchanged. | 14 GB |
| **Nymeria Head Camera Poses** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/camera/{S01..S20}/*.npz` | Aria `T_world_device` raw head camera trajectories sampled at 20 fps. | 757 MB |
| **Nymeria Upright Camera RGB** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/camera_rgb/{S01..S20}/*.npz` | Upright RGB-sensor world poses (OpenCV/Kimodo Y-up) used for all camera action training. | 502 MB |
| **Nymeria Video Clips** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/video/{S01..S20}/*.mp4` | Egocentric RGB video clips (640² @ 20 fps), frame-aligned to motion. | ~265 GB |
| **Nymeria Video Manifest** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl` | Master mapping aligning video, camera, motion, and narration captions. | 64 MB |
| **Train/Test Split** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/train_test_split.json` | Sequence-level split (71 held-out sequences). | 34 KB |
| **Floor Calibration** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/metadata/floor_calibration.json` | Per-sequence floor height offsets and dropped window list. | 490 KB |
| **Physical Quality Filter** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json` | T97 window exclusion mask (removes physical camera artifacts/jitter). | 1.8 MB |
| **SEED Proportional Motion** | `/mnt/projects/ll/jungbinc/weka/seed/soma_proportional_motions_20fps/` | BONES-SEED dataset converted to proportional SOMA format at 20 fps. | 136 GB |
| **SEED Proportional UniEgo** | `/mnt/projects/ll/jungbinc/weka/seed/soma_proportional_uniegomotion_20fps/` | BONES 283-D UniEgo motions + `Mean_uniego.npy` / `Std_uniego.npy`. | 24 GB |
| **SEED Uniform Motion & Stats** | `/mnt/projects/ll/jungbinc/weka/seed/soma_uniform_motions_20fps/`<br>`/mnt/projects/ll/jungbinc/weka/seed/stats/` | Root 369-D uniform motion tree and normalization arrays. | ~136 GB |
| **SEED Metadata** | `/mnt/projects/ll/jungbinc/weka/seed/metadata/`<br>`/mnt/projects/ll/jungbinc/weka/seed/multi_timeline.jsonl` | Single and multi-timeline narration annotations (`seed_metadata_v004.csv/parquet`). | ~400 MB |
| **Kimodo Benchmark 20fps** | `/mnt/projects/ll/jungbinc/weka/Kimodo-Motion-Gen-Benchmark-20fps/` | Kimodo motion generation benchmark resampled to 20 fps (`testsuite/`). | 14 GB |
| **Kimodo Benchmark Splits** | `/mnt/projects/ll/jungbinc/weka/Kimodo-Motion-Gen-Benchmark/splits/` | Split partition files for benchmark evaluation. | 5.9 MB |

---

## 2. Training Pairs & Window Indices (`$RUN_ROOT/joint_attention/` & Repo)

| File | Exact Path | Description |
|---|---|---|
| **Nymeria Pairs (Train)** | `/home/jungbinc/cosmos_motion_ft/motion_expert/pairs_train.jsonl`<br>`/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/nymeria_pairs_train.jsonl` | Active Phase-2/3 Nymeria text–motion paired training examples. |
| **Nymeria Pairs (Val)** | `/home/jungbinc/cosmos_motion_ft/motion_expert/pairs_val.jsonl`<br>`/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/nymeria_pairs_val.jsonl` | Active Phase-2/3 Nymeria validation pairs. |
| **BONES Pairs (Train)** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/bones_pairs_train.jsonl` | Resolved BONES 4th-overview caption pairs for joint-attention training. |
| **BONES Pairs (Val)** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/bones_pairs_val.jsonl` | Resolved BONES validation pairs. |
| **BONES Index (Train/Val)** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/bones_index_{train,val}.json` | Index lookup tables for fast multi-stream dataloading. |
| **Canonical 71 Windows** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/full71_windows.json` | 71 held-out test windows for reproducible benchmarking. |
| **Loss Bomb Blocklist** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/joint_attention/bomb_windows.json` | Known unstable/loss-bomb window exclusions. |

---

## 3. Model Checkpoints (`$RUN_ROOT/`)

### Phase 1: Native Camera / Generator DCPs
Located under: `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world/`

| Model Variant | Checkpoint Path | Type / Contents |
|---|---|---|
| **Phase 1 VQ-A** | `.../native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000/` | Full resumable DCP (`model/`, `optim/`, `scheduler/`, `trainer/`). Global LoRA + action head. |
| **Phase 1 VQ-B** | `.../native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000/` | Full resumable DCP. Variable-prefix training. |
| **Phase 1 VQ-D** | `.../native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000/` | Full resumable DCP. Cross-attention KV LoRA variant. |
| **Phase 1 Baseline Delta** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/portable/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt` | Compact 293-tensor EMA adapter (129 MB) for initializing Phase-3 bridge models. |

### Phase 2: Motion Expert Models
Located under: `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/`

| Model Variant | Checkpoint Path | Description |
|---|---|---|
| **Phase 2 Native Schedule** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt` | Production 200k-step Phase-2 motion expert (x₀-prediction, mROPE 3D). |
| **Phase 2 Contact Loss** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt` | Phase-2 model trained with auxiliary foot-contact loss. |

### Phase 3: Bidirectional Joint Attention Bridge Models
Located under: `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/`

| Model Variant | Checkpoint Path | Description |
|---|---|---|
| **Phase 3 Vanilla Bridge** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt` | 200k-step bidirectional co-generation bridge model (V2M + M2V). |
| **Phase 3 Head-Camera** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/ckpt_step115000.pt` | Bridge model with rigid head-joint→camera auxiliary loss. |
| **Phase 3 Multitask** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask/ckpt_step065000.pt` | Multi-task joint camera and motion generation. |
| **Phase 3 Contact Init** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k/ckpt_step035000.pt` | Phase-3 bridge initialized from Phase-2 contact checkpoint. |

---

## 4. Evaluation Models & Metric Weights

| Evaluator / Model | Exact Path | Role / Details |
|---|---|---|
| **DreamSim Backbone** | `/mnt/projects/ll/jungbinc/weka/model_cache/dreamsim/` | Perceptual visual distance metric. |
| **FVD (VideoMAE-g)** | `/mnt/projects/ll/jungbinc/weka/model_cache/cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth` | Fréchet Video Distance feature extractor. |
| **LPIPS AlexNet** | `/mnt/projects/ll/jungbinc/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` | Offline LPIPS backbone for image/frame quality metrics. |
| **C45 Shape-Aware Bundle** | `/mnt/projects/ll/jungbinc/weka/shape_aware_motion_eval_c45_20260715/` | Complete self-contained C45 evaluation suite (evaluator model, 190-D stats, LLM2Vec cache). |

---

## 5. Evaluation Fixtures (`$RUN_ROOT/`)

| Fixture Suite | Exact Path | Description |
|---|---|---|
| **Phase 1 Full 71 Suite** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2/` | Canonical held-out 71 sequence inputs for forward/inverse dynamics and policy evaluation. |
| **Phase 1 VQ Prefix-5 Suite** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1/` | Standard benchmark inputs across fixed prefix frames (1, 9, 17, 33, 49). |

---

## 6. Downloaded Upstream Model Assets

| Asset | Exact Local Path | Pinned Revision |
|---|---|---|
| **Cosmos3-Nano** | `/mnt/projects/ll/jungbinc/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/7a312c868bcce8e40b3eb40861300a9d0ba3fde1/` | `7a312c868bcce8e40b3eb40861300a9d0ba3fde1` |
| **Wan2.2 TI2V 5B VAE** | `/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth` | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| **Cosmos3-Edge model assets** | `/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge/` | Model index SHA-256 `ee48f9da9fbab206b6d2902eb109a842dde7b1347f5716a177f0be00968acf33` |
| **Cosmos3-Edge DCP** | `/mnt/projects/ll/jungbinc/weka/cosmos3_edge_dcp/` | DCP metadata SHA-256 `f0d19c6bdbe43663e3a1a6fcb3437a5718d92565fd5db2fdc1ec499bc9d5e1ec` |
| **Cosmos3-Edge framework** | `/mnt/projects/ll/jungbinc/cosmos-framework-edge/` | Commit `d4599e2e43fbd06168e9884205b9b66c3902d8f6` |

The Hugging Face cache is intentionally under `/mnt/projects/ll/jungbinc`, not the
full `/home` filesystem. Source the restored runtime configuration before any run:

```bash
cd /home/jungbinc/cosmos_motion_ft
source restored_env.sh
```

This selects `/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos`, configures the
framework/repository `PYTHONPATH`, and exports the model, checkpoint, data, and
cache locations above.

## 7. Interactive L40 Inference

Keep the allocation in `tmux 0` so several inference checks reuse one scheduled GPU:

```bash
tmux attach -t 0
srun --partition=batch --nodes=1 --ntasks=1 --gres=gpu:l40:1 \
  --cpus-per-task=16 --mem=128G --time=03:00:00 \
  --job-name=cosmos-l40-interactive --pty bash -l

cd /home/jungbinc/cosmos_motion_ft
source restored_env.sh
```

The smoke launchers can be run directly inside that shell; their `#SBATCH` lines
are ignored by Bash. Outputs and per-second GPU-memory logs are written below
`$RUN_ROOT/l40_smoke/`.

## 8. Camera-aligned Head V1

| Artifact | Exact Path |
|---|---|
| **Historical relative Head-to-camera calibration** | `/home/jungbinc/cosmos_motion_ft/motion_expert_joint_attention/head_camera_calibration_train.json` |
| **Build manifest** | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1/camera_head_recanonicalization_manifest.json` |
| **Matching mean** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy` |
| **Matching std** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy` |
| **Stats population summary** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/summary.json` |
| **Full quantitative report** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/quantitative/comparison_report.json` |
| **Qualitative gallery** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative/gallery_manifest.json` |
| **Restored standard SOMA visualization skin** | `/mnt/projects/ll/jungbinc/weka/shape_aware_motion_eval_c45_20260715/code/kimodo_open/kimodo/assets/skeletons/somaskel77/skin_standard.npz` |
| **Original-vs-aligned SOMA mesh gallery** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative_soma_mesh/gallery_manifest.json` |
| **SOMA mesh comparison MP4s** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative_soma_mesh/videos/` |
| **SOMA mesh peak-impact posters** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative_soma_mesh/posters/` |
| **Experimental absolute-lever report** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit/absolute_lever_report.json` |
| **Experimental absolute-lever calibration** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit/head_camera_calibration_train_absolute_lever_v1.json` |
| **Absolute-lever trajectory comparison** | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit/diverse_absolute_lever_trajectory_comparison.png` |
| **Method and results** | `/home/jungbinc/cosmos_motion_ft/motion_expert_joint_attention/CAMERA_HEAD_RECANONICALIZATION.md` |

Activate all three representation-dependent paths together only for a new model:

```bash
cd /home/jungbinc/cosmos_motion_ft
source restored_env.sh
source motion_expert_joint_attention/use_camera_head_v1.sh
```

Historical checkpoints must keep the original `uniego_rep` and original statistics.
Start camera-aligned training with `--bones_frac 0`; BONES lacks synchronized camera
poses and therefore cannot provide the same corrected Head definition.

The 2026-08-31 absolute-lever refit is complete but remains experimental and opt-in.
On the exact quality-filter-clean held-out population it changes corrected absolute
optical-center error from 3.7000 cm to 3.4082 cm, relative translation from 2.4304 mm
to 2.4531 mm, and leaves the 0.002043-degree rotation error unchanged. Even a leaky
per-test-sequence lever reaches only 2.5710 cm, so a constant lever does not solve the
positional residual. Do not replace the historical calibration by default; see the
method document and embedded report contract before using the experimental JSON.

The restored SOMA skin above is Kimodo's standard 77-joint visualization surface, not
subject-specific SOMA-X identity geometry. The mesh gallery applies the original and
`camhead_v1` UniEgo rotations to this same controlled skin. Its twelve 1920x1080,
10-fps H.264 MP4s and posters visualize the Head correction while separately reporting
that decoded joint positions, non-Head rotations, foot heights, and contact skating
remain unchanged to floating-point tolerance.

## 9. Cosmos3-Edge Phase-2 T2M + TI2M

The Edge Phase-2 implementation is isolated at
`/home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge`. It trains a new
T2M+TI2M motion expert from the released Edge base and does not consume
Phase-1 camera/video adapters or Nano Phase-2 checkpoints. TI2M uses the frozen
reasoner vision tower; it does not add generator rows or three-way attention.
Its pinned architecture is:

- Edge shared residual/head geometry: hidden 2048, Q heads 16, KV heads 8,
  head dim 128;
- seven trainable motion blocks: `[3,7,11,15,19,23,27]`;
- smaller original motion FFN: fresh SwiGLU width 3072 (Edge reasoner FFN is
  width 9216);
- task weights 0.75 Nymeria T2M / 0.25 Nymeria TI2M, with `bones_frac=0` by
  default; BONES remains available only as an explicit later ablation;
- Nymeria T2M uses native caption windows; TI2M uses 97 synchronized valid
  frames padded/masked to T=200 and a 256x256 frame-0 reasoner image;
- standalone uppercase `C` is expanded to sentence-aware `The camera wearer` /
  `the camera wearer` in both Phase 1 and Phase 2;
- `camera_head_recanonicalization_v1`, matching v1 stats, and floor calibration.
  BONES is a legacy motion-only source without synchronized camera data and is
  explicitly not labeled camera/head-equivalent.

All commands must use the wrapper so Nano and Edge framework imports cannot be
mixed:

```bash
bash motion_expert_t2m_edge/run.sh -m unittest discover \
  -s /home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge/tests -v
bash motion_expert_t2m_edge/sbatch_l40_smoke.sh
```

Run outputs and motion-only checkpoints live under
`/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/`.
The checkpoint contract pins the Edge base/DCP/framework identities, the v1
normalization hashes, and the exact stats population: all 120,929 retained
Nymeria train-caption windows, totaling 11,888,119 window-frame occurrences.
It rejects Nano or representation-mismatched weights.
The earlier schema-v1 T2M-only T=16 gate is in
`smoke_tmux0_l40s_v2_20260901/`; the T=200 T2M-only gate is in
`smoke_T200_tmux0_v2_20260901/`. Exact evidence and the excluded ECC-faulted
L40 attempt are documented in `motion_expert_t2m_edge/SMOKE_RESULTS.md`. Those
checkpoints predate the corrected TI2M+BONES scope and are rejected by schema
v2. The three-path schema-v2 T=16 and production-shape T=200/TI2M=97 L40S
smokes passed under `smoke_phase2_schema2_tmux0_20260901/` and
`smoke_phase2_schema2_T200_tmux0_20260901/`. They are now superseded by the
schema-v3 Nymeria-only gates at
`smoke_phase2_schema3_nymeria_tmux0_20260901/` and
`smoke_phase2_schema3_nymeria_T200_tmux0_20260901/`; both passed on one L40S.
Use `sbatch_l40_smoke.sh`, followed by the single-GPU
`sbatch_train_1gpu.sh`; a multi-GPU DDP gate is not required. The production
launcher writes the instrumented run under
`edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz/`, logs loss and pre-clip gradient
norm to `jungbinc-upenn/cosmos-motion-ft`, and persists `wandb_run_id.txt` so a
checkpoint resume continues the same dashboard. It freezes five Nymeria test
samples per task and their inference seeds under
`visualizations/fixed_samples/`, then uploads the same T2M `GT | generated`
and TI2M `conditioning image | GT | generated` cases at step 0 and every 5k
steps. Online scalar/media integration job `528071` passed; its repeatable
launcher is `sbatch_wandb_viz_smoke.sh` and its W&B run ID is `5lj0rq5m`.
The first instrumented production job `528072` used one L40S; its W&B run ID is
`bupzjaj5`. The API verified ten production-shape step-0 videos, five TI2M
condition images, the full-prompt table, and recurring loss/gradient rows at
steps 1 and 20. That job was later preempted twice before reaching the old 5k
checkpoint interval and was superseded while pending. The hardened launcher
now overwrites an atomic recovery checkpoint every 250 steps, persists RNG/data
epoch alongside model+optimizer, selects the newest complete recovery/regular
checkpoint with `--resume auto`, and requests Slurm requeue plus USR1 at 180
seconds. Clean preemption-safe job `528415` is submitted to `liu-compute` with
QoS `ll-med`, one `gpu:l40`, and the known ECC-faulted `ll-l40-1` excluded. It
runs under `edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz_preemptsafe`; at
submission it is pending for `Priority`. Pending `batch` job `528385` was
cancelled before it started and produced no output.
