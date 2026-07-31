# Data & Path Provenance (server hand-off document)

Written 2026-07-23 and re-audited 2026-07-31 while the a3ultra cluster
(`a3ultravis-slurm-login-001`, GCP project `lgair-a3-cluster`, zone
`us-east4-b`) was being decommissioned. This documents the active external
training/evaluation dependencies, what produced them, and whether they are
archived or regenerable. Access-gated raw captures and licensed body models are
an explicit boundary; the archived derived dataset is the operational source
for restoring this project.

How to read: each entry is *path → contents → generator (script @ repo) → regeneration
notes → size*. The generator scripts that lived **outside** this repo are vendored under
[`external/`](external/) (see §7). Existing deep-dive docs: `AGENTS_ALL.md` (project
context), `external/nymeria_kimodo_pipeline/README.md` +
`NYMERIA_PROPORTIONAL_PIPELINE.md` (the NymeriaPlus preprocessing pipeline, verbatim
copy), `native_phase_training/README.md` / `AUDIT.md`.

---

## 1. Repositories and pinned versions

| repo | origin | commit @ hand-off | role |
|---|---|---|---|
| `cosmos_motion_ft` (this repo) | `github.com/whwjdqls/cosmos_motion_ft` | use the final migration commit on `master`; also archived as `source/cosmos_motion_ft.bundle` in GCS | all experiments |
| `cosmos-framework` | `github.com/NVIDIA/cosmos-framework` | `82f8229` (2026-06-12 "Refactor datapackerdataloader…") **+ local patches** | native Cosmos-3 Nano training/inference. Checked out at `/home/jungbin_cho/cosmos-framework`; every launcher puts it on `PYTHONPATH` and native training `cd`s into it. The checkout carried **uncommitted, load-bearing patches** (`lora_keep_trainable_modules`, `SAVE_TRAINABLE_ONLY` LoRA-only DCP save, TensorBoardLog callback, pixel-path experiment) — captured with reapply instructions in `external/cosmos_framework_patches/`. |
| `kimodo` (a.k.a. `kimodo_open`) | `github.com/whwjdqls/kimodo` | `5e3daac` (= `483b3ca` + the previously-UNTRACKED uniego converters `nymeria_to_uniego.py` / `soma_proportional_to_uniego.py`, committed at hand-off) | SOMA-77 skeleton FK, uniego motion rep, TMR eval, BONES-SEED datasets. Lived at `/home/jungbin_cho/kimodo_open`. The converter scripts this repo's data depends on are also vendored in `external/kimodo_uniego_scripts/` (incl. `soma_proportional_to_uniego.py` + `UNIEGO_REPRESENTATION.md`). |
| `nymeria_kimodo_pipeline` | `github.com/whwjdqls/nymeria_kimodo_pipelin` (sic — repo name typo) | `cd1b3bc4048957b2ddd3a20fc1baacd29c5d6643` | raw NymeriaPlus → motion/video/camera/text preprocessing (see its README for the full 5-stage pipeline). Also vendored (minus weights/media) at `external/nymeria_kimodo_pipeline/`. |

Conda envs (miniforge, `~/miniforge3`): `cosmos` (torch 2.10.0+cu128, all
Cosmos training/inference; exact spec in `external/cosmos_env.yml` and
`external/cosmos_env_pip_freeze.txt`) plus `kimodo`, `soma`, `nymeria_plus`,
and `audio`. Relocatable exports for the latter four are tracked under
`external/nymeria_kimodo_pipeline/envs/`. Reapply
`external/cosmos_framework_patches/` after checking out Cosmos Framework at
`82f8229`.

---

## 2. Raw sources (external, re-downloadable)

| path | contents | origin |
|---|---|---|
| `/weka/jungbin/nymeriaplus/{Sxx}/{seq}/` | raw NymeriaPlus captures: `body/xdata_smpl_neutral.npz` (SMPL), `narration/`, `recording_head/data/data.vrs` (Aria ego video + trajectories), `metadata.json`. 19 subjects, 732 sequences (S01–S17, S19, S20; no S18). | Meta *Nymeria* dataset release (access-gated). `~/nymeria_dataset/` held downloader tooling. |
| `/weka/jungbin/Kimodo-SOMA-SEED-v1.1`, `/weka/jungbin/seed/` (release parts: `soma_proportional*`, `soma_uniform*`, `g1*`, `SEED-Timeline-Annotations`, `metadata/`, `multi_timeline.jsonl`) | NVIDIA Kimodo BONES-SEED motion-text release (SOMA skeleton), incl. `seed_metadata_v004.csv` and temporal-label jsonl | NVIDIA SEED release (HF/internal). `seed/custom_scripts/` = our small viz scripts only. |
| `/weka/jungbin/Kimodo-Motion-Gen-Benchmark`, `…-20fps` | official kimodo motion-gen benchmark + our 20 fps resample (`splits/train_split_paths.txt`, `testsuite/`) | NVIDIA release; 20 fps variant derived with kimodo scripts |
| HF cache `~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano` | Cosmos-3 Nano release (diffusers-layout weights, processor/`text_tokenizer`, standalone `vision_encoder/`) | `hf download nvidia/Cosmos3-Nano` |
| HF cache `models--Qwen--Qwen3-VL-8B-Instruct` | tokenizer fallback only | HF |
| HF cache `models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401` | TMR retrieval model for Phase-2 shape/TMR eval (`last_weights/text_encoder.pt`, `stats/motion`) | `hf download nvidia/TMR-SOMA-RP-v1` (snapshot hash pinned in `eval_phase2_shape_tmr.py`) |
| `/weka/jungbin/wan22_vae/Wan2.2_VAE.pth` (2.7 GB) | Wan2.2 video VAE used by ALL video paths | HF `Wan-AI/Wan2.2-TI2V-5B`, revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, file `Wan2.2_VAE.pth` (provenance: `external/cosmos3_nano_dcp_convert.log`). Env var `WAN_VAE_PATH`. |
| `/weka/jungbin/model_cache/` | FVD backbone (`cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth` = VideoMAE-g) + `dreamsim` weights, for video eval | public releases; exact tree is also archived on Drive and GCS |
| `/weka/jungbin/vggt_omega_ckpt/vggt_omega_1b_512.pt` | VGGT camera-estimation baseline ckpt (nymeria_world zero-shot comparisons) | internal/public VGGT release |
| `/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt` (also `/weka/jungbin/kimodo_caches/`) | LLM2Vec caption embeddings cache for BONES-SEED | built by kimodo scripts; regenerable |

---

## 3. `/weka/jungbin/nymeriaplus_kimodo_proportional/` — the main derived tree

Everything below derives from `/weka/jungbin/nymeriaplus` via
`external/nymeria_kimodo_pipeline/` (stages 1–4″ in its README) plus converters in this
repo and `kimodo`. **This tree is the expensive one to rebuild** (SMPL→SOMA fitting is
GPU-days).

| subpath | contents | generator | size |
|---|---|---|---|
| `{Sxx}/{seq}.npz` | shape-aware kimodo motion: `local_rot_mats (T,77,3,3)`, `root_positions`, `neutral_joints (77,3)`, 20 fps | pipeline stage 1 `process_nymeriaplus.py` (SMPL→SOMA fit, env `soma`; ~`smpl2soma_nymeria.py` prototype vendored too) → stage 2′ `soma_to_kimodo_proportional.py`. v1.1: per-seq `transl` recentred before fit (S17 ±300 m fp32 bug); betas = median over the 5-window artifact. | (per-seq npz) |
| `uniego_rep/{Sxx}/{seq}.npz` | **283-D uniego rep** consumed by all motion experts: `features (T,283)` = 30×(6D rot + 3D joint pos) ⊕ canon_delta(9) ⊕ foot contacts(4), Y-up, world-frame; `neutral_joints (30,3)`. 732 files. | `external/kimodo_uniego_scripts/nymeria_to_uniego.py` (from kimodo@483b3ca) run on the proportional kimodo npzs. Layout contract: `motion_expert_joint_attention/uniego_layout.py`. | 15 GB |
| `video/{Sxx}/{seq}.mp4` + `video/manifest_video.jsonl` | ego RGB 640² @20 fps, frame-aligned to motion; manifest joins video⟷camera⟷motion⟷text windows (`t2w_windows` with `caption`, `usable`, `start/end_frame`, `ground_offset_y`) | pipeline stage 4 `video/extract_ego_video.py` (env `nymeria_plus`) + `video/build_video_manifest.py` (env `kimodo`) | 277 GB |
| `camera/{Sxx}/{seq}.npz` | head Aria `T_world_device` sampled at body frames (Aria Z-up) | pipeline stage 4′ `camera/extract_camera_trajectory.py` | small |
| `camera_rgb/{Sxx}/{seq}.npz` | **upright RGB-camera** world poses (`cam_world_pos_upright`, `cam_world_rot_upright`, kimodo Y-up) — source of ALL camera actions (`rel_action_from_window` → (T-1,9) rel-SE(3), rot6d) | `external/nymeria_kimodo_pipeline/camera/preprocess_camera_rgb.py` (device→RGB-sensor extrinsic + upright + Z→Y-up), 728 files | 1.8 GB |
| `joint_latents_T97/{Sxx}/{uuid__seq}_{start}.npz` | precomputed Wan-VAE latents per T=97 window: `latents (48,25,16,16) fp16`, `camera_action (96,9)`, `image_size`, uuid/start/T/fps. 127,956 windows. | `motion_expert_joint_attention/precompute_latents.py` (writer; reader contract in `nymeria_joint_dataset.latent_path` and `native_phase_training/latent_nymeria_dataset.py`) | ~85 GB |
| `joint_latents_T97_720tier_640/{Sxx}/*.npz` | 720 model-tier cache: RGB transform 640x640, latent `(48,25,40,40)` fp16, T=97, 115,583 train windows | same precompute path; exact contract and completion record are tracked in `migration/cache_contracts/` | ~415 GB, regenerable and not cloud-backed |
| `joint_latents/` | older T=33 latent cache (superseded by `_T97`) | same script at T=33 | — |
| `train_test_split.json` | per-SEQUENCE train/test split (whole recordings held out; 71 test windows downstream) | `nymeria_world/make_train_test_split.py` | tiny |
| `metadata/metadata_atomic_action*.jsonl` (+floor variants) | 20 fps-aligned narration slices + per-slice GT floor (`ground_offset_y`), `usable`, foot-skating | pipeline stages 3/3′/3″ (`build_metadata.py`, `floor/extract_slice_floor.py`, `floor/fallback_floor_and_skating.py`) | 585 MB (dir) |
| `metadata/floor_calibration.json` | per-seq SOMA-fit floor delta (`d_minc − c0`, c0=+2.42 cm BONES convention) + dropped-window list (~5.3%) — folded into every joint-attention dataset `off` | `motion_expert_joint_attention/precompute_floor_calibration.py` (env `kimodo`, CPU) | tiny |
| `metadata/camera_motion_quality_filter_v1_T97.json` | versioned T97 physical-window exclusion artifact (camera speed/rot/head-cam coherence), sha256 `1fd64658…9848` pinned in the phase-1 launchers | `native_phase_training/build_camera_motion_quality_filter.py` | tiny |
| `visualization/` | skeleton+camera+ego side-by-side check renders | pipeline stage 4″ | — |

---

## 4. `/weka/jungbin/seed/` — BONES-SEED derived trees

| subpath | contents | generator | size |
|---|---|---|---|
| `soma_proportional_uniegomotion_20fps/` | SEED motions converted to the same 283-D uniego rep at 20 fps, + per-source `Mean_uniego.npy` / `Std_uniego.npy` | `external/kimodo_uniego_scripts/soma_to_uniego.py` + `compute_uniego_stats.py` | 23 GB |
| `cosmos_text_motion_full/shard_*/` | FULL 1,076,474 unique (text, motion[T,369]) pairs, sharded (sbatch array 2433) — used by the root text→motion finetune | `export_bones_seed_full.py` (repo root; subset variant `export_bones_seed_text_motion.py`) | 142 GB |
| `cosmos_text_motion_subset/` | small subset export (same schema) | `export_bones_seed_text_motion.py` | — |
| `stats/soma_uniform_motions_20fps/` | stats for the uniform (non-proportional) tree | kimodo stats script | 48 KB |

Joint-attention BONES pairs (`/weka/jungbin/cosmos_motion_ft_runs/joint_attention/
bones_pairs_{train,val}.jsonl` + `bones_index_{train,val}.json`) are built by
`motion_expert_joint_attention/build_bones_pairs.py` from the
`soma_proportional_uniegomotion_20fps` tree + SEED metadata (desc4 caption policy via
`filter_bones_pairs_desc4.py`).

---

## 5. Model checkpoints

| path | contents | provenance |
|---|---|---|
| `/weka/jungbin/cosmos3_nano_dcp` (30 GB) | Cosmos-3 Nano base weights in native **DCP** layout — `BASE_CHECKPOINT_PATH` for all native phase-1 training | Converted 2026-06-23 from HF `nvidia/Cosmos3-Nano` via cosmos-framework model build + DCP save; full log vendored at `external/cosmos3_nano_dcp_convert.log` (includes the exact model config used). |
| `/weka/jungbin/cosmos_motion_ft_runs/` (`IMAGINAIRE_OUTPUT_ROOT`) | ALL run outputs. Native runs under `cosmos3_camera/camera_world/<run>/checkpoints/iter_XXXXXXXXX` (DCP: `model/` `optim/` `scheduler/` `trainer/`; `model/` ≈ 85 GB holds net+EMA). Joint-attention runs save `ckpt_stepNNNNNN.pt`/`latest.pt` (trainable-delta only, small). | trainers in this repo |
| Named checkpoints referenced by eval scripts | e.g. `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt` (Phase-2 motion expert), `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k*` (Phase-3), `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/checkpoints/iter_000100000` (Phase-1 v1), `native_phase1_vq_{A..E}_*` (video-quality ablations) | their sbatch launchers in this repo record every hyperparameter |

The independent GCS and Drive hand-off contracts, retained-checkpoint lists,
restore drivers, and verifiers are under `migration/`. The GCS bucket roots
created on 2026-07-23 are:

```text
gs://mm-jinhyung_kim/jungbin_cho/nymeriaplus/
gs://mm-jinhyung_kim/jungbin_cho/nymeriaplus_proportional/
gs://mm-jinhyung_kim/jungbin_cho/seed/
gs://mm-jinhyung_kim/jungbin_cho/cosmos_motion_ft_runs/
```

The proportional backup contains all active derived motion, video, camera,
camera-RGB, metadata, T97 latent, split, floor-calibration, and quality-filter
artifacts. Selected runs retain their required resumable checkpoint, run
configuration/log state, and `eval*` directories. Standalone visualization
directories and intermediate checkpoints are outside the required contract.
No local source was deleted by migration commands. GCS last passed
`python migration/verify_migration.py gcs` on 2026-07-23. The supplementary
Drive archive is checked with `python migration/verify_drive.py
--include-trees`; its exact contract is `migration/DRIVE_ARTIFACT_MANIFEST.tsv`.

---

## 6. Eval fixtures under `/weka/jungbin/cosmos_motion_ft_runs/`

| path | contents | generator |
|---|---|---|
| `native_phase1_eval_inputs_viz5_256_T97_v2/`, `…_full71_256_T97_v2/` | held-out four-mode official-inference inputs (5-sample viz / all-71 windows), old single-prefix suite | `native_phase_training/prep_test_eval.py` (older revision) |
| `native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1/` | prefix-suite inputs (prefixes 1,9,17,33,49; quality-filtered; C→"A person" captions) | current `native_phase_training/prep_test_eval.py` |
| `joint_attention/full71_windows.json`, `bomb_windows.json` | canonical 71 test windows; loss-bomb window list | joint-attention eval tooling |
| `nymeria_camera_motion_source_audit/final/` | camera↔motion source-quality audit behind the quality filter | `motion_expert_joint_attention/audit_nymeria_camera_motion.py` |
| `/weka/jungbin/shape_aware_motion_eval_c45_20260715` | Phase-2 shape/TMR eval bundle (C45 TMR ckpt, fps 30) | `prepare_shape_tmr_eval.py` |
| head-camera calibration JSONs (paths in `motion_expert_joint_attention/config.py` / `head_camera_alignment.DEFAULT_CALIBRATION`) | rigid head-joint→upright-camera SE(3), train-split fit | `estimate_head_camera_calibration.py` |

Repo-internal: `motion_expert/stats/uniego283_{mean,std}.npy` and the
byte-identical `motion_expert_joint_attention/uniego283_{mean,std}.npy`. These
are the active Nymeria-grounded 283-D normalization arrays shared by current
Phase-2/3 training. BONES source `Mean_uniego.npy`/`Std_uniego.npy` and C45
evaluator 190-D stats have different roles and must not be substituted.

`motion_expert/pairs_{train,val}.jsonl` are not tracked because `.gitignore`
excludes them. Exact copies are archived on Drive as
`cosmos_data/joint_attention/nymeria_pairs_{train,val}.jsonl`; the Drive restore
script maps them back to the expected repo filenames.

---

## 7. What was vendored into `external/`

- `external/nymeria_kimodo_pipeline/` — the full preprocessing pipeline (scripts +
  READMEs; excluded: `__pycache__`, `.checkpoints` incl. 4.5 GB ImageBind, media/logs).
  Also pushed standalone to `github.com/whwjdqls/nymeria_kimodo_pipelin` (`a4094aa`).
- `external/kimodo_uniego_scripts/` — `nymeria_to_uniego.py`, `soma_to_uniego.py`,
  `kimodo_to_uniego.py`, `compute_uniego_stats.py` copied from kimodo@`483b3ca`
  (also on GitHub, but these four define this repo's data).
- `external/smpl2soma_nymeria.py` — standalone SMPL→SOMA prototype (the productionised
  version is `nymeria_kimodo_pipeline/process_nymeriaplus.py`).
- `external/cosmos3_nano_dcp_convert.log` — provenance of the DCP base checkpoint
  (exact model config + HF revisions of every sub-model, incl. the Wan VAE revision).
- `external/nymeria_kimodo_pipeline/envs/` — relocatable conda exports for
  `audio`, `kimodo`, `nymeria_plus`, and `soma`.

---

## 8. Archive checklist (if /weka is going away too)

Priority order, with sizes. Everything in tier 1–2 is either impossible or GPU-weeks to
regenerate; tier 3+ is regenerable from tiers above + this repo.

1. **Trained checkpoints you care about** (`cosmos_motion_ft_runs/...`): each
   native iter dir is about 86 GB. Full A/B/D DCPs and selected Phase-2/3
   checkpoints are verified on Drive. GCS retains the original baseline's full
   DCP; Drive carries its exact 129 MB EMA LoRA/action subset for Phase-3
   initialization.
2. **`nymeriaplus_kimodo_proportional`**: Drive contains the derived motion
   npzs, `uniego_rep`, `camera`, `camera_rgb`, metadata, split, and 277 GB
   video. GCS additionally contains the 256-tier T97 latent cache. The 720-tier
   cache is regenerable from these archived sources.
3. **`seed/soma_proportional_uniegomotion_20fps`** (23 GB incl. per-source stats) —
   regenerable from the SEED release + vendored scripts. `cosmos_text_motion_full`
   (142 GB) only matters for the root text→motion finetune.
4. **`cosmos3_nano_dcp`** (30 GB) — regenerable from HF in ~1 h (see convert log);
   `wan22_vae` (2.7 GB) — pinned HF revision, re-downloadable.
5. Eval fixtures (§6), evaluator models, LPIPS, and C45 stats are verified on
   Drive and GCS. Numeric run evaluation outputs are archived without redundant
   visualization media.

## 9. Regeneration DAG (from scratch)

```
raw NymeriaPlus ──(pipeline 1,2′: SMPL→SOMA→kimodo-proportional npz [env soma])──►
  {Sxx}/{seq}.npz ──(kimodo nymeria_to_uniego.py)──► uniego_rep/            ─┐
  ├─(pipeline 3,3′,3″)──► metadata/ (captions+floor)                         │
  ├─(pipeline 4 + build_video_manifest)──► video/ + manifest_video.jsonl     ├─► joint-attention
  ├─(pipeline 4′ + preprocess_camera_rgb)──► camera/, camera_rgb/            │   & native trainers
  └─(make_train_test_split.py)──► train_test_split.json                     ─┘
video+manifest ──(precompute_latents.py @T)──► joint_latents_T{T}/
uniego_rep+manifest ──(precompute_floor_calibration.py)──► floor_calibration.json
everything ──(build_camera_motion_quality_filter.py)──► quality_filter_v1_T97.json
SEED release ──(soma_to_uniego.py + compute_uniego_stats.py)──► seed uniego tree
             ──(build_bones_pairs.py + filter_bones_pairs_desc4.py)──► bones_pairs jsonl
HF Cosmos3-Nano ──(cosmos-framework convert, see log)──► cosmos3_nano_dcp
```

Env-var ↔ path map used by the native launchers: `BASE_CHECKPOINT_PATH=cosmos3_nano_dcp`,
`WAN_VAE_PATH=wan22_vae/Wan2.2_VAE.pth`, `IMAGINAIRE_OUTPUT_ROOT=cosmos_motion_ft_runs`,
`NYMERIA_LATENT_ROOT=…/joint_latents_T${NYMERIA_NUM_FRAMES}`,
`NYMERIA_QUALITY_FILTER=…/camera_motion_quality_filter_v1_T97.json`,
`NATIVEP1_EVAL_INPUT_DIR=…/native_phase1_eval_inputs_*`.
