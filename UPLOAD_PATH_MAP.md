# Server → Google Drive path map

Every backed-up item, with its **exact full source path** and **exact full Google Drive
destination**. All paths are literal and copy-pasteable (no truncation). rclone remote
is `data:` (personal Google Drive). Companion docs: `PROVENANCE.md` (what each item is /
how it was made / how to regenerate), `UPLOAD_MANIFEST.md` (tiering rationale).

**Download to a new server:** `rclone copy "<drive path>" "<local dest>"` — e.g.
`rclone copy data:cosmos_ckpts/native_phase1_vq_A/iter_000100000 ./iter_000100000`.
Use your own OAuth client_id to avoid the shared-client throttling (see PROVENANCE.md).

Status: ✅ done · ⬆️ pending · ❌ failed. All rows below are ✅ (backup complete 2026-07-29).

## Datasets → `data:cosmos_data/`

| server path | google drive path | size | status |
|---|---|---|---|
| `/weka/jungbin/nymeriaplus_kimodo_proportional/` (S01–S20 motions, uniego_rep, metadata, camera, camera_rgb, video, train_test_split.json) | `data:cosmos_data/nymeriaplus_kimodo_proportional/` | 401 GB | ✅ |
| `/weka/jungbin/seed/soma_proportional_motions_20fps/` | `data:cosmos_data/seed/soma_proportional_motions_20fps/` | 126 GB | ✅ |
| `/weka/jungbin/seed/soma_proportional_uniegomotion_20fps/` (BONES training input; incl. `Mean_uniego.npy`, `Std_uniego.npy`) | `data:cosmos_data/seed/soma_proportional_uniegomotion_20fps/` | 22.5 GB | ✅ |
| `/weka/jungbin/seed/multi_timeline.jsonl` | `data:cosmos_data/seed/multi_timeline.jsonl` | 173 MB | ✅ |
| `/weka/jungbin/seed/metadata/seed_metadata_v004.csv` | `data:cosmos_data/seed/metadata/seed_metadata_v004.csv` | 146 MB | ✅ |
| `/weka/jungbin/seed/metadata/seed_metadata_v004.parquet` | `data:cosmos_data/seed/metadata/seed_metadata_v004.parquet` | 4.5 MB | ✅ |
| `/weka/jungbin/seed/metadata/seed_metadata_v002_temporal_labels.jsonl` | `data:cosmos_data/seed/metadata/seed_metadata_v002_temporal_labels.jsonl` | 58 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_pairs_train.jsonl` | `data:cosmos_data/joint_attention/bones_pairs_train.jsonl` | 223 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_pairs_val.jsonl` | `data:cosmos_data/joint_attention/bones_pairs_val.jsonl` | 6.3 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_index_train.json` | `data:cosmos_data/joint_attention/bones_index_train.json` | 333 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_index_val.json` | `data:cosmos_data/joint_attention/bones_index_val.json` | 12.5 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json` | `data:cosmos_data/joint_attention/full71_windows.json` | 6.8 KB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bomb_windows.json` | `data:cosmos_data/joint_attention/bomb_windows.json` | 40 KB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1/` | `data:cosmos_data/eval_fixtures/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1/` | 11 MB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2/` | `data:cosmos_data/eval_fixtures/native_phase1_eval_inputs_full71_256_T97_v2/` | 130 MB | ✅ |
| `/weka/jungbin/shape_aware_motion_eval_c45_20260715/` | `data:cosmos_data/eval_fixtures/shape_aware_motion_eval_c45_20260715/` | 487 MB | ✅ |

## Checkpoints → `data:cosmos_ckpts/`

| server path | google drive path | size | status |
|---|---|---|---|
| `/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000` | `data:cosmos_ckpts/native_phase1_vq_A/iter_000100000` | 86 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000` | `data:cosmos_ckpts/native_phase1_vq_B/iter_000100000` | 86 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000` | `data:cosmos_ckpts/native_phase1_vq_D/iter_000100000` | 85 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt` | `data:cosmos_ckpts/ja_phase2_t2m_ti2m_native/ckpt_step200000.pt` | 5.8 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt` | `data:cosmos_ckpts/ja_phase3_bridge_native/ckpt_step200000.pt` | 9.1 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/ckpt_step115000.pt` | `data:cosmos_ckpts/ja_phase3_bridge_native_headcam/ckpt_step115000.pt` | 9.1 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask/ckpt_step065000.pt` | `data:cosmos_ckpts/ja_phase3_bridge_native_multitask/ckpt_step065000.pt` | 9.1 GB | ✅ |
| `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k/ckpt_step035000.pt` | `data:cosmos_ckpts/ja_phase3_bridge_native_contact/ckpt_step035000.pt` | 9.1 GB | ✅ |

## Small assets kept in git (NOT on Drive — restore via GitHub clone of `whwjdqls/cosmos_motion_ft`)

| repo-relative path | role |
|---|---|
| `motion_expert/stats/uniego283_mean.npy`, `..._std.npy` | 283-D z-score stats (also duplicated at `motion_expert_joint_attention/uniego283_{mean,std}.npy`) |
| `motion_expert_joint_attention/head_camera_calibration_train.json` | Phase-3 head-camera calibration (`DEFAULT_CALIBRATION`) |
| `skeleton_soma30.npz` | SOMA-30 skeleton |
| `motion_expert/pairs_train.jsonl`, `pairs_val.jsonl` | Nymeria text↔motion pairs |

## NOT backed up — regenerable / re-downloadable (see PROVENANCE.md §D)

`cosmos3_nano_dcp` (HF convert), `wan22_vae/Wan2.2_VAE.pth` (HF), TMR-SOMA-RP-v1 (HF),
`vggt_omega_ckpt` (public), `model_cache/` (FVD/dreamsim), `nymeriaplus_kimodo_proportional/joint_latents_T97/`
(regen via `precompute_latents.py`), SMPL/SMPL-X body models (license-gated).
