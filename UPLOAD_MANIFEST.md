# Upload Manifest — what to back up to Google Drive

Built 2026-07-28 by scanning the live filesystem. Companion to `PROVENANCE.md`
(which explains *what each thing is / how it was made*); this file is the
*checklist of what to move off the dying server*, with sizes and priority.

Drive layout convention: `data:cosmos_data/<tree>` for datasets,
`data:cosmos_ckpts/<run>` for checkpoints (matches what's already there).

Legend: ✅ already uploaded · ⬆️ upload (priority) · ⏭️ skip (re-downloadable/regenerable)

---

## ⚠️ Catch found during this scan

The seed tree already uploaded is **`soma_proportional_motions_20fps`** (126 GB,
rot-matrix motions) — but **BONES training actually consumes
`soma_proportional_uniegomotion_20fps`** (23 GB, the 283-D uniego rep), which is
the root every `bones_pairs_*.jsonl` `uniego_path` points at. **That 23 GB tree is
NOT uploaded yet** and is the training-critical one. See tier B below. (The 126 GB
rot-mat tree is the *source* it's derived from — nice to have, but the uniego tree
is what the trainer opens.)

---

## Tier 0 — tiny config/stats files (⬆️ HIGHEST priority, ~0.9 GB total)

Kilobytes-to-megabytes each, trivially lost, and **every training/eval run breaks
without them**. Upload these first — they fit in one tiny `rclone copy`.

| file | size | why |
|---|---|---|
| ✅ `motion_expert/stats/uniego283_{mean,std}.npy` | 1.3 KB ×2 | the 283-D z-score stats — **already in git** (repo) |
| ✅ `motion_expert_joint_attention/uniego283_{mean,std}.npy` | 1.3 KB ×2 | same, second copy — **already in git** |
| ⬆️ `seed/soma_proportional_uniegomotion_20fps/{Mean,Std}_uniego.npy` | 1.3 KB ×2 | per-source BONES stats (proportional_stats=True path) |
| ⬆️ `nymeriaplus_kimodo_proportional/train_test_split.json` | 33 KB | per-seq split — **also already inside the uploaded nymeria tree** ✅ |
| ⬆️ `nymeriaplus_kimodo_proportional/metadata/floor_calibration.json` | 490 KB | per-seq floor delta + drop list (✅ in uploaded nymeria/metadata) |
| ⬆️ `nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json` | 1.8 MB | window exclusion filter (✅ in uploaded nymeria/metadata) |
| ⬆️ `nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl` | 119 MB | captions+floor per slice (✅ in uploaded nymeria/metadata) |
| ⬆️ `nymeriaplus_kimodo_proportional/video/manifest_video.jsonl` | 64 MB | video⟷motion⟷camera⟷text windows (✅ in uploaded nymeria/video) |
| ⬆️ `cosmos_motion_ft_runs/joint_attention/bones_pairs_{train,val}.jsonl` | 223 MB + 6.3 MB | **BONES text↔motion pairs the JA trainer reads** |
| ⬆️ `cosmos_motion_ft_runs/joint_attention/bones_index_{train,val}.json` | 333 MB + 12.5 MB | BONES dataset index cache |
| ⬆️ `cosmos_motion_ft_runs/joint_attention/full71_windows.json` | 6.8 KB | canonical 71 eval windows |
| ⬆️ `cosmos_motion_ft_runs/joint_attention/bomb_windows.json` | 40 KB | loss-bomb window blocklist |
| ⬆️ `seed/multi_timeline.jsonl` | 173 MB | ✅ already uploaded to `data:cosmos_data/seed/` |
| ⬆️ `seed/metadata/{seed_metadata_v004.csv, *_temporal_labels.jsonl, *.parquet}` | 205 MB | SEED release metadata (caption source) |
| ✅ `motion_expert/pairs_{train,val}.jsonl` | — | Nymeria text↔motion pairs — **already in git** |

> Most nymeria/* rows above are already inside the uploaded `nymeria/metadata` and
> `nymeria/video` dirs (marked ✅) — the standalone rows matter only if restoring
> piecemeal. The **not-yet-anywhere** ones are the `joint_attention/*` pairs+indexes
> and `seed/metadata/*`.

---

## Tier A — checkpoints (⬆️, pick which; sizes vary)

### Native Phase-1 DCP runs (~86 GB each — big)
| run | iter | size | note |
|---|---|---|---|
| ✅ `native_phase1_vq_A_..._person` | 100000 | 86 GB | uploaded + verified |
| ⬆️ `native_phase1_vq_B_varprefix_global_lora_...` | 100000 | 86 GB | complete |
| ⬆️ `native_phase1_vq_D_varprefix_camera_kv_lora_...` | 100000 | 85 GB | complete |
| ◽ `native_phase1_vq_C_varprefix_action_only_...` | 65000 | ~86 GB | partial run |
| ◽ `native_phase1_vq_E_..._camera_kv_lora` | 5000 | — | smoke, skip |
| ◽ `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k` | 100000 | 86 GB | pre-ablation v1 baseline |
| ◽ `..._qfilterv1` / `..._qfilterv1_noi2v` | 35000 | ~86 GB | earlier partials |

> Each native DCP dir is `model/`(~85 GB net+EMA) + `optim/`(0.25 GB) + tiny
> scheduler/trainer. For inference-only you can upload just `model/` + its
> `.metadata`; to resume training keep the whole dir.

### Joint-attention `.pt` runs (5.8–9.1 GB each — the important science)
| run | ckpt | size | keep? |
|---|---|---|---|
| ⬆️ `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d` | step200000 | 5.8 GB | **Phase-2 production** (init_motion for Phase-3) |
| ⬆️ `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k` | step200000 | 9.1 GB | **Phase-3 production bridge** |
| ⬆️ `..._p2native200k_headcam` | step115000 | 9.1 GB | Phase-3 head-camera variant |
| ⬆️ `..._p2native200k_multitask` | step65000 | 9.1 GB | Phase-3 multitask variant |
| ⬆️ `..._p2contact200k` | step35000 | 9.1 GB | Phase-3 contact variant |
| ◽ `ja_t2m_ti2m_..._contact_c0p05_..._unipc35` | step200000 | 5.8 GB | contact-loss Phase-2 ablation |
| ◽ `ja_7task_full` | step200000 | 6.0 GB | 7-task joint run |
| ◽ `ja_phase3_7task` | step100000 | 6.0 GB | 7-task Phase-3 |
| ◽ `ja_t2m_x0_T200(_mrope3d)`, `ja_t2m_phase2_T200`, `ja_t2m_ti2m_..._T200_mrope3d` | 5.8 GB | earlier t2m ablations |
| ◽ `ja_phase1_camera` | step200000 | 185 MB | small camera-only run |

> Every run dir also holds `ckpt_step*` intermediates + `viz_step*` renders. Upload
> **only the final `ckpt_step<max>.pt` per run** unless you want the LR-schedule
> history; skip the viz dirs (regenerable).

---

## Tier B — training data not yet on Drive (⬆️)

| tree | size | why |
|---|---|---|
| ⬆️ **`seed/soma_proportional_uniegomotion_20fps/`** | 23 GB | **BONES training input** (bones_pairs point here) + its Mean/Std_uniego.npy. See the catch above. |
| ◽ `seed/cosmos_text_motion_full/` | 142 GB | root text→motion finetune shards (only if reviving that experiment) |
| ◽ `seed/soma_shapes/`, `seed/g1/` | 5.5 MB / — | SEED shape params / G1 retarget (small, cheap to grab) |
| ◽ `shape_aware_motion_eval_c45_20260715/` | 489 MB | Phase-2 TMR eval bundle |
| ◽ `cosmos_motion_ft_runs/native_phase1_eval_inputs_*` | 12 MB / 131 MB | native eval fixtures (prefix suite + full71) |

---

## Tier C — already on Drive ✅ (no action)

- `data:cosmos_data/nymeriaplus_kimodo_proportional/` — S01–S20 motions, uniego_rep,
  metadata, camera, camera_rgb, video, train_test_split.json (401 GB, verified)
- `data:cosmos_data/seed/soma_proportional_motions_20fps/` + `multi_timeline.jsonl`
  (126 GB rot-mat motions, verified)
- `data:cosmos_ckpts/native_phase1_vq_A/iter_000100000` (86 GB, verified)

---

## Tier D — skip (re-downloadable / regenerable, ⏭️)

| item | size | recover by |
|---|---|---|
| `cosmos3_nano_dcp` | 30 GB | convert from HF `nvidia/Cosmos3-Nano` (~1 h; `external/cosmos3_nano_dcp_convert.log`) |
| `wan22_vae/Wan2.2_VAE.pth` | 2.7 GB | HF `Wan-AI/Wan2.2-TI2V-5B` rev 921dbaf3 |
| TMR-SOMA-RP-v1 | — | `hf download nvidia/TMR-SOMA-RP-v1` (snapshot pinned in eval_phase2_shape_tmr.py) |
| `vggt_omega_ckpt/vggt_omega_1b_512.pt` | 4.6 GB | VGGT release (baseline only) |
| `model_cache/` (FVD, dreamsim) | — | public releases |
| `nymeriaplus_kimodo_proportional/joint_latents_T97/` | ~85 GB | regenerate: `precompute_latents.py` on video/ (~a GPU-day) |
| `nymeriaplus_kimodo_proportional/joint_latents/` (T33) | — | superseded |
| SMPL/SMPL-X body models | — | license-gated, re-download from smpl-x.is.tue.mpg.de |

---

## Suggested upload order & rough budget

1. **Tier 0** (config/stats/json) — ~0.9 GB, minutes. Do first; it's what silently breaks restores.
2. **Tier B uniego tree** (23 GB) — the missing training input.
3. **Tier A JA finals you care about** — ~5–9 GB each; the two "production" rows are the must-keeps (~15 GB).
4. **Tier A native B + D** — 86 GB each (~172 GB).
5. Optional: native C, extra JA ablations, cosmos_text_motion_full (142 GB).

Google download-side quota is ~10 TB/day; upload is 750 GB/day per account — so
tiers 0–4 (~285 GB) fit in one day. Use your own OAuth client_id (see
`PROVENANCE.md` / download notes) to avoid the shared-client throttling that
stalled the earlier uploads.
