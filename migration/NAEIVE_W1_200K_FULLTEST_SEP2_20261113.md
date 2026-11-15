# Naeive adaptation videoW1 native EMA200k full test, sep-2, 2026-11-13

## Completed and published, 2026-11-14

All11,677 unique physical windows /71 recordings completed across four tasks:
46,708 fresh predictions. Final global aggregation and AUDIT passed at04:01UTC.
Final Variations now contains the complete metric tables and attached full-precision
CSV, report, audit and cached-caption-alias provenance. Local
`NOTION_fulltest_PUBLISHED.json` and `NOTION_ALLCOMPLETE.json` record the successful
connected publication; the operation status is `complete_published`.

| Metric | M2V | V2M | T2M | TI2M |
|---|---:|---:|---:|---:|
| FVD | 77.996580 | — | — | — |
| PSNR | 18.336774 | — | — | — |
| SSIM | 0.542629 | — | — | — |
| LPIPS-Alex | 0.298548 | — | — | — |
| DreamSim | 0.172095 | — | — | — |
| MPJPE cm | — | 29.962385 | 44.810249 | 43.219268 |
| Motion FID | — | 0.062860 | 0.087151 | 0.087033 |

The directional-camera W1 EMA200k full-test FVD41.238070 uses the exact same ordered
IDs and bit-identical70,062 ground-truth I3D400 feature vectors, confirmed from the
saved global feature arrays. Its inference uses whole-bridge CFG1.1/UniPC35 versus
native CFG1/UniPC30 here. Naeive allocates25% of training samples to M2V versus50%
for the two-task Phase3 model. These are possible contributors, not an isolated
explanation of the quality gap. No additional sampling was started for this check.

The sections below retain the setup and recovery history; pending progress notes
are superseded by this final completion/publication record.

User requested full-test evaluation after finishing fresh W1 training. Correct checkpoint is the native DCP iter_000200000 under naeive_adaptation_x0_wv1_edge_T97_gbs32_lora_4tasks_200k_sep2_20261110, not older W10. Actual trainer iteration200000, raw/EMA metadata, W1 motion contract,424 native-source SHA and mean/std parity verified. EMA parameters are additionally read from DCP and compared bitwise after official load_ema_to_reg=True. Three selected action/LoRA EMA tensors passed on all three initial workers.

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_naeive_wv1_EMA200k_fulltest_20261113`.

All11677 unique physical aligned-T97 windows from71 held-out recordings, identical captions/seeds to the completed directional camera full test. Four supported tasks M2V,V2M,T2M,TI2M use same cohort;46708 fresh outputs. Old Naeive full test included11911 annotation entries with repeated physical windows; do not conflate denominators. The fixed1420 random20 cohort is contained with exact seeds. Native prepared inputs reused, never old predictions; no raw rescans/shared I/O. Model has no camera stream or Phase3 bridge.

Sampler native UniPC30/shift10/CFG1, matching old Naeive training previews. Native power EMA, not Phase3 EMA0.999. Direct clean x0 motion uses actual-sigma velocity conversion. Neutral prefix; first image M2V/TI2M; T2M fixed visual zeros. Text only T2M/TI2M. Motion/future-video targets are zeroed before noise sampling as appropriate. Model decoded motion shape97x283 and video latent48x25x16x16 preserved. Output RGB metrics exclude observed frame0. Actual native sampler prefix is retained and checked at native tolerance1e-4: normalization roundtrip produces ~1.44e-6 drift. Do not force-replace the returned first latent with GT to satisfy a Phase3-specific bitwise check. Matched native video scorer changes only that prefix comparison; copied source and all other native metric code hashed.

Six virtual shards, three independent workers on only empty coordinated0/1/3, batch32M2V/64motion/decode4 with OOM backoff, per-user UUID locks held by detached controller. Training on2/4 untouched, foreign jobs untouched. Current controller3386746, initial current workers3386758/3386828/3386843. Verify live PID/start identity and status; first two startup attempts (missing evaluator snapshot path and Phase3 prefix exactness check) are resolved/archived. No failed-attempt predictions were saved.

Native sampler/architecture code is from the frozen W1 training snapshot. Matched metric modules and shape_aware_TMR come from the exact frozen directional-camera evaluation snapshot; source manifests for both retained. Five focused CPU checks passed; GPU full-task acceptance and first scored chunks must be checked before claiming evaluation healthy.

M2V native PSNR/SSIM/LPIPS-Alex/DreamSim/I3D400 FVD six16-frame segments/window; three motion tasks Body23 physical/reconstruction/UPE/LPE/sJPE plus matched91f9d8 TMR FID/retrieval/matching/diversity and measured GT. Global aggregation across all windows, never shard-FID/FVD averages; recording bootstrap5000. Cached six explicit text aliases and full provenance are copied from the completed directional evaluation (not fresh exact raw-string embeddings), applied identically to model/GT evaluator lookup only. Reports/audits/ZIP include alias map. No GT retrieval assumed100%.

Detached manager writes final COMPLETE, fulltest/AUDIT, report.md, metrics.csv and NOTION_OUTBOX.json after global aggregation. Publish through connected Notion tools when active; no durable chat-cell uploader/local Notion token assumed. Destination Final Variations page412d2a44f44b80a2936aca27f96ac016. Queue entry already published, marker sep2-naeive-w1-EMA200k-fulltest-20261113. Do not claim final metrics published without a distinct receipt.

GPU startup acceptance now PASSED on all three workers: first64 cases for each of4 tasks complete on eachGPU (768 fresh predictions), checked shape/finite/step/EMA metadata/observed-prefix fields. M2V metrics advancing; no fresh errors at this verification. This is generation/startup acceptance, not final full-test completion. GPU_START_VERIFIED.json retained. Queue entry updated with this evidence.

Update2026-11-14: initial shards0/1/2 stopped at final26/27-case chunks because native vm.motion_scores called R32 with N<32. All23356 predictions saved (5839 windows x4);5760 windows fully scored. Fixed only per-chunk retrieval wrapper: N<32 explicitly unavailable, no fake/padded/dropped cases; original global R32 implementation unchanged/restored. Six preflight tests passed (including1/26/27/31 and unchanged64-case results). Manager4164143 resumed on0/1/3. Originals/errors archived under resolved_retrieval_tail_error_20261114; prediction inode/size/mtime inventory preserved. Sampling code unchanged. Need verify recovery past tail and new shards, then final aggregate/publication as before.
