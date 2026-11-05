# Naeive Adaptation: exact71 cached M2V windows

Report: https://app.notion.com/p/40fd2a44f44b80489a0bdecc5d9410bb .
Completed November5 using existing predictions only. The user clarified that
only the exact71 windows were wanted and the dataset must not be resampled.
No new video generation or neural evaluator inference was performed.

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_naeive_m2v_matched71_cached_20261105T022142Z`.
Pointer: `naeive_m2v_matched71_current_ops.txt` in its parent directory.

All71 recording/start pairs from the completed110k Phase3 comparison exist in
the completed11911-entry Naeive full-test evaluation. Their normalized GT
motion and neutral-shape inputs pass equality checks against the exact fixed
bundles (atol1e-6). For duplicate physical windows, the first entry in original
full-test manifest order is retained without looking at output quality.
`cases.json` and `per_case.csv` preserve the full fixed→source ID/seed mapping,
candidate counts and source-file hashes.

Checkpoint: native motion-head x0 Naeive Adaptation, EMA200k. Cached sampling
is retained: UniPC30,shift10,guidance1,T97/20fps/256square, one image and full
motion/neutral shape, blank caption, original full-test per-case seeds. The
Phase3 matched71 runs used35 steps and their original fixed seeds; these
sampling controls are not asserted identical. No35-step rerun was launched.

| Metric | Exact71 cached predictions |
|---|---:|
| PSNR dB | 20.814565 |
| SSIM | 0.652935 |
| LPIPS-Alex | 0.230158 |
| DreamSim | 0.137476 |
| I3D-FVD | 296.034300 |
| VideoMAE-v2 CD-FVD | 313.299364 |

Frame metrics reuse the cached frames1–96 scores and are averaged over only
these71 entries. Both FVD values are recomputed on CPU from exactly71 saved
GT/model feature pairs, using the original evaluator/statistics convention;
the full11911 distribution score is not reused. GT identity references and
source weight/metadata hashes accompany the report.

Six fixed GT/cached-prediction display clips use case indices0/14/28/42/56/70.
Re-encoding these existing pixels creates presentation media, not new model
samples. All seven attachments (six videos and the994204-byte evidence ZIP)
passed remote readback and full downloaded SHA256 verification.
ZIP SHA256:
`a617f8cf87572bb4878f2cf4814a81531ef2ab8c9f4b9e41c894f5b291054d35`.
Root `COMPLETE.json` and `publication.json` record verified publication.
