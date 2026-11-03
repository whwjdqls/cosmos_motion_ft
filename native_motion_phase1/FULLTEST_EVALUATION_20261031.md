# Final naeive-adaptation EMA200k evaluation on sep-2

## Final completion (2026-11-02 23:50 UTC)

All11911 cases per task and47644 saved predictions passed coverage checks.
Reconstruction/physics, video metrics, motion FID/diversity and text-dependent
TMR retrieval/matching are complete and published. The earlier missing-evaluator
and caption-cache notes below describe historical preparation snapshots.
The matched Nymeria caption cache was staged with SHA256
`6d065f2d402dc8da8cc6532ac65e25a492888aa78db0a5537230d7609e833eba`.
All required captions resolve, including six formatting-only sanitizer lookups;
generation captions and predictions were preserved. Retrieval uses11904 cases
per grouping seed0/1/2; matching distance uses all11911. The final evidence ZIP
is `remaining_metrics_gpu5_20261102T221146Z/fulltest_final_metrics_ema200k_20261102.zip`
under the operation below; its16723632 bytes and SHA256
`1c79e0aee79980dc4a3c017b1541429e483a8d282fae8f1d1cb036a57da3be73`
were verified by downloading the published artifact. COMPLETE.json and the
latest Notion section are authoritative. Camera-action tasks remain unsupported;
multimodality is unavailable with one generated sample per case.

The versioned preparation/inference/reconstruction/video-scoring scripts retain
their executed protocol. Final TMR postprocessing and archive/publication scripts
are preserved separately in the operation's
`remaining_metrics_gpu5_20261102T221146Z/`; the original report worker's pending
TMR fields are historical and are not the final semantic report.

## Preparation and execution record

The user requested the full Nymeria test evaluation and publication to
https://app.notion.com/p/40ad2a44f44b8017b16aeb87901f236d.
After confirming that this checkpoint has motion heads instead of camera heads,
the user selected its four trained tasks. Image+camera→video and video→camera
are unsupported and must not receive scores from another checkpoint.

`evaluate_fulltest.py prepare` enumerates the complete existing aligned-T97
test index, applies the checkpoint's original floor/canonicalization/statistics
and feature guard without random replacement, and saves immutable provenance.
There are 11938 candidates, 11911 eligible annotation entries, 11677 unique
physical windows and all71 held-out recordings. The27 feature-guard rejections
are retained individually. The same cohort is used for all four tasks; shorter
native T2M spans are outside this fixed97-frame model protocol. Annotation
multiplicity is retained and disclosed. The user explicitly requested that
text→motion and text+image→motion both use this smaller image-aligned cohort.
Both already read the same immutable `cases.json`, so no results or case IDs
need to change. `paired_text_cohort_verified.json` records its SHA256 and checks
completed pairs for matching case IDs, captions/GT bundles, shapes and seeds.

`evaluate_batched.py` loads the final native EMA DCP, preserves the x0 adapter,
and samples all four tasks with native UniPC30/shift10/CFG1, per-case seeds,
97frames/20fps/256×256. Batch size is8. Unobserved motion targets and future
M2V video latents are replaced with zero placeholders before native pure-noise
sampling; only the observed shape/image tokens are retained. Single-case and
eight-case GPU smoke checks passed all four tasks, finite outputs and observed
token preservation. Generation resumes from individual atomic result files.

`score_fulltest.py` incrementally scores generated-frame PSNR/SSIM/LPIPS-Alex,
DreamSim ensemble, I3D400 FVD and FP32 VideoMAE-v2 CD-FVD. Both distribution
scores use frames1–96 and are labeled separately. Feature extraction uses
content-debiased-fvd commit a1e037ab7cb087debd2221d14ae4a001ec054201.
The VideoMAE checkpoint SHA256 matches the existing pinned reference.
Model pixels are saved losslessly before qualitative video encoding.
Motion reconstruction and physics reuse the existing SOMA30/Body23 metrics,
with matched measured GT, raw coordinates and no prediction grounding/reset;
only PA-MPJPE uses a whole-clip similarity fit. Text reconstruction distances
are paired diagnostics rather than unique-target generative-quality measures.

The selected Nymeria TMR evaluator checkpoint/statistics/caption cache were not
found in the checked local/shared backups. Its motion FID, retrieval, matching
distance and embedding diversity remain pending that bundle. Do not substitute
the BONES C45 checkpoint or manufacture GT retrieval/physics values.

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_naeive_fulltest_20261031T215024Z`.
The pointer is `operations/eval_naeive_current_ops.txt`. Initially GPU4 hosted
generation and video scoring. At22:25UTC the user authorized a second idle GPU:
generation remains on GPU4 and video scoring resumes on GPU6. All training
jobs retain their existing GPUs. Dataset, caches, metric weights,
environments and outputs are on NVMe. The user approved the required evaluator
transfer:29 files /3960848670 bytes, all source-stream and local SHA256 verified.
DreamSim0.2.1 and Decord0.6.0 were installed in the isolated
`/mnt/local/jinhyung.kim/envs/naeive_eval` overlay, preserving training packages.

Tmux sessions: `cosmos_naive_fulltest_gpu4`, `cosmos_naive_motion_metrics`,
`cosmos_naive_video_metrics_gpu6`, `cosmos_naive_eval_report`. The old GPU4
video-scorer waiter was disabled before its active scorer was stopped. Its168
completed atomic metric/feature pairs are retained; hashes and the exact move
are under `gpu6_scoring_20261031T222550Z`. The CPU
report worker updates `report.md`, per-task CSVs and `quantitative_evidence.zip`.
Live logs, exit codes, `status.json`, metric status files and `report_status.json`
establish actual progress. A running job or uploaded preview is not completion.
Only `INFERENCE_COMPLETE.json` plus full coverage in metric summaries establish
the reconstruction/video evaluation's completion; TMR is tracked separately.

Six deterministic cases across different test recordings provide four-task
qualitative comparisons, GT beside predictions in every panel. Their exact
selection and captions are in `gallery_manifest.json`. Notion media upload and
remote checksum receipts are retained under the operation. Initial published
numbers must be labeled provisional and retain their exact scored coverage.
