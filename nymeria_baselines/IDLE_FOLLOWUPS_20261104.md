# Full-window forward-video and final shape-UEM evaluations

Operation on Yonsei:
`/scratch2/whwjdqls99/cosmos/operations/idle_gpu_followups_20261104`.
Read its README/status/receipts for exact live state. The user explicitly chose
all97-frame windows across71 recordings. The accepted cohort has11911
caption/window cases and11677 physical windows; annotations retain paired seeds.

## Completed 2026-11-05

All11911cases are generated and scored for Phase1 and each of the three final350
UniEgoMotion conditions. Full output audits passed on compute. Final Notion
tables were independently read back against current summaries after completion.
Receipts: `FULL_COMPLETION_VERIFIED.json`, `GOAL_COMPLETION_AUDIT.json`,
`phase1_videoio/audits/FULL_OUTPUT_AUDIT.json`, and the three full UEM output
audits in `uem_final350/audits/`. Videos, predictions, per-case metrics,
recording-balanced results, features and exact configurations are retained.

Phase1 uses the official native EMA200k sampler, camera plus first image, empty
semantic text, UniPC30/shift10/guidance1. Metrics score generated frames1..96.
FVD here uses the pinned TATS I3D400 logits described in `metric_assets.json`.

| Reference/model | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | DreamSim ↓ | FVD-I3D400 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT identity | 100 (cap) | 1 | 0 | 0 | 0 |
| Two-task LoRA + video I/O EMA200k | 19.815407 | 0.593452 | 0.256971 | 0.161340 | 44.063621 |

UniEgoMotion uses the same final350-epoch/555800-update checkpoint for all three
conditions, clean supplied SOMA30 shape, paired seeds and native1000-step
sampling. Each model's complete physical/FID evaluation covers11911cases.
R@3 uses batches32 and3seeds,11904cases/seed with7 dropped; values are mean±std.

| Conditioning | MPJPE cm ↓ | FID ↓ | R@3 % ↑ | Skate cm/s | Foot depth cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT | 0 | 0 | 91.70±0.14 | 14.076 | 1.048 |
| Trajectory | 12.749 | 0.041364 | 66.92±0.14 | 14.137 | 0.941 |
| Video | 41.392 | 0.059815 | 64.55±0.08 | 12.410 | 0.667 |
| Video + trajectory | 11.190 | 0.037073 | 79.82±0.17 | 14.409 | 0.896 |

[Phase1 results in Notion](https://app.notion.com/p/binne/Ablations-40dd2a44f44b8080aad3cda6b5fea7c8#40ed2a44f44b81478061fc4cb59b40ed)
and [UEM results in Notion](https://app.notion.com/p/binne/Ablations-40dd2a44f44b8080aad3cda6b5fea7c8#40ed2a44f44b81ad8b2df87962bb5383).

All temporary prefix-scoring coordinator holds were released;265cached cases
passed an additional artifact check. The pinned native source, checkpoint,
sampler and training remained unchanged. Follow-up queues are complete. Original
retained allocations/PRO requests remain; older `p3_hidden_film` inference on
node25 is a separate, still-running evaluation.

## Pipeline and provenance

Phase1 uses empty text, image+camera→video, the final two-task prefix1 Edge
LoRA+video-I/O EMA200k checkpoint, and official native inference. Exact checkpoint
identity, staged Yonsei location, resolved contract and acceptance gate
are mandatory; do not substitute the stopped action+video-projections-only run.

Update2026-11-05: exact checkpoint is now located in Drive folder
`cosmos_data/phase1_twotask_prefix1_phase23_20261031/phase1_video_io_ema200k`.
Download/checksum/extraction on owned node34 CPUs stages its native run root
under the operation's `assets/`. The restorer releases the dispatcher only
after archive/payload/metadata/contract/DCP/EMA projection verification. Read
`PHASE1_STAGED.json` and `phase1_checkpoint.json` for actual completed staging.
`native_phase_training/fulltest_forward.py` provides resumable generation,
PSNR/SSIM/LPIPS/DreamSim/FVD-I3D400, per-case/recording aggregates and verified
Notion publication. The observed first frame is excluded from metrics.

Tmux `phase1_fulltest_inputs_20261104` prepared the entire input manifest on owned
node25 CPUs (no GPU), exit0 in38.1minutes. All11911 passed official schema/action
loading; `ALL_INPUTS_COMPLETE.json` plus `VERIFIED_INPUTS.json` prove input
completion and exact case/seed/settings correspondence, not sampled metrics. Native imports
use the base Cosmos environment; the private metric venv is only for metric
dependencies. No training environment was changed. Check live progress/exit/logs.

`uem_final_conditions.py` waits for supplied-shape three-mode training to finish
350epochs, snapshots555800 updates, and evaluates trajectory/video/both with
identical seeds and always clean supplied SOMA30 shape. It preserves native1000
steps and benchmarks the existing noise-preserving batch adapter before sampling.
Scoring retains separate generator/evaluator statistics and measured GT results.

The independent compute-CPU audit exercised the actual conditional evaluator
input method against the current three-mode training Dataset for all11911 cases
and all3modes. Camera, visual and shape tensors matched exactly; max inverse-
normalization error was2.384e-7. Operation receipt `UEM_INPUT_AUDIT_COMPLETE.json`
plus per-case evidence are checked against current source/cohort/stats before
final preparation. GPU/native-sampler acceptance remains a separate required gate.

`idle_followup_worker.py` is a compute-only process/claim-aware dispatcher for
already owned allocations. It does not cancel jobs, stop workers or enter CUDA
while waiting. Live CUDA users and existing full-test pool claims take priority;
per-GPU/task leases protect handoffs. Inspect actual processes and receipts for
live state; the follow-up dispatch queues above have now completed.
Detailed task receipts/failures prove progress; setup alone is not completion.

The previous full-test V2M pool now excludes only `p3_joint` and `p3_camera` at
the user's request. Specific inference processes were stopped; original sruns
and partial outputs remain. The remaining target is8models. Manually removed
Notion rows are preserved as removed and do not imply canceled model evaluations.
