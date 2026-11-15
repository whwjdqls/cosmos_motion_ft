# Latest retained Phase3 fixed71 evaluation, sep-1 (2026-11-09)

User authorizes evaluating latest checkpoints on free GPUs. GPUs5/6 were empty
and explicitly allocated here; small owned CUDA guards maintain the allocation
across evaluator handovers. No training GPU or another user's process is shared
or stopped. Existing six training ranks retain their original identities.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_latest_all_fixed71_20261109T000922Z`.

Four latest full checkpoints were hardlinked outside rotating trainer folders,
CPU metadata/fingerprint checked and SHA256 pinned. These evaluation pins do
not alter any independent100k archive. All are EMA:

| Stable model ID | Step | Camera | Native CFG1 M2V reusable |
| --- | ---: | --- | --- |
| sep1_bidir5050_w1_camera_nocamloss_EMA0 | 30000 | True | True |
| sep1_dir5050_w1_EMA0 | 150000 | False | True |
| sep1_dir5050_w10_lateEMA | 190000 | False | False |
| sep1_bidir5050_w10_lateEMA | 150000 | False | False |

Eight settings/model: M2V CFG1/1.1/1.5/2; V2M and video+text→motion CFG1.5/2.
Same71 windows/seeds0–70, native35-step UniPC, shifts3/10, T97/20FPS/256px,
supplied shape, groupedBF16/TF32off. V2M contiguous64+7; M2V native virtual
rank even32+4/odd32+3; decode4. All inputs and repeated reads remain on local NVMe.

Each process uses its own immutable training source snapshot. EMA/source checks,
CFG1 native single-case parity and available cached32-case native parity,
whole-bridge-reference source invariance, no reference bridge calls, text route
and unchanged bridge state are acceptance gates. Camera and directionalW1
passed both modes'32-case exact native parity before full generation.

Whole-bridge-off reference; video+text inference routes GT caption only to the
frozen motion expert, identically in both branches. Generator remains empty.
This never changes caption-free training. The camera model alone uses the
native camera-aware sampler from independent private Gaussian, never GT camera.
Target motion/video output receives CFG; auxiliary camera solver uses the
conditioned camera readout without separate camera CFG. Non-reused camera
outputs are saved alongside their targets for audit.

Native completed EMA CFG1 M2V reuse: directionalW1/150k and camera/30k,142 clips.
All other2130 predictions are newly generated. Total32settings/2272 outputs.
No partial stopped-run EMA validation is treated as completed/reusable.

Metrics: matched91f9d8... Nymeria TMR/Body23 adapter and stats; positional,
UPE/LPE/sJPE, retrieval/FID and physical measures. M2V lossless RGB frames1–96,
PSNR/SSIM/LPIPS/DreamSim/native426-segment I3D FVD. Portable cohort signature
matches the80k protocol, but these results must not be labeled matched80k.
Camera30k vs150k/190k and differing EMA histories preclude a causal attribution.
No model or default guidance selection, per the user's explicit instruction.

Scheduler dispatches two GPU workers, then the remaining two; check RUNNING.json,
ERROR.json and per-model worker.log before any restart. CPU finalizer waits for
ALL_METRICS_COMPLETE, audits all32settings, builds quantitative CSV/ZIP/figure
and renders9 preselected comparison videos/contact sheets (cases0,28,70).
COMPLETE.json is written only after those steps succeed. FINALIZE_ERROR.json
records any audit/report issue; never report completion from process liveness.

Scope defaults to all4 retained sep-1 variants. An optional asynchronous question
asks whether to include sep-2; no remote filesystem availability is assumed.

## Completion and publication

Completed locally2026-11-09T01:14:59Z; published2026-11-09T01:18:41Z.
All32 settings/2272 outputs scored and independently audited.497 newly generated
auxiliary camera arrays are finite with correct shape.9 comparison videos are
97 frames at20FPS, all media hashes/dimensions verified;9 contact sheets included.
Quantitative archive,CSV,protocol,figure,qualitative manifest and18 media files
are23 successful Notion attachments. Training identities unchanged; own evaluation
workers and GPU5/6 allocation guards exited.100k preservation watcher unaffected.

Report: https://app.notion.com/p/binne/devugged-MMdiT-410d2a44f44b8098b1b7c7af42d2353b
(top latest-checkpoint section; existing content preserved). No model or CFG chosen.
Local expanded report:`REPORT_PUBLISHED.md`; CSV:`metrics.csv`; full quantitative
provenance:`quantitative_evidence.zip`.COMPLETE.json and PUBLISHED.json record success.

| Stable model ID | Step | M2V CFG1.1 DreamSim ↓ | V2M CFG1.5 MPJPE cm ↓ | Video+text CFG1.5 MPJPE cm ↓ |
|---|---:|---:|---:|---:|
| sep1_bidir5050_w1_camera_nocamloss_EMA0 | 30000 | 0.14085 | 17.7123 | 12.4518 |
| sep1_dir5050_w1_EMA0 | 150000 | 0.13509 | 13.6566 | 11.0466 |
| sep1_dir5050_w10_lateEMA | 190000 | 0.13347 | 15.1146 | 11.6581 |
| sep1_bidir5050_w10_lateEMA | 150000 | 0.13359 | 18.7038 | 12.9799 |

These are checkpoints pinned at evaluation setup, with different training ages
and EMA histories. They are not a matched-step causal ablation or a selection.
Full32-setting tables and paired per-window guidance intervals are in the report.

The initial report builder missed the two reused CFG1 M2V per-window metric JSONs.
Copied from the exact native completed validation directories; case IDs/order and
all four summary means verified. No generation/scoring changed. Retained
RESOLVED_REPORT_ISSUE.json and REUSED_PER_CASE_VERIFIED.json explain the old
traceback in the append-only build_report.log; retry succeeded. No GPU rerun needed.
