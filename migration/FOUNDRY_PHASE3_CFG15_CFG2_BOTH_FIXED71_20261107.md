# Latest sep-1 EMA comparison: whole-bridge CFG1.5/2

Completed and published on2026-11-08. COMPLETE.json/PUBLISHED.json/AUDIT.json
and REPORT_PUBLISHED.md in the operation record all852 predictions, full metric
coverage, 12videos/12contact sheets, source checks and successful Notion write.
All evaluator processes exited; all six training ranks remained active.

CFG1.5 is better than2 on all reported M2V fidelity metrics in every checkpoint.
V2M MPJPE1.5/2: directionalW10 16.59/15.56cm; bidirectionalW10 19.33/19.27;
directionalW1 14.27/13.97. Jerk and acceleration error worsen at2 for all three.
Only directionalW10 has an additional paired MPJPE interval excluding0.
Qualitative galleries explicitly retain wrong crouched-posture failures.
Training/EMA histories differ, so this is not a causal video-loss-weight study.

User requests quantitative V2M/M2V evaluation and matched qualitative examples
on the same fixed71 windows for all three active Phase3 models. Destination:
https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b .

Operation:
`/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_latest_cfg15_cfg2_both_fixed71_20261107T231929Z`.

Latest saved checkpoints were pinned with hardlinks at selection time:
directional videoW10 120k, bidirectional videoW10 120k, directional videoW1 80k.
Use plan.json/checkpoint hashes rather than subsequently advancing live steps.

EMA bridge weights, no extra Phase3 LoRA; original frozen P1 video-I/O EMA200k
and P2 200k. No captions or inference camera tokens. Native35-step UniPC,
motion/video shifts3/10, T97/20FPS/256square, fixed seeds0–70, supplied shape.
CFG reference uses bridges_enabled=False: all seven complete bridge updates,
including within-modality attention, are bypassed. Same current inputs enter
both passes. Guidance = reference + scale*(conditioned-reference).

Sampling32 follows native even/odd virtual ranks32+4/32+3 so native CFG1
controls are exactly matched. RGB decode4. Acceptance requires native/wrapped
CFG1 parity, 32-case cached parity in each direction, source/EMA verification,
bridge-off source invariance, no reference bridge calls, finite outputs,
fixed first video latent and unchanged trainable parameters. All live training
continues; already allocated own GPU1/3/7 co-evaluation slots are used,
with30% process memory caps and nice10. GPU1 was added to run the third model
concurrently after checking its only resident was our directionalW1 trainer.
The CPU scheduler adopted the first two evaluators without restarting them;
GPU_ALLOCATION_EXTENSION.json preserves that handover. Never stop/share foreign jobs.

852 new predictions:3models*2directions*2CFGscales*71windows. Native metrics
include TMR motionFID/retrieval, Body23 reconstruction/physics, UPE/LPE/DisCoRD
sJPE, DreamSim/LPIPS/PSNR/SSIM and native426-segment I3D400 FVD. Do not rank
native426-descriptor FVD against pooled71-descriptor or VideoMAE conventions.

Fixed qualitative indices0/14/28/42/56/70, before seeing new results. M2V:
three model rows, columnsGT/CFG1.5/CFG2. V2M: three model rows, observed
video/GTmotion/CFG1.5/CFG2; fixed common display projection/extent per case,
unaligned world coordinates, GT ghosts. Raw arrays remain local; rendering
does not alter quantitative inputs.

Check scheduler.log, per-model status/acceptance/GENERATION_COMPLETE/COMPLETE,
then ALL_METRICS_COMPLETE. Do not duplicate launches. Finalize with
render_comparisons.py and build_report.py. PUBLISHED/COMPLETE publication
receipts establish actual completion; the setup details above preserve the
executed protocol. The completed outcome is recorded at the top of this file.
