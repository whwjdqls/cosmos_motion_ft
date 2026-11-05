# sep-2 MMDiT joint40/40/20 replacement runs

Both directional and bidirectional experiments use40% video→motion,40%
motion→video and20% joint prediction. Joint sees only one initial image and
the supplied neutral shape; its caption is empty and both future video and
motion are targets. Frozen specialists are the standard Phase1 video-I/O
EMA200k and selected Phase2 200k.

- Bidirectional: GPUs0,3; previous workers3620614/3620615.
- Directional: GPUs1,7; previous workers3620873/3620874.
- Fresh200k, global32=2×16, accumulation1; clean-motion E(0), video weight10,
  native motion losses, bridge LR2e-4/1000 warmup/cosine, no added LoRA/EMA.
- Seven MMDiT bridges, grouped attention, bf16_bridge projections/attention;
  FP32 norms/softmax/master/Adam. Shape excluded from bridges, retained by
  the frozen motion expert. Both arms use the same paired data/noise seeds.
- Fixed71 V2M/M2V monitoring every10k, UniPC35/CFG1, shifts3/10, seeds0–70,
  sampling32/GPU with OOM backoff, VAE decode4. I3D400 FVD uses426 segment
  descriptors; it differs from the71-descriptor CDFVD protocol.
- Training/validation use the existing node-local naeive_eval venv, which
  inherits exactly cosmos's Torch2.10.0+cu128 and adds DreamSim. CUDNN_HOME,
  CUBLAS_HOME and library paths explicitly select the same base environment
  libraries; no environment packages or other users' files were modified.

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_mmdit_joint_handover_20261105`.

[Experiment report](https://app.notion.com/p/40fd2a44f44b814fb925c3d86c250284).

The launcher defaults to empty GPUs. An explicit --shared-gpu-pids admits only
the exact selected owned residents, pins process start times, rejects any new
resident or PID reuse, and requires64GiB free. A separate handover lease allows
the existing trainers to retain their allocation leases. Both new launchers
freeze all scientific/lazy evaluator sources before acceptance.

The manager must verify BOTH replacements before stopping either old run:
native two-rank preflight including joint gradients/caption isolation, fresh
four-update checkpoint, strict two-update continuation, at least20 finite
production updates with all three task streams, and remote W&B progress.
Only then does it send SIGUSR1 to each old rank0, await both ranks' graceful
exit, and verify/hardlink full bridge+optimizer+two-rank-RNG recovery states.
It preserves the previous regular checkpoint too. Failed startup leaves both
old runs active.

30 focused CPU/worker/policy/validation checks passed. Real GPU evaluator
acceptance reproduces all71 previous GT retrieval results, zero GT MPJPE/FID,
video identity metrics, and reviewed I3D features within1.72e-5. The reviewed
I3D weights are rgb_imagenet.pt SHA256
2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538.
The existing CDFVD I3D weights are different and are not substituted.

Read the operation's status.json, per-arm launch logs, PRODUCTION_RUNNING.json,
REPLACEMENTS_HEALTHY.json, OLD_RUNS_PRESERVED.json and COMPLETE.json for actual
state. This document records the launch contract, not a claim of completion.

## Verified handover completion

Both replacements passed native DDP preflight, fresh save4/strict resume6,
at least20 finite production updates with all three tasks and remote W&B.
Both continue training after graceful shutdown of all old workers/loggers.

- Bidirectional: GPUs0/3, W&B
  https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-b6b71e1630ea628544fe .
- Directional: GPUs1/7, W&B
  https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-5caf5e94d92b872de27e .
- Verified12:56UTC:both160 updates; recent1.324/1.277seconds per update.
  Read live metrics for current progress. Target remains200k.
- Old bidirectional stopped134993, old directional137999. Both final full
  recovery states and previous130k regular milestones are locally hardlinked
  under the operation's preserved/{attention}/. Each old recovery has56 bridge
  tensors,56 optimizer states, two rank RNG states and the exact32×step cursor.
- Root COMPLETE/RUNNING_VERIFIED/REMOTE_WANDB_VERIFIED/OLD_RUNS_PRESERVED
  and verified Notion report establish the completed handover. The earlier
  statement about not claiming completion applies to the launch contract.
- Source implementation is commit e55efe4; all349 frozen Python sources per arm
  passed snapshot hash checks. Main checkout/new commits cannot alter these jobs.
