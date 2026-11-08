# Directional camera prediction, two tasks, video weight 1

Fresh Phase 3 MMDiT bridges from the standard frozen Phase 1 video-I/O EMA200k
and Phase 2 200k specialists. Production samples V2M50% / M2V50% / joint0%,
with empty captions, clean-motion E_M(0), global batch32 = two GPUs ×16,
no gradient accumulation, EMA0.999 from step0, and a fresh200k schedule.
Native physical motion losses/noise schedules are retained; video weight1.

Both variants retain the existing camera ablation:96 noisy GT camera tokens
during training, native raw9 SE(3) padded64/domain2, no clean camera tokens.
Camera clock follows motion sigma for V2M and video sigma for M2V.
Camera readout is detached and camera loss is zero; native projections and
both specialists remain frozen. Camera hidden states participate in the
native generator attention. Bridges connect only video and motion rows.
Residual information in noisy GT camera inputs is part of this experiment.

`--attention-scope temporal_local` keeps the existing Wan temporal groups
for all bridge attention. `--attention-scope global` permits all temporal
same-modality edges and all target-query→conditioning-key edges across time.
Clean queries still cannot read noisy rows from the other modality. Shape
still bypasses every bridge without entering Q/K/V. The native specialist
attention is identical in both variants. Global execution uses the same BF16
projections/matmuls, FP32 normalization/softmax and FP32 parameter/Adam storage
as the grouped run; it is actual full temporal attention, not a dense
implementation of the same local mask. The override is process-local and
restored on context exit. Scope and all extension source hashes are contracted.

The two recipes differ only in experiment identity and attention scope.
Local/global mask, gradient, far-frame-dependency, clean-direction, shape
bypass and checkpoint-scope restoration tests live in `test_experiment.py`.
Launch gates additionally require native two-rank/microbatch16 preflight,
zero camera loss and detached prediction checks, batched camera-aware sampling
from private noise, and fresh4→strict-reload6 full-state probes.
Diagnostic joint checks in disposable preflight are not production tasks.

Fixed71 validation every10k uses EMA, native35-step UniPC, empty captions,
motion/video shifts3/10 and the same metrics/assets as current runs.
The camera-aware batched sampler starts camera from independent Gaussian
noise, never test GT camera; target and camera schedules update together.
It preserves the initial image latent and per-case target RNG streams.
The ordinary camera-free validation guard remains unchanged for other runs.

Node-specific operation and overlap/cutover evidence:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_camera_directional_nojoint_local_global_20261108`.
Only after BOTH replacements pass live production and remote W&B checks
may the owned old rank0 workers receive checkpointed SIGUSR1 stops.
Consult operation COMPLETE.json for actual cutover, never infer it from this README.

Launch through the isolated package:

```bash
python -m motion_expert_phase3_edge_camera_directional_nojoint.launch \
  --attention-scope temporal_local --camera-cache "$CAMERA_CACHE" \
  --gpus 3,5 --shared-gpu-pids "$COORDINATED_OWN_OLD_RANK_PIDS" \
  --phase1 "$PHASE1" --phase2 "$PHASE2" --cache "$TRAIN_CACHE" \
  --out "$NEW_RUN" --validation-config "$FIXED71_CONFIG"
```

Use `global` and GPUs1,7 for the paired global variant. The normal launcher
defaults to MMDiT/BF16, EMA0.999 and EMA validation. It permits only explicitly
coordinated owned resident PIDs, freezes complete sources, and refuses a
pre-existing output directory. Never reuse this command to resume an old run.
