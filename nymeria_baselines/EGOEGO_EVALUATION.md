# EgoEgo final checkpoint: whole Nymeria test evaluation

Requested on2026-11-07. The final supplied-shape camera-to-motion EgoEgo finished
350 epochs /555,800 updates at15:06KST under minsu1206. Its source checkpoint is
`/scratch2/minsu1206/cosmos/operations/egoego_shape_20261104/training/epoch=349-step=555800.ckpt`.

The evaluation operation is
`/scratch2/whwjdqls99/cosmos/operations/egoego_fulltest_20261107`.
Read its `PREFLIGHT_ACCEPTED.json`, rank progress/completion files and final
`COMPLETE.json`; code or a launcher is not evidence of completed evaluation.

## Frozen comparison

- Reuse the exact full-test cases, original per-case seeds, targets and evaluator
  from `operations/fulltest_v2m_ablations_20261104`:11,911 annotation cases,
  11,677 physical windows,71 held-out recordings,97 frames at20FPS.
- Use final EMA0.999, matching EgoEgo training validation. Recover all79 EMA
  tensors from the original optimizer parameter order, checking count, shape,
  update555800, finiteness and exact copy. This policy is fixed before scoring;
  do not silently substitute raw weights or choose using test results.
- Use the trained FP32 native cosine1000-step x0 DDPM, TF32 disabled, no output
  clipping, no guidance or schedule shortcut. Each case retains an independent
  CUDA generator, the original seed, and native initial/conditioning/step noise
  draw order. Benchmark batch sizes8/32/64/128/256 on the owned GPU. Require
  bitwise native singleton parity and deterministic same-batch replay; record
  singleton-versus-batch FP32 kernel roundoff separately.
- Inputs are measured camera trajectory and supplied centered SOMA30 neutral
  skeleton only. No text, RGB or target motion enters sampling. Predict original
  global body23 positions and row-convention6D rotations; no post-hoc FK replaces
  generated positions.
- Map the trained camera-yaw/horizontal-origin frame into the common benchmark
  frame using camera plus the already fitted578-training-record head/camera
  calibration. GT is used only for an independent cross-frame audit, never for
  prediction origin, rotation, grounding or shape fitting. Audit the first case
  of every test recording before accepting inference.

The driver is `nymeria_baselines/evaluate_egoego_fulltest.py`, outside the
fingerprinted `egoego_shape/` training package. Training code/checkpoints are
unchanged. Prepared configuration pins its own driver hash and the original
package/upstream/statistics/checkpoint hashes.

## Metrics and saved results

The shared `fulltest_v2m_score.py` computes MPJPE, whole-clip similarity
PA-MPJPE, Root ADE/FDE and relative ADE, UPE14/LPE9, head/non-head error,
acceleration error/magnitude, foot skating, foot/full-body penetration depths
and frequencies, foot height, airborne fraction and DisCoRD sJPE. The latter
includes four-end-effector mean, static/noise parts, native joint-sum and
ratio-of-sums versions plus jerk magnitudes. MPJVE remains omitted per the
user's earlier correction.

Nymeria shape-aware TMR uses its original evaluator mean/std and training-only
body23-to-SOMA30 leaf completion. It reports FID and paired R@1/2/3 in32-candidate
groups with seeds0/1/2:11,904 cases per seed, seven dropped. Physical/FID metrics
use all11,911 cases. The finalizer adds matching distance and fixed-pair embedding
diversity without changing original FID/retrieval definitions. This diversity is
not repeated-condition multimodality. No video-image or point-cloud scene
collision score is implied by this single-sample camera-to-motion benchmark.

Identical frozen GT embeddings/targets are reused with an explicit provenance
receipt after matching cases, evaluator, normalization, adapter and text cache.
The local GT summary records the model-list-only configuration change; the
original GT artifacts are unchanged. Every headline report includes GT.

Final files are `RESULTS.md`, `comparison.csv`, `summary.json` and
`COMPLETE.json`; complete per-case physical scores, TMR embeddings/ranks,
window-weighted and recording-balanced aggregates are retained. Predicted
features, body positions/rotations, supplied shape, cameras and placement
transforms are saved in `models/egoego_camera_shape_ema_final555800/predictions`.

Use only owned compute. Evaluation is assigned the previously idle2xPRO6000
allocation2622173/node58; current Phase3 training2616204 is untouched. Slurm
RPC intermittently rejected new steps before loading any model; the retained
allocation's original tmux compute shell is also available. A failed Slurm
launch is not a failed model preflight. Preserve logs of each attempt.

## Completed results

Root COMPLETE.json is complete_verified:all11,911 predictions and scored cases,
71 recordings,zero failures,worker exit0. Native two-GPU sampling took823.7s.

| Metric | EgoEgo EMA555.8k | Matched GT |
| --- | ---: | ---: |
| MPJPE cm |17.410861|0|
| Whole-clip PA-MPJPE cm |11.859729|approximately0|
| Root ADE cm |13.536644|0|
| Root FDE cm |14.302576|0|
| UPE14 / LPE9 cm |14.909022 /21.302609|0 /0|
| sJPE mean |0.343693|0|
| Skating cm/s |17.641749|14.076369|
| Foot penetration cm |1.023239|1.048054|
| Acceleration error m/s² |2.361200|0|
| FID |0.062487|0|
| R@1 % |30.353943 +/-0.110917|70.805332 +/-0.491940|
| R@2 % |45.558916 +/-0.491437|86.074709 +/-0.189773|
| R@3 % |55.393145 +/-0.458503|91.703069 +/-0.139897|

The operation's `all_metrics.csv` contains every stored native metric key, all
three retrieval seeds, semantic diagnostics and recording-balanced aggregates,
alongside GT. This is the complete accepted floor-clean annotation cohort, not
the earlier one-window-per-recording71-case evaluation.
