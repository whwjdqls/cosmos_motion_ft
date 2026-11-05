# Paired motion errors and DisCoRD sJPE

The user's final metric request is PA-MPJPE, Root ADE, UPE, LPE and **sJPE**.
MPJVE was explicitly withdrawn; it is not added.

The current backfill covers every retained model in the11911-case Nymeria
full-test V2M comparison and all3 final555800-step supplied-shape UEM conditions,
including matching GT. Stopped p3_joint/p3_camera remain excluded. The unfinished
hidden-FiLM result is marked partial until the original full scoring completes;
a metadata watcher then dispatches the additive scoring pass to compute. Phase1
image/camera-to-video has no predicted body joints, so these metrics are N/A.

## Definition and source

DisCoRD source is pinned to commit
`782deffca1a263a2241b6321cfe6532a022f284e`,
[evaluation/eval_t2m.py](https://github.com/whwjdqls/DisCoRD/blob/782deffca1a263a2241b6321cfe6532a022f284e/evaluation/eval_t2m.py).
The frozen source copies and SHA256 manifest are under operation `sources/DisCoRD`.
Paper equations5–8: https://arxiv.org/html/2411.19527v4 .

- Common decoded body23 joints, original canonical-world coordinates,20fps,
  all97frames. No inference, normalization, resampling, smoothing, grounding or
  prediction alignment is changed.
- PA-MPJPE preserves the existing **whole-clip similarity** Umeyama fit,
  including scale. It is not silently changed to per-frame Procrustes.
- Root ADE is mean absolute Hips trajectory error, existing `root_err_m`.
- UPE is raw MPJPE of14 upper joints: two spine joints, chest, two neck joints,
  head, and four shoulder/arm/forearm/wrist joints on each side.
- LPE is raw MPJPE of9 lower joints: pelvis plus hip/knee/ankle/toe on each
  side. The14+9 partition is disjoint and exhaustive; its weighted average
  must equal all23-joint MPJPE. Neither UPE nor LPE is root-centered.
- All position errors retain meters in JSON and are rendered in **cm**.
- sJPE uses **jerk magnitudes**, not signed jerk or jerk-vector subtraction:
  third forward difference times20^3, then Euclidean norm. The94 valid frames
  are compared at left/right toe bases and wrists, adapting upstream
  HumanML3D indices10/11/20/21 to body23 indices18/22/10/14.
- Primary **sJPE mean** averages the pointwise symmetric difference across
  time and4joints, retaining upstream epsilon1e-6. It is dimensionless0..1,
  without an extra2 or100 factor. Noise and static components sum to total.
- Upstream `calculate_area` sums4joints but divides only by time. Its exact
  native result is saved as `sjpe_discord_v1_jointsum` (4x the reported mean),
  with both components. `calculate_area_v2` is a distinct ratio-of-sums
  aggregation, also saved explicitly. Neither is silently mislabeled as the
  other. For completely zero jerk, v2's undefined0/0 is explicitly defined0.
- Original decoded GT itself is the zero-error reference, including sJPE.
  Raw GT jerk magnitude remains available; zero error does not mean zero jerk.

## Implementation and verification

`motion_error_metrics.py` defines the metrics and their full protocol. The
shared `evaluate_uem_phase3.measure` adds them to future body23 evaluations.
`backfill_motion_metrics.py` computes them from existing predictions, with an
exact original-decoder GT cache. Existing PA/MPJPE/root values are independently
recomputed and checked against every original scored case at tolerance1e-6m.
Native DisCoRD functions are extracted by AST, without importing its model
stack, and directly compared with the local implementation. Analytic checks
cover constant jerk, over/under jerk, GTidentity, static degeneracy and the
upper/lower partition. All numerical work runs on an owned compute node.

Original summaries, completion hashes, predictions and old Notion tables are
preserved. New per-case/batch/recording-balanced summaries go to
`extended_metrics/<model>/` beside each original evaluation. Combined CSV/JSON,
protocol, source copies and publication receipts go to:

`/scratch2/whwjdqls99/cosmos/operations/motion_metrics_20261105`.

Tmux `motion_metrics_20261105` runs the initial compute pass and then the
metadata-only watcher. Compute launcher uses existing owned2673623 CPUs;
`CUDA_VISIBLE_DEVICES` is empty, and no GPU allocation is canceled.
Notion receives a separate additional table on the established Ablations page.
Use actual COMPLETE/publication receipts rather than this setup note to claim
which models are finished.

## First completed pass (2026-11-05 10:37KST)

All10 already-completed model evaluations plus both GTreferences have full
11911-case new metrics and passed per-case MPJPE/PA/root parity. Hidden-FiLM
92879 is partial at10317cases in this first snapshot and updates only after
its original full scoring completes. Native/code and analytic tests passed;
Notion publication was independently read back. See
`BACKFILL_PASS_VERIFIED.json` and `notion_publication.json`. The metadata watcher
was moved to the existing Cosmos interpreter after the login interpreter lacked
NumPy; numerical workers completed0 and no inference was interrupted.

[New Notion metrics table](https://app.notion.com/p/binne/Ablations-40dd2a44f44b8080aad3cda6b5fea7c8#40fd2a44f44b81bfa42dd49ef86b2a77).
