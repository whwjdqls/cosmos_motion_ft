# Phase2 paper ablation menu

Proposal dated 2026-10-31, updated to reuse Kimodo369 and the central checkout
with minsu-only thin launchers. This document does not launch training. The
recommended first four are A0 unchanged control, A1 flow velocity, A2 existing
Kimodo369 and A3 image+text50% with BONES exposure fixed. The pending slots are described separately
in `ABLATIONS_PRO6000_20261031.md`. Central implementations and thin minsu launchers now exist; see that document
for CPU evidence, the exact user-selected dropout partition, full-stat preparation
and outstanding GPU acceptance.

The user selected **minsu1206** as the execution account/destination, with thin
launchers calling the shared whwjdqls99 checkout. Explicit follow-up: preserve
all existing whw allocation requests and allocations; none was cancelled.

Reference architecture: seven motion SA→CA→FFN blocks, final normalized frozen
Edge reasoner hidden states, learned per-block K/V projections, cross-key RMSNorm,
native RoPE, caption-only T2M packing, unchanged multimodal TI2M packing, shape
conditioning, camera/head-aligned UniEgo283, direct clean-x0 training.

## Existing predecessor representation identified

The repository's `motion_expert/BONES_SEED_POC.md:64` explicitly records the
transition from the original SOMABonesSeedDataset's **369-D KimodoMotionRep**
to the precomputed **283-D UniEgo** features. The existing implementation is
`kimodo/motion_rep/reps/kimodo_motionrep.py` in the pinned Kimodo dependency,
currently `/home/whwjdqls99/kimodo_open`. It has an encoder, an inverse decoder
and optional per-actor neutral joints; the shape-aware dataset passes these.

Its SOMA30 layout is smooth-root position3 + global heading2 + root-offset
joint positions90 + global 6D rotations180 + joint velocities90 + contacts4
=369. It also exposes local-root conversion helpers. This is Kimodo's
smoothed-root representation, distinct from native HumanML3D's SMPL22 features.
Use the existing Kimodo/SOMA representation as the natural predecessor
comparison before developing an additional new HumanML-style representation.
The queued slot is not implemented or launched by this identification alone.

Original source-tree pointers remain at
`/lustre/whwjdqls99/cosmos/weka/nymeriaplus_kimodo_proportional/S*` and
`/lustre/whwjdqls99/cosmos/weka/seed/soma_proportional_motions_20fps` (symlinked
to Scratch2). Listing these paths establishes source availability, not a
full-corpus validity audit. For a representation-only comparison, encode the
same currently calibrated/aligned physical motions and windows in both forms;
using historical raw source transforms unchanged would also change data
canonicalization/head orientation. New matched training statistics and the
existing physical losses still need integration and acceptance checks.

## High-value controlled comparisons

| Priority | Ablation | Comparison | Claim it can test |
|---|---|---|---|
| 1 | Motion representation | UniEgo283 versus the existing Kimodo369 on the identical SOMA skeleton | Suitability of the head/camera-centric representation for generation and image-conditioned motion |
| 1 | Prediction objective | Clean x0 versus flow velocity noise−x0 | Whether the motion prediction target and its noise weighting matter |
| 1 | Image-conditioned training mass | TI2M25% versus50%, with overall BONES37.5% fixed | Whether limited image training exposure limits visual conditioning |
| 1 | Physical losses | Full loss versus no contact/foot-velocity/foot-height terms; separately no decoded-joint/temporal terms | Which physical supervision improves contact and coherent motion |
| 1 | Attention organization | Joint attention versus SA→CA; additionally CA-only versus SA→CA using the same memory/projections/norms | Value of stream separation and motion-only temporal mixing |
| 2 | Reasoner memory source | Final hidden states versus layerwise K/V in the same SA→CA layout | Which frozen reasoner representation transfers best to motion |
| 2 | Conditioning prior | Frozen Cosmos conditioning versus a suitable frozen LLM2Vec/CLIP encoder with the same motion expert and inputs | How much the conditioning backbone contributes beyond the motion generator |
| 2 | Shape conditioning | Real neutral shape versus constant neutral-shape input | Whether the model responds to supplied body shape |
| 2 | BONES mixing | Nymeria-only versus mixed BONES/Nymeria, keeping TI2M25% | Whether additional text-motion data improves semantics/generalization or dilutes Nymeria specificity |
| 3 | Expert capacity | For example3/7/14 motion blocks with fixed widths and conditioning | Whether quality depends strongly on trainable motion capacity |

For the encoder experiment, match input modalities, training examples, optimizer,
motion-expert architecture and trainable adapter budget as closely as possible.
Encoder tokenization and frozen parameter counts may differ and must be reported.
A text-only encoder comparison evaluates T2M only; an encoder ablation making
TI2M claims needs an image-capable alternative. Do not call a POC comparison
controlled if decoder depth, prompt, data or losses differ too.

## Existing experiment families worth auditing/reusing

- Caption-only T2M versus the Edge instruction/chat wrapper. TI2M still uses
  the native multimodal template in the current caption-only model.
- Projection A joint attention, added SA before joint attention, final-hidden
  self/cross, layer-K/V self/cross and cross-key normalization variants.
- Cross-key RoPE and projected-V norm experiments. Toggle one factor at a
  time: an existing V-norm-plus-no-key-RoPE run alone cannot identify which
  change caused a difference. Frozen reasoner native positions stay intact.
- Centered versus absolute decoded-joint loss, global-feature weighting,
  batch128/256, LR-floor changes, BONES-only and longer-training experiments.

These source/run records establish that experiments were prepared or run, not
that every pair is a matched causal ablation. Audit actual checkpoint contracts,
losses, prompts, mean/std, batch sizes, steps and evaluation settings before
assembling a result table. Select new controlled reruns only for missing pairs.
Treat LR/batch/duration sweeps mainly as optimization controls or appendix
material unless they answer a central paper claim.

## Proposed first four jobs

Use a fresh unchanged control plus representation, prediction-objective and
image-proportion variants. Keep the same new hardware/topology, global256,
200k updates, LR/schedule, seed policy, prompts and physical loss coefficients.
Decide the feasible per-GPU microbatch after acceptance; keep it identical
across comparisons when possible. Do not compare a fresh ablation only against
an old control with different batch/optimizer settings.

Current pre-caption-dropout mixture:

| Sample type | Reference | Higher-image proposal |
|---|---:|---:|
| BONES text-only | 37.5% | 37.5% |
| Nymeria text-only | 37.5% | 12.5% |
| Nymeria image+text | 25% | 50% |

The proposal uses T2M:TI2M=50:50 and bones_frac0.75 within T2M. All TI2M is
Nymeria. Preserve historical BONES19% effective dropout and Nymeria T2M10%.
The user selected TI2M30% caption dropout for A3:35% image+text and15% image-only.
Aggregate A3:41.625% text-only,35% both,15% image-only,8.375% text/image-null;
shape and length remain supplied. A0/A1/A2 retain the historical TI2M10%.

## Interpretation and implementation constraints

Velocity prediction needs x0_hat=x_sigma−sigma*v_hat before decoded physical
losses. Direct velocity MSE changes sigma-dependent weighting: the current
clean-x0 feature MSE equals sigma² times the corresponding velocity MSE.
State which objective is used. Preserve the original noising/sigma distribution
and UniPC integration convention; never decode raw velocity as body pose.

The recommended A2 reuses Kimodo369, not a new HumanML3D-native adapter. Keep
SOMA, its real shape, original contact labels and the same calibrated windows.
Kimodo's encoder can recompute contacts; preserve the current labels so that
this does not become a contact-supervision ablation too. Use its own train-only
mean/std, verify round trips and integrate the appropriate differentiable decoder.
Report unavoidable input/output-head parameter changes. Preserve physical
coordinate alignment; do not introduce GT initial-state conditioning or fit
generated motion to GT for the main metrics.

Keep implementations in `/home/whwjdqls99/cosmos_motion_ft`. For minsu1206,
use only thin launch wrappers and account-owned output/cache/log paths, with
source revision recorded per run. Existing whw Slurm allocations belong to
whw; they are not transferable to minsu.

Removing SA, changing attention organization or adding depth changes parameter
count/FLOPs. Report these and use a capacity-matched control where practical.
The historical joint/self-cross runs alone do not isolate attention softmax
competition if projection layout, norms or prompt packing also differ.

## Evaluation

- Same held-out cases, captions, neutral shapes, lengths and paired seeds for
  all variants within each dataset. BONES and Nymeria have distinct test sets;
  compare every model on the same cases within each, with its matching evaluator.
- Correct generator statistics for decoding, then evaluator-specific statistics.
  Do not feed the new representation directly to an evaluator trained on another.
- Generation: FID, R@1/2/3 with the agreed32-candidate/three-grouping-seed
  protocol, matching distance, diversity, skating, penetration and airborne rate.
  Always include measured GT. Paired MPJPE is secondary for unconstrained
  generation and should not rank models by itself.
- Visual conditioning: correct image versus wrong image and no image, fixed
  caption/noise; assess visible action/interaction and first-frame body plausibility.
  Image-swap sensitivity alone does not establish useful image conditioning.
  Match prompt layout where practical and document native T2M/TI2M differences.
- Shape: shared prompt/noise with multiple neutral shapes; measure bone-length
  adherence and response to requested proportions, with qualitative examples.
- Fixed sampling settings for the primary comparison. Tune guidance only on a
  separate validation set. Show matched-step100k/200k results and multiple
  generation seeds; do not equate these with multiple independent training seeds.
- The learning-value test for downstream Phase3 is a separate follow-up:
  Phase2-pretrained versus random motion expert, same Phase1 and bridge recipe.
  A permanently frozen random expert is a nonviable control for a claim about
  pretraining; allow motion learning in both comparison arms under a matched
  downstream optimization policy and disclose that extra trainable scope.

## Primary references

- HumanML3D authors' representation implementation:
  https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb
- MDM motivates sample prediction through geometric motion losses:
  https://arxiv.org/abs/2209.14916

These references motivate tests; they do not establish outcomes for our model.
