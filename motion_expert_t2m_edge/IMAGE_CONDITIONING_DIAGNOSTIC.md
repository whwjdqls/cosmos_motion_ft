# Paired image-conditioning diagnostic

The report and selected playable comparisons are published under the Notion
[Experiments page](https://app.notion.com/p/407d2a44f44b81a5ad53dbb685cdbd9e).
The user requests all experiment reports in Notion; retain local evidence too.

## Full-test generative extension (2026-10-28)

`evaluate_image_conditioning.py` uses the existing immutable Nymeria TI2M
benchmark: **11,892 caption/window cases, 11,657 physical windows, 71 test
recordings**, all T97/20fps. Each case is sampled from pure noise under correct
image, two wrong-image controls, and native text-only conditions. Wrong-image
donors are two deterministic permutations across recordings, preserving the
same image distribution. All variants share the target caption, neutral shape,
length and initial noise. Native T2M also changes the prompt layout.

The checkpoint and its normalization are the same 200k specialist used below.
Sampling retains UniPC35/shift3/CFG2 and the established stable per-case seed;
TI2M null conditioning drops text and retains the corresponding image. There
is no GT motion input. Use the validation-selected **Nymeria** evaluator with
its independently verified statistics, not the BONES evaluator or generator
statistics. Report measured GT and generated R@1/2/3 in 32-candidate groups
(three grouping seeds), FID, matching distance, foot skating and floor metrics.
The three retrieval seeds quantify grouping variation, not sampling variance.

GT source arrays require their saved per-recording floor offset. Recomputed
GT embeddings and physical metrics are checked against the established full
test reference for every case before a batch is accepted. No generated motion
is grounded or fitted. All input arrays/images and checkpoints are hashed;
missing old image-cache keys are rebuilt from the exact source frame under
the new operation, preserving existing caches. Failed cases stop evaluation
rather than silently changing the test set. Completed batches can be resumed.

Yonsei artifacts: `/scratch2/whwjdqls99/cosmos/operations/phase2_image_generative_20261028`.
The retained 2x4090 allocation2608785 runs two independent shards. Consult
their `progress.json`, final `summary.json` and `completed_at` for actual status.
Configuration, donor mapping, per-case measures, candidate groups/ranks and
compact result tables are saved. The coordinator publishes the final table to
the same Notion page after successful aggregation; publication status is
recorded separately. The existing 32-case videos are labelled as diagnostic
examples, not as samples from this larger cohort.

GT-path correction during execution: the original reference obtains its GT
embedding through the dataset's CPU UniEgo decoder. One window produced a
2.38e-7m maximum CPU/GPU joint difference but an8.88e-4 maximum embedding
difference after evaluator preprocessing. Repeating the original CPU path
matched the saved embedding exactly at batch64 (about1e-7 at batch1/8).
The resumed worker uses this original GT path with unchanged tolerances;
generated sampling/scoring is unchanged. Final aggregation uses the one
immutable measured GT reference for all conditions. No cases are dropped.
`gt_mismatch_diagnostic.json`, `gt_path_fix.json` and
`same_test_set_audit.json` retain the evidence and coverage checks.

## Joint text/image CFG preview

`diagnose_joint_cfg.py` uses the exact32 cases and seeds0/1/2 from the first
diagnostic. The five sampled settings are current TI2M text-CFG2 and joint
CFG1/2/3/5. Current null conditioning retains the observed image and drops
text; joint null conditioning uses native T2M with empty text and no image.
Both use `x0_null + scale * (x0_cond - x0_null)` and the unchanged UniPC35
shift3 solver. Guidance is applied once. CFG1 directly uses the conditional
prediction without cancellation roundoff. Model weights/training stay fixed.

The native no-image branch has a different prompt layout. This is a joint
guidance intervention, not a same-layout image-only control. The32-case
preview is not the full11892-window generative benchmark and is not a reliable
FID estimate. Report actual GT plus sampled reconstruction/physics, with six
predetermined seed0 comparison videos. Larger visual changes alone do not
establish better motion quality or stronger useful conditioning.

Yonsei artifacts are `operations/phase2_joint_cfg_20261028` under Scratch2
Cosmos. Its coordinator uses one retained4090 while full-test shard0 proceeds,
then automatically resumes full-test shard1. Results and two playable examples
are appended to the same Notion report. Check logs/progress for actual state.

## Image-only CFG preview

`diagnose_joint_cfg.py --target image` retains the exact caption in the native
T2M reference branch and removes the image. For scales1/2/3/5 it applies
`x0_text + scale * (x0_text_image - x0_text)`, once per UniPC denoising step.
There is no additional text CFG term. The control remains current TI2M text
CFG2. Caption token IDs in each no-image reference are checked against native
tokenization of the full caption, and those branches must have zero image
tokens. T2M/TI2M prompt layouts still differ, as in the joint-CFG experiment.

This uses the identical32 windows, input hashes, shapes and seeds0/1/2 from
the earlier previews, with the same200k checkpoint, UniPC35 and shift3. The
first six seed0 examples are rendered. The completed joint-sweep source was
archived beside its outputs before adding this option, preserving its original
source hash. Existing training, generator weights and full-test evaluation
workers are unchanged.

Yonsei outputs: `operations/phase2_image_only_cfg_20261028` under Scratch2
Cosmos. The short preview shares retained2608785 GPU1 after confirming about
8.3GB use by the full evaluator; its own allocator is capped at40% of the4090.
It publishes its protocol, actual GT/model metrics and selected videos to the
same Notion page. Do not treat a shared-allocation launch as an idle GPU or as
proof of completion; consult logs and publication receipts.

`diagnose_image_conditioning.py` tests the completed Self→cross final-hidden,
cross-K RMSNorm, caption-only B256 Phase 2 specialist at 200k. It changes no
training code, weights, losses, reasoner operators, or normalization statistics.

The Yonsei execution on 2026-10-28 uses retained allocation 2608785 (two RTX4090s).
Configuration, frozen inputs, per-case measurements, and results are under:

`/scratch2/whwjdqls99/cosmos/operations/phase2_image_conditioning_20261028`

Runtime logs and `completed_at` determine whether the experiment is finished.
The script is site independent; invoke it through `motion_expert_t2m_edge/run.sh`
with `COSMOS_SITE=yonsei` or `grasp`, explicit checkpoint/output paths, the site's
CUDA environment, and `SHAPE_TMR_BUNDLE` for the existing physics definitions.
All preparation, inference, metric computation, and rendering require a compute
allocation. The launcher's source-image cache is allowed, while the frozen
image-embedding cache is disabled for this diagnostic.

## Quantitative result, 2026-10-28

Both inference workers completed all 32 recordings, with zero preparation
rejections, three seeds, four conditions, and four denoiser noise levels.
All same-input repeats were bitwise identical. Each TI2M condition contained
64 visual tokens; correct/wrong token IDs and positions matched exactly.

| Condition | Pure-noise x0 feature MSE | Sample MPJPE (m) | Sample PA-MPJPE (m) | Frame-0 MPJPE (m) |
|---|---:|---:|---:|---:|
| GT self | 0 | 0 | ~0 | 0 |
| Correct image | 0.692916 | 0.399072 | 0.198173 | 0.151046 |
| Wrong image 1 | 0.741236 | 0.406770 | 0.198598 | 0.183634 |
| Wrong image 2 | 0.763706 | 0.432288 | 0.201755 | 0.186513 |
| No image (native T2M) | 0.722096 | 0.454883 | 0.199866 | 0.159677 |

The correct image lowers pure-noise denoiser MSE by 6.5% and 9.3% relative to
the two wrong-image controls. Their other-minus-correct paired 95% intervals
are [0.00815, 0.09308] and [0.01306, 0.15847]. At sigma0.25/0.5 there is no
clear correct-image benefit; at0.75 it is small and not consistent across both
wrong-image controls. Native no-image differences also include prompt/layout
effects and should not be interpreted as a pure image-content intervention.

Frame-0 sampled error improves by 3.26cm and3.55cm over wrong images (paired
95% intervals [1.31,5.29]cm and [1.53,5.78]cm). Whole-clip reconstruction gains
are less consistent: MPJPE's interval crosses zero for wrong1, and PA-MPJPE's
interval crosses zero for both wrong controls. Swapping images changes sampled
joints by23–26cm on average, establishing sensitivity; the correct-image
advantages are the separate evidence that some of that sensitivity is useful.

Measured GT physics is retained, including foot-skate-from-height0.157536,
foot-skate ratio0.156860, mean foot penetration depth1.71032cm, and18.4278% of
frames with a foot below-1cm. Correct-image outputs have the corresponding
values0.122404,0.119179,1.34764cm,20.1031%. These do not establish that image
conditioning resolves floating or skating; the wrong-image conditions do not
consistently worsen those metrics.

Conclusion: the trained image-conditioning path is functional and has a
measurable, limited benefit, most clearly for initial pose and pure-noise
prediction. It is not evidence of strong image control throughout the sequence.

Prompt clarification: `nano_caption_only` affects **T2M only**. TI2M calls
`encode_reasoner_image_text`/`apply_chat_template(add_generation_prompt=True)`
during both training and sampling. Thus this experiment has no TI2M prompt
mismatch between training and inference.

## Controls and coverage

- Freeze 32 valid T97 windows from 32 distinct Nymeria test recordings, selected
  with seed 20261028 before seeing any model outputs. Use the existing loader's
  floor calibration, canonicalization, camera/head representation and shape.
  Failed candidates are recorded and replaced only during cohort preparation.
- Use the checkpoint's exact mean/std, with SHA256 verification and strict
  architecture/data-contract loading. Preserve the prepared input hashes.
- For each case, compare the correct first image, two images from different
  test recordings, and native T2M with no image. Captions, neutral shape, output
  length and initial noise remain fixed. Wrong-image donors are fixed cyclic
  permutations of the cohort, independent of their scores.
- Assert equal input IDs, image-token masks, and 3-D positions for each
  correct/wrong comparison; only image embeddings change. Check their actual
  embedding difference and repeat the same model input to measure numerical
  reproducibility. No-image changes the native token layout, so it cannot
  isolate image content as cleanly as the image swaps.
- Seeds 0/1/2; single-denoiser x0 predictions at actual sigma
  0.25/0.5/0.75/1.0. Sigma 1 contains no GT motion. This probe has no CFG.
- Also sample with the existing UniPC35 / shift3 / CFG2 implementation.
  TI2M's null branch removes text and retains the same condition image.
  Every condition receives an exact copy of the same initial noise.

## Measures and interpretation

`summary.json` and `per_case.csv` retain denoiser feature MSE, caption-fixed
image-swap sensitivity, sampled feature MSE, MPJPE, whole-clip Procrustes MPJPE,
frame-0 MPJPE, root error, acceleration, and the existing foot/floor metrics.
GT-self reconstruction and measured GT foot/floor metrics are included.
Paired differences average the three seeds within each recording; 95% bootstrap
intervals resample recordings, not individual correlated seeds.

An image-dependent output alone establishes sensitivity, not useful image
conditioning. Look for lower correct-image error than both wrong-image controls,
especially at pure noise, and consistent improvements in the sampled comparison.
Sampled reconstruction is only a diagnostic: image+caption allows multiple valid
motions. These 32 cases are not a reliable FID/retrieval benchmark. Foot contact
and floating are separate measurements, not implied by conditioning sensitivity.

`gallery/` renders the predetermined first six cases at seed 0: GT, correct image,
wrong image 1, and no image, with both input images shown. The shared renderer's
head axes, thin skeletons, floor layering and original motion coordinates are
preserved. Rendering skips alternate frames and writes at 10 FPS, preserving
the duration of the original 20 FPS motions.

## Commands

Within the configured compute environment, pass the same `--checkpoint` and
`--out` to each stage:

```bash
DIAGNOSTIC="$REPO_ROOT/motion_expert_t2m_edge/diagnose_image_conditioning.py"
bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$DIAGNOSTIC" prepare --checkpoint CKPT --out OUTPUT
# One independent process per GPU; no DDP or shared trainable state.
bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$DIAGNOSTIC" worker --shard 0 --shards 2 --checkpoint CKPT --out OUTPUT
bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$DIAGNOSTIC" worker --shard 1 --shards 2 --checkpoint CKPT --out OUTPUT
bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$DIAGNOSTIC" summarize --checkpoint CKPT --out OUTPUT
bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$DIAGNOSTIC" render --checkpoint CKPT --out OUTPUT
```

Use absolute script paths when `run.sh` changes into the framework directory.
Completed case outputs can be resumed; preparation refuses to overwrite an
existing cohort, and aggregation requires every distinct selected case.

## Completed image-only CFG follow-up (2026-10-28)

On the same32 cases and three seeds, retain the caption in both branches and
use `x0_text + s*(x0_text_image - x0_text)` for s=1/2/3/5. Training and
checkpoints are unchanged. The native no-image T2M branch changes prompt layout.

| Condition | MPJPE cm | PA-MPJPE cm | Frame-0 cm | Foot penetration cm |
| --- | ---: | ---: | ---: | ---: |
| GT | 0.00 | 0.00 | 0.00 | 1.7103 |
| Current text CFG2 | 40.01 | 19.92 | 15.10 | 1.3621 |
| Image CFG1 (unguided) | 41.15 | 20.17 | 14.97 | 0.9516 |
| Image CFG2 | 39.17 | 21.20 | 16.41 | 1.2743 |
| Image CFG3 | 41.23 | 22.27 | 18.11 | 1.6559 |
| Image CFG5 | 45.66 | 24.29 | 20.96 | 2.4317 |

There is no clear overall gain. Image CFG2 minus current text CFG2 has paired
MPJPE difference -0.84cm (95% bootstrap interval [-7.06,+4.02]), PA-MPJPE
+1.28cm [+0.21,+2.37], and frame-0 error +1.32cm [+0.40,+2.34]. Resample32
recordings after averaging three seeds,5,000 resamples. These exploratory
reconstruction measures are separate from generative FID/retrieval and do not
establish a floating/contact fix.

Across all96 case-seed pairs, unguided image/joint CFG1 outputs match bitwise.
Recomputed text CFG2 baselines differ slightly across differently packed
reference batches (mean joint difference0.481cm, maximum1.742cm); compare
settings to their own sweep's baseline. Caption retention and image absence
in the reference branch were asserted. Evidence: Scratch2 Cosmos operations
`phase2_image_only_cfg_20261028/{summary.json,audit_and_comparison.json,per_case.csv}`.
Two playable six-panel videos and metrics are verified on the existing
[Notion report](https://app.notion.com/p/407d2a44f44b81a5ad53dbb685cdbd9e).
This completed32-case diagnostic is distinct from the ongoing11,892-case benchmark.

## Actual input-modality generation comparison (2026-10-28)

The user's objective is generative quality, not reconstruction error. Earlier
"image-only CFG" retained text; it did not test image-only generation. The new
`evaluate_generation_modalities.py` tests three actual input settings:

| Setting | Caption supplied | First image supplied |
| --- | --- | --- |
| Image only | No (empty string) | Yes |
| Text only | Yes | No |
| Image + text | Yes | Yes |

All use `x0_unconditional + 2*(x0_conditioned-x0_unconditional)`, where the common
reference is native T2M with empty text and no image. Thus image+text uses joint
guidance, unlike the earlier text-CFG baseline that kept the image in the null
branch. This choice fixes guidance strength/reference across modalities. Native
TI2M/T2M prompt layouts still differ. No checkpoint, training or loss is changed.

Quantitative evaluation preserves the byte-identical11,892-case manifest,
71 test recordings, checkpoint and evaluator hashes, and per-case noise seeds
from the full-test image-conditioning evaluation. All three modes generate
97 frames at20fps from pure noise, with the same neutral shape and UniPC35 /
shift3. No GT pose, observed motion, or camera trajectory enters generation.
Generator normalization and evaluator normalization are separately verified;
GT uses the original CPU dataset path, with the existing reference guard intact.

Report measured GT and all three modes using Nymeria TMR FID, paired R@1/2/3
in32-candidate groups (seeds0/1/2), matching distance, embedding diversity,
foot skating, penetration and airborne fraction. Diversity uses1,000 fixed
disjoint pairs shared across modes and GT, seed20261028. Compare diversity to
GT rather than maximizing it. Image-only retrieval uses withheld captions and
measures image-implied semantics; it cannot be interpreted as obedience to text
that was not provided. No MPJPE ranking is used.

Six predetermined cases from the original32-case diagnostic provide side-by-side
GT / image only / text only / both previews with paired seeds0/1/2. Seed0 videos
retain the standard thin skeletons, first-frame head axes, and physical floor;
green borders identify the observed image. Captions shown in the banner go only
to text-only/both branches. These previews are not the full quantitative cohort.

Artifacts: Scratch2 Cosmos `operations/phase2_generation_modalities_20261028`.
Its retained2608785 coordinator publishes the first two videos and then the
full quantitative table automatically after complete, zero-skip aggregation.
Read `shard_*/progress.json`, `worker.exit_codes`, `final_summary_ready.json`, and
`publication.json` for actual progress/completion; this protocol is not itself
a claim that full-test inference has finished.

## Fixed text guidance with varying image guidance (2026-10-28)

`evaluate_text2_image_cfg.py` implements the requested decomposition:

```text
guided = unconditional + 2*(text - unconditional)
                       + image_weight*(text_image - text)
image_weight = [0, 0.5, 1, 2, 3, 5]
```

The three denoiser branches use native empty T2M, caption-only T2M, and the same
caption plus first image in TI2M. All predictions use the same current noisy
motion. Apply guidance exactly once; the native UniPC35/shift3 solver receives
guidance1. Weight0 returns exact text-only CFG2, and weight2 returns exact joint
CFG2 without cancellation roundoff. The text-first decomposition assigns
text-image interaction to the image increment; it does not estimate independent
causal contributions. Native prompt layouts differ.

Eight illustrative cases (slots6,16,14,27,30,24,7,19) from the original32-case
held-out diagnostic were selected by action variety and GT motion before this
sweep. They cover kneeling, high-step marching, walking/turning, bending to pick
objects up, kicking, reaching, upper-body exercise, and badminton. An initially
considered low-articulation marching case13 was replaced by case7 before any
sweep sampling. `selection.json` records the complete32-case GT motion audit.
This is explicit illustrative selection, not ranking by generated quality or
retrieval. Seeds0/1/2 are matched for all weights. An image-only CFG2 comparator
has an empty caption. GT is provided only for rendering/measurement.

Each selected case produces a four-panel GT/image-only/text-only/both video and
an eight-panel sweep video (GT,image-only,and six image weights). All preserve
the shared renderer and observed-image green border. Cases6,16,14,30 at seed0
are uploaded to the [follow-up Notion report](https://app.notion.com/p/407d2a44f44b817ba15ad37cef229360).

The full-test sweep preserves the original11,892-case manifest, checkpoint and
evaluator identities, neutral shapes and stable generation seeds. It reports
measured GT and six settings with Nymeria TMR FID,R32,matching distance,diversity,
and contact metrics, with no MPJPE ranking. The selected illustrations do not
determine the quantitative cohort. The dispatcher waits for the existing
full-test evaluations to finish before using their retained2×4090 allocation.
Operation: Scratch2 Cosmos `operations/phase2_text2_image_cfg_20261028`; live
progress, exit files, and verified publication state determine completion.

### Long root travel and visible-object supplements

The user additionally requested cases with substantial GT travel and cases
where a visible captioned object could resolve motion ambiguity. The operation
`phase2_root_object_cfg_20261028` scans all11,892 eligible GT windows and saves
`all_gt_root_travel.csv`. Long-travel candidates require net horizontal travel
at least4m in4.8s, mean speed at most3m/s, maximum speed at most5m/s,
path/net at most1.6, and root-height span below0.7m. These select clear sustained
locomotion rather than obvious jitter; they do not filter the full benchmark.

Four selected test windows cover walking, indoor turning, jogging and running:
source indices2706,10864,11747,4161, with5.44–6.82m paths and5.12–6.67m net
travel. Supplemental slots100–103 identify these windows. Slots200–203 use
original diagnostic cases26,9,12,31: laptop approach/trackpad reach, sliding
glass-door interaction, vanity-drawer pulling, and bag placement on a chair.
The named targets were checked in the original conditioning images. No
generated-output quality/error score was used to select these cases.

`visualize_conditioning_examples.py` delegates inference unchanged to the
fixed-text/image-weight sampler. It adds raw world-XZ root-trajectory figures
(no recentering or GT alignment) and object-input figures with manual amber
target outlines. Outlines are display-only; the generator receives unaltered
RGB. Long-travel frames retain the full-test cache resolution and go through
the normal image preprocessing; original object-diagnostic inputs are256px.
The two input sizes are not silently equated or manually cropped.

All eight supplemental cases use the same seven generation variants and
seeds0/1/2. Seed0 videos, trajectory figures and relevant input figures are
published in case-specific toggles on the existing follow-up Notion page.
These are illustrative tests of whether an image might help, not a claim
that it helps in every case; failures are retained. They do not change the
full-test quantitative cohort or launch another full-test sweep.
