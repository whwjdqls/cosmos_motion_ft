**Measured camera trajectories give the strongest reported position reconstruction. In the video-based comparisons, captions and bridge extrapolation help substantially; the evaluated FiLM snapshots remain weaker than the control on matched video-only reconstruction.**

Synthesis as of 2026-11-01 of [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3) and its [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2) child page, including the November 1 control-150k update. This consolidates the baseline, input-ablation, caption/guidance, joint-training and camera-FiLM results. Training updates and model availability are kept separate from evaluated-checkpoint results.

- **Video-only:** original 200k has the lowest native MPJPE among our reported Phase 3 checkpoints at 20.56 cm. Control 150k is close at 20.97 cm and has better native motion FID/retrieval:0.225692 and 51.56% R@3. The evaluated hidden-state 150k/trajectory 130k snapshots score 26.19/31.00 cm.
- **Video+text:** scale 2 gives 13.98 cm for joint 200k,14.13 cm for original 200k and 15.30 cm for control 150k. The original model’s scale 1.5 point has a better semantic tradeoff: FID 0.128778, R@3 72.92%, MPJPE 14.63 cm.
- **Measured camera:** UniEgoMotion recon/all with video+camera score 8.03/8.48 cm. Trajectory FiLM 130k reaches 9.44 cm with text+camera+video at scale 2. These systems observe different information and body-shape conditioning, so this is a comparison of complete systems rather than a controlled architecture ranking.
- **Joint training:** the joint 200k checkpoint has no clear matched reconstruction advantage over original 200k: all four reported paired MPJPE intervals include zero.
- **FiLM:** camera input and captions improve several within-model results, but camera-only hidden-state conditioning is mixed. Final hidden-state 200k/latest trajectory 180k weights now exist; the published FiLM scores still use 150k/130k.

**The common comparison uses 71 selected windows, one per held-out recording.**

Each window has 97 frames at 20 fps. These are the same fixed 71 cases, not every eligible Nymeria test window. Physical errors use Body23 in the native frame-0 head-yaw coordinates; only PA-MPJPE applies a whole-clip similarity fit. Reported outputs receive no generated-motion grounding, root reset or oracle head information unless explicitly labeled.

Phase 3 receives supplied neutral body shape; camera baselines predict shape and do not receive GT body shape. Camera-conditioned arms observe all 97 measured poses; video-only arms do not. Caption access is labeled separately. Thus matching cases does not imply matching observed inputs. Our motion sampling uses UniPC35/shift 3; UniEgoMotion retains its native 1000-step sampler.

Motion FID and retrieval use the same validation-selected Nymeria TMR evaluator and training-only Body23 → SOMA30 completion. FID uses all 71 windows. R@1/2/3 uses 32 candidates and grouping seeds 0/1/2, with 64 windows scored and 7 dropped per seed; “ ± ” is sample SD across those groups. This is a small-cohort semantic diagnostic, separate from full-test Phase 2 scores and video FVD. Common GT R@1/2/3 is 48.96 ± 0.90 /63.02 ± 0.90 /70.83 ± 1.80%; GT self-FID is 0. Sources: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3); [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2).

**Native video-only results favor the original/control checkpoints over the evaluated FiLM snapshots.**

All our Phase 3 rows below use video, supplied neutral shape, empty text and native bridge scale 1. UniEgoMotion RGB_recon uses its video specialist with predicted shape and different training/sampling. Source: [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2), native video-to-motion tables; [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), RGB and completed-GT rows.

| Model / evaluated checkpoint | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | Motion FID ↓ | R@3 % ↑ |
| --- | --- | --- | --- | --- | --- |
| GT reference | 0.00 | 0.00 | 0.00 | 0.000000 | 70.83 ± 1.80 |
| Original Phase 3, 200k | 20.56 | 9.98 | 18.98 | 0.259868 | 42.71 ± 4.77 |
| Joint Phase 3, 200k | 22.53 | 10.70 | 20.54 | 0.286935 | 45.31 ± 1.56 |
| Control, 150k | 20.97 | 9.74 | 19.12 | 0.225692 | 51.56 ± 2.71 |
| Hidden-state FiLM, 150k | 26.19 | 12.39 | 24.66 | 0.272464 | 46.35 ± 1.80 |
| Trajectory FiLM, 130k | 31.00 | 14.88 | 29.28 | 0.413816 | 35.42 ± 3.25 |
| UniEgoMotion RGB_recon, 341,420 | 44.22 | 19.80 | 42.22 | 0.281923 | 34.38 ± 1.56 |

Control 150k outperforms hidden-state 150k and trajectory 130k in MPJPE at all four video/text/scale settings. Its native control-minus-hidden-state 95% interval is[−8.02,−2.89]cm; control-minus-trajectory is[−13.72,−6.87]cm. These establish differences between the evaluated checkpoints. They do not isolate a FiLM architecture effect across training mixtures, task exposure and checkpoint budgets.

**Captions and stronger bridge extrapolation reduce video-conditioned position errors, with different optima for semantics.**

All entries in this compact matrix are MPJPE in cm; lower is better. GT self-error is 0.00 cm. “s” denotes bridge scale, with s1 the native prediction. Source: [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2), all prespecified video/caption/guidance settings.

| Model / evaluated checkpoint | Video s1 | Video s2 | Video+text s1 | Video+text s2 |
| --- | --- | --- | --- | --- |
| Original Phase 3, 200k | 20.56 | 16.43 | 16.31 | 14.13 |
| Joint Phase 3, 200k | 22.53 | 17.46 | 16.75 | 13.98 |
| Control, 150k | 20.97 | 17.33 | 17.05 | 15.30 |
| Hidden-state FiLM, 150k | 26.19 | 22.37 | 19.71 | 16.70 |
| Trajectory FiLM, 130k | 31.00 | 28.65 | 22.52 | 18.11 |

For original 200k, video+text at scale 1.5 gives MPJPE 14.63 cm, FID 0.128778, R@3 72.92%; scale 2 gives 14.13 cm, FID 0.140510, R@3 69.79%. The sweep tests 0,0.5,1,1.5,2,3,5; these are exploratory test-cohort tradeoffs, without an independently validation-selected winning scale. Source: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), full guidance sweep.

**Scale 2 is whole-bridge-disabled extrapolation.** Its reference bypasses the modality bridges, including within-motion attention, while retaining the caption and shape. It is not a trained null-video classifier-free branch or pure text CFG. In the original caption arm, scale 0 already has R@3 68.75% despite 43.66 cm MPJPE: strong caption retrieval alone does not establish video-aligned motion. Retrieval can also exceed the measured GT reference in these small groups without implying better physical reconstruction.

For joint 200k minus original 200k, native video MPJPE changes by+1.97 cm with 95% CI [-0.06, +4.01]; text+video/scale 2 changes by-0.16 cm with CI [-1.46, +1.22]. All four matched intervals span zero. Original training samples V2M and M2V at 50% each; joint training samples V2M, M2V and joint at one-third each. Equal update counts therefore have different task exposure. These rows evaluate V2M reconstruction, not joint video+motion generation quality. Source: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), joint-trained comparison.

**Measured camera input substantially improves geometric reconstruction.**

The following rows use native scale 1 and no text. Input and shape access differ across systems; compare the labels before interpreting a ranking. Sources: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), baseline/input-ablation tables; [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2), camera-conditioned native inference.

| Model / evaluated checkpoint | Observed input | MPJPE cm ↓ | Root ADE cm ↓ | Motion FID ↓ | R@3 % ↑ |
| --- | --- | --- | --- | --- | --- |
| GT reference | — | 0.00 | 0.00 | 0.000000 | 70.83 ± 1.80 |
| UniEgoMotion traj_recon, 555,800 | camera | 11.29 | 8.36 | 0.272554 | 45.83 ± 2.39 |
| EgoAllo, 160k | camera | 9.34 | 6.52 | 0.251218 | 49.48 ± 3.93 |
| UniEgoMotion recon, 339,832 | video+camera | 8.03 | 6.25 | 0.137530 | 57.29 ± 6.51 |
| UniEgoMotion all, 555,800 | video+camera | 8.48 | 6.51 | 0.130458 | 60.42 ± 2.39 |
| Hidden-state FiLM, 150k | video+camera | 18.90 | 17.22 | 0.221805 | 50.52 |
| Trajectory FiLM, 130k | video+camera | 12.21 | 9.67 | 0.252486 | 48.96 |

Within hidden-state FiLM, adding camera to video changes MPJPE 26.19 → 18.90 cm, with paired 95% CI [−10.82,−4.18]cm. Within trajectory FiLM it changes 31.00 → 12.21 cm, CI [−23.51,−14.60]cm. This measures the extra camera information and trained pathway together. The control 150k report currently covers video/text+video, so it does not supply the camera-conditioned control needed to isolate the FiLM component.

UniEgoMotion’s same-checkpoint missing-input ablations show strong camera dependence. All entries below are MPJPE in cm; GT self-error is 0.00 cm. Single-input rows are inference ablations of the full reconstruction setting. Source: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), UniEgoMotion input ablations.

| Same-checkpoint UEM ablation | Video+camera MPJPE | Camera-only MPJPE | Video-only MPJPE |
| --- | --- | --- | --- |
| UniEgoMotion recon, 339,832 | 8.03 | 8.18 | 55.93 |
| UniEgoMotion all, 555,800 | 8.48 | 9.51 | 43.17 |

For the recon checkpoint, MPJPE changes little when video is removed (8.03 → 8.18 cm), but loses accuracy when camera is removed (8.03 → 55.93 cm). Our native original V2M is much better than the dedicated UEM RGB specialist (20.56 versus 44.22 cm); differing shapes, training and samplers limit architectural attribution.

**Text helps camera+video for both FiLM variants, but does not uniformly improve camera-only reconstruction.**

All matrix entries are MPJPE in cm, using the same case/seed/shape inputs. GT self-error is 0.00 cm. Sources: [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2), native camera and text-conditioning follow-up.

| Model | Camera s1 | Text+camera s1 | Camera+video s1 | Text+camera+video s1 | Text+camera+video s2 |
| --- | --- | --- | --- | --- | --- |
| Hidden-state FiLM, 150k | 23.43 | 28.00 | 18.90 | 14.51 | 14.83 |
| Trajectory FiLM, 130k | 15.53 | 12.50 | 12.21 | 11.03 | 9.44 |

At native scale 1, adding text to camera+video changes hidden-state MPJPE by-4.39 cm (CI [-8.27, -1.12]) and trajectory by-1.18 cm (CI [-2.12, -0.28]). Hidden-state camera-only instead changes by+4.56 cm (CI [-2.70, 11.84]), spanning zero. FID and R@3 improve in all six native FiLM model/input caption comparisons, showing that semantic improvement and position improvement can disagree.

Trajectory 130k with text+camera+video has native FID 0.120673 and R@3 67.71 ± 3.93, versus hidden-state FID 0.124617 and R@3 68.23 ± 3.93. Trajectory’s scale 2 position result is 9.44 cm MPJPE,5.94 cm PA-MPJPE and 7.53 cm root ADE. Camera-scale 2 lacks a corresponding no-text scale 2 control in the report, so its improvement cannot be attributed to captions alone.

The compared recipes use empty captions for V2M training. Caption-conditioned V2M inference uses their frozen specialists’ existing text ability. Some saved captions describe more of a recording than the selected short window; this is an inference intervention, not evidence about text-conditioned retraining.

**Position, semantics and contact metrics answer different questions.**

GT skating is 18.51 cm/s, penetration 1.31 cm and airborne 16.42%; these are measured, not assumed zero. EgoAllo has good 9.34 cm MPJPE but 30.45 cm/s skating. UEM recon with only video has 55.93 cm MPJPE yet 2.65 cm/s skating and 0.00 cm penetration; the 83.91% airborne rate makes the low contact metrics an unreliable indicator of reconstruction quality. Sources: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3), Body23 physical metrics and UEM ablations.

The original oracle-head/global diagnostic improves MPJPE 20.56 → 13.59 cm and PA-MPJPE 9.98 → 7.74 cm. This uses privileged GT global information and belongs outside the fair-input model comparison. It supports investigating global/head placement, while leaving the source of remaining reconstruction error unresolved.

**The next useful comparisons need newer weights and matched camera controls.**

- Re-evaluate hidden-state final 200k and the latest saved trajectory 180k on this same 71-window protocol. Their newer shared weights do not replace the published 150k/130k metrics. The source page records both shared package paths and verified hashes.
- Add camera-only and camera+video evaluation of the control checkpoint to separate camera-observation benefits from the FiLM design.
- Keep conclusions about original and joint checkpoints tied to paired intervals and actual V2M task exposure. Report position, semantics and physics together; choose guidance with validation rather than selecting solely from these test scores.
- Retain the evaluated UEM RGB_recon/recon steps 341420/339832 until new inference results are available. The source records resumed training, without new evaluated scores for later checkpoints.
- The V → C → M row in the parent table is blank; no quantitative result is available there.

The original pages retain the complete tables, source/checkpoint identities, quantitative downloads and qualitative galleries: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3) and [camera FiLM](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2). This page summarizes those published evaluations; it adds no new sampling results.


**Does adding joint training help? Current evidence: no demonstrated improvement in the conditional tasks.**

The matched Body23 V2M evaluation compares the original and equal-thirds joint checkpoints at 200k on the same 71 windows, shapes, seeds and sampler. Positive joint-minus-original MPJPE is worse:

| Input / bridge scale | Original MPJPE cm | Joint MPJPE cm | Joint − original cm | Paired 95% interval cm |
| --- | --- | --- | --- | --- |
| GT reference | 0.00 | 0.00 | 0.00 | — |
| Video, scale 1 | 20.56 | 22.53 | +1.97 | [−0.06, +4.01] |
| Video, scale 2 | 16.43 | 17.46 | +1.04 | [−1.53, +3.79] |
| Video+caption, scale 1 | 16.31 | 16.75 | +0.44 | [−0.89, +2.07] |
| Video+caption, scale 2 | 14.13 | 13.98 | −0.16 | [−1.46, +1.22] |

Joint training has a worse point estimate in three settings and a tiny improvement in one. All four paired intervals include zero; this supports no clear demonstrated gain, rather than proof that the methods are equivalent. Some secondary scores improve, but not consistently: at text+video/scale 2, motion FID improves 0.140510 → 0.132200 while R@3 falls 69.79 → 65.10%. GT motion FID is 0 and R@3 is 70.83±1.80%. Source: [Video2motion comparison](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3).

The older published 71-case M2V benchmark also favors the original checkpoint on all three reported video scores. These are descriptive differences; no joint-versus-original paired interval is reported for this video comparison. Keep this benchmark separate from the newer selected qualitative guidance2 videos.

| Model / checkpoint | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ |
| --- | --- | --- | --- |
| GT reference | 100 cap | 1.0000 | 0 |
| Original M2V,200k | 18.218 | 0.5663 | 0.3087 |
| Joint-trained M2V,200k | 17.982 | 0.5590 | 0.3192 |

The M2V measurements score future RGB frames1–96. Source: [Phase 3 — bridges, joint generation and inference controls](https://app.notion.com/p/3fbd2a44f44b810597cae125e52b6d1f), video-results table. Its older motion table uses a different joint aggregate, so its 22.01/23.60 cm motion scores are not mixed into the Body23 table above.

**Interpretation:** joint training explicitly trains simultaneous video+motion generation, but the available reports do not establish better video-motion coherence, plausibility or diversity. Paired GT reconstruction scores on the less-constrained joint task cannot answer that question. The newer [motion2video](https://app.notion.com/p/40bd2a44f44b80c5a349ca6df079d9be) page contains selected original/joint/FiLM comparisons, without a new aggregate original-versus-joint metric table.

The training comparison changes more than the presence of a joint loss. Original draws V2M/M2V 50%/50%; equal-thirds joint draws each standalone task 33.3% and the joint task 33.3%. Direct clean-conditioned V2M/M2V examples therefore fall by roughly one-third at equal 200k updates. Joint examples still contribute both full specialist losses, so it would also be incorrect to call them unsupervised or count only standalone examples as total supervision. Task allocation and gradient weighting can contribute to the outcome; that explanation is plausible, not proven.

A clean next ablation would hold the frozen specialists fixed and compare the task mixtures with explicit control of compute and direct task exposure, while measuring conditional accuracy and joint coherence separately. The new [40/40/20 experiment](https://app.notion.com/p/40bd2a44f44b81e09744c4dc386106db) also changes the frozen Phase1 specialist, so its future outcome cannot by itself isolate the effect of joint-task probability. No new GPU tests or inference were run for this assessment.

