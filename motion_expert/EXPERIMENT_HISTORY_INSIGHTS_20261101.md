# Early motion / Phase 2 experiments: findings beyond prediction and physical losses

Reviewed 2026-11-01 from Git history, preserved experiment reports and current
decode/conditioning code. This is a historical review, not a new evaluation.
Historical metrics below are transcribed from the cited reports; old remote
checkpoints and result JSONs were not rerun or independently re-scored today.

## What actually preceded the current Phase 2

| Record | Experiment | Interpretation |
|---|---|---|
| Initial commit `f63d121`, June 25 | Root `train_motion_ft.py`: Kimodo369, BONES uniform skeleton, frozen Nano reasoner, motion heads plus generator LoRA/full generator | Different architecture/data/optimization from the later standalone expert; not a representation-only control |
| Initial commit, `motion_expert/README.md` | Nymeria UniEgo283 standalone expert v1/v2/v3; frozen Nano final hidden states, motion self-attention and separate cross-attention | Already contains several bundled fixes; Git cannot individually date or isolate all v1/v2 changes |
| `6ca6f76`, June 29 | BONES proportional UniEgo283 POC with pooled LLM2Vec in-context text, no Cosmos reasoner/generator | Demonstrates a viable motion-native model; changes dataset and conditioner as well as interface |
| `890eec6`, July 2 | BONES POC direct flow-velocity implementation with decoded clean-motion losses | A velocity branch existed; the early failed v1 does not establish that velocity prediction cannot work |
| July 11–15 records | Native-schedule BONES POC, solver and physical-loss evaluations | Provides controlled schedule and sampler evidence |
| September audit/ablations | Nano versus Edge, K/V projection and self/cross variants | Strong evidence that conditioning interface matters independently of using UniEgo |

Original phase numbering differed: initial README called motion attachment
Phase 1 and T2M Phase 2. Do not confuse that motion Phase 1 with today's
camera/video Phase 1. Commit dates are provenance bounds, not exact launch dates.

## 1. The original Kimodo reconstruction failure was a conversion bug

`external/nymeria_kimodo_pipeline/NYMERIA_PROPORTIONAL_PIPELINE.md:17`
identifies incompatible SOMA joint-orient/local-rotation conventions and a
Z-up/Y-up mismatch in the original Nymeria converter. Recorded head-above-feet:

| Source | Height | Meaning |
|---|---:|---|
| SOMA source GT | 1.20 m | Standing, Z-up |
| BONES proportional through Kimodo FK | 1.24 m | Standing, Y-up |
| Old Nymeria conversion through Kimodo FK | 0.13 m | Crumpled |
| Corrected Nymeria example | 1.22 m | Matches source geometry |

The corrected converter uses rest-to-posed world rotations, the proper up-axis
rotation and Kimodo parent-local rotations. Typical source-joint reconstruction
error is recorded as 2–3e-6 m. A standing-height heuristic also wrongly rejected
16 valid bent-over/floor sequences; activity-independent geometry validation
passed all 732. This is not evidence that Kimodo369 intrinsically reconstructs
poorly. Root `motion_decode.py` separately mirrors the uniform/canonical FK
path; that contract must not be silently reused for proportional actors.

## 2. Grounding and the statistics population were substantive fixes

The early UniEgo Nymeria POC v1/v2 used raw SLAM-world height. The v3 report
records foot-minimum mean -2.3 m, std 3.6 m, range -12.8 to 1.5 m. Atomic-window
floor subtraction changed mean/std to approximately -0.10/0.22 m; 1,088
ambiguous-floor windows were excluded and grounded training statistics were
recomputed (maximum channel std about 6.5 to 1).

This removes a nuisance world-height distribution; it is not a contact-loss
effect. Grounding uses annotated floor offsets, not forcing the lowest
predicted foot onto zero. Sitting, stairs and airborne motion require their
actual floor context. Round-trip geometry alone does not certify source quality:
the later July source audit found upstream jumps and head/camera mismatch even
in correctly converted, floor-filtered windows.

## 3. Timestep conditioning changed with the successful v2

The historical POC replaced a one-time additive time input with per-block
AdaLN-zero modulation. Recorded root-step magnitude improved from 0.37 m to
0.015 m (GT 0.015 m), but x0, decoded losses and caching changed together.
The documentation calls stronger timestep conditioning necessary; the evidence
supports it as a plausible contributor, not an isolated causal conclusion or
proof that every additive-time architecture fails. The later Nano joint expert
is a different architecture with additive time conditioning.

## 4. Noise schedule materially changed semantics with the losses fixed

July 13 POC comparison: same architecture, proportional UniEgo, shape/text
conditioning, x0 and 1/1/5 losses; native shifted-lognormal training plus its
inference schedule versus the legacy schedule. Both checkpoints 200k, 911/917
content-overview cases (six nonfinite-GT exclusions), C45 step5k, 100 sampling
steps, CFG2, identical per-case noise, no postprocessing.

| Metric | Legacy | Native shift3 | GT |
|---|---:|---:|---:|
| Protocol R@3, % | 54.34 | 69.15 | 89.13 |
| Plain R@3, % | 42.81 | 58.84 | 80.90 |
| FID | 0.08028 | 0.05255 | Not tabulated in historical report |
| Predicted-contact skating, cm/s | 10.23 | 13.85 | 1.90 |

The schedule package improved semantics substantially but did not improve every
physical metric. This does not isolate training distribution from inference
ladder because both changed. Source: `BONES_SEED_POC.md:285–387`.
Runs: `bs_incontext_v1` and
`bs_native_x0_logitnormal_shift3_w1_1_5_200k`, step200000; original report:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_vs_legacy.json`.

Same-native-checkpoint solver comparisons gave protocol R@3 69.15 (Euler100),
69.59 (Heun50, 99 evaluations), 69.81 (UniPC35); skating 13.85, 13.73, 13.54
cm/s. GT R@3 89.13, skating 1.90. Solver choice improved efficiency markedly,
but was not a cure for the physics gap. These numbers use content overview,
not the later BONES repetition overview benchmark.

## 5. The earliest CFG branch was not trained as an unconditional branch

`git show f63d121:SAMPLING_NOTES.md` explicitly says the original Kimodo369
Cosmos run had no caption dropout despite sampling CFG with an empty prompt
(default 2.5). The later standalone POC introduced caption dropout. Thus early
guided-generation failures are confounded with an unsupported null condition,
not just representation. No controlled historical CFG-dropout quality delta
was found. Subsequent Nano/Edge shared BONES loading also had two dropout
opportunities (approximately 19% at nominal 10%); provenance matters.

## 6. Conditioning interface can matter much more than representation labels

The September matched 64-case Nano/old-Edge probe used the same BONES normalized
GT, source statistics, shape and noise. At pure noise, raw correct/shuffled
caption single-denoiser feature MSE was 0.6848815/1.4592065 for Nano versus
1.0948293/1.1700791 for Edge; GT self-MSE zero. Correct captions reduced error
relative to shuffled captions by 53.06% versus 6.43%. This is not final sampled
quality and equal steps do not mean equal training exposure.

The September 18 evaluation snapshot supplies an independent sampled benchmark:
2,374 BONES repetition/overview cases, UniPC35, CFG2, shift3, independent C45
normalization. These rows are all mixed-data B128 Edge at 200k except Nano:

| Model | R@3 % | FID | Contact skate cm/s | Penetration cm |
|---|---:|---:|---:|---:|
| GT | 93.93 | 0 | 2.13 | 0.0352 |
| Nano contact reference, 200k | 82.31 | 0.00930 | 4.26 | 0.0335 |
| Edge parity baseline B128, 200k | 15.42 | 0.07580 | 4.41 | 0.1355 |
| Edge reasoner-K FP32 B128, 200k | 18.20 | 0.07049 | 4.55 | 0.0604 |
| Edge Projection A B128, 200k | 47.77 | 0.04050 | 4.47 | 0.0620 |

Run identities and completed table: `../YONSEI_PHASE2_EVAL_20260918_2045.md`.
Projection A learns a reasoner K/V interface; it retains UniEgo. Its improvement
supports trainable adaptation as useful. Additional parameters are part of
that intervention. It does not prove frozen Edge representations are intrinsically
bad, or that matching Nano key RMS/attention mass would suffice. Tiny skating
differences coexist with very large retrieval differences.

Later caption-only T2M packing is another explicitly isolated interface change
in `../motion_expert_t2m_edge/CAPTION_ONLY_ABLATION.md`: remove structural
BOS/EOS/generation prefix, retain native TI2M template. That file alone is a
configuration record, not a scored causal result; no later caption-only metric
is substituted into the September table here.

## 7. UniEgo solves one geometry problem but retains a trajectory-learning problem

In `decode_uniego_torch.py`, canonical transforms are composed across time,
then applied to direct local joint positions. This bypasses pose reconstruction
through the body FK tree, but small canonical-delta errors can accumulate into
global drift. Reversible GT decoding does not establish easy unconditional
generation. Local joint rotation channels do not determine the decoded joint
positions in this decoder; orientation fidelity therefore cannot be certified
by joint-position metrics alone. These are code-derived mechanisms, not isolated
experimental causes of a specific failure.

Floor-aware preprocessing likewise does not constrain every generated transform
to remain yaw-only/zero-height. Local foot-Y supervision is not identical to
world-foot-height supervision for unconstrained predictions. This shared issue
does not explain an Edge-only regression by itself.

Camera/head recanonicalization was separately audited over 13,553,128 frames:
maximum position change 0.0468 mm, foot-height change 9.43e-12 m. It did not
physically lift the stored corpus. It changed orientation features and potentially
learning difficulty; the old head-to-camera lever calibration was less accurate.
See `../EDGE_NANO_PHASE2_AUDIT_20260914.md:449`.

## 8. Shape and rendering checks prevented misleading diagnoses

Low bone-length MAE alone did not demonstrate use of shape conditioning. Later
same-caption/same-noise natural-shape swaps tested causal responsiveness; the
200k full-contact POC content-overview result reports shape correlation 0.965
and counterfactual slope 0.899 (911 cases), beyond merely generating an average
skeleton. These are separate metrics from text semantics.

Early Y-up/Z-up rendering errors also made motions look wrong without changing
training. The November 1 A2 sample audit found correct GT/image/normalization
alignment and left-then-right turning in both GT/prediction, but incorrect initial
body/head relation and under-turning in the prediction. It also found a truncated
display caption; the model had received the full text. That single case is not a
dataset-wide left/right conclusion. See `../motion_expert_t2m_edge/LEFT_RIGHT_AUDIT_20261101.md`.

## Conclusions and provenance limits

The historical sequence is not “Kimodo is bad, UniEgo is also bad.” It combines
a real source-conversion bug, nuisance world height, conditioning/optimization
changes and eventual successful UniEgo generators. The cleanest additional
findings concern conversion correctness, grounding, schedule choice and useful
text conditioning. AdaLN and CFG training are credible additional contributors
with weaker isolated historical evidence.

Do not conflate diffusion velocity `epsilon-x0`, physical joint velocity and
an x0 model's induced velocity-space objective. The July2 implementation prevents
categorically dismissing direct velocity, but no matched successful checkpoint
was identified in this review. Likewise the old README's universal claim of
an `E[epsilon^2]=1` flow-matching memorization floor is not valid in general;
do not use it to dismiss a finite-example overfit/debugging test.

For a present Kimodo369/UniEgo283 comparison, hold calibrated physical joints,
windows, caption/shape inputs, loss definitions, optimizer/exposure, conditioning
and sampler fixed, with representation-specific training stats. Existing current
A0/A2 work is the appropriate comparison to inspect; this review launches no
new training and changes no running jobs or model code.
