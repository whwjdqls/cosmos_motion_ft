# Shared Agent Context

This repository is a cluster-oriented research checkout for adapting NVIDIA Cosmos-3 Nano into human-motion and egocentric world-model variants. It is not a standalone Python package. Real runs usually require `/home/jungbin_cho/cosmos-framework`, the `cosmos` conda env, Slurm GPU nodes, and `/weka/jungbin/...` datasets/checkpoints.

Primary current work areas: `motion_expert_joint_attention/` and the isolated native-Cosmos Phase 1 path in `native_phase_training/`.

This file is the shared source of context for both Claude and Codex. Agent-specific bootstrap files such as `AGENTS.md` and `CLAUDE.md` intentionally point here and should stay small.

## Documentation Policy

`AGENTS_ALL.md` is the only canonical high-level context document for this repo. Keep detailed project context here, then verify behavior against code before editing.

Root `README.md`, `AGENTS.md`, `CLAUDE.md`, `motion_expert_joint_attention/README.md`, and `motion_expert_joint_attention/DESIGN_7TASK.md` are now short pointers/stubs. Do not re-expand them into parallel sources of truth.

Stale large docs were removed on 2026-07-06: root `COSMOS.md`, `DESIGN.md`, `SAMPLING_NOTES.md`, and `PLAN_ego_video_camera_world_model.md`. Their still-relevant content was consolidated here. `nymeria_world/` remains on disk but is older work and not the source of truth for current joint-attention training.

## High-Level Thesis

The project extends Cosmos-3 Nano into an omnimodal world model centered on human motion and egocentric camera action. The recurring pattern is: keep Cosmos's reasoner/understanding pathway frozen, treat text as conditioning rather than a target, and adapt the generation pathway or add motion-specific generation pathways for continuous per-frame outputs.

Text is always a condition, not a captioning target. Do not train `video -> text`, `motion -> text`, or VLM captioning tasks unless explicitly requested. For Cosmos-based motion runs, use raw text strings through the Cosmos processor/reasoner, not precomputed LLM2Vec embeddings, except in the separate BONES-only `motion_expert/bs_*` POC that intentionally uses cached LLM2Vec.

Motion can be both generated output and a control signal. Pure `motion -> video` without scene/image context is considered ill-posed; use `motion + image/text -> video`.

## Experiments

There are four experiment families historically. Current active work is `motion_expert_joint_attention/`; the others are legacy/reference unless the user explicitly asks for them. Do not mix their data contracts, trainers, or representations.

### Root Text-to-Motion Finetune

Files: `train_motion_ft.py`, `sample_motion.py`, `motion_decode.py`.

Goal: text to 369-d Kimodo/BONES-SEED motion. Cosmos reasoner is frozen. The model trains motion heads plus LoRA or full `_moe_gen`. The root run uses raw text through Cosmos's own processor, not LLM2Vec.

Motion representation: 369-d Kimodo/BONES-SEED at 20 fps:

`[0:3] smooth_root_pos`, `[3:5] heading(cos,sin)`, `[5:95] local joint positions`, `[95:275] global rot6d`, `[275:365] velocities`, `[365:369] foot contacts`.

Stats come from `/weka/jungbin/seed/stats/soma_uniform_motions_20fps/`. Decode and losses live in `motion_decode.py`, a pure-torch bit-exact port of Kimodo decode/FK. Known gaps: heading augmentation is off; fps fixed at 20.

Export uses the `kimodo` env; training/sampling uses the `cosmos` env. Full BONES export lives at `/weka/jungbin/seed/cosmos_text_motion_full`.

Sampling convention: training path is `x_t = (1-t)x0 + t*noise`, target `v = noise - x0`, where `t=1` is noise and `t=0` is clean. Euler sampling integrates from `t=1` to `t=0` with `x <- x - dt * v_hat`. Sample-time packing must pack the current `x_t` and current `t`; do not re-noise inside the sampler. CFG uses an empty caption as approximate unconditional.

Healthy root checkpoint overlay in `sample_motion.load_model` previously loaded 406 trainable tensors: 397 `_moe_gen`, 5 motion heads, and 4 `time_embedder`.

### `motion_expert/` POC

Files: `motion_expert/README.md`, `motion_expert/BONES_SEED_POC.md`, and `motion_expert/*.py`.

This is an older standalone MotionExpert text-to-motion POC. It does not instantiate the Cosmos generator.

Main Nymeria POC: text goes through a frozen Cosmos reasoner once, hidden states `H_R` are cached, and a small trainable MotionExpert transformer one-way cross-attends to `H_R`. Information flow is `text -> frozen reasoner -> cached H_R`, plus `neutral_joints -> shape token`, plus noised motion tokens. MotionExpert predicts x0. There is no generator token or generator attention edge.

Representation: 283-d UniEgo/SOMA-30:

`[0:270]` local SE(3) per joint as 30 x `[rot6d + trans]`, `[270:279]` canonical delta, `[279:283]` foot contacts. Decode uses `decode_uniego_torch.py`, bit-exact to Kimodo, no FK. World convention is Y-up, +Z-forward; visualization remaps to matplotlib Z-up and negates X.

Important POC lessons: v1 velocity + feature MSE crumpled/spun; v2 fixed quality with x0 prediction, cached `H_R`, decoded losses, and AdaLN-zero; v3 added floor grounding and viz coordinate fix. Loss is feature MSE + centroid-relative decoded-joint pose + decoded joint-velocity smoothness. Absolute joint loss was unstable and drift-dominated.

BONES-only `bs_*` POC: uses BONES-SEED proportional 283-d UniEgo, cached LLM2Vec pooled text embeddings, an in-context text token plus shape token, no Cosmos and no reasoner. Runs in `kimodo`, not `cosmos`. It uses `SOMABonesSeedDataset` indexing by subclassing and overriding motion I/O. Text is dropped to the cached `""` embedding for CFG; shape is never dropped. This POC is intentionally different from the Cosmos raw-text runs.

**Machine boundary (confirmed 2026-07-14): `motion_expert/bs_*` is A100-machine-only.** Do not run, debug, or validate these BONES scripts on the H200 `a3ultra` machines. They work in the intended A100 environment, whose data and run roots use `/mnt/shared/...`. On the H200 machine, the `kimodo` environment does not provide `cosmos_framework` or `diffusers`, and the local Cosmos checkout exposes UniPC under `cosmos_framework.model.vfm` while the pulled BONES UniPC code expects the older `cosmos_framework.model.generator` namespace. Consequently the BONES official-UniPC paths fail there. This restriction does not apply to `motion_expert_joint_attention/`, native Phase 1, or H200 Phase-3 job `2870`.

Native-schedule BONES Phase-2 POC: `motion_expert/bs_native_flow.py`, `bs_train.py`, and `bs_sample.py` keep the BONES model/data/x0/loss recipe fixed while replacing the unshifted schedule with Cosmos's action-like shifted logit-normal training sigma and shifted 1000-step inference ladder. Default motion shift is 3. Training-time sigma and diagnostic Euler/Heun integration are local formulas pinned to NVIDIA Cosmos Framework commit `3d9c0878fd0dde76eac98161aed0493d85a036fd`; the later official-UniPC path imports the audited Cosmos scheduler supplied by the A100 `kimodo` environment. The BONES model itself still has no Cosmos reasoner or generator. Launch through `motion_expert/sbatch_bs_native_phase2.sh` on the A100 machine, never on an H200 or login CPU.

Native-schedule POC launch status on 2026-07-11: Slurm smoke job `10386` passed five real-data optimizer steps on one A100 with finite losses/gradients, shifted sigma mean about `0.70-0.74`, and 10.3 GB peak memory. Production job `10387` runs `bs_native_x0_logitnormal_shift3_w1_1_5_200k` for 200k steps, batch 128, loss weights `1/1/5`, reusing `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_incontext_v1/bs_train_index.json`. Its run directory is `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_1_5_200k` and Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-bsnatp2-10387.out`.

`motion_expert/bs_tmr_eval.py` is the in-memory generation evaluator for comparing native- and
legacy-schedule BONES checkpoints with the shape-aware C45 TMR. It follows Kimodo's benchmark
metric contract without writing generated motions or embeddings: generate normalized proportional
283-D UniEgo at 20 FPS, unnormalize and decode to SOMA-30 joints, resample joints to 30 FPS and
featurize with C45's official-stat `TMRMotionRep`, and pass the same centered proportional
`neutral_joints` to both the generator and C45 shape encoder. It reports protocol and plain
R-precision, TMR FID, benchmark foot skate/contact metrics on native 20-FPS joints, and bone-length
MAE versus the conditioned skeleton. Generated four-channel foot contacts are thresholded at 0.5.
Generation uses the benchmark LLM2Vec cache because the training cache intentionally omits almost
all held-out benchmark captions; both caches store outputs of the same frozen text representation.
The full content/overview comparison launcher is `motion_expert/sbatch_bs_tmr_eval.sh`; its default
evaluator is C45 step 5000, selected as C45's best full-overview checkpoint.

Full in-memory C45 comparison completed as Slurm job `10570` on 2026-07-13. It discovered 917
content/overview cases and scored 911; the six exclusions were proportional GT UniEgo windows with
non-finite values. Both step-200k generators used 100 steps, CFG 2, identical case-seeded noise,
and C45 step 5000. Native versus legacy: protocol R@3 `69.15` vs `54.34`, plain R@3 `58.84` vs
`42.81`, and FID gen-GT `0.05255` vs `0.08028`. Physical quality favored legacy: predicted-contact
skate `13.85` vs `10.23` cm/s, contact consistency `0.818` vs `0.879`, skate ratio `0.260` vs
`0.212`, and conditioned-skeleton bone MAE `0.360` vs `0.269` cm. GT references were `1.90` cm/s,
`1.000`, `0.102`, and `0.181` cm respectively. Aggregate result:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_vs_legacy.json`.

Native Heun sampler follow-up: `bs_native_flow.sample_x0_heun` implements explicit trapezoidal
RK2 on the same shifted native ladder, with an Euler final interval because its endpoint is sigma
zero. `bs_sample.py --native_solver heun` exposes it for normal sampling, and `bs_tmr_eval.py
--native-solver heun` exposes it for in-memory evaluation. A 50-step run has 99 denoiser
evaluations, or 198 model forwards with two-branch CFG, versus Euler-100's 100/200. Five focused
native-flow tests pass and GPU smoke job `10606` passed. Full job `10607` scored all 911 cases:
Heun-50 versus Euler-100 protocol R@3 `69.59` vs `69.15`, plain R@3 `57.96` vs `58.84`, FID
gen-GT `0.05193` vs `0.05255`, predicted-contact skate `13.73` vs `13.85` cm/s, contact consistency
`0.820` vs `0.818`, skate ratio `0.258` vs `0.260`, and bone MAE `0.352` vs `0.360` cm. This is
effectively a tie: Heun gives marginal FID/physical improvements but does not fix foot skating and
slightly lowers plain R@3. Result:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_heun50.json`.

Official Cosmos-3 UniPC follow-up: `bs_native_flow.sample_x0_unipc` does not reimplement UniPC. It
imports NVIDIA `cosmos-framework==1.2.2`'s `FlowUniPCMultistepScheduler` from the `kimodo` env and
calls its `step` method directly; installed scheduler SHA-256 is
`03aef1959f273b704ca4954f69b2a34df0fdd412f6acc8ab91625eeed78cf4fe`, byte-identical to audited
Cosmos Framework commit `3d9c0878fd0dde76eac98161aed0493d85a036fd`. The only local adapter converts
guided x0 to the scheduler's required flow velocity as `(x - x0_cfg) / sigma`. Preserve the real
wrapper's defaults: 35 steps, order 2, `bh2`, `flow_prediction`, `predict_x0=True`, lower-order
final, shift 3, and no dynamic shifting. The official code shifts its constructor sigma range and
then shifts the inference ladder again in `set_timesteps`; do not replace this with the earlier
single-shift Euler/Heun diagnostic ladder.

UniPC GPU smoke job `10608` passed. Full job `10609` evaluated all 911 cases with C45 step 5000,
CFG 2, and 35 denoiser evaluations/70 CFG model calls: protocol R@3 `69.81`, plain R@3 `59.06`,
FID gen-GT `0.05143`, predicted-contact skate `13.54` cm/s, consistency `0.821`, skate ratio
`0.257`, and bone MAE `0.352` cm. Relative to Euler-100, this is +0.66 protocol R@3, +0.22 plain
R@3, 2.13% lower FID, and slightly better physical metrics with 65% fewer denoiser evaluations.
Result: `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_unipc35.json`.

Foot-skating loss follow-up launched 2026-07-13. `motion_expert/bs_losses.py` contains tested,
configurable contact-aware losses; `bs_train.py` records/logs them, and
`sbatch_bs_native_phase2.sh` exposes all weights through `BS_NATIVE_W_*`. The SOMA-30 contact order
is `[LeftFoot, LeftToeBase, RightFoot, RightToeBase]`, indices `[24,25,28,29]`. Contact BCE is
class-balanced and centered on raw contact threshold 0.5; horizontal physical foot speed and raw
foot-height reconstruction are masked with GT contacts so the model cannot avoid them by predicting
no contact. Raw Y channels are used for height because they exactly equal decoded Y under UniEgo's
yaw-only canonical frame and avoid unstable cumulative-decoder gradients. Four focused loss tests
pass; final contact-aware GPU smoke `10616` passed with gradient norms `2.7-5.5`.

Two 200k controlled runs train from scratch. Job `10622` is
`bs_native_x0_logitnormal_shift3_w1_10_100_inline10k_200k`, changing existing
feature/joint/smooth weights to `1/10/100`. Job `10623` is
`bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_inline10k_200k`, retaining `1/1/5`
and adding contact/foot-velocity/foot-height weights `0.05/1/10` with contact logit scale 2.
Separate dependent evaluations `10619/10620` were canceled. `bs_train.py` now initializes C45 and
all 911 GT references once, then evaluates the live model in-process every 10k checkpoint and at
the final step. Results are `<run>/inline_eval/step_XXXXXX.json` plus `history.json`; no motions or
embeddings are saved. CPU/CUDA RNG states and training mode are restored after each callback.
Integration smoke `10621` passed the train/save/live-eval/write path. Native checkpoint
visualizations also use official UniPC-35. Full rationale, paths, smoke ablations, and evaluation
outputs are in `motion_expert/BONES_SEED_POC.md` section 18.

Early 2026-07-14 results motivated two paired hybrid follow-ups. Job `10644` keeps the stronger
`1/10/100` reconstruction weights and adds foot velocity/height weights `0.5/5` without contact
BCE. Job `10645` is identical but adds contact BCE weight `0.025` at logit scale 2. Their run
directories are respectively `bs_native_x0_logitnormal_shift3_w1_10_100_foot_v0p5_h5_inline10k_200k`
and `bs_native_x0_logitnormal_shift3_w1_10_100_softcontact_c0p025_v0p5_h5_s2_inline10k_200k` under
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/`; both use the same inline 10k C45 evaluation.

Shape-awareness evaluation was strengthened on 2026-07-14. `bs_tmr_eval.py` now supplements
conditioned-skeleton bone MAE with actor-centered bone correlation/slope/variance and an optional
paired counterfactual pass. The counterfactual uses identical text, duration, and initial noise but
conditions on the most different natural held-out skeleton, then measures generated/requested bone
delta alignment, target advantage, and retrieval retention. New inline evaluators default to
`farthest`; processes already running before this edit retain the old imported callback. Initial
smoke `10650` failed before Python because Slurm used `/bin/sh` for a `pipefail` wrapper, and blocked
job `10651` was canceled. Corrected explicit-Bash smoke `10679` gates final-200k backfill `10680`,
which also waits for training jobs `10644/10645`; output is
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_shape_counterfactual_final200_ablation_runs.json`.
Smoke `10679` completed successfully with all five metric tests and an eight-case paired
UniPC/C45 integration pass. Foot-only hybrid job `10644`, soft-contact job `10645`, and final shape
backfill job `10680` all completed. The backfill report is
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_shape_counterfactual_final200_ablation_runs.json`.

Full-contact continuation/all-benchmark follow-up launched 2026-07-15. `bs_train.py` supports a
strict model-only warm start with global checkpoint numbering and a restart-local LR schedule. The
source step-200k checkpoint has no optimizer or RNG state, so job `10747` is explicitly a fresh
AdamW warm start rather than an exact resume: unchanged full-contact recipe, steps `200k->500k`,
LR `5e-5`, 1k local warmup, cosine decay, and seed `200000`. It keeps 10k in-process 911-case C45
overview evaluation. Run:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_continue200to500k_lr5e-5_seed200000`.
`sbatch_bs_all_t2m_eval.sh` evaluates all six applicable content/repetition overview/single/multi
text-to-motion suites using UniPC-35, physical metrics, and paired shape intervention, then
`bs_all_benchmark_summary.py` writes one structured report. Step-200k job `10746` writes under
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/full_contact_200k_all_text2motion_unipc35_shape_cf`;
dependent final-500k job `10748` writes the corresponding `full_contact_500k...` directory.
Constraint-conditioned suites are marked not applicable because this generator has no constraint
input. Continuation/evaluation/unit smokes `10742/10743/10744` all passed. Initial job `10746`
revealed that benchmark readers looked up raw text in a cache keyed by Kimodo-sanitized prompts;
this was fixed in `bs_tmr_eval.py`, `st_inline_eval.py`, and `official_tmr_eval.py`. Audit `10751`
recovered all 36 valid timeline prompts, and correction job `10752` rebuilt the report with
9,124/9,162 discovered cases; only 32 non-finite GT motions and six sub-10-frame requests remain.
Per-suite protocol/plain R@3 is `70.58/59.60`, `63.93/54.28`, `70.06/62.55`, `77.89/65.33`,
`64.74/52.59`, and `77.30/68.89` in the order above. The case-weighted suite means are
protocol/plain R@3 `71.56/60.80`, FID `0.02195`, contact skate `3.44 cm/s`, bone MAE `0.382 cm`,
shape correlation `0.969`, and counterfactual response slope `0.906`; these are means of
suite-level metrics, not one merged retrieval computation.

Generator-normalization ablation launched 2026-07-15 as job `10860`. It reproduces the
full-contact 200k recipe, including the historical global DataLoader RNG and pre-increment periodic
checkpoint indexing, but replaces the proportional-data stats with
`motion_expert/stats/uniego283_{mean,std}.npy` (tag `nymeria_grounded_uniego283`, SHA prefixes
`bd1d6bdc`/`ee069e3a`). Run:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_nymeria_grounded_stats_inline10k_200k`.
Control smoke `10859` with the original proportional stats reproduced the archived baseline's first
update exactly to printed precision; alternate-stat smoke `10857` passed five finite updates.
`bs_normalization.py` now pins paths, hashes, shape, dtype, and a tag in configs/checkpoints.
`bs_sample.py` and `bs_tmr_eval.py` resolve stats from checkpoint metadata by default and reject
silent mismatches; the evaluator resolves each generator independently, allowing valid mixed-stat
comparisons. Full rationale and job audit are in `motion_expert/BONES_SEED_POC.md` section 21.

### `nymeria_world/` Camera World Model

Files: `nymeria_world/*`.

Goal: egocentric video + text + ego-camera world model. This is older native-camera work and is not the current source of truth for joint-attention training. Human body motion and audio are out of scope there. This experiment uses Cosmos's native training/inference stack (`cosmos_framework.scripts.train` and `cosmos_framework.scripts.inference`), not `train_motion_ft.py`.

Camera action is Cosmos native `camera_pose`, domain id 2, raw dim 9 `[pos(3), rot6d(6)]`, zero-padded to 64 only for Cosmos action heads. Do not z-score camera actions. The representation is relative SE(3): `Delta T_t = T_{t-1}^{-1} T_t`, implemented through `pose_abs_to_rel(rotation_format="rot6d", pose_convention="backward_framewise")`.

Coordinate convention: source camera npz is raw Aria device frame, but the chosen training/eval convention is RGB optical plus `Rz(-90deg)`, matching the upright video. Finetuning can learn any consistent frame, but OpenCV/upright-video frame aligns best with Cosmos zero-shot predictions.

Zero-shot inverse dynamics showed a 7-17x translation scale gap. The diagnosis is action temporal-step/training-distribution mismatch, not coordinate frame, fps, de-normalization, or metric-scale error. The model emits action magnitudes similar to about 7-frame GT displacement. Finetuning pretrained camera action heads should adapt the convention.

Native modes: `forward_dynamics`, `inverse_dynamics`, and `policy`. `image2video` was tried in a 4-task mixture but is dropped for multi-GPU training because it has no action, so action-head parameters can get no gradients on all-i2v steps and distributed collectives desync.

Training modes:

- LoRA default: reasoner frozen; LoRA on generator attention plus pretrained camera action heads kept trainable. Action heads are not re-initialized in this camera experiment.
- Full-gen: reasoner frozen; full generation pathway trainable. Requires FSDP sharding (`dp_shard=8`) because replicated optimizer state OOMs.

Checkpoints are LoRA/action-head deltas. Native inference cannot overlay LoRA directly, so evaluation is merge-then-sample: `export_merge_lora.py` -> `prep_test_eval.py` -> `run_infer_merged.sh` or `sbatch_infer_3tasks.sh` -> `viz_eval_samples.py`.

Inference must run from `cosmos-framework`, use `--no-guardrails`, use `.jsonl` inputs, set local Wan VAE override when needed, and set `lora_enabled=false` for merged checkpoints.

### `native_phase_training/` Official-Compatible Phase 1

Read `native_phase_training/README.md` before editing this path. That README is the detailed runbook for the current official-compatible Phase 1 camera/video generator run.

Purpose: train a new Nymeria camera/video generator LoRA that keeps the training and sampling contract close to NVIDIA's native Cosmos-3 Nano action/video setup. This path exists because the older joint-attention Phase 1 and custom sampler showed poorer visual quality than base Cosmos with the official sampler. The new path uses saved Wan-VAE latents for training speed, but it preserves native Cosmos sequence construction, rectified-flow noising, loss masks, losses, action heads, checkpoint format, and official inference compatibility. Batch cardinality is an optimization choice and does not affect sampler compatibility.

This directory is intentionally isolated from `motion_expert_joint_attention/`. It has no motion expert, no 3-way joint attention, and no modality bridge. It should produce a frozen video/camera specialist that future bridge or motion experiments can reuse.

Main files:

- `native_phase_training/latent_omni_model.py`: `LatentOmniMoTModel(OmniMoTModel)`. If `video_latents` is present in the training batch, it uses those cached clean VAE latents instead of encoding pixels. Without `video_latents`, it falls back to native `OmniMoTModel`, which is why official inference still works.
- `native_phase_training/latent_nymeria_dataset.py`: cached-latent Nymeria camera dataset. It emits native action-SFT fields plus dummy video metadata and real `video_latents`. Forward/policy prompts use official action JSON, inverse text is exactly empty, and image-to-video uses the official generic duration/resolution prose.
- `native_phase_training/latent_nymeria_dataset.py` also defines `CyclingDataLoader`, which is required for long runs. Native `IterativeJointDataLoader` assumes child streams are infinite; finite map-style `DataLoader` streams can otherwise exhaust and make the trainer silently spin without advancing `global_id`.
- The same file defines `LatentAwareIterativeJointDataLoader`. The stock counter sees only the `[3,97,1,1]` dummy video; the local override adds the real `25*8*8=1600` cached-latent patch tokens. Production defaults to exactly four clips per GPU (`NATIVEP1_CLIPS_PER_GPU=4`); setting that environment variable to `0` restores the audited 45,056-token-budget packing mode.
- `native_phase_training/experiment.py`: registers Hydra experiment `world_camera_nymeria_latent_nano`, resolves local tokenizer/VAE paths, sets `resolution=256`, and builds the four native streams.
- `native_phase_training/AUDIT.md`: records the finite-loader, prompt, cached-latent packing, and evaluation-contract audits and fixes.
- `native_phase_training/prep_test_eval.py`: held-out official-inference input builder for forward/inverse/policy/image-to-video, pinned to T97/action96, 20 FPS, 256, and shift 3.0. Use this instead of the historical 480/shift-10 helper.
- `native_phase_training/visualize_checkpoint.py`: validates four mode-specific official outputs, creates GT/generated videos for forward/policy/image-to-video, camera plots for inverse/policy, and a manifest.
- `native_phase_training/evaluate_forward_dreamsim.py`: optional canonical-full71 forward evaluator using official DreamSim 0.2.1 over all 96 generated suffix frames, with early/middle/late summaries.
- `native_phase_training/evaluate_forward_cdfvd.py`: optional canonical-full71 set evaluator using CVPR 2024 content-debiased FVD with VideoMAE-v2-SSv2. This is deliberately not legacy TensorFlow/I3D FVD; it uses all 96 generated suffix frames for the full score and all 32 frames per horizon. See `native_phase_training/FORWARD_VIDEO_METRICS.md` for the pinned revision and exact contract.
- `native_phase_training/checkpoint_eval_callback.py` plus `sbatch_checkpoint_eval.sh`: production rank 0 submits an isolated official four-mode EMA/UniPC evaluation after every successful checkpoint save. The stock in-training generation callback stays disabled.
- `native_phase_training/run_contract.py`: persists architecture-critical Phase-1 settings in each run and resolves/validates them before evaluation. Manual C/D/E evaluation must never rely on default adaptation environment values.
- `native_phase_training/world_camera_nymeria_latent.toml`: production TOML.
- `native_phase_training/run_latent_train.py`: training entrypoint. It assigns TensorBoard to `$TB_LOG_DIR` if set, otherwise `${job.path_local}/tensorboard`, avoiding the old shared fallback directory.
- `native_phase_training/inference_config.py`: official inference shim; use it with `--config-file native_phase_training/inference_config.py`.
- `native_phase_training/sbatch_phase1_native_camera.sh`: production Slurm launcher.

Default data contract:

- latent root: `/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97`;
- `NYMERIA_NUM_FRAMES=97`, `NYMERIA_RESOLUTION=256`, fps 20;
- latent `.npz` keys: `latents [48,25,16,16]`, `camera_action [96,9]`, `image_size`;
- camera action is raw Cosmos `camera_pose`, domain id 2, raw dim 9, padded to 64 only for the native action projection. Do not z-score it.

Task mix:

- `forward_dynamics`: 40%;
- `inverse_dynamics`: 25%;
- `policy`: 20%;
- `image2video`: 15% video regularizer.

Prompt contract:

- forward dynamics and policy: `ActionPromptJsonFormatter` JSON, matching official action inference;
- inverse dynamics: exactly empty prompt;
- image-to-video: plain caption plus official generic duration/FPS and resolution prose, without action-viewpoint prose;
- 10% CFG dropout is applied after formatting and drops the whole prompt to empty-string conditioning.

Native RF/loss settings are intentionally inherited from Cosmos Nano:

- video train-time distribution: `waver`;
- image train-time distribution: `logitnormal`;
- `independent_action_schedule=false`, so action reuses the sample's vision/video sigma; configured `train_time_action_distribution=logitnormal` is dormant;
- train-time loss weight: `uniform`;
- action loss weight: `10.0`;
- shift config: `{256:3, 480:5, 720:10}`.

Trainable default:

- generator LoRA on `q/k/v/o_proj_moe_gen`, rank 16, alpha 32;
- action heads `action2llm`, `llm2action`, and `action_modality_embed`;
- frozen reasoner and frozen base generator weights.
- native PowerEMA remains enabled. The optimizer updates regular LoRA/action parameters; after each step Cosmos updates an FP32 EMA model, and official checkpoint evaluation uses `--use-ema-weights`. Because the generic implementation duplicates and traverses the full frozen model, it is memory-inefficient for LoRA-only training, but it is retained for this official-compatible baseline.

Vision-head recommendation as of 2026-07-10: keep `vae2llm` and `llm2vae` frozen for the first native baseline. The goal is to preserve base Cosmos visual quality while adapting camera control through generator LoRA and action heads. A later ablation can unfreeze `llm2vae`/`vae2llm` and possibly `time_embedder` at lower LR if the baseline underfits, but that increases the risk of perturbing the pretrained visual distribution.

Schedule/LR recommendation as of 2026-07-12: do not use the old 200k-step Phase-1 schedule or uniform `2e-4` LoRA/action-head LR. The fixed-four baseline runs 100k max steps with `NATIVEP1_LORA_LR=5e-5`, `NATIVEP1_ACTION_LR_MULT=4.0`, and checkpoints every 5k. With `f_start=f_max=0.4`, the first 500 steps are a flat plateau at `2e-5` effective LR for generator LoRA and `8e-5` for the action modules, followed by linear decay. Keep these rates for the first fixed-four baseline: they are already 10x/2.5x below the old custom run's effective LoRA/action rates, and changing batch size plus LR simultaneously would obscure the comparison. Evaluate early checkpoints and lower the LoRA rate only if visual quality drifts.

Official inference syntax gotcha: the config must be passed as the relative Python file path:

```bash
--config-file native_phase_training/inference_config.py
--experiment world_camera_nymeria_latent_nano
```

Do not pass an absolute `.py` path, and do not pass module form without `.py`.

Required local env:

```bash
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

Smoke status on 2026-07-10:

- One-step post-patch train smoke passed on `a3ultravis-a3ultranodeset-1`, GPU 1.
- Smoke checkpoint: `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_smoke_train_postpatch/checkpoints/iter_000000001`.
- Official inference loaded that exact checkpoint and successfully sampled `forward_dynamics`, `inverse_dynamics`, and `policy`.
- Each smoke output had `sample_outputs.json`, `vision.mp4`, action content shaped `96 x 9`, and 97-frame 256x256 MP4 at 20 fps.
- Official inference output dir: `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_postpatch`.

Production OOM incident:

- Slurm job `2792` (`nativep1`) failed on 2026-07-10 after reaching the first training forward pass.
- It was CUDA OOM on rank 2, not a data/checkpoint bug.
- Evidence in `/home/jungbin_cho/cosmos_motion_ft/slurm-nativep1-2792.out` lines 1552 and 1700: GPU 2 had only 623 MiB free; our process used 123.29 GiB; external PID `2951793` used 15.88 GiB.
- PID `2951793` was owned by `jmleeluck` and was running `scripts/train_latent_grpo_libero.py ... --device cuda:2` on bus `00000000:96:00.0`, which maps to GPU 2.
- The production config also had `model.config.compile.enabled=true`, while successful smokes had compile disabled. TorchInductor compile increased first-step memory pressure.

Patch after job `2792`:

- `sbatch_phase1_native_camera.sh` now requests `#SBATCH --exclusive`;
- it prints `nvidia-smi` state before launching;
- it runs a memory preflight and fails before torchrun if any GPU has less than `NATIVEP1_MIN_FREE_MIB` free, default 132000 MiB;
- it sets `model.config.compile.enabled=false` in the production override.

Replacement job after OOM:

- Slurm job `2799` (`nativep1`) was submitted on 2026-07-10.
- It was pending on resources after submission. Launcher state at submission: 100k max steps, default LoRA LR `2e-4`, compile disabled.
- This was superseded by the lower-LR split recipe: base LoRA LR `5e-5`, 4x LR multiplier for `action2llm`, `llm2action`, and `action_modality_embed`, run name `native_phase1_camera_latent_lora5e5_action4x_100k`.
- If job `2799` still exists, cancel/resubmit before treating it as the active baseline because Slurm copied the old launcher at submission time.
- Job `2800` was submitted with the lower-LR split recipe on 2026-07-10. At submission it was pending on resources.
- Job `2800` was cancelled before start after the finite-dataloader audit. Resubmit only after verifying the `CyclingDataLoader` wrapper remains active in `native_phase_training/experiment.py`.
- After the cycling fix, a forced-exhaustion smoke (`NYMERIA_MAX_SAMPLES=8`, 20 iterations) passed and saved `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_cycling_exhaust_smoke/checkpoints/iter_000000020`. Official inference loaded it and produced successful forward/inverse/policy samples under `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_cycling_exhaust_smoke`.
- Job `2801` was submitted on 2026-07-10 with the lower-LR split recipe plus `CyclingDataLoader`. At submission it was pending on priority.
- The 2026-07-11 audit found job `2801` used legacy prose action prompts and fixed 32-sample packing based on dummy pixels. It was superseded by the from-base token-budget run `native_phase1_camera_json_tokpack_lora5e5_action4x_100k`; do not resume job `2801` into either the token-budget or fixed-four recipe.
- Corrected smoke checkpoint `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/cosmos3_camera/camera_world/native_phase1_json_tokpack_smoke_20260711/checkpoints/iter_000000004` completed train/save/load. Official inference output `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/official_inference_4mode_v2_20260711` passed all four modes with separate directories, UniPC shift 3, 97-frame 256x256/20-FPS MP4s, and `[96,9]` actions for action modes. Its `viz/manifest.json` covers all four modes.
- Job `2838` started on 2026-07-12 with the token-budget configuration and was intentionally cancelled at step 4,399 on 2026-07-12 before its first 5k checkpoint; it produced no training checkpoint. Replacement job `2852` was submitted from the fixed-four launcher under `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k`, with exactly four clips per GPU and PowerEMA retained. At submission it was pending on scheduler priority, with node 3 as the estimated backfill target.
- Keep the stock `EveryNDrawSample` callback disabled for cached-latent training; it sees 1x1 dummy GT, covers only the selected stream, lacks action visualization, and uses mismatched generic sampler defaults. Production sets `NATIVEP1_AUTO_EVAL=1`: the rank-0 post-save callback submits a separate one-GPU Slurm job that runs official EMA/UniPC inference and `visualize_checkpoint.py` for all four modes. Set it to 0 for smoke/debug runs.

Final fixed-four Phase-1 status (2026-07-14): job `2852` completed 100k steps at `/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k`. The final **100k EMA** DCP at `checkpoints/iter_000100000`, not a historical joint `.pt`, is the Phase-3 generator initialization. Its full official-inference evaluation used all 71 held-out forward windows and all 71 held-out inverse windows on node 3 GPU 2; outputs are under `eval_full71_inverse_forward/iter_000100000`. All 142 samples passed, and analysis wrote 71 inverse camera plots plus 71 forward comparison MP4s. Inverse means: rotation `0.213036 deg`, direction cosine `0.838294`, length/scale ratio `1.007017`, normalized translation error `0.003177 m`, and Sim(3) ATE `0.023574 m`. Forward means, excluding the conditioned frame 0: PSNR `19.4988 dB`, SSIM `0.612583`, and LPIPS-Alex `0.285336`. Early/middle/late horizon means are PSNR `23.2029/18.5262/16.7672 dB`, SSIM `0.724805/0.589087/0.523856`, and LPIPS `0.159399/0.306662/0.389947`. The 100k checkpoint is the best of the 12 fully evaluated checkpoints (5k through 100k) on rotation, direction cosine, translation error, ATE, PSNR, SSIM, and LPIPS. Relative to 90k, it improves PSNR by `0.1223 dB`, SSIM by `0.00405`, LPIPS by `0.00386` lower, and ATE by about 3.1%; mean scale ratio moves slightly farther from ideal 1.0 (`1.00515` to `1.00702`), although its median improves to `0.99860`.

Phase-1 video-quality ablation evaluation correction (2026-07-22): the automatic `checkpoint_evals` suite contains only five fixed held-out sources expanded over prefixes `[1,9,17,33,49]`; its `n=5` aggregates are diagnostics, not the original benchmark. The five underlying GT clips/actions/poses are byte-identical to the first five historical full-71 instances, although the compact ablation captions intentionally replace standalone `C` with `A person`. A-E now also configure a distinct 10k-cadence `eval_full71_inverse_forward` callback using the exact historical 71 prefix-1 forward and 71 inverse records, EMA, official shift-3 UniPC, 30 steps, and guidance 1 on an exclusive eight-GPU node. The full-71 launcher and legacy all-checkpoint driver resolve the saved architecture contract before model import. Only `eval_full71_inverse_forward` should be compared to the original Phase-1 metrics above; retain `checkpoint_evals` for variable-prefix, policy, I2V, and qualitative inspection.

A/B were already running when the full-71 callback was added, so their in-memory callback sets remain the old compact-only version. Explicit 70k full-71 backfills are Slurm jobs `3052` (A) and `3053` (B); compact 70k jobs are `3050/3051`. D/E were pending and will load the dual-callback configuration when they start. Future A/B 10k checkpoints need explicit full-71 submission unless those training jobs are restarted; do not mistake the continued compact callback submissions for canonical evaluation.

Verification bar for this directory is higher than a config import. After changes, run pycompile, TOML dryrun, one-step train smoke, final DCP save, official inference load, and official samples for forward/inverse/policy.

### `motion_expert_joint_attention/` Current Main Work

Read order: this file first, then `motion_expert_joint_attention/task_plan.py`, `joint_motion_model.py`, `mot_joint_attention.py`, `mot_joint_layer.py`, and the specific trainer/sampler/dataset file being edited.

This is a self-contained model wrapper around frozen Cosmos-3 Nano reasoner+generator. It adds a trainable motion expert as a true third MoT pathway (`_moe_motion`) with shared joint attention. Historical/default runs use 3-way joint attention. New bridge experiments are flag-gated with `--coupling bridge_local` and keep the native generator path separate from the motion expert, then apply a trainable local gen-motion bridge.

Default/historical packing is:

`[reasoner text | generator image/video/camera | motion shape+frames]`

Reasoner rows use causal attention over reasoner only. Generator and motion rows share the full attention block and attend densely over `{reasoner, generator, motion}`. The reasoner never reads generator or motion rows. This asymmetric attention is load-bearing.

`mot_joint_attention.py` owns only the two-call attention mask:

1. causal call over reasoner rows only;
2. full call with queries from generator+motion rows and keys/values from all rows;
3. scatter outputs back.

`mot_joint_layer.py` owns weight routing by role:

- reasoner rows use plain frozen reasoner weights;
- generator rows use `_moe_gen` frozen/LoRA/full weights;
- motion rows use trainable `_moe_motion` weights.

The mask mode and the weight role are separate concepts. Motion shares generator full-attention behavior but uses its own weights.

Sparse-depth motion is default. `MOTION_LAYER_STRIDE=3` gives motion layers `{2,5,...,35}` for 12 motion blocks. `stride=6` gives 6 blocks. Plain layers own no `_moe_motion` params, run only reasoner+generator, and pass motion rows through unchanged.

Bridge-local mode:

- CLI: `--coupling bridge_local`.
- At former motion layers, reasoner+generator runs without motion K/V, and reasoner+motion runs without generator K/V.
- `modality_bridge.py` then applies a zero-initialized gated bridge between generator and motion rows.
- Directional masking prevents clean condition tokens from reading noisy target tokens. Cross-modal residual updates are role-driven: only noisy target query rows may read source rows in another modality. For `video2motion`, noised motion may attend local clean video. For `motimg2video`, noised video may attend local clean motion. Experimental joint-target tasks permit bidirectional updates because both generator and motion rows are targets. The causal Wan 4x mapping for 97 source frames is latent groups `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`; source frame `m` maps to latent frame `(m+3)//4`. Camera action row `i` covers source interval `[i,i+1]`, including direct local camera-motion edges.
- The old 3-way path remains available as `--coupling joint` and is the default for old checkpoints.

Motion weights are freshly initialized, never copied from `_moe_gen`. `_reset_motion_params` uses small normal init; qk norms/layer norms are fresh; `llm2motion` is zero initialized. Motion FFN width defaults to `MOTION_INTERMEDIATE_SIZE=3072`. Shared attention fixes hidden width 4096 and Q/K/V head geometry.

`FrozenCosmos` must build with `action_gen=True`, because camera tasks require `action2llm`, `llm2action`, and `action_modality_embed`.

## Base Seven-Task Contract and Experimental Tasks

Task names come from `task_plan.py` and must be used verbatim.

The single conditioning mechanism is `condition_mask`: `True` means clean/condition/no loss; `False` means noised/supervised. Text with instruction gets 10% CFG drop during training. `inverse_dynamics` and `video2motion` use empty text. Empty text still tokenizes to a structural EOS token, so the reasoner segment remains non-empty.

Tasks:

- `inverse_dynamics`: video -> camera. Clean: all video. Target: all camera action channels `[:9]`. Empty text.
- `forward_dynamics`: camera + text + image -> video. Clean: frame-0 image and all camera. Target: future video latent frames.
- `policy`: text + image -> camera + video. Clean: frame-0 image. Targets: all camera and future video.
- `text2motion`: text -> motion. Clean: text and shape token. Target: valid motion frames.
- `textimg2motion`: text + image -> motion. Active/correct path is `--textimg_condition reasoner`, which sends frame 0 through the Qwen-VL reasoner image path and packs no generator rows for this task. Historical `--textimg_condition generator` packs the image as one clean generator latent frame and is deprecated; keep it only for old checkpoint/run compatibility. Phase-2 bridge-era pretraining should use `text2motion + textimg2motion` with `--textimg_condition reasoner` before freezing the motion expert.
- `motimg2video`: motion + text + image -> video. Clean: motion, text, frame-0 image. Target: future video latent frames.
- `video2motion`: video -> motion. Clean: all video and shape token. Target: valid motion frames. Empty text.

Phase-3-only opt-in tasks have zero default sampling weight and therefore do not alter historical runs:

- `video2camera_motion`: video -> camera + motion. Clean: all video and shape token. Targets: camera action channels `[:9]` and valid motion frames. Empty text. Each target branch has relative loss weight 0.5 so the two-target sample retains a total branch budget of 1.
- `camimg2video_motion`: camera + image -> video + motion. Clean: frame-0 image, all camera actions, and shape token. Targets: future video latent frames and valid motion frames. Empty text. Each target branch has relative loss weight 0.5.

For generator-conditioned tasks, image is not a separate modality; it is video-latent frame 0. A task with image but no video packs exactly one clean latent frame. The exception is `textimg2motion`, which should use `--textimg_condition reasoner` for new runs: raw frame-0 pixels are preprocessed by the Qwen-VL reasoner processor and appear in reasoner rows instead of generator rows. Do not start new TI2M runs with `--textimg_condition generator` unless explicitly reproducing a historical checkpoint.

Reasoner-side image conditioning implementation fact as of 2026-07-11: the local `nvidia/Cosmos3-Nano` snapshot materializes `net.language_model` as `Qwen3VLTextForCausalLM`, so it does not expose `language_model.visual` by default. The snapshot does include a standalone visual tower under `vision_encoder/` (`model.safetensors`, `config.json`) plus a Qwen3VL processor. `motion_expert_joint_attention/cosmos_loader.py` lazily loads the 576.4M-param `Qwen3VLVisionModel`, freezes it, attaches it as `net.language_model.visual`, and then uses NVIDIA's `prepare_multimodal_reasoner_inputs` helper. Normal text-only and generator-latent paths do not load this tower.

The released processor accepts 256x256 images. Its configured minimum area is 65,536 pixels. A direct processor check on node 3 measured: source 224 -> processor grid `[1,16,16]` -> 64 merged visual tokens; source 256 -> the same 64 tokens; source 640 -> grid `[1,40,40]` -> 400 tokens. New Phase-2 TI2M uses `--reasoner_image_size 256` at both dataset and encoder boundaries, reducing the visual sequence by 6.25x versus raw 640 frames. This size is saved in checkpoints and replayed by `sample.py`; old reasoner-image checkpoints without the field retain historical 640 behavior.

Motion tokens include one clean shape token from `neutral_joints` plus valid motion frames. Camera tokens are Cosmos camera pseudo-actions with domain id 2 and loss only on channels `[:9]`; zero-padded channels are ignored. Camera loss is weighted by `ACTION_LOSS_WEIGHT=10`.

## Joint-Attention Data

`nymeria_joint_dataset.py` builds two NymeriaPlus indices:

- `_index`: aligned windows where video, camera, and motion share the same `(uuid, start)`. Normally its aligned length equals output/padding `T`. The corrected Phase-2 T2M+TI2M run is the one guarded exception: output `T=200`, but `_index` uses 97 valid aligned frames for reasoner-image TI2M and pads/masks its remaining 103 batch rows. This exception is rejected if generator/video/camera tasks are active.
- `_t2m_index`: native captioned motion windows for Nymeria `text2motion`, independent of T and padded/masked to T. Used only when mode is `text2motion` and no video/camera are needed.

Alignment invariant:

- Motion-output tasks with no video need no video-frame alignment. `text2motion` should use native motion windows.
- `textimg2motion` uses the same 97-frame NymeriaPlus window as the source video and sends frame 0 to the reasoner. It does not use generator rows or a full 200-frame video window.
- Tasks that pack video with camera or motion must use `_index`.

BONES-SEED is motion-only and is routed only to `text2motion`; it cannot supply image/video/camera tasks. BONES text pools are overview/natural, single-timeline, and multi-timeline. Active pair files use only `content_natural_desc_4` for overview/natural rows; single-timeline and multi-timeline rows are kept as-is. `--bones_frac` controls the fraction of text2motion mass drawn from BONES. A 2026-07-11 ordered-source audit checked every emitted natural row against `seed_metadata_v004.csv`: train has 120,074/120,074 exact desc-4 matches and val has 6,541/6,541; there were zero other overview captions. Train has 585,312 total rows and val has 18,459 total rows. New training fails fast if BONES was requested but could not load; it no longer silently changes the source mixture.

Both Nymeria and BONES motion use shared 283-d UniEgo stats (`uniego283_mean.npy`, `uniego283_std.npy`) in the joint-attention run. This is a documented approximation for BONES. Only motion is z-scored. Camera stays raw. Video stays in VAE latent space.

The dataset humanizes Nymeria captions by rewriting standalone `C` to `a person`.

Training-time caption dropout is controlled by `--cfg_dropout` and defaults to 0.10. It is applied by `nymeria_joint_dataset.py` to all instruction-caption tasks during train only, and is forwarded into the BONES `UniegoPairsDataset` for BONES text2motion rows. It does not apply to validation/test data. For `textimg2motion --textimg_condition reasoner`, the current implementation drops text but keeps the frame-0 image condition; there is no image-condition dropout flag. Sampling CFG for this task uses real text + same image for the conditional branch and empty text + same image for the null branch.

Feature guard skips extreme motion windows with `|z|max > 20`.

Precomputed Wan-VAE latents are preferred for video. `precompute_latents.py` writes `.npz` files under a latent root such as `joint_latents` or `joint_latents_T97`, with latents, camera action, and metadata. For non-default T, latent roots are suffixed like `joint_latents_T97`. Set `WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth`.

## Floor Calibration

NymeriaPlus UniEgo motion has per-sequence vertical bias after GT grounding. BONES convention leaves contacted feet about +2-3 cm above y=0, while Nymeria initially had median foot penetration around -5.4 cm.

`precompute_floor_calibration.py` computes per-sequence deltas:

`delta_seq = d_minc(seq) - c0`

where `d_minc` is the median contacted-foot min-y after GT grounding and `c0 = +2.42 cm` is the BONES target convention. Contact channels `[279:283]` map to joints `[24,25,28,29]`. Heights use Y channel `j*9+7`.

At dataset index-build time:

`entry["off"] = ground_offset_y + delta_seq`

This is the calibrated total vertical shift that `ground_features` subtracts. Original values are recoverable as `off_gt` and `delta`. Future camera-motion absolute alignment must use the calibrated total `off`.

Window drops from `floor_calibration.json` are skipped in both `_index` and `_t2m_index`:

- `wrong_floor`: enough contacts and the contact median is more than 0.20 m from expected floor.
- `residual_penetration`: calibrated min-foot-y below -0.20 m.
- `extreme_y`: a later defense-in-depth scan found post-calibration joint-Y magnitudes above 2.5 m that survived the first two criteria.

The completed 713-sequence calibration file is noisier than the early ~2% estimate: among usable captioned windows, train drops 7,173/128,102 (5.60%: 4,640 wrong-floor, 2,181 residual-penetration, 352 extreme-Y) and test drops 714/13,487 (5.29%: 488/198/28). The active Phase-2 and native Phase-3 logs confirm these exact train drops at dataset construction, leaving 120,929 native T2M windows and 112,937 aligned T97 windows. Every train/test sequence has its own measured delta; no sequence uses the global fallback. The full71 camera-oriented list was not motion-quality filtered and happens to contain five dropped windows (7.04%): four wrong-floor and one residual-penetration. Direct diagnostics show these are gross, not borderline: wrong-floor contact-height errors are 0.58, 2.64, 0.66, and 1.24 m, and the residual case reaches 0.415 m below floor. Use the floor-valid result or select replacement windows for motion evaluation.

Filtering does not prove that every retained SOMA fit is clean. The calibration summary over retained windows has median min-foot-y +0.34 cm and no windows below -20 cm, but 15.22% have at least one min-foot observation below -5 cm; that minimum-over-window statistic can include transient fit/contact artifacts and is not itself a wrong-floor label. `_load_motion` therefore also rejects normalized `|z|max > 20` at runtime before the sample reaches the loss. In the completed 200k Phase-2 run this guard rejected about 247,961 attempts out of approximately 51.2M sampled items (~0.48%), concentrated in 574 unique starts. This adds retry/log overhead but prevents those extreme samples from training the model. Moderate residual motion-fitting noise remains a real data-quality risk and should not be described as fully solved.

Normalization provenance matters for existing checkpoints. The active `motion_expert/stats/uniego283_{mean,std}.npy` files were generated on 2026-06-24 from the older raw-`ground_offset_y` pair set; the per-sequence calibration/drop file was generated on 2026-07-02. `audit_motion_stats.py` recomputes read-only, versioned candidate stats after current calibration/filtering and the runtime `|z|max > 20` guard, and refuses to overwrite the active stats. The 2026-07-15 train audit starts from 120,929 floor-filtered windows, removes 616 runtime-guard failures, and computes candidate stats over 120,313 windows/11,828,053 frames under `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/motion_quality_audit_20260715`: clean joint-Y mean shifts only +3.24 cm, but median joint-Y std changes from 0.628 m to 0.228 m (clean/old ratio 0.364); foot-Y std changes from about 0.585 m to 0.066-0.069 m. All 30 Y channels have clean/old std below 0.8, while the non-Y median ratio is 0.973. A separate guarded 71-sequence test audit gives joint-Y std 0.227 m, so this is not a train/test shift. Phase 2's physical decoded losses mitigate relative pose and velocity errors, but its centroid-relative pose loss removes global translation, so the inflated old Y scale still underweights global/floor-height feature error. Existing Phase-2/Phase-3 checkpoints remain internally consistent with the old stats and must continue using them. A clean-stats experiment needs versioned mean/std paths saved in checkpoint metadata and a Phase-2 retrain from scratch; never silently repoint the global config because that would reinterpret old checkpoint outputs.

`precompute_floor_calibration.py` now includes `extreme_y` directly and deduplicates blacklist output by physical `(uuid,start,end)` while retaining annotation-occurrence statistics. A full low-priority reproduction on 2026-07-15 matched the active c0, all 713 deltas, and all 7,611 effective drop windows exactly; there are 7,887 dropped annotation-row occurrences because the source manifest contains duplicate interval annotations.

If calibration JSON is missing, the dataset warns and proceeds uncalibrated. BONES samples are untouched.

## Objectives and Timestep Convention

In `motion_expert_joint_attention/train.py`, motion objective is selected by `--objective`; the current argparse default is `x0`, while vision and camera are always velocity objective to match Cosmos rectified flow.

Velocity objective: `x_t = (1-t)x0 + t*noise`, target `v = noise - x0`, sample by Euler from `t=1` to `t=0`.

X0 objective: model predicts clean x0. Historical `--motion_schedule legacy` uses unshifted logit-normal sigma plus a linear DDIM-in-sigma ladder. New `--motion_schedule native` uses Cosmos's shifted logit-normal training sigma and shifted 1000-step inference ladder while retaining the empirically better x0 target.

For each target modality, the same per-sample time used to create that modality's noisy state must be passed to `model.forward`. Ordinary one-target tasks can use the compatibility `t_or_sigma` argument. Joint-target tasks must pass independent `motion_t_or_sigma` and `gen_t_or_sigma`; sharing one sampled training time would change both frozen specialists' pretrained marginals.

Critical Cosmos timestep scale: Cosmos heads multiply by `TIMESTEP_SCALE=1e-3` internally. If raw `t in [0,1]` is passed directly, the embedder sees nearly constant `t*1e-3`, loss can drop while samples remain noise. `JointMotionModel.forward` divides `t_or_sigma` by `TIMESTEP_SCALE` exactly once before passing to heads, so the internal embedder sees true flow time. Train and sample must both go through this pre-scale. Do not additionally scale in callers.

Motion losses in joint-attention training: masked 283-d feature loss, decoded-joint loss, and smoothness loss, with documented weights `1/10/50`. Vision loss is flow MSE on latent channels. Camera loss is flow MSE on action channels `[:9]` times 10.

For current Phase 2, base seven-task, and experimental joint-target runs, all motion-supervised tasks train x0 prediction by default. `text2motion`, `textimg2motion`, `video2motion`, `video2camera_motion`, and `camimg2video_motion` train `motion_pred` directly against clean normalized motion `x0` via `flow.add_noise_x0_masked` unless `--objective velocity` is explicitly passed. `motimg2video` packs motion as a clean condition only and has no motion target/loss. The checkpoint records `args["objective"]`, and `sample.py` dispatches motion sampling to `sample_x0` or `sample_velocity` from that saved value.

Native motion schedule implementation as of 2026-07-11:

- `flow.sample_sigma_native_logitnormal` matches current Cosmos action-style training: CPU standard-normal draw, sigmoid, `sigma=1-t_raw`, rational shift `s*sigma/(1+(s-1)*sigma)`, default shift 3, then `x_sigma=(1-sigma)x0+sigma*eps` with clean x0 as target.
- `flow.native_inference_schedule` is bit-identical to `FlowUniPCMultistepScheduler.set_timesteps`: float32 `(N-1)/N` endpoint, linear base ladder, rational shift, integer `sigma*N` model timesteps, final zero sigma.
- Historical `--motion_native_solver euler` is the BONES POC-proven first-order x0/straight-path update on that exact native ladder. It remains available for reproducing old checkpoints and evaluations, but it is schedule-identical rather than solver-identical to generator UniPC.
- New native-schedule motion runs default to `--motion_native_solver unipc`. It converts `x0_hat` to `v_hat=(x_sigma-x0_hat)/sigma` and drives NVIDIA's official `FlowUniPCMultistepScheduler`, making integration solver-identical to native generator sampling while retaining the motion expert's x0 prediction target. The adapter is covered by official-ladder, fixed-noise, perfect-x0, and dispatch tests.
- Training and sampling both feed the quantized/shifted normalized sigma through the same `JointMotionModel.forward` timestep pre-scale. Vision/camera objectives and schedules are unchanged.

The BONES POC uses `shift(sigmoid(z))`, while current Cosmos code computes `shift(1-sigmoid(z))`; these are the same distribution by logistic-normal symmetry. The real joint-attention implementation follows the current framework ordering exactly.

Checkpoint metadata now stores `args`, `task_weights`, and an explicit `data_policy` block. The latter records native T2M versus aligned TI2M temporal policy, `ti2m_frames`, caption dropout, no image dropout, reasoner-image size, BONES overview caption policy, motion schedule/shift/timestep count, and native solver. Resume warns on drift in schedule, mRoPE, coupling, image size, and task-specific TI2M length.

## Trainability and Checkpoints

For joint attention, motion is always fully trained unless `--freeze_motion` is used for Phase 1. Reasoner and generator each choose one of frozen, LoRA, or full.

Important toggles:

- `--gen_lora`: LoRA on generator q/k/v/o projections.
- `--reasoner_lora`: LoRA on reasoner q/k/v/o projections.
- `--gen_full`: train all `_moe_gen` plus gen I/O heads (`vae2llm`, `llm2vae`, `action2llm`, `llm2action`, `action_modality_embed`).
- `--freeze_gen`: keep generator LoRA/full/action-head params frozen even when `--gen_lora` or `--gen_full` is used to instantiate/load them. This is for bridge-only Phase 3 runs that warm-start Phase-1 generator LoRA but train only bridges.
- `--freeze_motion`: build motion expert but exclude it from optimization.
- `--init_gen <ckpt>` and `--init_motion <ckpt>`: warm-start disjoint subsets.
- `--motion_intermediate`, `--motion_layer_stride`, `--tasks`, `--task_weights`, `--bones_frac`, `--objective`.

Generator/reasoner LoRA and `gen_full` params live under `cosmos.net`, so use `model.named_all_parameters()` and `trainable_parameters()` instead of plain `model.parameters()`.

Checkpoints store trainable deltas only (`trainable_state_dict`). Pure DDP manually all-reduces every trainable param, materializing zero grads for unused task-specific params to avoid multi-task NCCL desync. Rank-0-only visualization is wrapped in pre/post barriers so other ranks cannot enter the next backward while rank 0 samples. Non-finite optimizer-step decisions are reduced across ranks so every replica/shard steps or skips together; the non-finite loss guard uses a graph-connected finite zero rather than `NaN*0`. Bridge-only checkpoints with `--freeze_gen --freeze_motion` are not self-contained: their `model` state contains only bridge tensors. `sample.py` / eval must first replay recorded `args["init_gen"]` and `args["init_motion"]`, then overlay the bridge checkpoint delta; otherwise V2M samples can be exactly zero because the frozen motion expert is missing.

Known discrepancy: `config.TASK_WEIGHTS` and `task_plan.TASK_WEIGHTS` differ. `train.py` uses `config.TASK_WEIGHTS`; dataset normalizes weights when sampling. `task_plan` weights are only for its self-check.

Known stale flag: old docs/scripts may mention `--data_mix`; current `train.py` uses `--tasks`, `--task_weights`, and `--bones_frac`.

Base-weight key remapping was fixed on 2026-07-02. Runs before that may have random reasoner attention and random action heads despite docs claiming pretrained reuse. The fixed loader remaps reasoner attention and action I/O heads correctly.

Camera mRoPE packing in `gen_heads.build_gen_segment` was fixed on 2026-07-02 to pack camera tokens at the vision segment start temporal offset, matching native Cosmos, instead of sequentially after video.

Motion mRoPE has an explicit `--motion_mrope {legacy,cosmos3d}` flag in train/sample/eval model loading:

- `legacy`: old behavior and default for backward compatibility. Motion rows use sequential positions after the generator segment, expanded across all three mRoPE axes.
- `cosmos3d`: official-style 3D mRoPE for motion. Motion frame tokens are a `T x 1 x 1` temporal grid with temporal compression factor 1, so motion frame `k` shares the same temporal coordinate family as video/camera frame `k` in video+motion tasks. The shape token is non-temporal conditioning and is placed on the last text-time plane with distinct spatial axes to avoid collisions with text, video patches, and motion frame 0.

## Planned Joint-Attention Curriculum

The same model/mask supports a 3-phase curriculum:

1. Phase 1: camera tasks only, `--tasks inverse_dynamics forward_dynamics policy --gen_lora --freeze_motion`. Motion expert built but not stepped; no motion tokens are packed.
2. Phase 2: text2motion, historically `--tasks text2motion`, optional BONES via `--bones_frac`. Motion pathway trains; generator frozen. Bridge-era corrected Phase 2 should train `text2motion + textimg2motion` with `--textimg_condition reasoner` so the motion expert has seen reasoner visual tokens before it is frozen.
3. Phase 3 has two distinct experiment families. Historical joint runs train all seven tasks with 3-way attention. The active bridge experiment trains only `video2motion + motimg2video`, recreates and freezes the Phase-1 generator LoRA and Phase-2 motion expert, and optimizes only `LocalModalityBridge` modules.

The new Phase-2 3D-mRoPE variant run is `ja_t2m_x0_T200_mrope3d`, launched by `motion_expert_joint_attention/sbatch_t2m_both_mrope3d.sh`. It is the same recipe as `ja_t2m_x0_T200` except `--motion_mrope cosmos3d`; it trains `--tasks text2motion --bones_frac 0.5 --T 200 --objective x0 --batch_size 32 --steps 200000` on 8 GPUs. Slurm job `2765` was submitted on 2026-07-06 and started on `a3ultravis-a3ultranodeset-1`. Its run dir is `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_x0_T200_mrope3d`.

The native-schedule corrected Phase-2 launcher is `motion_expert_joint_attention/sbatch_t2m_ti2m_native_mrope3d.sh`. Its run is `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d` with:

- T2M + TI2M weights 0.75/0.25 and `bones_frac=0.5`, giving effective mass 37.5% Nymeria T2M, 37.5% BONES T2M, and 25% Nymeria TI2M;
- output/padding `T=200` for full/native T2M capacity, but exactly 97 aligned valid Nymeria frames for TI2M;
- reasoner-side TI2M image at 256x256, no image dropout, text CFG dropout 0.10;
- x0 target, native shifted logit-normal schedule, shift 3, 1000 timesteps, and the historical POC-proven native-ladder Euler sampler;
- `motion_mrope=cosmos3d`, joint coupling, stride 3 / 12 motion blocks, batch 32 per rank, 8 GPUs, 200k steps;
- 96-hour Slurm wall limit. The previous T2M+TI2M run measured about 1.26 s/step, so 200k steps alone need about 70 hours; a 48-hour request predictably stops near 137k before visualization/checkpoint overhead;
- in-training visualization every 2k steps with two T2M and two TI2M samples. TI2M MP4s show conditioning image | GT | generated; rendering uses stride 2 at 10 FPS while raw arrays keep all frames.

Pre-submit verification on node 3 passed: exact native schedule/UniPC ladder checks; real 97-valid-frame TI2M padded to 200; full 993.73M-trainable-parameter T2M and TI2M forward/backward with frozen-gradient assertions; checkpoint optimizer save/reload with zero skipped tensors; post-load 200-frame T2M and 97-frame TI2M sampling; condition/GT/generated MP4 creation; and a two-GPU run with zero parameter divergence.

Production submission: Slurm job `2844` (`t2mtinat`) was submitted on 2026-07-11 with no dependency. It requested one exclusive `a3ultra` node, 8 GPUs, 64 CPUs, and 96 hours. Slurm output is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mtinat-2844.out`; run outputs go to `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d`. It completed successfully with exit code 0 on 2026-07-14 after 2d 2h 28m. The final named checkpoint is `ckpt_step200000.pt` (about 6.19 GB); bridge launchers name this file explicitly so they cannot silently consume a moving `latest.pt`.

The contact-aware native Phase-2 variant launcher is `motion_expert_joint_attention/sbatch_t2m_ti2m_native_mrope3d_contact.sh`. It keeps job `2844`'s architecture, native x0 training schedule, Cosmos3D motion mRoPE, T2M/TI2M mixture, aligned 97-frame TI2M policy, 256x256 reasoner image, batch size, LR, cosine schedule, and 200k steps. Its training-objective change is the proven BONES contact objective: `1*L_feature + 1*L_joint + 5*L_smooth + 0.05*L_contact_BCE + 1*L_contact_horizontal_foot_velocity + 10*L_contact_foot_height`, with contact-logit scale 2 and motion FPS 20. It also selects official NVIDIA UniPC-35 for in-training sampling/visualization and records that solver in checkpoints; this changes sampling, not the training noising or x0 loss. Contact BCE and physical terms use ground-truth contact masks; the four contacts map to SOMA-30 joints `[24,25,28,29]`, and foot height uses raw local-pose Y channels to avoid cumulative-decoder gradient spikes. The new terms are opt-in (`train.py` defaults them to zero), logged to TensorBoard, checkpoint metadata, and `data_policy`.

This variant deliberately uses the original pre-floor-filtering NymeriaPlus shared stats for both Nymeria and BONES: `/home/jungbin_cho/cosmos_motion_ft/motion_expert/stats/uniego283_mean.npy` and `uniego283_std.npy`. The launcher verifies their SHA-256 hashes (`bd1d6bdc...b400d3` and `ee069e3a...65f28c`) before starting. It does not use the 2026-07-15 cleaned candidate statistics. In-training visualization remains enabled every 2k steps and includes T2M plus reasoner-image TI2M; the launcher requires visualization setup/rendering to succeed.

UniPC conversion verification on 2026-07-15 passed the native training-sigma, official scheduler-ladder, perfect-x0 endpoint, fixed-initial-noise, and default-dispatch contracts on node 2. A real one-update smoke at `/weka/jungbin/cosmos_motion_ft_runs/_smoke_phase2_contact_native_unipc35_115_20260715` then trained with all requested losses, saved a 149-tensor checkpoint plus optimizer state and TensorBoard event, and completed one T2M-200 and one reasoner-image TI2M-97 visualization with official UniPC-35. Both generated arrays were finite, both MP4s passed `ffprobe`, the TI2M condition image was 256x256, and checkpoint/config/manifest metadata all recorded `motion_native_solver=unipc`. Pending Euler-configured job `2995` was canceled before start. Replacement production job `2997` was submitted with the UniPC-35 launcher; at submission it was pending for Slurm priority. Its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35`, and its log is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mticnt-2997.out`.

Job `2997` later failed on 2026-07-17 at logged step `102280`, not from OOM or a model/loss failure. Losses remained finite, peak allocation stayed at `67.1 GB/GPU`, and the valid periodic checkpoint/latest pair was completely written at step 100k. A DataLoader worker then reported `could not unlink the shared memory file /torch_*: No such file or directory`; an eight-rank 16M-element gradient all-reduce stopped completing and hit the 600-second NCCL watchdog. Evaluator smoke processes had also been launched by SSH on the same Slurm-exclusive node/GPU during exactly this interval. Slurm exclusivity does not prevent SSH processes from bypassing Slurm, so never run evaluation or smoke workloads on a node occupied by this eight-GPU training job even when `nvidia-smi` appears to show spare memory.

Recovery hardening on 2026-07-18 keeps the scientific recipe unchanged but reduces DataLoader IPC pressure from `8 workers x prefetch 2` to `4 workers x prefetch 1` per rank, adds a 300-second DataLoader timeout, saves every 5k steps, and gives torchrun two bounded worker-group restarts. A timeout now becomes a process failure that torchrun can restart instead of waiting indefinitely for NCCL; `--resume auto` reloads the newest complete weights, optimizer, and step after either a Slurm resubmission or an in-job worker restart. Parser, four-worker DataLoader, shell-syntax, compile, and diff checks passed. Resume job `3000` was submitted with the same run name and 200k target; it will load `latest.pt` at step 100k and continue from step 100001. At submission it was pending for priority because all four nodes were allocated. Its Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mticnt-3000.out`.

Phase-2 motion training does not maintain a model-weight EMA. `motion_expert_joint_attention/train.py` optimizes the regular motion-expert parameters and saves those directly; its `_loss_ema` is only a scalar used by the explosive-batch guard. The BONES native-schedule POC likewise has no model EMA. Adding motion-weight EMA is a separate future experiment, not part of job `2844`.

The final Phase-2 checkpoint has a dedicated shape-aware C45 evaluation path in `prepare_shape_tmr_eval.py`, `precompute_nymeria_tmr_text.py`, `eval_phase2_shape_tmr.py`, and `sbatch_phase2_shape_tmr_eval.sh`. It evaluates T2M on all six official BONES-SEED text-to-motion suites and on every floor-filtered, direct-guard-valid Nymeria test annotation window from all 71 held-out sequences. Different captions for the same physical Nymeria start remain distinct evaluation cases. TI2M is evaluated only on Nymeria's aligned 97-frame windows. The active floor calibration removes known wrong-floor/residual/extreme-Y rows first; the old-stat training guard then rejects `|z|max > 20` directly, without dataset retry/substitution.

Sampling uses the checkpoint's native x0 schedule (shift 3, 1000 training timesteps) with official Cosmos UniPC for 35 steps and deterministic per-case initial noise. T2M uses text CFG 2 and a same-text/same-noise farthest-natural-skeleton intervention. TI2M is sampled twice with identical image and noise: `ti2m_cfg2` keeps the image in both conditional/null branches and drops only text in the null branch, while `ti2m_no_cfg` uses guidance 1 and performs only the conditional image+text branch. Image dropout is not part of either TI2M evaluation. TI2M does not use the skeleton counterfactual because keeping the original actor image while replacing body proportions would create contradictory conditions.

The C45 bundle is `/weka/jungbin/shape_aware_motion_eval_c45_20260715`; evaluator checkpoint `artifacts/evaluator/c45_step_00005000.pt` and `artifacts/evaluator/stats/motion` are paired. Those C45 stats are used only inside the frozen evaluator. Phase-2 outputs are first unnormalized with `motion_expert_joint_attention/stats/uniego283_{mean,std}.npy`, decoded to SOMA-30 joints at 20 FPS, then resampled to C45's 30 FPS representation. The benchmark LLM2Vec cache is retained byte-for-byte and extended with every valid BONES/Nymeria evaluation caption absent from the base cache, only after a local encoder parity check against known cached rows. Reported outputs include protocol/plain retrieval, TMR FID, foot/contact metrics, bone-shape metrics, counterfactual response for T2M, per-cohort JSON, a case-weighted six-suite BONES summary, and representative MP4s. The production Slurm script gates the full 8-GPU sweep on a one-GPU end-to-end smoke covering all three cohort types, both TI2M CFG modes, C45 metrics, counterfactual sampling, checkpoint loading, and rendering.

Contact-aware Phase-2 step-100k BONES evaluation completed on 2026-07-17 at `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/eval_bones_shape_tmr_c45_unipc35_step100000`. It uses the same six manifests, cached benchmark text, C45 checkpoint/stats, native shift-3 x0 schedule, UniPC-35, CFG 2, and deterministic per-case noise as the earlier final-checkpoint evaluation. All six suites and 9,124 cases completed. The case-weighted suite means are: bone MAE `0.47281 cm`, contact consistency `0.85979`, predicted-contact skate `5.0693 cm/s`, height skate `19.2935 cm/s`, maximum contact velocity `26.2800 cm/s`, skate ratio `0.10715`, FID gen-GT `0.03350`, plain R@3 `58.0776`, protocol R@1/2/3/5/10 `46.779/61.584/69.028/76.304/83.659`, shape centered correlation `0.93496`, shape response slope `0.83560`, shape variance ratio `0.89367`, counterfactual delta cosine `0.97583`, counterfactual response slope `0.85504`, and counterfactual target advantage `2.3752 cm`. All 42 JSON and 42 NPZ rank shards, 12 finite visualization arrays, and six MP4s were validated.

Against the original non-contact Phase-2 final checkpoint at step 200k, the contact-aware 100k checkpoint reduces predicted-contact skate by `55.66%`, height skate by `27.74%`, maximum contact velocity by `51.62%`, and skate ratio by `49.38%`; the skating reduction occurs in every suite (`53.7-56.7%`). It is worse on FID (`+52.90%`), protocol R@3 (`-5.48` points), plain R@3 (`-6.46` points), bone MAE (`+6.26%`), and contact consistency (`-0.0144`). Shape tracking is mostly stable: centered correlation changes `-0.005`, response slope `+0.003`, and variance ratio `+0.008`. This comparison is useful but not controlled because it changes both the loss and training duration (`100k` contact-aware versus `200k` non-contact); do not attribute all regressions or gains solely to contact losses without a same-step comparison.

## Bridge Hypotheses and Representation Risks

These notes are experiment hypotheses, not established results.

Modality-bridge hypothesis: full generator-motion joint self-attention is expensive and may perturb the pretrained generator distribution by adding motion tokens directly into generator attention. For frame-aligned physical translation tasks (`video2motion`, `motimg2video`), a trainable gen-motion bridge with temporal locality may be a better inductive bias. The active Phase-3 bridge experiment freezes the Phase-1 generator/video/camera model and the Phase-2 motion expert, then trains only `modality_bridge.py` so generator and motion hidden states can exchange local information without changing either expert's weights. Directional masking is task-specific: for `video2motion`, noised motion reads local clean video while video cannot read noisy motion; for `motimg2video`, noised video reads local clean motion/shape while motion cannot read noisy video. Same-modality bridge attention remains enabled. Twelve bridges are inserted after the native reasoner+generator update and the reasoner+motion update at sparse motion layers 2,5,...,35. Their generator/motion residual gates are zero-initialized, making the bridge an exact no-op before training. This is still a hypothesis; successful optimization and useful alignment are not established by the smoke tests.

The head-camera Phase-3 variant adds an explicit, flag-gated relative-geometry contract without changing either frozen specialist or adding a trainable adapter. `head_camera_alignment.py` decodes UniEgo's SOMA `Head` joint (index 6) and maps its frame-to-frame transform into the upright-RGB camera-action frame using a robust global train-split calibration. For `T_world_camera = T_world_head X`, it computes `R_camera = R_X^T R_head R_X` and `t_camera = R_X^T(t_head + (R_head-I)r_X)`. Absolute translations are intentionally unused. `estimate_head_camera_calibration.py` fitted `head_camera_calibration_train.json` from all 642 train sequences: 61,632 relative-action samples, train median/mean translation error 1.54/3.09 mm per frame, and train median/mean rotation error 1.44/1.58 degrees. On all 71 test sequences, 11,892 T97 windows that pass the same floor and `|z|` guards have median/mean translation error 2.61/3.57 mm and rotation error 1.28/1.38 degrees; the lever arm improves held-out median translation from 4.80 mm. This validates the relative approximation but does not establish reliable absolute head-camera alignment.

With `--head_camera_alignment`, data/model flow is deliberately asymmetric and leakage-guarded. For M2V, the model derives a clean 96x9 camera-action condition only from its clean motion input and packs it through the frozen Phase-1 camera pathway; synchronized GT camera action is used only for a no-grad calibration diagnostic. For V2M, the model still receives only clean video. Synchronized camera action is carried in the separate `camera_alignment_action` batch field and supervises relative camera action decoded from predicted motion x0 with robust translation/rotation losses; it is never packed as a V2M condition. `task_plan.resolve_sample` rejects derived camera conditioning for every task except M2V. The intended production weights are `w_head_camera_trans=0.05`, `w_head_camera_rot=0.05`, with 2 cm and 5 degree normalization scales. This remains an experiment: the geometry is verified, but improvement in V2M/M2V metrics must be measured rather than assumed.

`eval_all.py` automatically writes `motion_recon/video2motion/head_camera_alignment_metrics.json` for head-camera checkpoints. It reports mean/median/p90 V2M translation and rotation action errors plus the GT-motion calibration floor on the identical windows; floor-valid output is split when needed. `--eval_head_camera_alignment` enables the exact same train-calibrated evaluation-only metric for a historical baseline checkpoint without changing that model's inputs, weights, or sampling. This is required for a paired baseline/headcam comparison. The metric measures relative action consistency, not absolute camera pose. M2V keeps the existing PSNR/SSIM/LPIPS video metrics because camera motion cannot be read directly from generated pixels without a separate inverse-dynamics estimator. `merge_phase3_clean71.py` also merges the 66 floor-valid rows plus five deterministic replacements for this metric.

Phase-1 camera-coordinate reminder: all native Phase-1 action targets are from the preprocessed **upright RGB optical-camera** trajectory, not the raw Aria device trajectory. `nymeria_camera_rgb_dataset.py` rewrites the manifest's `/camera/` path to `/camera_rgb/`, loads `cam_world_pos_upright` and `cam_world_rot_upright`, and computes the 9D metric relative action with Cosmos `pose_abs_to_rel(..., pose_convention="backward_framewise")`. Preprocessing first forms `T_world_rgb = T_world_device @ T_device_rgb` from the recording's VRS device-to-RGB extrinsic and then applies the optical-axis rotation matching the upright video. Inverse dynamics, forward dynamics, and policy therefore all use relative upright-RGB-camera translation `(3)` plus rotation-6D `(6)`. The head mapping and all metrics below target this same frame.

The 2026-07-20 **test-actor oracle** is a separate, explicitly leaky diagnostic requested to answer how well GT motion can reconstruct camera after calibrating each held-out actor from GT. `estimate_test_actor_head_camera_calibration.py` fits one fixed `R_X,r_X` for each of the 17 actors represented by the motion-clean71 benchmark. It uses the exact 71 held-out windows (66 original floor-valid windows plus five deterministic replacements), synchronized GT UniEgo motion, and synchronized GT upright-RGB camera actions. Each six-parameter actor transform starts from the train-global calibration and minimizes the same normalized Phase-3 geometry objective: SmoothL1 translation at a 2 cm scale plus rotation-matrix chord at a 5 degree scale, with a bounded rotation correction and lever arm. The fit and reported clean71 windows are identical. Consequently this is an **in-sample oracle floor**, not a leakage-free evaluation, deployable calibration, training target, or model-selection metric. The loader requires `kind=oracle_test_actor_head_camera_calibration`, `split=test`, and explicit `uses_test_gt_motion`, `uses_test_gt_camera`, and `diagnostic_only` flags before accepting the artifact.

The metric components are:

- `translation_m` / `rotation_deg`: V2M-predicted motion mapped with the production train-global rigid calibration, compared with synchronized GT upright-RGB camera action.
- `gt_calibration_translation_m` / `gt_calibration_rotation_deg`: GT motion mapped with that same train-global calibration. This is the leakage-free calibration/representation/data floor paired to the model rows.
- `oracle_actor_translation_m` / `oracle_actor_rotation_deg`: V2M-predicted motion mapped with the actor's test-GT-fitted transform. This diagnoses whether an oracle actor transform transfers from GT motion to model output; it is not a fair model score.
- `gt_oracle_actor_translation_m` / `gt_oracle_actor_rotation_deg`: GT motion mapped with its own test-GT-fitted actor transform. This is the requested in-sample actor-calibrated floor.

For every 97-frame window, translation is the mean Euclidean error over 96 relative translations in metres, and rotation is the mean SO(3) geodesic angle over 96 relative rotations in degrees. JSON aggregation then reports mean/median/p90 across windows. `per_actor_aggregate` first groups those window rows by actor; the overall aggregate remains window-weighted, not actor-balanced. Absolute pose, absolute origin, and accumulated trajectory error are not part of these metrics.

Vanilla Phase-3 step-200k motion-clean71 results (`head_camera_alignment_metrics_motion_clean71.json`):

| Motion source | Rigid calibration | Translation mean | Rotation mean |
|---|---|---:|---:|
| V2M prediction | train-global | 6.394 mm | 1.281 deg |
| GT motion | train-global | 4.445 mm | 1.235 deg |
| V2M prediction | test-actor oracle | 6.709 mm | 1.259 deg |
| GT motion | test-actor oracle | 4.056 mm | 1.205 deg |

Per-actor GT floors are:

| Actor | Windows | Train-global GT | Test-actor-oracle GT |
|---|---:|---:|---:|
| S01 | 7 | 5.490 mm / 1.040 deg | 5.164 mm / 1.017 deg |
| S02 | 9 | 3.558 mm / 1.180 deg | 3.789 mm / 1.147 deg |
| S03 | 3 | 2.624 mm / 0.886 deg | 2.087 mm / 0.882 deg |
| S04 | 1 | 2.795 mm / 0.875 deg | 1.423 mm / 0.864 deg |
| S05 | 5 | 5.319 mm / 1.193 deg | 4.865 mm / 1.179 deg |
| S06 | 3 | 7.923 mm / 1.715 deg | 6.719 mm / 1.723 deg |
| S07 | 6 | 5.297 mm / 1.168 deg | 5.191 mm / 1.175 deg |
| S08 | 4 | 3.638 mm / 1.289 deg | 3.379 mm / 1.250 deg |
| S09 | 1 | 6.123 mm / 1.261 deg | 2.815 mm / 1.236 deg |
| S10 | 5 | 5.374 mm / 1.417 deg | 4.994 mm / 1.430 deg |
| S11 | 1 | 6.442 mm / 1.373 deg | 2.252 mm / 1.335 deg |
| S12 | 7 | 3.698 mm / 1.175 deg | 3.606 mm / 1.141 deg |
| S13 | 2 | 3.476 mm / 0.937 deg | 2.335 mm / 0.937 deg |
| S14 | 3 | 4.752 mm / 1.461 deg | 3.589 mm / 1.232 deg |
| S16 | 7 | 4.095 mm / 1.296 deg | 3.942 mm / 1.228 deg |
| S17 | 2 | 2.414 mm / 0.658 deg | 1.636 mm / 0.672 deg |
| S19 | 5 | 3.548 mm / 1.686 deg | 3.857 mm / 1.655 deg |

The oracle reduces the aggregate GT floor by 8.76% in translation and 2.46% in rotation, so actor-specific rigid geometry explains a small but measurable part of the mismatch. It does not explain the main V2M translation problem: applying the GT-fitted actor transforms to predicted motion worsens translation by 4.93% (`6.394 -> 6.709 mm`) while improving rotation by only 1.70%. Some actors also worsen even on GT, because the optimizer minimizes the combined normalized robust objective rather than either reported mean independently and the SOMA Head-to-worn-camera relation is not perfectly rigid. Single-window actors S04/S09/S11 are especially in-sample and must not be used to claim generalization. Do not subtract the GT floor as though errors were independent; use paired rows and report the calibration contract.

The same step-200k motion-clean71 V2M output confirms that its dominant decoded error is global/root trajectory: MPJPE is `0.187697 m`, root error is `0.163340 m`, root-centered MPJPE is `0.078462 m` (custom diagnostic, no rotation/scale fit), and PA-MPJPE is `0.089861 m`. Per-window MPJPE and root error correlate at `0.9738`. Mean root error grows from `0.0776 m` at frame 0 to `0.1033/0.1447/0.2170/0.3112 m` at frames 24/48/72/96 and is mostly horizontal. UniEgo hybrid decodes further localize the problem: predicted local features with GT global delta give `0.098156 m` MPJPE; GT local features with the predicted full global delta give `0.141005 m`; keeping all GT features except predicted delta translation gives `0.136229 m`; replacing only predicted delta rotation with GT barely changes full prediction (`0.185798 m`). Therefore global delta translation is a major contributor, but local body error also remains; this is not a pure rigid head-calibration problem.

For a direct camera-prior comparison, recomputation on the identical floor-valid66 windows used raw relative and accumulated camera actions rather than mixing the Phase-1 scale-normalized headline with raw Phase-3 errors. Phase-1 inverse dynamics gives `3.317 mm` raw per-step translation, `0.209 deg` rotation, raw trajectory mean/RMSE/endpoint `0.0728/0.0857/0.1492 m`, and Sim(3)-aligned ATE `0.02335 m`. Phase-3 V2M motion mapped through the train-global head calibration gives `6.146 mm`, `1.261 deg`, `0.1418/0.1727/0.3100 m`, and `0.03740 m`. GT motion through that global calibration already gives `4.519 mm`, `1.207 deg`, `0.1322/0.1618/0.2839 m`, but only `0.00328 m` Sim(3) ATE; the large raw-versus-Sim(3) gap reflects calibration/frame bias rather than trajectory-shape failure alone. The Phase-1 camera model is clearly the stronger metric-camera prior, but vanilla `video2motion` does not run or expose its frozen camera-action prediction path, so that prior cannot automatically correct UniEgo root motion. A head-camera auxiliary loss or an explicit joint `video -> camera + motion` objective is required to couple them.

The 2026-07-20 **relative-mapping** GT visualization makes that raw-versus-Sim(3) gap concrete. `visualize_gt_head_camera_alignment.py` uses the exact motion-clean71 windows and no model output, but it is not a direct overlay of the two stored world trajectories. For each window it decodes the floor-calibrated/canonicalized GT SOMA skeleton, forms the global-calibration camera pose `H_t X`, and applies one constant left world transform that makes the stored Phase-1 upright-RGB camera pose exactly equal at frame 0. All later measured camera poses remain unchanged. Its divergence therefore measures the production global Head-frame mapping after frame-0 pose alignment. Static plots and 97-frame/20-FPS MP4s show the skeleton, both trajectories, and both upright optical-camera `+Z` directions. Outputs are under `eval_full71_step200000_unipc30/gt_camera_vs_gt_motion_viz`; `summary.json` records all 71 rows and transform/action reproduction checks, `all71_summary.png` is the aggregate diagnostic, and `cases/` contains seven ranked PNGs plus worst-endpoint, worst-local-error, and median MP4s. All three MP4s are valid H.264 `1320x880`, 97-frame clips.

This visualization shows that the large accumulated GT floor is a coherent frame-direction effect amplified by locomotion, not 96 independent millimetre errors or a motion-scale failure. Across clean71, mean/median per-step translation error is `4.445/3.639 mm`, while mean/median frame-96 position gap is `0.276/0.176 m` (p90 `0.686 m`, max `1.687 m`). The observed sequence-window SOMA-Head-to-upright-camera rotation differs from the train-global `R_X` by mean/median `14.73/12.79 deg`; measured net displacement is mean/median `1.321/1.141 m`. The simple rotation-chord term `2*d*sin(theta/2)` correlates `0.875` with endpoint gap, compared with `0.820` for per-step translation, `0.654` for displacement alone, and `0.408` for frame rotation alone. The worst window moves `3.40 m`, has `26.2 deg` median frame error, predicts a `1.542 m` chord gap, and observes `1.687 m`; the second moves `2.67 m`, has `32.5 deg`, predicts `1.494 m`, and observes `1.519 m`. Yet clean71 Sim(3)-aligned trajectory RMSE is only `3.24 mm` mean (`2.92 mm` median), with fitted scale `1.0028` mean/`0.9998` median. Thus GT motion usually preserves trajectory shape and metric scale very closely, but a global head-camera frame is not accurate enough per sequence to express translation in Phase-1's local camera coordinates. A per-sequence rigid alignment can visually remove most of the raw path error, but it is unavailable at inference and must not be treated as a leakage-free solution.

While implementing the visualization, a separate absolute-orientation diagnostic bug was found and fixed in `calibration_sample_from_arrays`: directly decoding a nonzero-start UniEgo slice incorrectly treated its first relative `canon_delta` as a sequence-absolute transform. The utility now decodes the sequence prefix before selecting absolute head orientations, and `_verify_head_camera_alignment.py` has a synthetic nonzero-start regression. Relative actions, model training, existing head-camera losses, and the visualization trajectories were never affected because they canonicalize the window and use frames 1+ deltas. The saved train-global calibration was also effectively unaffected: 637/642 fit windows start at zero, and a full corrected recomputation changes its rotation by only `0.0117 deg` and lever arm by `0.052 mm`; relative fit metrics are unchanged. Keep the active calibration JSON unchanged for checkpoint compatibility. The test-actor oracle's final transforms were optimized from relative actions, so only its auxiliary absolute-frame-deviation diagnostic was exposed to this bug.

The 2026-07-21 source-level audit corrected an important interpretation. `audit_nymeria_camera_motion.py` independently decodes only the UniEgo Head and compares it directly with `camera_rgb` after the fixed Aria-Z-up to Kimodo-Y-up basis change, with no fitted calibration or frame-0 alignment. Across all 728 sequences that have camera sidecars, proportional motion/UniEgo/raw camera/upright camera timestamps match exactly, every sequence's step-lag search selects zero frames, median MPS sampling error is about `0.246 ms`, and the per-sequence median camera-origin-to-Head-joint distance is tightly centered at `0.139 m`. The direct position streams therefore normally do share a metric world frame. In the William clean71 worst mapping case, the measured camera, decoded Head trajectory, and `Head + global lever` trajectory overlap directly; the apparent `1.687 m` endpoint divergence appears only after the noisy global Head-to-camera orientation is forced to equal the measured camera pose at frame 0. Corrected plots are under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/plots`.

The corrected direct visualization set in that `plots/` directory includes `heldout_motion_clean71_direct_head_camera.png` for all 71 selected windows and two H.264 `1320x880`/20-FPS animations: `direct_head_camera_william_start0.mp4` (97 frames) and `direct_head_camera_kevin_source_jump_1450_1600.mp4` (150 frames). Every view uses the stored upright RGB camera pose and independently decoded UniEgo SOMA Head pose after only the fixed world-basis change; there is no fitted transform and no frame-0 pose alignment. Blue/orange paths are camera origin/Head joint, arrows are each frame's own `+Z`, and the lower panels show direct separation and translation-step continuity. William stays at `0.133 m` median/`0.140 m` maximum separation with maximum camera/Head steps `0.070/0.060 m`; Kevin reaches `1.003 m` separation and `1.644/0.956 m` steps. `direct_head_camera_manifest.json` records the contract, all 71 per-window metrics, and animation paths.

The same audit independently reproduces the camera preprocessor, ruling out a hidden device-to-RGB/action transform bug: over all 728 valid sequences, worst raw-to-upright-RGB position/rotation differences are `1.53e-5 m`/`6.34e-8`, and worst stored relative-action translation/rotation reconstruction errors are `3.23e-5 m`/`0.03 deg` (float storage precision). Two tooling issues were fixed in `/home/jungbin_cho/nymeria_kimodo_pipeline/camera` without rewriting active data. `preprocess_camera_rgb.py` had the diagnostic `K_rgb_upright` principal-point mapping for the opposite image-rotation direction; the checked old sidecar differs from the correct Pillow-clockwise K by up to `16.3 px`. Intrinsics are never loaded by current training/evaluation and camera poses/actions were already correct. New sidecars carry `preprocess_version=2`, corrected K, continuity arrays, and version-aware freshness checks; existing sidecars remain legacy until an explicit rebuild. `extract_camera_all.py --motion-root` previously changed discovery but workers still used the default hard-coded root; root/output paths now propagate into every worker. Alternate-root, current-sidecar, VRS/K, and temporary full `camera_rgb` preprocessing smokes pass on node 2.

The broad non-rigidity is rotational and is already present in the source SMPL fit. Over 728 sequences, the per-sequence fixed SOMA-Head-to-camera relation has mean/median full-rotation residual `15.06/14.57 deg` and horizontal-heading residual mean/median `13.05/12.27 deg`. Even within non-overlapping T97 windows the sequence-level mean residual is `8.80 deg` full rotation and `7.81 deg` heading. On all 71 test sequences, the original SMPL Head has `14.53 deg` mean residual to camera, the converted SOMA Head has `14.42 deg`, and SMPL-to-SOMA Head differs by only `0.923 deg` mean. Thus the converter is not creating the main orientation problem; the fitted source Head orientation is not a rigid proxy for the worn RGB camera. Across 642 train sequences, sequence relations differ from a train-global relation by `10.58 deg` mean; grouping by named actor reduces sequence-to-actor-mean disagreement only to `6.57 deg`, so actor calibration cannot remove the remaining session/time/pose variation. The fitted translation lever is much more stable: train mean Head-frame offset is approximately `[-0.0133, 0.0629, 0.1236] m`, with a typical sequence-mean residual of only `4-5 mm` after excluding source discontinuities.

There is also a sparse but real source-data discontinuity problem that must not be conflated with the broad orientation issue. Forty-two of 728 camera-bearing sequences contain at least one interval where direct Head-camera separation exceeds `0.5 m`; 47 trip a conservative camera-step gate of at least `0.25 m` or `30 deg` in one 20-FPS frame. Split by component, only 12 sequences have >=`0.25 m` camera translation steps while 39 have >=`30 deg` rotations; similarly, 51 sequences have >=`0.25 m` Head translations but 227 trip the angular gate. The angular count can include genuine fast turns and should be treated as a conservative exclusion criterion, not proof that all 227 recordings have coordinate resets. The metre-scale translations are unambiguous: four train sequences contain `1.54-3.34 m` camera jumps with rotations up to `168.8 deg`; held-out `S17/20230918_s0_kevin_shaw_act2_5g4k0z` has a `1.644 m` camera jump and a simultaneous `0.935 m` raw-SMPL Head jump at transition `1519->1520`, despite MPS `quality_score=1.0` and sub-millisecond sampling error. This proves that at least the severe events are upstream source/registration behavior, not UniEgo decoding.

Exact T97 impact depends on the consumer. The unfiltered native Phase-1 index has 722/119,632 train windows (`0.6035%`) in the conservative camera-or-motion union, but only 95 (`0.0794%`) actually trip the camera gate relevant to its camera targets: 40 translation and 74 rotation windows, with overlap. The motion Phase-2/3 dataset first removes the existing floor-calibration drop list; its remaining aligned population has 539/112,937 affected train windows (`0.4773%`): 89 camera-gate windows (38 translation, 70 rotation), 457 Head-gate windows (70 translation, 452 rotation), 84 direct cross-modal step disagreements, and 93 >0.5 m separation windows. Only 120/112,937 (`0.1063%`) hit the stricter union of translation jumps, cross-modal translation disagreement, or >0.5 m separation; the other 419 are rotation-gate-only. Its corresponding conservative test count is 53/11,938 (`0.4440%`), of which nine are strict translation/separation cases. None of the selected motion-clean71 or replacement-five evaluation windows intersects these masks, so their reported root divergence remains an orientation/representation result rather than a catastrophic-jump artifact. Canonical artifacts are `summary_all.json`, `details_all.jsonl`, `training_window_impact_T97.json`, and complete unfiltered/Phase-2/3-filtered affected-window JSONLs under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/`. Existing floor filtering does not remove every source jump; future training should additionally exclude the listed rows through a versioned manifest/cache. Do not mutate old checkpoints or cached data in place.

The 2026-07-21 ranked clean-window visualization adds a stricter warning: even floor filtering plus the 539-window conservative source mask does **not** make every position trajectory reliable. `visualize_ranked_head_camera_windows.py` reconstructed all 112,937 Phase-2/3 train T97 rows, removed the conservative union, and ranked the remaining 112,398 rows. Within-window fixed Head-to-camera SO(3) residual has mean/median/p90/p95/p99/max `8.70/8.14/15.41/17.94/23.52/48.26 deg`. A separate no-fit translation score computes `RMSE((p_camera[t]-p_camera[0])-(p_head[t]-p_head[0]))` directly in the shared world frame: mean/median/p90/p95/p99/max is `7.01/5.73/14.79/17.39/21.28/45.59 cm`. This raw score is not expected to be zero even for rigid data because the physical camera lever arm rotates around the Head, but its worst cases expose smooth source drift that the single-step and 0.5 m separation gates miss. The worst is `S09/20230620_s0_marie_vasquez_act4_dhbf58@7759`: raw RMSE/endpoint/max `45.6/56.5/57.0 cm`, mean step disagreement `10.6 mm`, and fitted-lever residual mean `93.6 mm`, despite direct separation remaining below the existing 0.5 m gate. The worst rotational case is `S04/20230711_s0_frederick_young_act1_2imlal@12779`, with `48.26 deg` within-window residual. Best visual cohorts require at least 0.5 m raw camera path so static clips cannot win trivially. Outputs are under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/ranked_clean_train_T97`: 80 unique-window detailed PNGs, four 20-case contact sheets, four H.264 2000x1400/20-FPS/97-frame montages, and `summary.json`. Blue camera and orange Head paths are always raw; the green fitted-lever path appears only as a separate detailed diagnostic. The active dataset still applies only its historical floor drop list, not this conservative source mask or a new smooth-drift mask.

Filtered native Phase-1 follow-up (2026-07-21): `native_phase_training/build_camera_motion_quality_filter.py` reconstructs the exact usable+captioned+cached T97 Phase-1 population and writes a strict physical-window exclusion artifact. The pinned file is `/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json`, SHA-256 `1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848`. It excludes hard camera/Head/cross-modal jumps, >0.5 m direct separation, >25 degree mean residual after a best fixed per-window Head-to-camera rotation, or >5 cm RMS residual after a best fixed Head-frame lever. The 25-degree rotation cutoff is just above train p99 `24.68`; the 5 cm lever cutoff is over four times train p99 `11.49 mm`. The separate 30-degree one-frame angular gates are conservative and can include genuine fast turns; they are not proof of source resets. Train drops `1,583/119,632` rows (`1.323%`, `1,524` unique) and retains `118,049`; test drops `113/12,613` (`0.896%`) and retains `12,500`. Duplicate captions for one physical window are all removed. Global-extrinsic disagreement, raw relative-path RMSE, and the motion floor blacklist are explicitly not filter criteria. Full thresholds, overlap counts, nearby-cutoff sensitivity, per-subject rates, source hashes, and exclusions are embedded in the artifact and documented in `native_phase_training/AUDIT.md`.

`latent_nymeria_dataset.py` now validates filter kind/version/T/split/duplicates/summary counts and logs exact per-reason row removal. `sbatch_phase1_native_camera.sh` requires a pinned SHA for any filtered launch. The two controlled launchers are `sbatch_phase1_native_camera_qfilter.sh` (original `40/25/20/15` four-task mix) and `sbatch_phase1_native_camera_qfilter_no_i2v.sh` (drops only I2V, leaving raw `40/25/20`, effectively `47.06/29.41/23.53%`). Both remain 100k fixed-four, global batch 32, LoRA LR `5e-5`, action-head LR 4x, action loss 10, native RF, PowerEMA, save every 5k, and official shift-3 UniPC inference. Auto-eval is disabled for these launches to avoid forty uncontrolled one-GPU submissions while multi-node jobs are pending; evaluate selected checkpoints manually. Acceptance smokes under `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_qfilter_20260721` exercised every active training stream, saved and reloaded both EMA DCPs, sampled all four inference modes for both with NVIDIA's official entrypoint, and validated eight `256x256`/97-frame/20-FPS outputs, `96x9` action arrays, and two complete four-mode visualization manifests.

Filtered Phase-1 production launched 2026-07-21 as job `3017` (`np1qf4`, node 1, four tasks) and job `3018` (`np1qf3`, node 0, no I2V). Run directories end in `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1` and `_qfilterv1_noi2v`; logs are `slurm-np1qf4-3017.out` and `slurm-np1qf3-3018.out`. Launch evidence confirms eight ranks, the exact filter SHA/counts, intended stream ratios, and finite updates for both. Old jobs `3011`, `3010`, and `3003` were canceled one node at a time around the start of other-user job `3016`; `3003` was kept until `3017` completed all-rank model/filter initialization, preserving the requested queue fairness for pending two-node job `3014`.

The four-task filtered LR ablation is `sbatch_phase1_native_camera_qfilter_lr1e5.sh`, job `3021`, run name `native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1`. Its only scientific change from job `3017` is optimizer base/LoRA LR `1e-5` instead of `5e-5`; action-head multiplier remains 4x (`4e-5` effective), and filter/tasks/batching/EMA/native RF/scheduler/100k-save contract stay fixed. It was submitted with `afterany:3016` because 21 foreign SSH processes occupied every GPU on the sole Slurm-idle node; do not clear that dependency until all eight GPUs pass the existing 132-GB-free preflight.

Native Phase-1 video-quality ablations A-D were implemented on 2026-07-21. They all use the pinned qfilterv1 population, whole-token `C -> A person`, global batch 32, 100k horizon, save every 5k, base/LoRA LR `5e-5`, action modules at 4x, PowerEMA, action loss weight 2, and per-sample active-suffix normalization. Exact cached Wan causal boundaries are RGB `[1,9,17,33,49]` -> latent `[1,3,5,9,13]`; do not silently change these to the originally requested `[1,8,16,32,48]`, because those cut through causal four-frame latent groups and require either target-frame leakage or dropped clean frames. Visual RF noising and MSE indexes cover only the suffix; camera sequence offsets remain unchanged.

A is prefix-1 global generation Q/K/V/O LoRA with all four tasks. B is variable-prefix global LoRA with all four. C is variable-prefix action-interface-only. D is variable-prefix camera-token-only generation K/V LoRA; optional E is its prefix-1 factorization. C/D/E explicitly drop I2V training: native dummy action loss prevents a backward/FSDP crash, but I2V video loss has no trainable path in C and D's camera mask is empty, so those would be zero-update optimizer/scheduler/EMA steps. Their active `40/25/20` streams normalize to `47.06/29.41/23.53%`; I2V remains a mandatory frozen-prior evaluation at every prefix. Config import fails if action-only/camera-KV mode leaves I2V active.

`native_phase_training/camera_token_lora.py` preserves NVIDIA LoRA/DCP key names and masks the K/V residual to finalized packed action-token rows; text/video rows use exactly frozen base projections and zero-initialized B preserves the base output. `prefix_inference.py` is a narrow official-inference shim that validates contiguous explicit latent prefixes for forward/policy/I2V. `evaluate_prefix_suite.py` reports suffix-only PSNR/SSIM/LPIPS at full and relative early/middle/late horizons, inverse camera metrics, and policy full/suffix-reanchored camera metrics. Each 10k checkpoint evaluation uses official EMA UniPC (action 30/guidance 1/shift 3; I2V 35/guidance 6/shift 3), five held-out sources x five prefixes, all pair videos and GT/five-prefix grids, and writes `COMPLETE.json` only after metrics and visualization succeed.

Native Phase-1 pair/grid videos encode frame provenance directly. Green borders and `GT REFERENCE`/`GT CONDITION` headers identify reference or clean conditioning frames; red borders and `GENERATED` identify sampled suffix frames. The switch occurs at RGB frame `prefix_length` (prefix 9 means GT frames 0-8, generated frames 9-96). `viz/manifest.json` stores the same zero-based inclusive ranges under `video_frame_provenance`; do not remove this marking when changing visualization code.

Prefix-evaluation JSONLs retain `source_name`, `rgb_prefix_length`, and `latent_prefix_length` for local grouping, but NVIDIA's Pydantic inference schema forbids those extra fields. `sanitize_prefix_inference_inputs.py` validates and strips only those three fields into each output's `inference_inputs/` directory before the official CLI runs; `condition_frame_indexes_vision` remains intact. Visualization and metrics must continue to read the original enriched JSONLs. Passing the enriched files directly to `cosmos_framework.scripts.inference` fails before sampling.

Launchers are `sbatch_phase1_video_quality_{A,B,C,D,E}.sh`. Production jobs are A `3025`, B `3026`, C `3027`, and D `3028`; E was not submitted. A showed finite updates at about 1.0-1.1 s/step after launch. B-D perform a two-step eight-GPU DCP train/save/restart preflight on first allocation, then exact-resume into 100k. Status is dynamic; inspect `squeue`, the Slurm logs, and run-local completion markers before reporting it.

On 2026-07-22, A/B/C 10k and 20k evaluations were recovered directly on spare node-3 GPUs because Slurm copies `3029`-`3034` remained pending. Every one of the six run-local outputs has 80 successful official EMA-UniPC samples, 95 visualization records, all fixed-prefix video/camera metrics, and `COMPLETE.json` under `checkpoint_evals/iter_000010000` or `iter_000020000`. The redundant queued jobs were canceled only after those artifacts were validated; jobs `3025`-`3028` were not canceled. This real run exposed and verified the inference-schema sanitizer described above.

A/30k and C/30k were completed the same way on 2026-07-22, after which pending copies `3035/3036` were canceled. Both `iter_000030000` evaluation directories have 80 successful samples, 95 marked visualization records, all five-prefix metrics, and `COMPLETE.json`; jobs `3025`-`3028` were left unchanged. A/30k inverse means are `0.4047 deg`, direction cosine `0.8935`, translation error `6.764 mm`, and ATE `4.102 cm`; C/30k is `0.6466 deg`, `0.7533`, `14.628 mm`, and `11.543 cm`. These are fixed five-source diagnostics. B had no completed 30k DCP at that observation time, so no B/30k result should be inferred.

The final artifact is `eval_full71_step200000_unipc30/head_camera_calibration_test_actor_motion_clean71_joint_oracle.json` under the vanilla Phase-3 run. `backfill_oracle_actor_head_metrics.py` reproduced the pre-existing global metrics before writing (maximum discrepancy `0.00000004 m / 0.000156 deg`), appended actor IDs and four oracle keys to saved rows without resampling, and updated all71, floor-valid66, replacement5, motion-clean71, and summary JSON. Provenance records overlap as 66/71 for the original all71 set, 66/66 for floor-valid, 5/5 for replacements, and 71/71 for motion-clean71. `eval_all.py --oracle_test_actor_calibration PATH` supports the same fields in future evaluations without changing sampling or model inputs. The discarded attempts to fit actor orientation from unaligned full-sequence frames or from absolute orientation averages were rejected because they produced 14-degree or larger inconsistencies; only aligned windows and the joint relative-action objective are retained in code/results.

The isolated launcher is `sbatch_phase3_bridge_native_v2m_m2v_headcam.sh`. It keeps the exact baseline Phase-1 100k EMA and Phase-2 200k specialists, 50/50 V2M/M2V mixture, T97, global batch 32, 12 local bridges, LR/schedule, save cadence, and required two-per-direction visualization. The only learning-path changes are the derived M2V camera condition and V2M auxiliary losses; both motion and video checkpoint sampling use their native shift-3 UniPC solvers.

Verification on 2026-07-18 used free H200 node 0. Synthetic geometry, gradient, task-plan leakage, collate separation, native bridge locality, Python compile, launcher syntax, and diff checks passed. An exact-checkpoint one-GPU smoke loaded all 293 Phase-1 generator tensors and 149 Phase-2 motion tensors with zero missing/shape mismatches. V2M produced finite loss 0.1521, nonzero bridge gradients, zero frozen-specialist gradient leakage, and initial head-camera error 1.4 cm/2.48 degrees; the weighted auxiliary contribution was about 0.014, so it did not dominate. M2V produced finite loss 0.1404, nonzero bridge gradients, zero leakage, and its clean motion-derived camera condition was 1 mm/0.36 degrees from synchronized GT for that sample. The full-path smoke at `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_headcam_full_20260718` performed an optimizer step, wrote a 96-tensor checkpoint plus optimizer/TensorBoard state, and produced valid 97-frame V2M and M2V media. A separate reload smoke at `smoke_phase3_headcam_reload_20260718` restored all 96 tensors and optimizer state with zero mismatches, then reran both UniPC samplers, VAE decode, motion/video rendering, and manifests successfully.

Production job `3003` (`p3brheadcam`) was submitted on 2026-07-18 and started on `a3ultravis-a3ultranodeset-0`. Its 8-rank startup loaded all specialist tensors cleanly, built the required two-V2M/two-M2V held-out visualization set, and advanced through step 220 with finite losses, no skipped optimizer updates, nonzero bridge gates, about 1.94 seconds/iteration after startup, and 122.6 GB peak allocated memory (well below the 143 GB H200 limit). Its run-local TensorBoard event and config were also created with the expected alignment and solver fields. Run directory: `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam`. Slurm log: `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brheadcam-3003.out`. This status is dynamic; inspect `squeue -j 3003`, `sacct -j 3003`, and the log before claiming current progress or success.

Step-10k paired evaluation completed on 2026-07-19. Both checkpoints use the exact same T97, seed-0, CFG-1, native shift-3 UniPC-30, full-71 plus deterministic replacement-5 contract. Outputs are `eval_full71_step010000_unipc30` and `eval_motion_clean_replacement5_step010000_unipc30` under each run. All 304 MP4s per run pass `ffprobe`; each run also has 152 finite `[97,283]` motion arrays and 76 finite `[48,25,16,16]` generated-latent files. The paired report is `compare_step010000_vs_baseline_step010000.json` in the headcam run.

On motion-clean71, headcam versus baseline changes V2M MPJPE `0.24662 -> 0.23522 m` (4.62% better; 38/71 wins), PA-MPJPE `0.13585 -> 0.12090 m` (11.00%; 55/71), feature MSE `0.48969 -> 0.43709` (10.74%; 49/71), acceleration error `0.006723 -> 0.006813` (1.35% worse; 34/71), and root error `0.21677 -> 0.20701 m` (4.50%; 35/71). M2V changes PSNR `15.3880 -> 16.4445 dB`, SSIM `0.48695 -> 0.51738`, and LPIPS `0.45467 -> 0.40057`; headcam wins 62/71, 62/71, and 60/71 rows. Relative head-camera translation improves `8.740 -> 7.286 mm` (16.64%; 60/71), and rotation improves `1.515 -> 1.014 degrees` (33.04%; 71/71). All-71 and floor-valid66 show the same direction on every metric except acceleration error. This supports the alignment intervention at 10k, especially for rotation and M2V, while showing that decoded root/MPJPE gains are modest and not universal.

Operational provenance: the first SSH full-71 attempt was terminated when another user's Slurm allocation populated node 2. Exclusive Slurm job `3004` safely reran both full sets and completed sampling/visualization, but exited during postprocessing because the launcher called system Python without NumPy. A second merger bug placed `_video_payload`'s return after another function's return. Both launcher and control-flow bugs are fixed and helper-tested; the completed outputs were merged successfully without resampling. Do not run SSH evaluators on occupied training nodes.

The latest comparison was pinned at request time to headcam 30k. Exclusive six-GPU job `3008` (`p3hdcmp`) runs the same full-71/replacement-5 protocol through `sbatch_compare_phase3_headcam.sh` for three checkpoints: headcam 30k, baseline 200k (literal latest-to-latest), and baseline 30k (matched training budget). It was pending for resources on 2026-07-19. It will write `compare_headcam_step030000_vs_baseline_step200000.json` and `compare_step030000_vs_baseline_step030000.json` in the headcam run. Superseded pending job `3007` was canceled before start. Treat this status as dynamic and verify Slurm/output markers before reporting 30k results.

The corrected causal 4x locality for a 97-frame clip and 25 Wan-VAE latent frames is `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`. In code, motion frame `m` maps to latent frame `ceil(m/4)`, implemented as `(m+3)//4`. The old rule `[4*g, 4*g+3]` was wrong: it assigned source frames 1..3 to latent 0 and omitted source frames 94..96. `_verify_native_bridge_contracts.py` asserts every one of the 97 motion frames has exactly one bridge-local latent.

Bridge risk: if the frozen motion expert only learned text/text-image-to-motion and never learned to interpret generator/video features, the bridge must translate generator hidden states into useful motion-expert activations through residual edits alone. This may be enough, but should be evaluated as a capacity/alignment question rather than assumed. Monitor bridge gate values, bridge output norms, `video2motion` reconstruction, and `motimg2video` qualitative control. Because generator weights stay frozen in this hypothesis, Phase-1 camera/video metrics should be checked for non-regression.

Sampler compatibility for a native Phase-1 generator plus frozen motion expert:

- Native Phase 1 and native-schedule Phase 2 share a normalized shifted RF noise coordinate and 1000-step scale, but intentionally do not share the training-time base distribution or prediction target. Generator targets use Waver clean-time, convert to noise-time, apply resolution shift 3, and predict velocity. Motion targets use logit-normal noise-time, apply shift 3, and predict clean x0. New motion runs convert x0 to velocity at each step and use the same official UniPC scheduler class as the generator; historical job `2844` and its derived Phase-3 run recorded Euler and remain reproducible. Historical Phase-2 checkpoints without `motion_schedule=native` still use the legacy unshifted contract.
- The bridge should not force these into one global diffusion process. Treat each bridge task as a target-modality task. For `video2motion`, video is clean conditioning and motion is the noised target, so use the motion expert's motion `x0` noiser/sampler. For `motimg2video`, motion is clean conditioning and video is the noised target, so use the native Cosmos generator RF noiser/sampler.
- This is acceptable because the bridge aligns hidden states, not raw sampler states. It is wrong to train bridge video targets with the old custom `motion_expert_joint_attention` video sampler and then evaluate with official Cosmos inference. For the experimental joint-target objectives, sample motion and generator training times independently from their native marginals and pass both into the model; do not force one sampled scalar on both targets.
- `train.py`/`sample.py` now checkpoint independent `motion_schedule` and `gen_schedule` contracts. `gen_schedule=native` uses the exact CPU float32 Waver transform plus shift during training and `FlowUniPCMultistepScheduler` during video sampling; `legacy` preserves uniform-time Euler behavior for old checkpoints. For standalone Phase-1 visual-quality evaluation, the authoritative path remains `cosmos_framework.scripts.inference` through `native_phase_training/inference_config.py`; the joint sampler is required once bridge layers participate.

The 2026-07-11 Phase-2 run implements the earlier proposed inference-coordinate unification: native action-style logit-normal sigma with x0 prediction. Joint-target bridge training still uses target-modality training distributions rather than forcing simultaneously noised video and motion into one sampled scalar process. At inference, both native specialists can share one shift-3/1000 UniPC state because their inference noise coordinate is the same. The coupled sampler makes one model call per UniPC evaluation, passes the common quantized model timestep to both specialists, and converts motion x0 to velocity with the scheduler's actual sigma (which need not equal the quantized model timestep divided by 1000). Video-style `waver` for motion remains an unimplemented ablation.

The old bridge run/launcher (`ja_phase3_bridge_v2m_m2v_from_t2m3d_p1cam`, `sbatch_phase3_bridge_v2m_m2v.sh`, historical job `2774`) is retained only for provenance. It used historical custom specialists/schedules and the incorrect local frame map; do not resume it into the native experiment.

Active native bridge launcher: `motion_expert_joint_attention/sbatch_phase3_bridge_native_v2m_m2v.sh`. It uses 50/50 V2M/M2V, `T=97`, batch 4/GPU on 8 GPUs, 200k steps, LR `2e-4`, 1k warmup, cosine decay, and saves every 5k. The Slurm request is 120 hours: the measured historical bridge rate of about 1.738 s/step requires about 96.6 hours for optimizer steps alone, so the former 96-hour request could not cover checkpoint I/O and required visualization. It loads the final Phase-1 100k **EMA** from native DCP through `checkpoint_utils.py`, recreating LoRA rank 16 / alpha 32; using the old alpha-16 default would halve the trained adapter residual despite a shape-clean load. It loads the explicit Phase-2 200k `.pt`, freezes both specialists, and trains 805.40M bridge parameters. Generator-facing samples use native `[BOS,text,EOS,SOG]` reasoner packing, FPS-modulated float 3D-mRoPE at source FPS 20, and temporal modality margin 15000; this is required in addition to matching Waver/shift-3/UniPC. Mixed-task loss is normalized by local batch size: motion aggregate is scaled by the number of motion-target samples over `B`, while each vision/camera sample contributes its loss divided by `B`. The previous implementation summed per-sample vision/camera losses but used one aggregate motion mean, so effective task weight changed with batch composition.

Contact-Phase-2 vanilla bridge ablation (submitted 2026-07-20): `motion_expert_joint_attention/sbatch_phase3_bridge_native_v2m_m2v_contactp2.sh` changes only the frozen Phase-2 specialist and its matched V2M loss recipe relative to the naive native bridge. It initializes motion from the completed contact-aware checkpoint `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt`. The historical non-contact Phase 2 and its vanilla Phase 3 both used motion feature/joint/smooth weights `1/10/50` with no contact terms; this new pair consistently uses `1/1/5`, contact BCE `0.05`, GT-contact-masked foot velocity `1`, foot height `10`, contact-logit scale `2`, and 20 FPS. These losses apply only to V2M because M2V carries clean condition motion and has no motion target. Everything else remains the naive baseline: Phase-1 100k EMA, 50/50 V2M/M2V, no head-camera supervision, no joint-target tasks, frozen specialists, 12 local bridges, T97, global batch 32, LR `2e-4`, 1k warmup, cosine decay, 200k steps, and required two-per-direction visualization every 5k. Motion and video visualization use native shift-3 UniPC-30. Focused native-bridge, native-motion-flow/UniPC, and four contact-loss contracts passed before submission. Production job `3011` (`p3brp2ct`) requests one exclusive 8-GPU node for five days; its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k` and log is `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brp2ct-3011.out`. At submission it was pending for resources; verify dynamic state with `squeue -j 3011` and `sacct -j 3011`.

Checkpoint visualization is required, not best-effort, for this launcher. Every 5k checkpoint and the final 200k state sample two fixed held-out V2M and two fixed held-out M2V windows with 30 native solver steps and guidance 1.0. V2M writes the clean source video, GT|predicted motion, and a combined comparison MP4. M2V writes GT/generated latent arrays, separate decoded MP4s, and a GT|generated comparison. `--require_viz` synchronizes setup/runtime failures across ranks and terminates the run coherently rather than silently training without results. `--viz_only` exists as a no-training/no-checkpoint verification seam.

Pre-submit verification on node 3 used actual Phase-1 90k EMA and Phase-2 190k weights. All 293 generator tensors and 149 motion tensors loaded with zero missing/shape mismatches. One real V2M and one real M2V forward/backward passed finite-loss and frozen-gradient checks. A metadata-checkpoint reload then sampled one held-out V2M case with native x0/Euler and one M2V case with native velocity/UniPC and wrote motion arrays/metrics, generated video latents, and `summary.json`. A second built-in smoke on node 2 repeated the two real forward/backward paths while Phase 2 was running: V2M loss `0.1500`, M2V loss `0.1581`, nonzero bridge gradients, and zero frozen-specialist gradient leakage. Final-pair verification then used Phase-1 100k EMA plus Phase-2 200k: 293/293 generator and 149/149 motion tensors loaded cleanly; V2M loss was `0.3095`, M2V loss was `0.1846`, bridge gradients were nonzero, and frozen-gradient checks passed. A separate `--viz_only` run sampled one held-out example per direction and produced valid V2M source/GT-prediction comparison media plus M2V GT/generated/comparison media under `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_bridge_viz_exact_p1ema100k_p2native200k/viz_step000000`. These checks prove wiring, loading, noising, gradient routing, both sampler executions, VAE decoding, and media writing; they do **not** prove bridge quality. Residual risks include the Phase-2 expert's changed reasoner context inside Phase 3, prompt-format differences between native Phase 1 and joint Phase 3, UniEgo's incomplete camera/world placement information for M2V, and the large bridge capacity/zero-gate optimization dynamics.

Phase-3 joint-target multitask ablation (implemented 2026-07-19): `motion_expert_joint_attention/sbatch_phase3_bridge_native_multitask.sh` is the third directly comparable bridge run after (1) vanilla V2M/M2V and (2) the head-camera-supervision variant. It uses the same Phase-1 100k EMA, Phase-2 200k native checkpoint, frozen-specialist topology, 12 local bridges, T97, batch 4/GPU, global batch 32, LR `2e-4`, 1k warmup, cosine decay, 200k steps, and checkpoint/required-visualization interval 5k. It deliberately does **not** enable head-camera losses, so its intervention is the two added objectives rather than a combination of the second and third variants. The production mixture is 35% `video2motion`, 35% `motimg2video`, 15% `video2camera_motion`, and 15% `camimg2video_motion`. Each joint task's two target branches have relative weight 0.5. The resulting expected branch mass is motion 0.50, video 0.425, camera 0.075, with total loss budget 1/sample; this preserves the vanilla motion exposure while retaining 70% direct corner-task sampling.

The two joint objectives are `video -> camera + motion` and `camera + frame-0 image -> future video + motion`, both without text. During training, motion uses its shifted logit-normal x0 marginal and generator targets use their shifted Waver velocity marginal, sampled independently. During joint inference, `sample.py` uses one coupled shift-3 UniPC state and one forward call per solver evaluation. Bridge masking is derived from clean/noisy token roles rather than hard-coded task names: only target rows receive cross-modal residuals; both target modalities can update in joint tasks; the original V2M/M2V one-sided masks remain unchanged. Joint inverse visualization writes source video, GT/pred motion, GT/pred camera arrays, and a camera trajectory plot. Joint forward visualization writes GT/generated video plus GT/pred motion. With `--viz_n 8`, each checkpoint gets two held-out records for each of the four active modes.

Pre-submit verification for this ablation used the exact final specialists. CPU contract tests `_verify_phase3_multitask_contracts.py` and `_verify_native_bridge_contracts.py` passed. Real T33 and T97 one-GPU backward smokes exercised all four tasks with finite losses, nonzero bridge gate/core gradients after gate opening, and zero frozen-specialist gradient leakage. A two-step train/save/reload run restored all 96 bridge tensors plus optimizer state exactly and executed all four native samplers, VAE decoding, motion rendering, camera plotting, and manifest writing; all arrays were finite and all MP4 frame counts were valid. A two-rank ordinary-training smoke on node 1 GPUs 2/7 completed four mixed-task steps at 51.0 GB peak per added process with no skipped updates or collective hangs; `COSMOS_DDP_PARAM_CHECK=1` reported bit-identical replicated parameters after every optimizer step. Smoke output is `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_multitask_ddp2_t33_20260719`. These checks establish execution and serialization contracts, not that the added objectives improve held-out quality.

Production job `3010` (`p3brmulti`) was submitted on 2026-07-19 with no dependency. It requests one exclusive 8-GPU `a3ultra` node, 64 CPUs, and five days. Its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask`, and its Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brmulti-3010.out`. At submission it was pending for `Priority`; the log is not created until allocation. The launcher fails before model construction if any allocated GPU already uses more than 2 GiB, guarding against SSH processes that bypass Slurm exclusivity. Treat job state as dynamic and inspect `squeue -j 3010`, `sacct -j 3010`, and the log before reporting progress.

Interpret the ablation cautiously. Joint-target success would demonstrate that one self-attention bridge can express both directional and simultaneous cross-modal updates, strengthening the flexibility argument. It would not by itself prove self-attention is the best bridge architecture; that claim requires a matched-capacity bidirectional cross-attention baseline. `video -> camera + motion` may also admit two mostly independent predictions from the clean video, so improvement on that task alone is weak evidence that the generated camera and motion communicate with each other. The independent off-diagonal training times and the joint forward objective apply stronger pressure to use cross-target context, but whether that pressure improves ordinary V2M/M2V remains an empirical question. Evaluate all three variants with identical held-out windows, native UniPC-30 overrides, seeds, and matched training steps.

Native Phase-3 production job `2865` (`p3brnative`) ran normally through step 5,000 on 2026-07-14, then failed during the required checkpoint visualization. Training itself was healthy: finite losses, no skipped updates, about 1.811 s/step, and peak allocated GPU memory about 117.5 GB. It successfully wrote both `ckpt_step005000.pt` and `latest.pt` under `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k`; a metadata/mmap load confirmed step 5,000, 96 model tensors, optimizer state, and the expected V2M/M2V task list. The partial `viz_step005000` directory is **not** a complete evaluation: it contains the first V2M motion artifacts only and must not be treated as a successful four-sample visualization.

The job-2865 failure was a distributed collective mismatch, not OOM, bad training data, timeout, or checkpoint corruption. Only rank 0 runs `do_viz`, but `precompute_latents.load_vae` constructed `Wan2pt2VAEInterface`, whose `_video_vae` unconditionally called the framework's world-group `sync_model_states(model)`. Ranks 1-7 were already waiting in `run_checkpoint_viz`'s scalar status broadcast. The incompatible collective ordering made a nonzero/garbled status reach peer ranks; rank 2 raised the misleading `required checkpoint visualization failed ... on rank 0`, then torchrun terminated rank 0 before it could log its own traceback. `load_vae(..., rank_local=True)` now suppresses only that constructor synchronization for rank-0-only visualization and restores the framework function in `finally`; all ordinary VAE training/evaluation/precompute callers keep synchronized construction. The temporary visualization VAE is released and the CUDA cache cleared before rank 0 broadcasts status, so it is not retained during later optimizer steps. Visualization setup/barrier/status now use a separate CPU/Gloo group with a two-hour timeout because four 30-step samples may legitimately exceed the default 10-minute NCCL timeout; gradient synchronization remains on the normal NCCL group. A local contract test verified VAE-sync suppression/restoration, and Python syntax/diff checks passed.

The exact 8-rank fix gate is Slurm job `2869` (`p3brvizsmk`), launched by `sbatch_phase3_bridge_native_viz_ddp_smoke.sh`. It reloads the saved step-5,000 bridge plus the exact final specialists and must produce one held-out V2M and one held-out M2V visualization through `--viz_only`. Production resume job `2870` (`p3brnative`) has dependency `afterok:2869` and uses the regular launcher with `--resume auto`, so it will continue at step 5,001 only if the distributed visualization gate succeeds. At documentation time both are pending: `2869` for cluster priority/resources and `2870` for its intentional dependency. Logs are `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brvizsmk-2869.out` and `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brnative-2870.out`. Treat all later status as dynamic and check `squeue`, `sacct`, and the logs before making current-progress claims.

Representation risk: camera action is raw metric relative SE(3) delta (`pos(3)+rot6d(6)`, loss on `[:9]`) and is naturally action-like. Current motion is normalized 283-d UniEgo, with body-centric/canonicalized pose features and foot contacts; it is not the same kind of raw metric state as camera action. This representation is probably reasonable for `video2motion` because the target is UniEgo features. It may be more ambiguous for `motimg2video`, because a normalized/canonicalized body motion condition may not fully specify absolute camera-frame body placement or the body-camera relation. If `motimg2video` is weak, representation alignment should be considered before blaming only the bridge architecture. Possible future auxiliary conditions include camera/world-frame root or pelvis deltas, root velocity, or camera-relative body trajectory, but these are not implemented.

Main launch wrappers include `run.sh`, `sbatch_t2m_both.sh`, `sbatch_t2m_both_mrope3d.sh`, `sbatch_t2m_ti2m_native_mrope3d.sh`, `sbatch_phase3_7task.sh`, `sbatch_phase3_bridge_native_v2m_m2v_contactp2.sh`, `sbatch_phase3_bridge_native_multitask.sh`, `sbatch_precompute.sh`, `sbatch_precompute_T97.sh`, `run_eval.sh`, and `run_camera_eval.sh`.

## Visualization and Evaluation

Joint-attention motion renderer: `render_motion.py`, runs in `cosmos`, no Kimodo import. It renders GT left/blue and generated right/red, with root-tracking XZ viewport, floor grid, trajectory trail, SOMA-30 parents, skipped fingertip-end joints, Y-up/+Z-forward input, negated X to match Kimodo display, and mplot3d Z-up remap.

`render_viz.py` is a Kimodo-env alternative. `_autorender.sh` watches viz dirs.

Phase-1 camera eval/viz note: the existing `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/camera_eval` contains inverse-dynamics outputs and metrics. On 2026-07-06 an additional FD/policy qualitative eval for `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/ckpt_step200000.pt` was run on the T2M node into `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/fd_policy_camera_eval` with 8 test windows. It produced raw generated videos under `fd_out/` and `policy_out/`, and 16 GT|generated side-by-side mp4s under `viz/` (`*_fd.mp4`, `*_policy.mp4`). This was intentionally kept separate from the existing inverse-dynamics `camera_eval` directory to avoid overwriting old metadata.

Phase-3 held-out evaluation must pass `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json` to `eval_all.py --windows_json`. As of 2026-07-15, that argument constrains camera, motion, and video tasks in the listed order and requires all requested windows to resolve before evaluation starts. Older `eval_all.py` revisions forwarded the list only to `eval_camera`; V2M/M2V silently used the ordinary dataset index instead (an inspected historical output contained 71 windows from one S01 sequence). Do not label old V2M/M2V outputs from that path as the canonical 71-sequence benchmark. The corrected M2V evaluation excludes conditioned frame 0 and reports PSNR, SSIM, and LPIPS-Alex over frames 1-96 using the same GT resize/pad and metric implementation as native Phase-1 forward evaluation. It writes GT/generated comparison MP4s for every evaluated window. Corrected V2M evaluation writes GT/prediction motion MP4s. Both tasks report the canonical all-71 aggregate plus a 66-window aggregate excluding the five windows flagged by floor calibration; both are retained so benchmark coverage and motion-GT/conditioning-quality sensitivity remain explicit.

Phase-3 step-35k canonical evaluation completed on 2026-07-15 for `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step035000.pt`, using T97, 30 native steps, guidance 1, exact `full71_windows.json`, motion native shift-3 Euler/x0, and video native shift-3 UniPC/velocity. Output is `eval_full71_step035000_native30`. All 71 V2M and 71 M2V cases completed; motion/video arrays are finite and have expected shapes, and all 284 MP4s (142 primary comparisons plus M2V GT/generated component clips) pass `ffprobe`. All-71 V2M is MPJPE `0.26389 m`, PA-MPJPE `0.09835 m`, feature MSE `0.35335`, acceleration error `0.00622`, and root error `0.23721 m`. The floor-valid 66 subset is MPJPE `0.20526 m`, PA-MPJPE `0.09491 m`, feature MSE `0.32637`, acceleration error `0.00609`, and root error `0.17788 m`. All-71 M2V is PSNR `16.1292 dB`, SSIM `0.51260`, and LPIPS-Alex `0.42438`; the floor-valid 66 values are `16.1430 dB`, `0.51585`, and `0.42295`.

For a sequence-balanced motion-clean comparison, the five flagged rows were replaced deterministically by same-sequence held-out T97 windows that have cached video latents, are absent from the floor blacklist, satisfy old-stats `|z|max <= 10`, have at least ten contact frames, contact median within 5 cm of the calibrated floor, and minimum foot height at least -10 cm. The replacement-only evaluation is `eval_motion_clean_replacement5_step035000_native30`; it completed all five V2M/M2V cases and wrote 20 valid MP4s. Combined with the retained 66, `eval_full71_step035000_native30/motion_clean71_provenance.json` records the exact 71-row composition. Motion-clean71 V2M is MPJPE `0.21364 m`, PA-MPJPE `0.10098 m`, feature MSE `0.34501`, acceleration error `0.00637`, and root error `0.18647 m`; M2V is PSNR `16.1189 dB`, SSIM `0.51157`, and LPIPS-Alex `0.42318`. Preserve all three reports: all-71 exposes sensitivity to the original camera-oriented list, floor-valid66 avoids invalid motion GT without replacement, and motion-clean71 restores one valid row per held-out sequence. In V2M, text is disabled in both training and sampling; manifest captions shown by older evaluation renders were metadata labels only. New renders explicitly label them as metadata-only to avoid implying text conditioning.

Phase-3 step-85k canonical evaluation completed on 2026-07-16 with the identical step-35k protocol and deterministic windows/noise: T97, 30 steps, guidance 1, motion native shift-3 Euler/x0, and video native shift-3 UniPC/velocity. Outputs are `eval_full71_step085000_native30` plus `eval_motion_clean_replacement5_step085000_native30`; the merged clean71 metrics and provenance are stored in the full-eval directory. All 71 full rows and five replacements completed, all motion/video arrays are finite with expected shapes, and all 304 MP4s pass `ffprobe`. Compared with step 35k, motion-clean71 V2M changes from MPJPE `0.21364` to `0.20938 m`, PA-MPJPE `0.10098` to `0.10079 m`, feature MSE `0.34501` to `0.36916`, acceleration error `0.00637` to `0.00644`, and root error `0.18647` to `0.18304 m`: decoded pose/root errors improve modestly, while normalized feature and temporal errors worsen. Motion-clean71 M2V changes from PSNR `16.1189` to `16.3416 dB`, SSIM `0.51157` to `0.51311`, and LPIPS-Alex `0.42318` to `0.40071`, a clearer perceptual improvement. The same direction holds for all-71 and floor-valid66. Treat step 85k as better for M2V but mixed, not categorically better, for V2M.

Motion-UniPC correction on 2026-07-16: `eval_all.py` now exposes an inference-only `--motion_native_solver {euler,unipc}` override, forwards it through `sample.load_joint_model`, and records the resolved motion/generator schedules and solvers in `summary.json`. It also hard-errors when an explicit `--ckpt` path is missing; BASE smoke is permitted only when `--ckpt` is omitted. This fixes a dangerous prior behavior where a deleted checkpoint silently evaluated an untrained BASE model. The attempted step-35k UniPC directories `eval_full71_step035000_motion_unipc30` and `eval_motion_clean_replacement5_step035000_motion_unipc30` are invalid for exactly this reason: checkpoint retention had removed `ckpt_step035000.pt`, their logs say `no ckpt ... BASE model`, the static predictions have near-zero acceleration, and both directories are marked with `INVALID_EVALUATION.json`. Do not use those metrics or videos. The historical step-35k Euler evaluation remains valid because it ran while the checkpoint existed, but a genuine step-35k UniPC rerun requires restoring that checkpoint externally.

The valid step-85k V2M UniPC-30 rerun is `eval_full71_step085000_motion_unipc30`, with five clean replacements in `eval_motion_clean_replacement5_step085000_motion_unipc30`. It uses the same windows, initial noise, steps, and guidance as the Euler evaluation; all 71 predictions are finite `[97,283]` arrays and all 71 comparison MP4s pass `ffprobe`. All-71 UniPC metrics are MPJPE `0.26657 m`, PA-MPJPE `0.09955 m`, feature MSE `0.38859`, acceleration error `0.00642`, and root error `0.24030 m`. Floor-valid66 values are `0.20575`, `0.09658`, `0.36255`, `0.00629`, and `0.17918`; motion-clean71 values are `0.21382`, `0.10325`, `0.38616`, `0.00655`, and `0.18668`. Compared with step-85k Euler on motion-clean71, UniPC is respectively `2.12%`, `2.43%`, `4.61%`, `1.69%`, and `1.99%` worse; only 9/71 cases improve MPJPE and 2/71 improve feature MSE. The UniPC motions are dynamic (`accel_pred` mean `0.00536`, versus `0.00524` for Euler and about `2.9e-7` for the invalid BASE fallback), so this is a small solver-quality regression rather than static collapse. M2V does not invoke the motion sampler because motion is clean conditioning, so its previously reported metrics are unchanged.

Phase-3 step-110k canonical evaluation completed on 2026-07-17 in `eval_full71_step110000_unipc30`, with the five deterministic clean replacements in `eval_motion_clean_replacement5_step110000_unipc30`. Step 110k was the latest complete checkpoint when this evaluation was launched; step 115k was saved while it ran. The contract is T97, exact `full71_windows.json`, 30 steps, CFG 1, seed 0, native shift-3 x0/UniPC for V2M, and native shift-3 velocity/UniPC for M2V. All 142 motion arrays are finite `[97,283]`, all 71 video NPZ outputs are finite, and all 304 full-plus-replacement MP4s pass `ffprobe`.

Step-110k all-71 V2M is MPJPE `0.24743 m`, PA-MPJPE `0.09890 m`, feature MSE `0.38056`, acceleration error `0.00635`, and root error `0.22310 m`; floor-valid66 is `0.18831`, `0.09629`, `0.35434`, `0.00622`, and `0.16328`; motion-clean71 is `0.19644`, `0.10271`, `0.37741`, `0.00648`, and `0.17098`. Against the valid step-85k UniPC motion-clean71 result, these improve by `8.13%`, `0.52%`, `2.26%`, `1.13%`, and `8.41%`, respectively; per-case wins are 49/71 MPJPE, 39/71 PA-MPJPE, 49/71 feature MSE, 44/71 acceleration error, and 47/71 root error. Predicted acceleration remains dynamic (`0.00522` mean), so this is not the static BASE-fallback failure.

Step-110k all-71 M2V is PSNR `16.4682 dB`, SSIM `0.51868`, and LPIPS-Alex `0.39873`; floor-valid66 is `16.4969`, `0.52278`, and `0.39571`; motion-clean71 is `16.4863`, `0.51861`, and `0.39553`. Against step 85k on motion-clean71, PSNR improves `+0.1448 dB`, SSIM `+0.00550`, and LPIPS `-0.00518`, with per-case wins of 46/71, 48/71, and 49/71. Step 110k is therefore consistently better than step 85k on the aggregate V2M and M2V metrics under the corrected sampler, though PA-MPJPE improves only marginally.

Phase-3 steps 140k, 150k, and 160k were evaluated on 2026-07-18 with the exact step-110k contract: T97, exact `full71_windows.json`, seed 0, CFG 1, 30 steps, native shift-3 x0/UniPC for V2M, and native shift-3 velocity/UniPC for M2V. Their full outputs are `eval_full71_step{140000,150000,160000}_unipc30`; deterministic replacement outputs are `eval_motion_clean_replacement5_step{140000,150000,160000}_unipc30`. For every checkpoint, all 71 original rows plus five replacements completed, all 152 full-plus-replacement motion arrays are finite `[97,283]`, all 76 generated video-latent files are finite `[48,25,16,16]`, and all 304 MP4s are 97-frame files that pass `ffprobe`. Each summary records both resolved solvers as UniPC, all specialist loads had zero missing/shape-mismatched tensors, all 96 bridge tensors overlaid, and no BASE fallback occurred. `run_phase3_recent_eval.sh` is the repeatable serial launcher; `merge_phase3_clean71.py` combines floor-valid66 plus replacement5, updates provenance/summary files, and was regression-checked to reproduce the existing step-110k aggregates exactly.

Recent V2M results (lower is better):

| Step | Set | MPJPE m | PA-MPJPE m | feature MSE | accel error | root error m |
|---|---|---:|---:|---:|---:|---:|
| 140k | all71 | 0.24468 | 0.08915 | 0.35504 | 0.00616 | 0.21883 |
| 140k | floor-valid66 | 0.18381 | 0.08619 | 0.32646 | 0.00604 | 0.15741 |
| 140k | motion-clean71 | 0.19191 | 0.09227 | 0.35029 | 0.00629 | 0.16511 |
| 150k | all71 | 0.23895 | 0.09036 | 0.34215 | 0.00624 | 0.21461 |
| 150k | floor-valid66 | 0.17822 | 0.08744 | 0.31428 | 0.00612 | 0.15341 |
| 150k | motion-clean71 | 0.18458 | 0.09342 | 0.34000 | 0.00637 | 0.15906 |
| 160k | all71 | 0.25007 | 0.08786 | 0.34407 | 0.00621 | 0.22465 |
| 160k | floor-valid66 | 0.18879 | 0.08446 | 0.31592 | 0.00608 | 0.16298 |
| 160k | motion-clean71 | 0.19551 | 0.09032 | 0.34056 | 0.00632 | 0.16924 |

Recent M2V results (PSNR/SSIM higher is better; LPIPS lower is better):

| Step | Set | PSNR dB | SSIM | LPIPS-Alex |
|---|---|---:|---:|---:|
| 140k | all71 | 16.5244 | 0.51963 | 0.39651 |
| 140k | floor-valid66 | 16.5524 | 0.52368 | 0.39340 |
| 140k | motion-clean71 | 16.5482 | 0.51987 | 0.39287 |
| 150k | all71 | 16.4897 | 0.51880 | 0.39275 |
| 150k | floor-valid66 | 16.5106 | 0.52293 | 0.38954 |
| 150k | motion-clean71 | 16.4985 | 0.51890 | 0.38953 |
| 160k | all71 | 16.6157 | 0.52177 | 0.38835 |
| 160k | floor-valid66 | 16.6436 | 0.52602 | 0.38472 |
| 160k | motion-clean71 | 16.6188 | 0.52169 | 0.38460 |

There is no single best recent checkpoint. On the sequence-balanced motion-clean71 report, 150k is best for MPJPE (`0.18458 m`), feature MSE (`0.34000`), and root error (`0.15906 m`); 160k is best for PA-MPJPE (`0.09032 m`) and every M2V metric; 140k has the lowest acceleration error (`0.00629`). Against 110k, 150k lowers clean71 MPJPE by `6.04%`, feature MSE by `9.91%`, and root error by `6.98%`; 160k lowers PA-MPJPE by `12.06%` and improves M2V by `+0.1325 dB` PSNR, `+0.00308` SSIM, and `-0.01094` LPIPS. Mean predicted accelerations remain dynamic at approximately `0.00509`, `0.00521`, and `0.00515` for 140k/150k/160k, so none exhibits the invalid static BASE-fallback behavior.

`sample.py` provides standalone per-task samplers. `eval_all.py`, `eval_camera.py`, and `eval_motion_recon.py` are main evaluation helpers.

In-train Phase-2 visualization balances T2M and TI2M. T2M also balances Nymeria/BONES where quota permits. It saves full-frame generated and GT `.npy` arrays plus MP4s; TI2M additionally saves the 256x256 conditioning PNG and renders conditioning image | GT motion | generated motion. Rank 0 samples while all other DDP ranks wait at barriers. Rendering is best-effort and must not break training.

## Cosmos Reference Constraints

The project-relevant Cosmos 3 constraints are consolidated here:

- Cosmos has AR/reasoner and DM/generator subsequences.
- AR tokens use causal attention over AR only.
- DM tokens use full attention with keys/values from AR+DM.
- AR/reasoner is not updated from DM/generator tokens.
- Non-language modalities use projection heads plus modality embeddings.
- Vision generation uses Wan2.2 VAE latents; VAE is frozen.
- Action uses domain-aware projections, relative pose pseudo-actions for ego/effector pose, 6D rotations, and domain ids.
- Cosmos3-Nano has 36 layers, hidden 4096, 32 attention heads, 8 KV heads, head dim 128, FFN 12288.
- Generator training uses rectified-flow velocity objective and masks clean conditioning tokens out of loss.
- Generator pre/mid-training freezes the reasoner and updates generation-specific params.
- Cosmos mid-training action mixture includes forward dynamics, inverse dynamics, and policy.
- Mid-training uses action loss scale 10, FusedAdamW lr `1e-4`, wd 0.05, grad clip 1.0, LambdaLinear start factor 0.4/cycle 100k, and shift 3/5/10 for 256p/480p/720p.
- Robot policy post-training is a structural analog for adding a continuous action-like modality: fresh heads, 5x LR on new action params, lr `2e-4`, shifted schedule around 5 for action sampling. However, root 369-d motion is absolute kinematic motion, not Cosmos delta action.
- For action generation, default Cosmos sampling uses steps 50, guidance 1 for forward/inverse dynamics, shift 5; policy uses fewer steps and guidance 3. Audio/visual generation uses different CFG/shift.

Project-specific assessment: the old root 369-d motion finetune structurally matches Cosmos but diverges in LR, head LR multiplier, timestep/noise schedule, LoRA usage, constant LR, and no CFG text dropout.

## Environment and Launch Rules

Most real commands must run on cluster nodes. This checkout alone cannot import Cosmos unless the external framework/env are available.

Invariant `cosmos` preamble:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework
```

The working directory must be `/home/jungbin_cho/cosmos-framework` for Cosmos config paths such as `QWEN_JSON`. Invoke scripts in this repo by path or through wrappers.

Use `motion_expert_joint_attention/run.sh` for Python entrypoints in that directory. `run.sh` sets env, cwd, and `PYTHONPATH`.

Single-GPU `srun` needs `--ntasks=1`; otherwise `srun` may fan out tasks per CPU and OOM. Do not let `srun` and `torchrun` both fan out ranks.

Inference often needs `--no-guardrails` because nodes lack `uvx` and guardrail downloads fail.

Python stdout can be block-buffered in logs. Monitor real training via TensorBoard.

Runs/checkpoints usually live under `/weka/jungbin/cosmos_motion_ft_runs/<run_name>/`. Joint-attention runs usually write TensorBoard events directly under their run directory. Native Phase 1 now writes events under `${job.path_local}/tensorboard` unless `TB_LOG_DIR` is set. Older native job `2801` wrote to the shared fallback `/weka/jungbin/cosmos_motion_ft_runs/tensorboard`.

Generated logs, Slurm outputs, cached latents, `.npy/.npz` stats, mp4s, and checkpoints are artifacts. Do not edit them unless explicitly requested.

## Environments

- `cosmos`: Py 3.13 / torch cu128-ish environment for Cosmos-3 Nano, training, sampling, inference, and `cosmos_framework`.
- `kimodo`: Py 3.10 / torch 2.4 environment for Kimodo datasets, BONES export, and some decode/render workflows.
- `nymeria_plus`: projectaria/VRS preprocessing for camera extraction.

The `kimodo` and `cosmos` environments cannot share one process. Typical root pipeline is export in `kimodo`, train/sample in `cosmos`, decode/render in `kimodo` or with pure-torch ports.

## Verification

Use the smallest relevant verification after edits.

Joint-attention CPU checks:

```bash
python motion_expert_joint_attention/mot_joint_attention.py
python motion_expert_joint_attention/mot_joint_layer.py
```

Joint-attention GPU smoke:

```bash
bash motion_expert_joint_attention/run.sh motion_expert_joint_attention/smoke_7task.py --gen_lora --steps 3
```

Other diagnostics in `motion_expert_joint_attention/` include `_verify_*.py`, `_diag_*.py`, `compare_uniego_stats.py`, and objective/floor/load checks.

Root checks include `verify_export.py`, `verify_full_export.py`, and `verify_motion_decode.py`.

Native Phase 1 checks:

```bash
python -m unittest native_phase_training.test_contracts
python -m py_compile native_phase_training/*.py
bash -n native_phase_training/sbatch_phase1_native_camera.sh
```

TOML/config dryrun:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth \
BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp \
NYMERIA_RESOLUTION=256 \
bash native_phase_training/run_latent_train.sh --dryrun \
  trainer.max_iter=1 \
  checkpoint.save_iter=1000 \
  job.name=native_phase1_config_check \
  model.config.parallelism.data_parallel_replicate_degree=8 \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1
```

Full native Phase 1 verification after code changes means more than import success: run a GPU train smoke on a node, let the trainer save a DCP checkpoint, then load that exact checkpoint through `cosmos_framework.scripts.inference` using `--config-file native_phase_training/inference_config.py --sampler unipc --use-ema-weights` and sample `forward_dynamics`, `inverse_dynamics`, `policy`, and `image2video`. Verify separate output names, `status=success`, action shape `96 x 9` for action modes, and a 97-frame 256x256 MP4 at 20 fps for every mode; run `visualize_checkpoint.py` and verify its four-mode manifest.

Do not assume imports or training work on the login/head machine outside the cluster env.

## File Map

Root:

- `AGENTS_ALL.md`: canonical context for agents and the only high-level source of truth.
- `AGENTS.md`: Codex bootstrap pointer to `AGENTS_ALL.md`.
- `CLAUDE.md`: Claude bootstrap pointer to `AGENTS_ALL.md`.
- `README.md`: short pointer to `AGENTS_ALL.md`.
- `train_motion_ft.py`: root text-to-motion trainer.
- `sample_motion.py`: root sampler.
- `motion_decode.py`: 369-d decode/loss.
- `verify_visual_tower/`: local checks for the Cosmos Nano standalone Qwen3-VL visual tower and reasoner-image smoke path.

`native_phase_training/`:

- `README.md`: official-compatible Phase 1 runbook, smoke evidence, OOM postmortem, and launch/eval commands.
- `latent_omni_model.py`: native `OmniMoTModel` subclass with optional cached-video-latent training input.
- `latent_nymeria_dataset.py`: Nymeria cached-latent native action/video dataset.
- `build_camera_motion_quality_filter.py`: exact T97 camera/Head discontinuity and rigid-consistency filter builder.
- `prep_test_eval.py`: native held-out T97/256/shift-3 official-inference input builder.
- `visualize_checkpoint.py`: four-mode official-output visualizer and manifest writer.
- `checkpoint_eval_callback.py`, `sbatch_checkpoint_eval.sh`: post-save isolated four-mode evaluation submission and worker job.
- `test_contracts.py`: prompt, packing, and evaluation-contract CPU tests.
- `experiment.py`: Hydra experiment registration for `world_camera_nymeria_latent_nano`.
- `world_camera_nymeria_latent.toml`: production TOML for the native camera/video LoRA run.
- `run_latent_train.py`, `run_latent_train.sh`: training entrypoints. `run_latent_train.py` sets a run-local TensorBoard log dir.
- `inference_config.py`: official inference config shim.
- `sbatch_phase1_native_camera.sh`: production Slurm launcher with exclusive-node request, memory preflight, and compile disabled.
- `sbatch_phase1_native_camera_qfilter.sh`, `sbatch_phase1_native_camera_qfilter_no_i2v.sh`: pinned filtered four-task and no-I2V launchers.
- `sbatch_phase1_native_camera_qfilter_lr1e5.sh`: exact filtered four-task LR ablation with base/LoRA LR `1e-5` and unchanged 4x action-head multiplier.

`motion_expert_joint_attention/`:

- `README.md`: short pointer to `AGENTS_ALL.md` and active code read order.
- `DESIGN_7TASK.md`: redirect stub kept for code docstring compatibility; task truth lives in `task_plan.py` and this file.
- `mot_joint_attention.py`: two-call mask.
- `mot_joint_layer.py`: role weight routing and sparse-depth layer.
- `joint_motion_model.py`: packed assembly, sparse 3-way/2-way interleave, timestep pre-scale, freeze/toggle logic, sampling.
- `cosmos_loader.py`: builds/materializes/freeze-loads Cosmos, tokenization, layer handles. Only file that should touch framework net directly.
- `motion_heads.py`: motion I/O heads and local timestep embedder.
- `gen_heads.py`: video/image/camera encode/decode via frozen Cosmos net.
- `task_plan.py`: task specs, condition masks, loss specs.
- `flow.py`: noising, losses, samplers.
- `nymeria_joint_dataset.py`: 5-modality data loader and collate.
- `dataset.py`: 283-d BONES/Nymeria motion pair loader.
- `build_bones_pairs.py`: offline BONES pair export.
- `precompute_latents.py`: offline Wan-VAE latent precompute.
- `precompute_floor_calibration.py`: per-seq floor deltas and drop list.
- `decode_uniego_torch.py`, `uniego_layout.py`: 283-d decode/layout.
- `head_camera_alignment.py`, `estimate_head_camera_calibration.py`, `head_camera_calibration_train.json`: relative SOMA-Head to upright-camera mapping, reusable rigid fitter, train-only estimator, and production constants.
- `audit_nymeria_camera_motion.py`, `summarize_nymeria_alignment_quality.py`, `visualize_nymeria_alignment_audit.py`: source-level timing/world-frame/continuity audit, exact Phase-1 versus floor-filtered Phase-2/3 window impact, and corrected direct-world visualizations.
- `estimate_test_actor_head_camera_calibration.py`, `backfill_oracle_actor_head_metrics.py`: explicitly test-leaky per-actor GT oracle fitting and offline augmentation of already-sampled V2M metrics; never use the oracle artifact for training/model selection.
- `merge_phase3_clean71.py`, `compare_phase3_evals.py`: corrected clean-71 aggregation and paired baseline/candidate comparison.
- `config.py`: dims, paths, task weights, defaults.
- `train.py`, `sample.py`, `eval_all.py`, `eval_camera.py`, `eval_motion_recon.py`.
- `prepare_shape_tmr_eval.py`, `precompute_nymeria_tmr_text.py`, `eval_phase2_shape_tmr.py`: final Phase-2 BONES/Nymeria C45 evaluator.
- `render_motion.py`, `render_viz.py`.
- `sbatch_phase3_bridge_native_v2m_m2v_headcam.sh`: isolated head-camera Phase-3 production launcher.
- `sbatch_phase3_bridge_native_multitask.sh`: isolated joint-target Phase-3 ablation launcher (35/35/15/15 V2M/M2V/joint-inverse/joint-forward).
- `_verify_phase3_multitask_contracts.py`: CPU contracts for joint task plans, role/locality masks, independent training sigmas, coupled state splitting, and scheduler compatibility guards.
- `sbatch_compare_phase3_step10k_headcam.sh`: exclusive paired step-10k full-71/replacement-5 evaluation launcher.
- `sbatch_compare_phase3_headcam.sh`: parameterized exclusive paired evaluation launcher (`PHASE3_COMPARE_STEP`).
- `run.sh`, other `sbatch_*.sh`, `_verify_*.py`, `_diag_*.py`.

`motion_expert/`:

- `README.md`: frozen-reasoner cross-attention POC.
- `BONES_SEED_POC.md`: BONES-only LLM2Vec in-context POC.
- `reasoner.py`, `precompute_hr.py`, `hr_cache.py`: reasoner hidden-state cache.
- `motion_expert.py`, `flow.py`, `train.py`, `sample.py`, `viz.py`.
- `bs_*`: BONES-only in-context LLM2Vec POC; `bs_normalization.py` owns checkpoint-pinned
  generator mean/std provenance and mismatch checks.

`nymeria_world/`:

- `README.md`: native camera-action world-model docs.
- `camera_to_action.py`, `nymeria_camera_dataset.py`, `nymeria_camera_rgb_dataset.py`.
- `extract_camera_opencv.py`, `prep_*`, `viz_*`, `run_vggt.py`.
- `launch_camera_phase2.sh`, `sbatch_camera_phase2.sh`.
- `export_merge_lora.py`, `prep_test_eval.py`, `run_infer_merged.sh`, `sbatch_infer_3tasks.sh`.

## Editing Guidance

Prefer existing local patterns and wrappers. Do not modify `/home/jungbin_cho/cosmos-framework` unless the user explicitly asks and the change is necessary.

Keep generated artifacts, Slurm logs, checkpoints, cached latents, stats files, mp4s, and `.npy/.npz` outputs untouched unless asked.

When changing joint-attention code, preserve these invariants:

- reasoner never attends to gen/motion;
- generator and motion attend densely over reasoner+generator+motion;
- mask ownership stays in `mot_joint_attention.py`;
- weight routing stays in `mot_joint_layer.py`;
- plain layers do not create `_moe_motion` params and pass motion rows through;
- motion init stays fresh unless explicitly changing the experiment;
- timestep pre-scale happens exactly once in `JointMotionModel.forward`;
- camera action stays raw 9-d with loss on `[:9]`;
- BONES remains text2motion-only.

When changing data code, be explicit about which index is used (`_index` vs `_t2m_index`) and why. When touching floor grounding, keep `entry["off"]` semantics as calibrated total vertical shift.

When changing training, use `named_all_parameters()` / `trainable_parameters()` for optimizer, grad clip, all-reduce, and checkpoint logic.

When changing samplers, keep train/sampling noising conventions pinned to the relevant trainer and do not re-noise already-current sample states.

When changing `native_phase_training/`, preserve these invariants:

- cached `video_latents` are a training input optimization only;
- official inference must still work with ordinary `vision_path` image/video inputs and no `video_latents`;
- camera actions stay raw 9-d `camera_pose` deltas and are padded only inside the native action path;
- native Cosmos RF distributions and action loss weight stay unchanged unless explicitly running an ablation;
- action prompts must match official `ActionPromptJsonFormatter`, inverse text must remain exactly empty, and image-to-video must stay on the official generic prompt template;
- cached-latent batches must use the active fixed-four contract (`max_samples_per_batch=4`, `max_sequence_length=None`); any optional 45,056-token run must use `LatentAwareIterativeJointDataLoader`, never stock counting of the 1x1 dummy pixels;
- native Phase 1 evaluations must explicitly use 256/shift 3/T97/action96/20 FPS;
- production checkpoint visualization must stay out of the training process and use the post-save official-inference Slurm job; smoke runs must disable auto-eval;
- do not reintroduce Torch compile into the production Slurm launcher unless a clean-node multi-GPU smoke proves it fits;
- do not start production training on a node with leftover GPU memory consumers; use the launcher preflight or inspect `nvidia-smi`.
