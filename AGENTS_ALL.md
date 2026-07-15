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
UniPC/C45 integration pass. Foot-only hybrid job `10644` also completed at 200k; job `10645`
remains the only unfinished training dependency for the full shape backfill.

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
input. Continuation/evaluation/unit smokes `10742/10743/10744` all passed.

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
- `native_phase_training/checkpoint_eval_callback.py` plus `sbatch_checkpoint_eval.sh`: production rank 0 submits an isolated official four-mode EMA/UniPC evaluation after every successful checkpoint save. The stock in-training generation callback stays disabled.
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
- Directional masking prevents clean condition tokens from reading noisy target tokens. For `video2motion`, noised motion may attend local clean video. For `motimg2video`, noised video may attend local clean motion. The causal Wan 4x mapping for 97 source frames is latent groups `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`; source frame `m` maps to latent frame `(m+3)//4`.
- The old 3-way path remains available as `--coupling joint` and is the default for old checkpoints.

Motion weights are freshly initialized, never copied from `_moe_gen`. `_reset_motion_params` uses small normal init; qk norms/layer norms are fresh; `llm2motion` is zero initialized. Motion FFN width defaults to `MOTION_INTERMEDIATE_SIZE=3072`. Shared attention fixes hidden width 4096 and Q/K/V head geometry.

`FrozenCosmos` must build with `action_gen=True`, because camera tasks require `action2llm`, `llm2action`, and `action_modality_embed`.

## Seven-Task Contract

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

If calibration JSON is missing, the dataset warns and proceeds uncalibrated. BONES samples are untouched.

## Objectives and Timestep Convention

In `motion_expert_joint_attention/train.py`, motion objective is selected by `--objective`; the current argparse default is `x0`, while vision and camera are always velocity objective to match Cosmos rectified flow.

Velocity objective: `x_t = (1-t)x0 + t*noise`, target `v = noise - x0`, sample by Euler from `t=1` to `t=0`.

X0 objective: model predicts clean x0. Historical `--motion_schedule legacy` uses unshifted logit-normal sigma plus a linear DDIM-in-sigma ladder. New `--motion_schedule native` uses Cosmos's shifted logit-normal training sigma and shifted 1000-step inference ladder while retaining the empirically better x0 target.

The same per-sample `t_or_sigma` must be used for noising and `model.forward`.

Critical Cosmos timestep scale: Cosmos heads multiply by `TIMESTEP_SCALE=1e-3` internally. If raw `t in [0,1]` is passed directly, the embedder sees nearly constant `t*1e-3`, loss can drop while samples remain noise. `JointMotionModel.forward` divides `t_or_sigma` by `TIMESTEP_SCALE` exactly once before passing to heads, so the internal embedder sees true flow time. Train and sample must both go through this pre-scale. Do not additionally scale in callers.

Motion losses in joint-attention training: masked 283-d feature loss, decoded-joint loss, and smoothness loss, with documented weights `1/10/50`. Vision loss is flow MSE on latent channels. Camera loss is flow MSE on action channels `[:9]` times 10.

For current Phase 2 and 7-task runs, all motion-supervised tasks train x0 prediction by default. `text2motion`, `textimg2motion`, and `video2motion` train `motion_pred` directly against clean normalized motion `x0` via `flow.add_noise_x0_masked` unless `--objective velocity` is explicitly passed. `motimg2video` packs motion as a clean condition only and has no motion target/loss. The checkpoint records `args["objective"]`, and `sample.py` dispatches motion sampling to `sample_x0` or `sample_velocity` from that saved value.

Native motion schedule implementation as of 2026-07-11:

- `flow.sample_sigma_native_logitnormal` matches current Cosmos action-style training: CPU standard-normal draw, sigmoid, `sigma=1-t_raw`, rational shift `s*sigma/(1+(s-1)*sigma)`, default shift 3, then `x_sigma=(1-sigma)x0+sigma*eps` with clean x0 as target.
- `flow.native_inference_schedule` is bit-identical to `FlowUniPCMultistepScheduler.set_timesteps`: float32 `(N-1)/N` endpoint, linear base ladder, rational shift, integer `sigma*N` model timesteps, final zero sigma.
- Production `--motion_native_solver euler` is the BONES POC-proven first-order x0/straight-path update on that exact native ladder. It is schedule-identical but not solver-identical to generator UniPC.
- Optional `--motion_native_solver unipc` converts `x0_hat` to `v_hat=(x_sigma-x0_hat)/sigma` and drives NVIDIA's official UniPC scheduler. This was checked with a perfect-x0 reconstruction test but remains an inference ablation, not the production default.
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
- x0 target, native shifted logit-normal schedule, shift 3, 1000 timesteps, POC-proven native-ladder Euler sampler;
- `motion_mrope=cosmos3d`, joint coupling, stride 3 / 12 motion blocks, batch 32 per rank, 8 GPUs, 200k steps;
- 96-hour Slurm wall limit. The previous T2M+TI2M run measured about 1.26 s/step, so 200k steps alone need about 70 hours; a 48-hour request predictably stops near 137k before visualization/checkpoint overhead;
- in-training visualization every 2k steps with two T2M and two TI2M samples. TI2M MP4s show conditioning image | GT | generated; rendering uses stride 2 at 10 FPS while raw arrays keep all frames.

Pre-submit verification on node 3 passed: exact native schedule/UniPC ladder checks; real 97-valid-frame TI2M padded to 200; full 993.73M-trainable-parameter T2M and TI2M forward/backward with frozen-gradient assertions; checkpoint optimizer save/reload with zero skipped tensors; post-load 200-frame T2M and 97-frame TI2M sampling; condition/GT/generated MP4 creation; and a two-GPU run with zero parameter divergence.

Production submission: Slurm job `2844` (`t2mtinat`) was submitted on 2026-07-11 with no dependency. It requested one exclusive `a3ultra` node, 8 GPUs, 64 CPUs, and 96 hours. Slurm output is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mtinat-2844.out`; run outputs go to `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d`. It completed successfully with exit code 0 on 2026-07-14 after 2d 2h 28m. The final named checkpoint is `ckpt_step200000.pt` (about 6.19 GB); bridge launchers name this file explicitly so they cannot silently consume a moving `latest.pt`.

Phase-2 motion training does not maintain a model-weight EMA. `motion_expert_joint_attention/train.py` optimizes the regular motion-expert parameters and saves those directly; its `_loss_ema` is only a scalar used by the explosive-batch guard. The BONES native-schedule POC likewise has no model EMA. Adding motion-weight EMA is a separate future experiment, not part of job `2844`.

## Bridge Hypotheses and Representation Risks

These notes are experiment hypotheses, not established results.

Modality-bridge hypothesis: full generator-motion joint self-attention is expensive and may perturb the pretrained generator distribution by adding motion tokens directly into generator attention. For frame-aligned physical translation tasks (`video2motion`, `motimg2video`), a trainable gen-motion bridge with temporal locality may be a better inductive bias. The active Phase-3 bridge experiment freezes the Phase-1 generator/video/camera model and the Phase-2 motion expert, then trains only `modality_bridge.py` so generator and motion hidden states can exchange local information without changing either expert's weights. Directional masking is task-specific: for `video2motion`, noised motion reads local clean video while video cannot read noisy motion; for `motimg2video`, noised video reads local clean motion/shape while motion cannot read noisy video. Same-modality bridge attention remains enabled. Twelve bridges are inserted after the native reasoner+generator update and the reasoner+motion update at sparse motion layers 2,5,...,35. Their generator/motion residual gates are zero-initialized, making the bridge an exact no-op before training. This is still a hypothesis; successful optimization and useful alignment are not established by the smoke tests.

The corrected causal 4x locality for a 97-frame clip and 25 Wan-VAE latent frames is `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`. In code, motion frame `m` maps to latent frame `ceil(m/4)`, implemented as `(m+3)//4`. The old rule `[4*g, 4*g+3]` was wrong: it assigned source frames 1..3 to latent 0 and omitted source frames 94..96. `_verify_native_bridge_contracts.py` asserts every one of the 97 motion frames has exactly one bridge-local latent.

Bridge risk: if the frozen motion expert only learned text/text-image-to-motion and never learned to interpret generator/video features, the bridge must translate generator hidden states into useful motion-expert activations through residual edits alone. This may be enough, but should be evaluated as a capacity/alignment question rather than assumed. Monitor bridge gate values, bridge output norms, `video2motion` reconstruction, and `motimg2video` qualitative control. Because generator weights stay frozen in this hypothesis, Phase-1 camera/video metrics should be checked for non-regression.

Sampler compatibility for a native Phase-1 generator plus frozen motion expert:

- Native Phase 1 and the new native-schedule Phase 2 share a normalized shifted RF noise coordinate and 1000-step scale, but intentionally do not share the training-time base distribution, prediction target, or solver. Generator targets use Waver clean-time, convert to noise-time, apply resolution shift 3, predict velocity, and sample with official UniPC. Motion targets use logit-normal noise-time, apply shift 3, predict clean x0, and sample with the POC-validated first-order x0 Euler update. Historical Phase-2 checkpoints without `motion_schedule=native` still use the legacy unshifted contract.
- The bridge should not force these into one global diffusion process. Treat each bridge task as a target-modality task. For `video2motion`, video is clean conditioning and motion is the noised target, so use the motion expert's motion `x0` noiser/sampler. For `motimg2video`, motion is clean conditioning and video is the noised target, so use the native Cosmos generator RF noiser/sampler.
- This is acceptable because the bridge aligns hidden states, not raw sampler states. It is wrong to train bridge video targets with the old custom `motion_expert_joint_attention` video sampler and then evaluate with official Cosmos inference, or to use one shared scalar schedule for simultaneously noised video and noised motion. Current task plans avoid noising video and motion as targets in the same sample; preserve that invariant.
- `train.py`/`sample.py` now checkpoint independent `motion_schedule` and `gen_schedule` contracts. `gen_schedule=native` uses the exact CPU float32 Waver transform plus shift during training and `FlowUniPCMultistepScheduler` during video sampling; `legacy` preserves uniform-time Euler behavior for old checkpoints. For standalone Phase-1 visual-quality evaluation, the authoritative path remains `cosmos_framework.scripts.inference` through `native_phase_training/inference_config.py`; the joint sampler is required once bridge layers participate.

The 2026-07-11 Phase-2 run implements the earlier proposed time-coordinate unification: native action-style logit-normal sigma with x0 prediction. A future bridge still must use target-modality schedules rather than forcing simultaneously noised video and motion into one scalar process. Video-style `waver` for motion remains an unimplemented ablation.

The old bridge run/launcher (`ja_phase3_bridge_v2m_m2v_from_t2m3d_p1cam`, `sbatch_phase3_bridge_v2m_m2v.sh`, historical job `2774`) is retained only for provenance. It used historical custom specialists/schedules and the incorrect local frame map; do not resume it into the native experiment.

Active native bridge launcher: `motion_expert_joint_attention/sbatch_phase3_bridge_native_v2m_m2v.sh`. It uses 50/50 V2M/M2V, `T=97`, batch 4/GPU on 8 GPUs, 200k steps, LR `2e-4`, 1k warmup, cosine decay, and saves every 5k. The Slurm request is 120 hours: the measured historical bridge rate of about 1.738 s/step requires about 96.6 hours for optimizer steps alone, so the former 96-hour request could not cover checkpoint I/O and required visualization. It loads the final Phase-1 100k **EMA** from native DCP through `checkpoint_utils.py`, recreating LoRA rank 16 / alpha 32; using the old alpha-16 default would halve the trained adapter residual despite a shape-clean load. It loads the explicit Phase-2 200k `.pt`, freezes both specialists, and trains 805.40M bridge parameters. Generator-facing samples use native `[BOS,text,EOS,SOG]` reasoner packing, FPS-modulated float 3D-mRoPE at source FPS 20, and temporal modality margin 15000; this is required in addition to matching Waver/shift-3/UniPC. Mixed-task loss is normalized by local batch size: motion aggregate is scaled by the number of motion-target samples over `B`, while each vision/camera sample contributes its loss divided by `B`. The previous implementation summed per-sample vision/camera losses but used one aggregate motion mean, so effective task weight changed with batch composition.

Checkpoint visualization is required, not best-effort, for this launcher. Every 5k checkpoint and the final 200k state sample two fixed held-out V2M and two fixed held-out M2V windows with 30 native solver steps and guidance 1.0. V2M writes the clean source video, GT|predicted motion, and a combined comparison MP4. M2V writes GT/generated latent arrays, separate decoded MP4s, and a GT|generated comparison. `--require_viz` synchronizes setup/runtime failures across ranks and terminates the run coherently rather than silently training without results. `--viz_only` exists as a no-training/no-checkpoint verification seam.

Pre-submit verification on node 3 used actual Phase-1 90k EMA and Phase-2 190k weights. All 293 generator tensors and 149 motion tensors loaded with zero missing/shape mismatches. One real V2M and one real M2V forward/backward passed finite-loss and frozen-gradient checks. A metadata-checkpoint reload then sampled one held-out V2M case with native x0/Euler and one M2V case with native velocity/UniPC and wrote motion arrays/metrics, generated video latents, and `summary.json`. A second built-in smoke on node 2 repeated the two real forward/backward paths while Phase 2 was running: V2M loss `0.1500`, M2V loss `0.1581`, nonzero bridge gradients, and zero frozen-specialist gradient leakage. Final-pair verification then used Phase-1 100k EMA plus Phase-2 200k: 293/293 generator and 149/149 motion tensors loaded cleanly; V2M loss was `0.3095`, M2V loss was `0.1846`, bridge gradients were nonzero, and frozen-gradient checks passed. A separate `--viz_only` run sampled one held-out example per direction and produced valid V2M source/GT-prediction comparison media plus M2V GT/generated/comparison media under `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_bridge_viz_exact_p1ema100k_p2native200k/viz_step000000`. These checks prove wiring, loading, noising, gradient routing, both sampler executions, VAE decoding, and media writing; they do **not** prove bridge quality. Residual risks include the Phase-2 expert's changed reasoner context inside Phase 3, prompt-format differences between native Phase 1 and joint Phase 3, UniEgo's incomplete camera/world placement information for M2V, and the large bridge capacity/zero-gate optimization dynamics.

Native Phase-3 production job `2865` (`p3brnative`) ran normally through step 5,000 on 2026-07-14, then failed during the required checkpoint visualization. Training itself was healthy: finite losses, no skipped updates, about 1.811 s/step, and peak allocated GPU memory about 117.5 GB. It successfully wrote both `ckpt_step005000.pt` and `latest.pt` under `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k`; a metadata/mmap load confirmed step 5,000, 96 model tensors, optimizer state, and the expected V2M/M2V task list. The partial `viz_step005000` directory is **not** a complete evaluation: it contains the first V2M motion artifacts only and must not be treated as a successful four-sample visualization.

The job-2865 failure was a distributed collective mismatch, not OOM, bad training data, timeout, or checkpoint corruption. Only rank 0 runs `do_viz`, but `precompute_latents.load_vae` constructed `Wan2pt2VAEInterface`, whose `_video_vae` unconditionally called the framework's world-group `sync_model_states(model)`. Ranks 1-7 were already waiting in `run_checkpoint_viz`'s scalar status broadcast. The incompatible collective ordering made a nonzero/garbled status reach peer ranks; rank 2 raised the misleading `required checkpoint visualization failed ... on rank 0`, then torchrun terminated rank 0 before it could log its own traceback. `load_vae(..., rank_local=True)` now suppresses only that constructor synchronization for rank-0-only visualization and restores the framework function in `finally`; all ordinary VAE training/evaluation/precompute callers keep synchronized construction. The temporary visualization VAE is released and the CUDA cache cleared before rank 0 broadcasts status, so it is not retained during later optimizer steps. Visualization setup/barrier/status now use a separate CPU/Gloo group with a two-hour timeout because four 30-step samples may legitimately exceed the default 10-minute NCCL timeout; gradient synchronization remains on the normal NCCL group. A local contract test verified VAE-sync suppression/restoration, and Python syntax/diff checks passed.

The exact 8-rank fix gate is Slurm job `2869` (`p3brvizsmk`), launched by `sbatch_phase3_bridge_native_viz_ddp_smoke.sh`. It reloads the saved step-5,000 bridge plus the exact final specialists and must produce one held-out V2M and one held-out M2V visualization through `--viz_only`. Production resume job `2870` (`p3brnative`) has dependency `afterok:2869` and uses the regular launcher with `--resume auto`, so it will continue at step 5,001 only if the distributed visualization gate succeeds. At documentation time both are pending: `2869` for cluster priority/resources and `2870` for its intentional dependency. Logs are `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brvizsmk-2869.out` and `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brnative-2870.out`. Treat all later status as dynamic and check `squeue`, `sacct`, and the logs before making current-progress claims.

Representation risk: camera action is raw metric relative SE(3) delta (`pos(3)+rot6d(6)`, loss on `[:9]`) and is naturally action-like. Current motion is normalized 283-d UniEgo, with body-centric/canonicalized pose features and foot contacts; it is not the same kind of raw metric state as camera action. This representation is probably reasonable for `video2motion` because the target is UniEgo features. It may be more ambiguous for `motimg2video`, because a normalized/canonicalized body motion condition may not fully specify absolute camera-frame body placement or the body-camera relation. If `motimg2video` is weak, representation alignment should be considered before blaming only the bridge architecture. Possible future auxiliary conditions include camera/world-frame root or pelvis deltas, root velocity, or camera-relative body trajectory, but these are not implemented.

Main launch wrappers include `run.sh`, `sbatch_t2m_both.sh`, `sbatch_t2m_both_mrope3d.sh`, `sbatch_t2m_ti2m_native_mrope3d.sh`, `sbatch_phase3_7task.sh`, `sbatch_precompute.sh`, `sbatch_precompute_T97.sh`, `run_eval.sh`, and `run_camera_eval.sh`.

## Visualization and Evaluation

Joint-attention motion renderer: `render_motion.py`, runs in `cosmos`, no Kimodo import. It renders GT left/blue and generated right/red, with root-tracking XZ viewport, floor grid, trajectory trail, SOMA-30 parents, skipped fingertip-end joints, Y-up/+Z-forward input, negated X to match Kimodo display, and mplot3d Z-up remap.

`render_viz.py` is a Kimodo-env alternative. `_autorender.sh` watches viz dirs.

Phase-1 camera eval/viz note: the existing `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/camera_eval` contains inverse-dynamics outputs and metrics. On 2026-07-06 an additional FD/policy qualitative eval for `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/ckpt_step200000.pt` was run on the T2M node into `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/fd_policy_camera_eval` with 8 test windows. It produced raw generated videos under `fd_out/` and `policy_out/`, and 16 GT|generated side-by-side mp4s under `viz/` (`*_fd.mp4`, `*_policy.mp4`). This was intentionally kept separate from the existing inverse-dynamics `camera_eval` directory to avoid overwriting old metadata.

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
- `prep_test_eval.py`: native held-out T97/256/shift-3 official-inference input builder.
- `visualize_checkpoint.py`: four-mode official-output visualizer and manifest writer.
- `checkpoint_eval_callback.py`, `sbatch_checkpoint_eval.sh`: post-save isolated four-mode evaluation submission and worker job.
- `test_contracts.py`: prompt, packing, and evaluation-contract CPU tests.
- `experiment.py`: Hydra experiment registration for `world_camera_nymeria_latent_nano`.
- `world_camera_nymeria_latent.toml`: production TOML for the native camera/video LoRA run.
- `run_latent_train.py`, `run_latent_train.sh`: training entrypoints. `run_latent_train.py` sets a run-local TensorBoard log dir.
- `inference_config.py`: official inference config shim.
- `sbatch_phase1_native_camera.sh`: production Slurm launcher with exclusive-node request, memory preflight, and compile disabled.

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
- `config.py`: dims, paths, task weights, defaults.
- `train.py`, `sample.py`, `eval_all.py`, `eval_camera.py`, `eval_motion_recon.py`.
- `render_motion.py`, `render_viz.py`.
- `run.sh`, `sbatch_*.sh`, `_verify_*.py`, `_diag_*.py`.

`motion_expert/`:

- `README.md`: frozen-reasoner cross-attention POC.
- `BONES_SEED_POC.md`: BONES-only LLM2Vec in-context POC.
- `reasoner.py`, `precompute_hr.py`, `hr_cache.py`: reasoner hidden-state cache.
- `motion_expert.py`, `flow.py`, `train.py`, `sample.py`, `viz.py`.
- `bs_*`: BONES-only in-context LLM2Vec POC.

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
- cached-latent batches must use latent-aware 45,056-token packing, not fixed sample-count packing based on dummy pixels;
- native Phase 1 evaluations must explicitly use 256/shift 3/T97/action96/20 FPS;
- production checkpoint visualization must stay out of the training process and use the post-save official-inference Slurm job; smoke runs must disable auto-eval;
- do not reintroduce Torch compile into the production Slurm launcher unless a clean-node multi-GPU smoke proves it fits;
- do not start production training on a node with leftover GPU memory consumers; use the launcher preflight or inspect `nvidia-smi`.
