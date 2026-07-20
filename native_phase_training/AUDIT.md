# Native Phase Training Audit

Initial audit: 2026-07-10. Correctness update: 2026-07-11.

This audit covers the isolated cached-latent Phase 1 path in `native_phase_training/` and was checked against the local `/home/jungbin_cho/cosmos-framework` source.

## Fixed Issues

### Finite child-loader livelock

`IterativeJointDataLoader` creates child iterators once and assumes they are effectively infinite. A finite map-style child can exhaust; the joint iterator then returns without advancing `global_id`, and the trainer can repeatedly select the same exhausted stream and silently spin.

`CyclingDataLoader` restores that invariant by cycling each finite PyTorch loader forever. It forwards the Cosmos-injected `collate_fn` and calls `DistributedSampler.set_epoch(epoch)` on each pass so the shuffle order changes. The historical pixel experiment in `cosmos-framework/.../world_camera_nymeria_nano.py` still has the finite-loader risk; this repository does not patch the framework copy.

### Train/inference prompt mismatch

The first cached-latent implementation applied legacy viewpoint/duration/resolution prose to every task. Official Cosmos action inference instead uses `ActionPromptJsonFormatter` for non-empty action prompts, while an empty inverse prompt remains exactly empty. Official generic image-to-video inference uses plain duration/resolution prose and does not add action-viewpoint prose.

The dataset now applies the contract per mode before tokenization and CFG dropout:

- forward dynamics and policy: official action JSON dictionary, serialized by `TextTokenizerTransform`;
- inverse dynamics: exactly empty text;
- image-to-video: official generic duration/FPS and resolution prose.

The image-to-video formatter is implemented explicitly because the framework training augmentor truncates `97/20` to `4.0` seconds, while official inference formats the true ratio as `4.8` seconds. A CPU contract test compares both local prompt paths directly against the official inference helpers.

### Dummy-video packing undercount

The stock joint loader counted the metadata tensor `[3,97,1,1]`, producing zero spatial patch tokens and only the two vision boundary markers. Training then used a fixed 32 samples per packed batch even though each real cached latent `[48,25,16,16]` contributes 1,600 patch tokens after 2x2 patchification. A typical 32-sample action batch therefore exceeded the model's native 45,056-token target.

`LatentAwareIterativeJointDataLoader` retains the parent count for text, EOS, vision markers, actions, and sound, then adds `25 * 8 * 8 = 1600` real vision patches per cached latent. This fixed the original undercount and remains required for token-budget runs.

As of 2026-07-12 production defaults to `NATIVEP1_CLIPS_PER_GPU=4`, which selects `max_samples_per_batch=4` and `max_sequence_length=None`. This is an intentional optimization-policy change, not a sampler-contract change. It is bounded safely because text is truncated to 4,096 tokens: four worst-case T97 action samples remain below 24k tokens, and a real resolved batch measured 7,323. `NATIVEP1_CLIPS_PER_GPU=0` restores the audited 45,056-token mode, where typical batches contain about 25-26 samples depending on prompt length. Token-budget job `2838` was cancelled at step 4,399 without a checkpoint; fixed-four replacement job `2852` was then submitted.

### Evaluation resolution/shift mismatch

The historical `nymeria_world/prep_test_eval.py` defaults to image size 480 and shift 10. That is not the current Phase 1 contract: training uses the 256 tier, whose native shift is 3, with T97/action96 at 20 FPS.

`native_phase_training/prep_test_eval.py` now pins resolution/image size 256, shift 3.0, T97, action96, and 20 FPS. It creates forward/inverse/policy/image-to-video JSONL files from one usable window per held-out UUID, keeps inverse prompts empty, and reports unavailable UUIDs instead of silently reducing the test set. Action modes use NVIDIA's 30-step/guidance-1 defaults; image-to-video uses 35 steps/guidance 6; all use the official UniPC path.

NVIDIA's bundled modality defaults contain shift 10 for the high-resolution release setting. The local evaluator uses the same official inference/UniPC implementation but overrides shift to Nano's native 256-tier value of 3, matching this run's training distribution.

Each record uses a mode-specific sample name so official inference cannot overwrite another mode's files. `visualize_checkpoint.py` requires every requested record in all four JSONL files to succeed, generates three GT/video comparisons plus inverse/policy camera plots per source sample, and only then writes a completion manifest.

## Other Parity Fixes

`LatentOmniMoTModel.get_data_and_condition` returns `raw_state_action` and `raw_state_sound`, matching native `OmniMoTModel`. The training packer consumes `x0_tokens_*`, so this was not known to affect loss, but it restores parity for visualization and slicing helpers.

`build_cached_index` logs kept, missing-latent, and candidate-window counts. Missing latent files are still skipped, but any shrinkage is visible.

## Configuration Truth

- Video training uses the `waver` timestep distribution and the resolution-256 shift of 3.
- `independent_action_schedule=false`, so action noising reuses each sample's vision/video sigma. The configured `train_time_action_distribution=logitnormal` is inactive unless independent action scheduling is enabled.
- Image-to-video uses the image `logitnormal` distribution.
- Loss weighting is uniform and action loss weight is 10.0.
- The LambdaLinear schedule has `f_start=f_max=0.4`: its first 500 steps are a flat 0.4x plateau, not an increasing warmup, followed by linear decay over the 100k cycle.
- LoRA mode trains generator `q/k/v/o_proj_moe_gen` LoRA plus `action2llm`, `llm2action`, and `action_modality_embed`. It leaves `time_embedder`, `vae2llm`, and `llm2vae` frozen.
- PowerEMA is enabled and official evaluation uses it. The generic framework builds and updates a full FP32 `net_ema`, although only LoRA/action parameters differ from their frozen-base values; this is retained for the baseline despite the avoidable memory cost.

## Remaining Deliberate Deviations

Raw video remains a `[3,T,1,1]` metadata tensor; clean training vision tokens come from `video_latents`. Latent-aware packing fixes the token-budget consequence, but MFU/VAE-FLOP estimates based on raw pixels remain cosmetic. Do not enable generation callbacks until their condition-image extraction is reviewed.

The stock `EveryNDrawSample` callback is deliberately disabled. It uses the training batch rather than independent real held-out media, covers only the selected stream, does not visualize action predictions, and defaults to sampling arguments that do not match this Phase 1 action evaluation. Production uses a rank-0 post-save hook to submit a separate official-inference Slurm job after each completed checkpoint. This avoids pausing or duplicating the model inside the training process while still producing all four qualitative modes.

Cached latents may be lower precision than a fresh VAE float32 encode. This is accepted for the speed-oriented path but remains a difference from pixel-native training.

## Restart Boundary

Production job `2801` started before the prompt and latent-aware packing fixes. Its checkpoint has already learned from a different prompt representation and fixed-32 batch composition. The corrected run must start from the base Cosmos checkpoint under the new default name `native_phase1_camera_json_tokpack_lora5e5_action4x_100k`; it must not auto-resume job `2801`'s DCP state.

## Verification

Focused tests live in `native_phase_training/test_contracts.py`. The 2026-07-11 acceptance smoke completed train/save/load and official forward/inverse/policy/image-to-video UniPC sampling, with all outputs and visualizations validated. Repeat that bar after future changes because import and config checks cannot validate the distributed native model path.
