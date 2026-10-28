# naeive adaptation

This ablation puts the existing Nymeria camera/head-aligned 283-dimensional
motion representation directly in Cosmos3-Edge's native action stream. It uses
global generator Q/K/V/O LoRA (rank16, alpha32) and fresh motion input/output
projections. There is no separate motion expert or Phase-3 bridge. Initialization
is the official Edge base, with a fresh optimizer; incompatible camera action
heads and their modality embedding are deliberately omitted from that load.
The native BF16 grouped GEMM requires aligned channels, so the projection width
is288. Five zero-padding channels are excluded through `raw_action_dim=283`;
they are not additional motion features or supervised outputs.

The training samples are the same 112,503 validated, normalized T97 windows in
the local Phase-3 cache. These use all eligible training recordings, matching
Phase3's cohort rather than the camera run's exclude5 cohort. Existing floor
calibration, frame-zero canonicalization, normalization and centered neutral
skeletons are preserved. No raw RGB is decoded during training.

| Task | Probability | Observed inputs | Predicted output |
| --- | --- | --- | --- |
| Text → motion | 25% | Caption, neutral shape | 97 × 283 motion |
| Text + image → motion | 25% | Caption, frame0, neutral shape | 97 × 283 motion |
| Video → motion | 25% | Full97-frame video, neutral shape | 97 × 283 motion |
| Motion → video | 25% | Full motion, frame0, neutral shape | RGB frames1–96 |

The native VFM implementation requires a visual stream, so text-only examples
carry a fixed zero latent with every visual token marked observed. It contains
no example image information and contributes no visual loss. Text+image examples
carry only the first causal Wan latent. The other two tasks have empty captions.
Caption dropout is0.1 for the two text tasks.

Neutral shape is one always-observed prefix token: the centered30×3 coordinates
occupy its first90 channels, with the remaining193 zero. The following97 tokens
are the unchanged motion features. `action_start_frame_offset=-1` aligns motion
frame0 with RGB frame0. Domain31 is used for this new motion stream. Inference
removes the prefix token from the returned motion. This shape-conditioning
adapter and the fixed visual placeholder are part of the ablation contract.

On sep-2, `train_foundary.sh` accepts one or two explicitly assigned GPUs. Batch32,
accumulation1, BF16, full activation checkpointing and FP32 EMA are the defaults.
The native Adam settings and200k-update horizon are retained:
LoRA LR2e-5 and action-head LR8e-5, flat for500 updates then linear decay to0;
betas0.9/0.99, epsilon1e-8, weight decay0.05 and gradient clipping1.

The current objective is `x0_reconstruction_shared_waver_v1`. Motion predicts
clean normalized `x0` directly, using the exact Phase-2/3 reconstruction
implementation: feature1, centered decoded joints1, temporal displacement5,
contact0.05, planted-foot velocity1 and foot height10. These weights are used
without an additional global multiplier. The observed shape token and five
padding channels are excluded before decoding and loss computation. Motion is
conditioning only in motion→video, so that task has no motion loss. Video keeps
native velocity prediction and its loss scale10.

Training noise is unchanged: native video WAVER, resolution shift3, with
`independent_action_schedule=False`. Motion therefore reuses the video's sigma;
the configured action logit-normal distribution is inactive. The interpolation
remains `x_sigma = (1-sigma)*x0 + sigma*epsilon`. At inference only the predicted
motion segment is converted to velocity `(x_sigma-x0_hat)/sigma` before native
CFG/UniPC. Conversion uses the scheduler's actual integration sigma, not its
quantized integer model timestep divided by1000. Observed tokens remain fixed.
This aligns the motion prediction target and reconstruction loss with Phase2/3;
it intentionally retains the shared native noise distribution for this ablation.

Four fixed held-out cases (one per task) produce in-process EMA UniPC30 previews
after launch/resume and every5k updates, in the training W&B run. Motion videos
show ground truth and generated skeletons; video generation shows ground truth
beside generated RGB. Metrics score normalized motion MSE, centered joint error,
and generated-frame video PSNR. These are test-set monitoring, not model selection.
The callback preserves training RNG and module modes.

Source `.site.local.sh`, set `FOUNDRY_GPU_INDEX`, a distinct `NATIVEP1_RUN_NAME`,
`MOTIONP1_OBJECTIVE=x0_reconstruction_shared_waver_v1`,
and `MOTIONP1_PREVIEW_DATA` to the prepared local directory, then run
`bash native_motion_phase1/train_foundary.sh`. Resume requires an explicit
same-run `RESUME_CHECKPOINT`. All data, model weights, caches and outputs must
resolve under `/mnt/local`.
For two GPUs, set `FOUNDRY_GPU_INDICES=1,7` and
`NATIVEP1_CLIPS_PER_GPU=16` to retain global batch32. The contract computes
global batch from both data-parallel dimensions. Each rank audits its own
gradients; both sample previews, while only rank0 renders, writes and logs.
For the original velocity ablation, use `MOTIONP1_OBJECTIVE=native_velocity`
(the compatibility default). Never resume one objective from the other's run;
`motion_contract.json` rejects that mismatch.

Operational logs for the2026-10-28 preparation and GPU acceptance test are under
`/mnt/local/jinhyung.kim/cosmos_storage/operations/native_motion_phase1`.
Live logs and the acceptance record establish launch status.

The original [velocity run](https://wandb.ai/jungbinc-upenn/cosmos3_camera/runs/h4r7mobn)
was stopped after verifying its complete step600 checkpoint, including model,
EMA, optimizer, scheduler and trainer state. Its source snapshot and artifacts
are retained in the original operations directory. The revised x0 objective
starts separately from the official Edge base with a fresh optimizer.
Its tests, acceptance evidence and launch record are under
`/mnt/local/jinhyung.kim/cosmos_storage/operations/naeive_adaptation_x0_20261028`.
Live logs and `plan.json` establish current status.

The revised [x0 run](https://wandb.ai/jungbinc-upenn/cosmos3_camera/runs/m8owvunc)
uses tmux `cosmos_naeive_adaptation_x0_gpu7` on sep-2. Nine focused CPU tests,
all four batch32 GPU backward tasks, full checkpoint4 recovery/update5, and
four finite97-frame EMA previews passed before launch. The production model,
optimizer and scheduler match that accepted recipe, with fresh base loading,
200k updates and normal equal task sampling restored after the short test.
All four initial production videos were verified uploaded to W&B, along with
the individual motion-loss terms. Training continued beyond step50, and the
four-task production gradient/loss audit passed. `running_verified.json`
records the remote media paths and the verification snapshot.

After the interrupted runs on2026-10-28, the user authorized any unoccupied
GPUs and requested two GPUs per experiment. The x0 run resumes full checkpoint
6237 on sep-2 GPUs1/7, still global32/accumulation1, in tmux
`cosmos_naeive_adaptation_x0_gpu1_7`. Phase1 separately resumes checkpoint20048
on GPUs0/5. Both keep their existing W&B IDs. Current resume plans, source
snapshot, checks and logs are under
`/mnt/local/jinhyung.kim/cosmos_storage/operations/resume_two_gpus_each_20261028`.
The operations pointer `naeive_adaptation_current_ops.txt` selects the motion
run's current logs. Moving from one to two ranks restores model/EMA/optimizer/
scheduler/iteration; the new rank uses its initialized RNG because the old
checkpoint only contains rank0 RNG. This is not a bit-identical continuation.
