# Phase1 versus Phase3: training losses and optimization

Exact Phase1 specialist: [51xcizf1](https://wandb.ai/jungbinc-upenn/cosmos3_camera/runs/51xcizf1), global_lora_video_io, prefix1, final EMA200k, used by all four current Phase3 runs. Source is the saved Phase1 config, native loss/optimizer/scheduler code, full W&B scalar history and local Phase3 metrics/contracts. This is an audit; no training setting or process changed.

## Loss definitions and normalization

Phase1 camera/action is native camera SE(3) velocity flow matching, not body motion. Both video and action RF losses have weight10. Video uses waver time sampling, uniform time weighting and shift3; action reuses the sampled schedule, and clean modalities are masked. normalize_loss_by_active=false and sample_level_loss_averaging=false. Action padding is cropped to raw_action_dim before the mean; the small action scalar is not explained by averaging all64 padded channels.

Phase3 video uses the same native velocity-FM formula/noise/time weighting, with video weight10 or1. Phase3 has no action/camera loss, including the camera-token variants (camera loss0, frozen/detached readout). The body-motion objective is clean x0 reconstruction plus physical terms, using weights feature1, centered joints1, smoothness5, contact0.05, foot velocity1, foot height10. It is not numerically interchangeable with the Phase1 camera RF loss.

Native Phase1 detailed loss logs average across the task batch, including clean/inactive zeros. Phase3 video_loss and motion_loss logs average only active samples. To match batch normalization, each Phase3 term is multiplied by its active sample count/32, then averaged. The native video temporal denominator retains the observed latent frame; only the numerator masks it.

## Last5000 updates ending at evaluated snapshots

| Stage/run | Endpoint | Raw video, whole-batch mean | Weighted video contribution | Weighted camera/body contribution | Video fraction of these terms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase1 video-I/O | 200000 | 0.11092708 | 1.10927078 | camera 0.00179350 | 99.839% |
| Image-joint baseline | 130000 | 0.11655730 | 1.16557303 | body 0.03713869 | 96.912% |
| Camera joint, video weight10 | 90000 | 0.11797068 | 1.17970678 | body 0.03852870 | 96.837% |
| Camera joint, video weight1 | 50000 | 0.12083302 | 0.12083302 | body 0.03542596 | 77.329% |
| Camera no-joint, video weight1 | 40000 | 0.09730750 | 0.09730750 | body 0.02837234 | 77.425% |

Phase1 raw camera/action mean is 0.000179350; weighted video/action ratio is 618.49. This describes scalar objectives, not gradients or quality. Different targets, masks, noising and conditioning limit cross-stage interpretation.

| Phase1 window endpoint | Averaging window | Raw video | Raw camera/action | Video ×10 | Camera ×10 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000 | (0,1000] | 0.12619717 | 0.000564833 | 1.26197170 | 0.00564833 |
| 10000 | (5000,10000] | 0.11394034 | 0.000264486 | 1.13940342 | 0.00264486 |
| 40000 | (35000,40000] | 0.11375787 | 0.000191634 | 1.13757870 | 0.00191634 |
| 50000 | (45000,50000] | 0.11435119 | 0.000208023 | 1.14351195 | 0.00208023 |
| 100000 | (95000,100000] | 0.11381584 | 0.000157026 | 1.13815838 | 0.00157026 |
| 200000 | (195000,200000] | 0.11092708 | 0.000179350 | 1.10927078 | 0.00179350 |

## Actual Phase3 log values before whole-batch adjustment

| Run | Step | video_loss, active mean | video_weighted, active mean | motion_loss, composite active mean | LR at endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-joint baseline | 130000 | 0.19440679 | 1.94406793 | 0.06161753 | 5.5093959e-05 |
| Camera joint, video weight10 | 90000 | 0.19788040 | 1.97880397 | 0.06396644 | 0.00011650044 |
| Camera joint, video weight1 | 50000 | 0.20127903 | 0.20127903 | 0.05881363 | 0.00017154293 |
| Camera no-joint, video weight1 | 40000 | 0.19623335 | 0.19623335 | 0.05640396 | 0.00018163757 |

## Optimizer/config comparison

| Setting | Phase1 video-I/O | Current Phase3 |
| --- | --- | --- |
| Trained parameters | LoRA rank16/alpha32 + video I/O + camera I/O/modality embedding; 15,669,440 | Seven full bridge modules, 117,469,198; both specialists frozen |
| Optimizer | Native FusedAdam, AdamW mode, FP32 master | torch.optim.AdamW, FP32 bridge parameters/states |
| LR | Base5e-5 × schedule max0.4 =2e-5 for LoRA/video I/O; action4× =8e-5 | Peak2e-4 for every bridge parameter |
| Schedule | LambdaLinear: first500 flat at initial effective LR, then linear decay tozero by200k; not an increasing warmup | Linear warmup0→2e-4 over1000, then cosine tozero at200k |
| Adam betas / epsilon | 0.9,0.99 /1e-8 | 0.9,0.99 /1e-8 |
| Weight decay | 0.05 | 0 |
| Gradient clipping | Global L2 max1 across trainable net parameters; video bucket in logger is batch type, not only video parameters | Global L2 max1 across trainable bridges |
| Global batch / accumulation | 32=2×16, accumulation1 | 32=2×16, accumulation1 |
| Duration / EMA | 200k updates; FP32 PowerEMA rate0.1 | 200k updates; no bridge EMA |
| Compute | BF16; FP32 master parameters/EMA | BF16 native/model computation; FP32 motion I/O/bridges |
| Task probabilities | Forward40:inverse25 renormalized to61.54%/38.46%; no captions | 40/40/20 V2M/M2V/joint, except no-joint50/50/0 |
| Targets | Video and camera velocities | Video velocity and clean body-motion reconstruction; no camera loss |

Peak Phase3 LR is10× the Phase1 LoRA/video-I/O effective LR and2.5× the Phase1 action-head effective LR. Comparisons of LR alone do not equalize update size: trainable parameter sets, gradient statistics, clipping, objectives, initialization and weight decay differ.

## Clipping and a matched40k comparison

| Run/window | Mean pre-clip total norm | Median | Fraction of logged updates >1 |
| --- | ---: | ---: | ---: |
| Phase1 195k–200k | 4.907340 | 2.508150 | 61.59% |
| Image-joint baseline last5k to130000 | 32.688278 | 23.239467 | 99.60% |
| Camera joint, video weight10 last5k to90000 | 32.176950 | 24.789772 | 100.00% |
| Camera joint, video weight1 last5k to50000 | 3.000802 | 2.229791 | 80.40% |
| Camera no-joint, video weight1 last5k to40000 | 2.105272 | 1.560068 | 68.60% |

| Camera joint at35k–40k | Raw video, active | Weighted video, global batch | Body, global batch | Mean gradient norm | Clipped logged updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| video weight10.0 | 0.20014656 | 1.19404087 | 0.04015545 | 28.464556 | 100.00% |
| video weight1.0 | 0.20046006 | 0.11959082 | 0.03588184 | 2.755966 | 78.40% |

Both matched40k camera runs use LR0.00018163757214143992 at the endpoint. Their raw video errors are nearly identical at0.200, but weighting changes scalar balance and observed gradient/clipping behavior. Per-objective gradient decomposition was not measured; these data do not establish an optimal LR or weight.

## Evidence integrity

Downloaded 100040 W&B scalar rows, containing 40 identical duplicate-step records. All duplicates were checked identical and removed, leaving100000 unique even-step records2..200000. Phase1 final loss window contains2500 records; each Phase3 loss window contains500 records every10 updates. Phase1 optimization window contains4999 available logged records.
Phase1 trainable-parameter counts were read from native DCP metadata: {'lora': 6422528, 'action': 8458240, 'video_io': 788672}. The LR/optimizer/RF values come from saved config, not generic README defaults (older README sections describe100k schedules).
Native detailed-component logging reconstructs the average total to within small numerical/logging residuals (maximum5.73e-4 in the final window); Phase3 reconstructed total errors are below1e-6. These small residuals do not affect the order-of-magnitude scalar comparison.

Files: comparison.json (full computed statistics/common40k comparisons), analyze.py, phase1_loss_history.jsonl, phase1_final_optimization_history.jsonl, phase1_wandb_summary.json, phase1_history_keys.json, native trainable metadata count receipt, loss_and_lr_comparison.png and .svg. No credentials are recorded in these artifacts.



## Interpreting the user's Phase1-to-Phase3 video-quality concern

The concern is preservation of Phase1 generation quality. Baseline raw video training loss decreases from0.26670 in updates1–1000 to0.20665 in5001–10000 and0.19441 in125001–130000. Thus the training log does not establish progressively worsening video denoising loss. Lower teacher-forced/random-sigma MSE does not guarantee better multi-step samples or perceptual/control quality.

Frozen native generator weights do not freeze video activations. Every bridge applies a learned residual to video states after native layers. The bridge permits video→video self-attention as well as motion keys; removing motion keys for guidance leaves that learned video-side transformation active. V2M/joint gradients also train the same bridge parameters. The objective has no explicit distillation/identity constraint preserving Phase1 video predictions. These are concrete routes for changed or worse video behavior despite frozen specialists; they do not prove which route causes the observed quality difference.

Optimization also differs: fresh117.47M full bridge parameters at peak2e-4, versus15.67M Phase1 LoRA/action/video-I/O parameters with effective video peak2e-5; Phase3 has no weight decay or EMA. Weight10 video contributes≈97% of Phase3 scalar loss and clips virtually every logged update. The issue is not demonstrated to be a lack of scalar video weight. Loss fractions alone cannot establish per-objective gradient influence or optimal hyperparameters.

The previously published five-model Phase1 reference uses the September5 original four-task LoRA EMA200k. The current four Phase3 runs instead use the October28 two-task LoRA+video-I/O EMA200k. Phase1 camera→video also receives explicit GT camera actions, whereas the Phase3 image-joint baseline receives body motion and no camera tokens. These source/conditioning differences prevent attributing the published quality gap solely to Phase3 bridge training.

Camera variants add a separate train/inference issue: noisy GT camera tokens condition training, but inference samples camera from noise with no camera supervision during Phase3. Their native camera readout is frozen/detached. Errors in sampled cameras may affect video; this mechanism does not apply to the camera-free image-joint baseline.

A clean next diagnostic should hold the CURRENT frozen specialist, first image, initial noise,35-step/shift10 solver and guidance1 fixed while declaring separate arms: native Phase1 with GT camera; camera-free generator with whole bridges disabled; the same generator with learned video-only bridge behavior (motion keys blocked); full motion-conditioned Phase3; and then its guidance2 output. These distinguish camera-condition removal, generator-side bridge drift, motion-conditioning effects and guidance. They have not yet been run as a matched full71 decomposition. A reduced-LR or preservation-loss training ablation would be a subsequent controlled test, not a conclusion from scalar loss magnitudes. No training changes were made.
