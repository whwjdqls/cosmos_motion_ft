# Latest Phase3 snapshots: full71 M2V at motion-stream guidance2

[Verified quantitative summary on Notion](https://app.notion.com/p/40fd2a44f44b80489a0bdecc5d9410bb). Completed 2026-11-05T03:34:18.158646+00:00 on sep-1.

User confirmed motion/shape-stream guidance2, matching the pilot, rather than EgoControl image-null guidance. Four newest complete saved checkpoints at selection were pinned: image-joint130k, camera joint weight10 90k, camera joint weight1 50k, no-joint camera weight1 40k. All284 fresh predictions scored over the exact71 recording windows, seeds0–70; zero exclusions.

| Model | Step | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DreamSim ↓ | FVD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image-joint baseline | 130,000 | 19.319510 | 0.606039 | 0.254713 | 0.138213 | 359.052016 |
| Camera joint, video weight10 | 90,000 | 18.865897 | 0.593307 | 0.271381 | 0.143167 | 357.086086 |
| Camera joint, video weight1 | 50,000 | 19.142216 | 0.591958 | 0.270293 | 0.143751 | 358.000990 |
| Camera no-joint, video weight1 | 40,000 | 18.924024 | 0.592433 | 0.269894 | 0.144234 | 347.714759 |

Image-joint130k has the highest mean PSNR and lowest LPIPS among these snapshots; no-joint40k has the lowest observed FVD. Updates/task exposure differ, so this is a snapshot recipe comparison, not an equal-budget architectural ranking. Prior CFG1 evaluations used older checkpoints and cannot isolate the effect of CFG on the latest weights.

## Exact guidance and controls

Video velocity is `v_reference + 2*(v_conditioned-v_reference)`. Reference blocks only video-query→motion/shape-key edges in all seven bridges; generator-only attention, motion-query rows, all weights and the first image remain. Motion stream includes neutral shape. Both branches receive the same current noisy video and, for camera models, noisy camera state. Only video velocity is guided; camera retains native conditional velocity and can change indirectly. No measured camera is supplied. There is no trained dropped-motion null; this is experimental inference-only bridge guidance.

All four share frozen Phase1 LoRA+video-I/O EMA200k, selected Phase2 200k, dataset and unchanged parent/shared scientific source identities. Same first image/GT motion/shape, original case order and seeds0–70, and identical realized initial video Gaussian. Native UniPC35, video shift10, T97/20fps/256²; blank captions, BF16 with FP32 motion I/O/bridges, TF32 off. Samplers remain unchanged behind the local guidance wrapper.

Acceptance passes: scale1/native bit equality, reference invariance to motion+shape, conditioned sensitivity to permuted motion, preserved observed latent, and no future-video-input effect. Each CFG2 trajectory executes35 normal/35 reference calls and245 reference mask interventions. Checkpoint/source/specialist identities and all71 GT I3D features match the existing fixed71 benchmark (max difference0 for all four models); framework commit d4599e2e43fbd06168e9884205b9b66c3902d8f6 is clean and unchanged.

## Metrics and baseline context

Only lossless RGB frames1–96 are scored before MP4 compression; supplied frame0 excluded. Same PSNR/SSIM/LPIPS-Alex/DreamSim backend and weights; I3D Kinetics400 logits, six16-frame suffix segments per window,224 crop,[-1,1], global float64 Frechet over426 features/model. Report includes population SD and paired5000-recording bootstrap differences against baseline130k, seed20261105; intervals do not cover training-replica/generation-seed uncertainty. No FVD confidence interval or motion-adherence claim.

EgoControl and Naive Adaptation published references are summarized separately with their protocols, preserving original sections/media. Downloaded Naive evidence confirms71×400 I3D descriptors (one/window), while these Phase3 and EgoControl use426×400 (six/window). Do not rank Naive cached I3D-FVD296.0343 or VideoMAE CD-FVD313.2994 directly against the426-feature FVD values. Naive also retains30 steps, guidance1 and cached per-entry seeds; EgoControl uses image-reference CFG2 mapped to Edge3 and a different trained EMA200k adaptation.

All training continued while evaluators temporarily shared our GPUs0/1/2/5. All four evaluators exited and GPU memory was released; no shared-storage transfer. Heavy data, caches and outputs remain node-local.

## Evidence

Local sep-1 operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_phase3_latest_m2v_motioncfg2_matched71_20261105`. Root COMPLETE.json is published_verified. 26 numeric rows were read back from Notion and the uploaded evidence ZIP downloaded and matched by byte count/SHA256.
ZIP SHA256 `753ba26c684465bb4be41a07c69c2a7cbe248642f0f30009ae0d61ee23ee203b` (7,332,307 bytes). Contains aggregate/per-frame/per-window metrics, I3D features, paired contrasts, native/CFG acceptance, framework/source/input/checkpoint identities, baseline source evidence and reproducible code.

- baseline: `/mnt/local/jinhyung.kim/cosmos/operations/eval_phase3_latest_m2v_motioncfg2_matched71_20261105/checkpoints/baseline/step_000130000.pt`; SHA256 `c83d7320084834a101471041edb116db2f5f6498581554c886e3547b1b51dee7`.
- joint_w1: `/mnt/local/jinhyung.kim/cosmos/operations/eval_phase3_latest_m2v_motioncfg2_matched71_20261105/checkpoints/joint_w1/step_000050000.pt`; SHA256 `cc6f725541a124c8eefa43a7fb02fb882405db83effc9c225fbca02e83e2f62d`.
- nojoint_w1: `/mnt/local/jinhyung.kim/cosmos/operations/eval_phase3_latest_m2v_motioncfg2_matched71_20261105/checkpoints/nojoint_w1/step_000040000.pt`; SHA256 `c390553abcf2e6bd24bb4ee817f4ceee2123afc743e8344a8adf36f4593564b7`.
- joint_w10: `/mnt/local/jinhyung.kim/cosmos/operations/eval_phase3_latest_m2v_motioncfg2_matched71_20261105/checkpoints/joint_w10/step_000090000.pt`; SHA256 `c67e9ffdc4daaae14ab052fa79334fac7ebc2fa6ed2c85b9f6a1d810819f362b`.
