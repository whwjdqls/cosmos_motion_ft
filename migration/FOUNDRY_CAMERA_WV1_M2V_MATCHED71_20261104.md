# Camera video-loss ablation: matched motion-to-video evaluation

Completed 2026-11-04T23:34:27.410800+00:00. [Notion report and eight comparison videos](https://app.notion.com/p/40bd2a44f44b80c5a349ca6df079d9be).

213 predictions completed: three pinned checkpoints × the original 71 held-out recording windows, zero skipped cases. These are the same joint-weight1 40k and no-joint-weight1 30k snapshots used in the latest V2M comparison, plus the joint-weight10 40k checkpoint. Training continued throughout temporary sharing of our GPUs 1,2,5. All evaluation/scoring workers have exited and released GPU memory.

| Model | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | DreamSim ↓ | Video FVD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint weight 10 (40k) | 18.641265 | 0.580397 | 0.291918 | 0.155999 | 384.048269 |
| Joint weight 1 (40k) | 18.948038 | 0.590950 | 0.278942 | 0.147993 | 386.814031 |
| No joint weight 1 (30k) | 19.118725 | 0.594887 | 0.277810 | 0.147205 | 384.412457 |

At matched 40k, lowering video weight from 10 to 1 improves all four paired pixel/perceptual metrics. PSNR changes by +0.3068 dB (paired 95% CI +0.1059 to +0.5237), LPIPS by −0.0130 (−0.0207 to −0.0052). FVD is slightly higher, 386.814 versus 384.048; no FVD confidence interval is claimed. This cohort does not show a clear overall quality collapse from lower video weight. One training seed and one generation per case do not establish replica robustness. No-joint 30k versus joint 40k has unequal budgets and task exposure.

## Protocol and evidence

Same full GT normalized motion, neutral shape, first cached video latent, original seeds and realized initial Gaussian for every model. All future video inputs are zeroed before sampling. No GT camera is passed: camera tokens start from independent Gaussian noise and follow the native video clock. Captions blank, native guidance 1, UniPC 35, video shift 10, T97/20fps/256². Observed latent and clean motion preservation, future-input invariance, checkpoint/source identities and initial-noise parity all pass. The same Phase1 video-I/O EMA200k and selected Phase2 200k are frozen.

Lossless RGB frames 1–96 are scored before MP4 compression; supplied frame0 is excluded. PSNR/SSIM/LPIPS/DreamSim use the established EgoControl backend. FVD uses Kinetics400 I3D logits, six 16-frame suffix segments per window, 224 crop and [−1,1] normalization; one global float64 Fréchet estimate over 426 segments/model. Metric source/weights match the earlier M2V report and first-case GT I3D features reproduce exactly. Older guidance-2 results should not be ranked directly against these guidance-1 scores.

Eight fixed qualitative indices 0,10,20,30,40,50,60,70 were selected before generation. Each 97-frame/20fps video shows GT and all three checkpoints. Pixel/FVD scores do not directly measure motion adherence, hand articulation or object interaction; previews show viewpoint/hand-occlusion mismatches across methods.

Local sep-1 operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_wv1_m2v_matched71_20261104`. Root `COMPLETE.json` is `published_verified`; 17 numeric table rows and all eight videos plus the evidence ZIP were read/downloaded back from Notion, with matching bytes and SHA256. Source contracts, exact sampling/scoring code, per-window scores, embeddings, paired bootstrap contrasts and media identities are attached. No shared-storage copies were performed.
