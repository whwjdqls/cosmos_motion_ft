# Motion-stream guidance pilot for camera checkpoints

[Verified Notion results and eight scale-comparison videos](https://app.notion.com/p/40bd2a44f44b80c5a349ca6df079d9be). Completed 2026-11-05T01:16:58.732346+00:00.

User requested CFG on motion after the native-guidance1 M2V evaluation. Three pinned checkpoints (joint weight10 40k, joint weight1 40k, no-joint weight1 30k), eight fixed windows at original indices 0,10,20,30,40,50,60,70, guidance scales 1/1.5/2. All 72 predictions scored: 48 new guided outputs and 24 reused native outputs; zero skipped cases. This pilot is separate from the complete 71-window benchmark.

## Guidance definition

Video velocity is `v_reference + scale * (v_conditioned - v_reference)`. The reference blocks only video-query→motion/shape-key edges in every bridge, preserving all other bridge attention edges and weights. This removes the whole motion stream including shape. Both passes receive the same first-image conditioning, blank text, current noisy video and noisy camera state. Camera velocity retains the native conditional prediction; it is not directly guided, but its trajectory can change indirectly as video changes. No GT camera is passed.

**No dropped-motion null condition was used in checkpoint training. This is experimental inference-only bridge guidance.** It differs from the earlier whole-bridge-disabled guidance2 gallery and from text CFG. No training checkpoint, source contract or default training setting was changed.

Same original seeds and realized initial Gaussian; UniPC35, video shift10, T97/20fps/256², same GT motion/neutral shape/first cached latent, BF16 with FP32 motion I/O and bridges, TF32 off. All three first-case checks pass bit-exact scale1 reproduction, reference video/camera invariance to changing motion+shape, nonzero conditioned/reference prediction difference, exact observed-latent preservation and no future-video-input leakage.

## Eight-window results

| Model | Guidance | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DreamSim ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint weight 10 (40k) | 1.0 | 18.273324 | 0.593283 | 0.312095 | 0.177632 |
| Joint weight 10 (40k) | 1.5 | 18.381090 | 0.598779 | 0.306378 | 0.174919 |
| Joint weight 10 (40k) | 2.0 | 18.555618 | 0.602987 | 0.303943 | 0.171408 |
| Joint weight 1 (40k) | 1.0 | 18.289040 | 0.592991 | 0.305506 | 0.164062 |
| Joint weight 1 (40k) | 1.5 | 18.406822 | 0.597421 | 0.297852 | 0.163364 |
| Joint weight 1 (40k) | 2.0 | 18.474733 | 0.596774 | 0.296773 | 0.166417 |
| No joint weight 1 (30k) | 1.0 | 18.430041 | 0.593722 | 0.306531 | 0.167303 |
| No joint weight 1 (30k) | 1.5 | 18.158991 | 0.589254 | 0.309219 | 0.167538 |
| No joint weight 1 (30k) | 2.0 | 17.988632 | 0.585908 | 0.309867 | 0.163786 |

On this pilot, joint weight1 has modest LPIPS improvement at scales1.5/2; its paired PSNR intervals include zero. No-joint PSNR worsens at both higher scales. No universal quality improvement or motion-adherence improvement is established. Pixel/perceptual scores exclude observed frame0 and use lossless frames1–96 before MP4; FVD is not reported for eight windows. Paired5000-recording bootstrap intervals, seed20261105, are included in the Notion report and archive. These means must not be ranked directly against the earlier71-window aggregates.

Eight videos show all scales and models for each fixed case. Rows are joint weight10/joint weight1/no-joint weight1; columns GT/scale1/scale1.5/scale2. All videos passed 97-frame/20fps/1024×864 checks.

All training continued during temporary sharing of our GPUs1,2,5. Evaluation/scoring workers exited and released GPU memory. Numeric tables and all eight videos plus evidence ZIP were downloaded/read back from Notion and matched by byte count and SHA256. All repeated reads and outputs are local NVMe; no shared-storage transfers.

Local sep-1 operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_motioncfg_pilot8_20261105`. Root `COMPLETE.json` is `published_verified`; the attached archive includes exact guidance/sampling/scoring code, checkpoint/source contracts, cases/seeds, per-frame/per-window metrics, paired contrasts, null-reference acceptance and media hashes.
