# Latest one-frame Phase 1 comparison and camera CFG

Report: https://app.notion.com/p/407d2a44f44b81608660d1fe4a51036e

Yonsei artifacts:
`/scratch2/whwjdqls99/cosmos/operations/phase1_latest_vs_baseline_20261029`.
The operation's `configuration.json` pins complete checkpoint paths, inputs,
hashes and evaluation settings. Its completion markers and publication records
are authoritative; preparing this document does not imply completion.

## Matched checkpoint comparison

The candidate is the two-task, one-observed-frame, no-text Phase 1 checkpoint
at iteration 194,233, frozen when this evaluation began. The reference is the
original four-task Phase 1 checkpoint at iteration 200,000. Compare:

- Original 200k with its usual caption (verified existing predictions).
- Original 200k with an empty caption (fresh predictions).
- Candidate 194,233 with an empty caption (fresh predictions).

Use the same 71 fixed windows, one per held-out recording, not every window
in those recordings. Both empty-caption conditions receive identical inputs.
Sampling uses official EMA/UniPC, 30 steps, shift 10, guidance 1, seed 0,
256 px, 97 frames and 20 FPS. Saved training contracts are resolved first;
shift 10 is the shared inference override. These primary predictions retain
the native inference caches used by the earlier reference evaluation.

PSNR, SSIM and LPIPS-Alex score generated frames 1–96, with equal recording
weights and measured GT identity references. The numerical PSNR convention
caps the GT self-reference at 100 dB. Camera metrics include scale-matched
translation and Sim(3)-aligned ATE; neither is an unaligned trajectory error.
Paired bootstrap intervals resample recordings and do not include uncertainty
over generation seeds. Training history, task mix, caption use and step count
all differ, so this comparison is not a single-factor training ablation.

The completed 71-case means (generated frames only) are:

| Condition | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| GT identity | 100.0000 | 1.00000 | 0.00000 |
| Original 200k + caption | 19.4302 | 0.60560 | 0.29834 |
| Original 200k, no caption | 19.4231 | 0.60258 | 0.30258 |
| New 194,233, no caption | 19.9362 | 0.62302 | 0.28847 |

Candidate-minus-reference paired 95% intervals exclude zero for all three
video metrics against both references. Against the original empty-caption
model, inverse rotation improves from 0.36061 to 0.31431 degrees, scale-matched
translation from 5.754 to 5.155 mm, and Sim(3) ATE from 3.623 to 3.159 cm.
The machine-readable report retains measured camera GT self-references too.

## Additional camera-only CFG

`camera_cfg_inference_config.py` is an isolated official-inference configuration.
Its `guidance` argument denotes **camera** guidance, while native text guidance
is fixed at 1. With the same first image, empty caption, noisy vision and
timestep in both branches, it computes:

```python
v = v_camera + (scale - 1) * (v_camera - v_camera_absent)
```

The null branch removes the camera action tokens and rebuilds the packed
sequence; it does not substitute a stationary or zero-valued trajectory. It
does not modify training, model weights, losses or the framework source.
Only camera-conditioned video generation is supported. Other guidance hooks,
CFG rescaling and parallel CFG are rejected to avoid combining interventions.

The sweep uses the candidate checkpoint, scales 1, 1.5, 2, 3 and 5, and fixed
case indices 0, 7, 14, 28, 42 and 63. All sampling settings and seeds are held
constant. Both diffusion caching and text-KV reuse are disabled for every
weight because the two branches have different token layouts. Consequently,
its scale-1 result must be compared with its **fresh native uncached control**,
not treated as bitwise identical to the cached 71-case benchmark prediction.
The initial separate-process decoded-pixel equality check failed. Independent
native repeats also differed (mean absolute pixel changes 0.42–1.91 on a
0–255 scale), so cross-process bitwise reproducibility is not established.
An additional control samples native and camera-CFG-1 predictions in the same
loaded model with identical inputs and restored RNG state: all six final
vision-latent tensors match bit-for-bit. The completion script requires these
paired controls and checks conditional/null call counts. All five sweep
weights were generated in the same process. Preserve the initial failed audit
and repeated controls; the source of cross-process variation is not isolated.

Each qualitative video contains GT and all five guidance weights. Green marks
the observed first frame; red marks generated frames. These six examples are
a qualitative diagnostic, not a full-test guidance benchmark or evidence that
larger guidance is better. The candidate dropped the image-to-video training
task, so the camera-absent branch is also outside its two-task fine-tuning mix.

`camera_cfg/configuration.json`, `audit.json`, `gallery/`, `publication.json`
and `exit_code` preserve settings, controls, media and verified upload status.
Main comparison media and early previews have separate publication records.
Completed at 2026-10-29 04:37 KST: the main report has six comparison videos
and two camera figures; the camera sweep has six videos; two early previews
are also retained. All 16 media assets were downloaded from Notion and their
SHA256 hashes verified. Main `exit_code` is 0. The camera sweep's original
`exit_code=1` records the failed cross-process pixel assertion; use its
`recovery.json` (exit 0) and completed publication record for final status.
All temporary evaluation processes exited, leaving the Phase 1 trainer active.
All inference, scoring and rendering ran on owned Yonsei compute allocations.
