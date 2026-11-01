# Zero-shot Edge: camera CFG and text/image-to-video comparisons

Report: https://app.notion.com/p/binne/409d2a44f44b80899b2bec1fc10ba9bc

Operation: `/scratch2/whwjdqls99/cosmos/operations/edge_zeroshot_camera_cfg_20261031`.
Driver: `native_phase_training/compare_edge_zero_camera.py`.

## Existing-result audit

The October14 `official_camera_fiveway_scale10_20261014/Edge` has71 successful
zero-shot forward-dynamics results. They use another fixed71 cohort, captions,
per-case seeds,35steps and translation gain10. They cannot be inserted into a
matched comparison using the two-task model's usual camera inputs. The earlier
`official_camera_nano_edge_20261014/Edge` contains an ERROR.json rather than
completed sample outputs. The inventory records paths, counts and example args.

Reuse six two-task/prefix1/no-text/EMA200k cameraCFG2 outputs from the completed
October29 report. Checkpoint, input files, camera-CFG source, outputs and prior
branch-call audit are pinned. The194233 sweep and prefix17/action-only models
are not substitutes for this200k checkpoint.

## Camera experiment

Six fixed71 indices0,7,14,28,42,63; identical first images, native96 camera
transitions, empty captions and seed0. No translation amplification or fitting.
Both zero-shot Edge and two-task200k are evaluated at camera guidance1,1.5,2.
This produces36 predictions:30 new,6 verified reuse. Six cases are a diagnostic,
not a full-test evaluation.

Official UniPC30, shift10, T97/20fps/256. Diffusion and text-KV caches disabled
for every camera weight. Text guidance fixed at1. Existing camera-CFG hook:

`v = v_camera + (scale - 1) * (v_camera - v_camera_absent)`

Null removes only camera tokens and preserves image/text/noisy video/timestep.
Scale1 invokes no camera-free branch. `zero_camera_inference.py` installs the
hook before the official CLI resolves the public HF Edge YAML. The DCP-only
`zero_camera_cfg_config.py` uses the resolved native Phase1 configuration for
the fine-tune. Saved runtime configs must confirm LoRA disabled
for zero-shot and enabled for two-task. Weights/training are never changed.

The first scale1 sample in each loaded camera model is also generated through
native uncached inference with restored CPU/CUDA RNG. Final latents must match
bit-for-bit. Branch counts must be30/0 at1 and30/30 at1.5 or2. Camera-absent
guidance is an inference intervention, not a claim of trained camera dropout.

Camera comparison videos show GT above, zero-shot in the middle row, two-task
below, with columns for cameraCFG1/1.5/2. Observed frame0 green; generated1–96
red; GT blue. Same native camera representation/units for both models.

## Text + image experiment

Reuse the exact10 inputs from `phase1_twotask_i2v_20261031`: six captions and
left/right prompt pairs on two images. Generate10 zero-shot samples with the
official I2V entrypoint. Reuse20 audited original/two-task200k samples.
UniPC35/textCFG6/shift10, structured prompts, seed0, native diffusion-cache
default, same first images, no camera input. Each qualitative video has
zero-shot/original200k/two-task200k panels. This is not pure text-only synthesis.

Report measured GT identity and PSNR/SSIM/LPIPS-Alex over frames1–96, on the six
original-caption cases and all camera weights. These measure recorded-future
resemblance, not text adherence or scene consistency. Controlled prompts have
no corresponding GT future. Preserve per-case/per-frame results and source IDs.

## Completion

Trust model-specific `*_VERIFIED.json`, guidance audits, METRICS_COMPLETE.json,
MEDIA_COMPLETE.json and finally COMPLETE.json/publication.json. The final
publisher verifies Notion tables and remote hashes for16 videos plus an evidence
ZIP. Configuration and queued/running inference do not establish completion.
All model/data/rendering work runs on allocated compute, not login nodes.

## Launch corrections

The initial public-HF launch incorrectly supplied the DCP module config and
failed before sampling (`Missing key ema`). The corrected wrapper leaves HF
configuration resolution to the official CLI. The first successful HF arm
produced all18 samples and passed native parity, then its reporting stage
expected a DCP-only `config.yaml`. The driver now reads the exact resolved
configuration recorded by `OmniMoTModel` in the HF debug log. Recovery validates
existing completed samples and their original sampling log without regenerating
or overwriting them. Failed logs are retained for provenance.

## Measured results

Six matched windows, generated frames1–96, equal window weighting. GT identity
PSNR is capped at100dB. No text adherence score is inferred from pixel metrics.

| Camera model | Camera CFG | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| GT | None | 100.0000 | 1.00000 | 0.00000 |
| zero | 1.0 | 15.6191 | 0.45213 | 0.47872 |
| zero | 1.5 | 15.2400 | 0.43895 | 0.47974 |
| zero | 2.0 | 14.9282 | 0.42778 | 0.48035 |
| two_task | 1.0 | 20.0775 | 0.62922 | 0.27824 |
| two_task | 1.5 | 20.0352 | 0.63308 | 0.26727 |
| two_task | 2.0 | 20.0358 | 0.63279 | 0.26226 |

| Text + first image model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| GT | 100.0000 | 1.00000 | 0.00000 |
| zero | 12.7174 | 0.36594 | 0.56303 |
| original | 15.1166 | 0.45773 | 0.54657 |
| two_task | 15.3715 | 0.46869 | 0.52460 |

Camera scales are guidance weights, not translation multipliers. Two-task
LPIPS improves modestly with stronger camera guidance, whereas zero-shot mean
PSNR/SSIM decline and LPIPS is essentially unchanged. Do not choose a universal
guidance scale from these six one-seed windows. Text/image results quantify
recorded-future resemblance, not general generation quality.

## Visual review

Visual inspection of the six camera grids: two-task tracks the recorded kitchen/hallway views more closely in cases0,14,28,42. It still misses unseen content, notably the bedroom/bed in case63, and some rooms/people are distorted. Zero-shot often retains the initial view too much or invents unseen layout. These are visual observations, not measured camera-trajectory accuracy.

Text+image visual inspection includes all10 initial/final-frame comparisons and temporal contact sheets for the directional examples. Two-task retains clear left/right response on both tested images. Original also responds, more weakly on image0. Zero-shot image0 turns are weak; image14 right-turn moves toward a cabinet on the left instead of preserving the requested turn. All three can hallucinate hands, people and geometry; original/two-task hands can be visibly distorted. No claim of universal superiority or general prompt adherence follows from two directional pairs and one seed.

All16 MP4s decode to97frames. Observed/generated border checks pass.
The local gallery is `gallery/index.html`; full five-timestamp contact sheets
are alongside each MP4. The official model/sampler, checkpoint identities,
resolved runtime configs and reuse hashes are retained in the evidence bundle.

Final status (2026-10-31): inference/metrics/rendering exited0; all16 videos
and the evidence ZIP were uploaded and remotely SHA256 verified. Notion tables
were read back and matched. `COMPLETE.json` reports17 verified assets and
`publication.json` is `complete_verified`. The GPU worker exited; the Habitat
application on the other card and the retained allocation remain untouched.
