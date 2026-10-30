# Two-task200k versus original200k: inverse and guided forward dynamics

Requested report:
https://app.notion.com/p/binne/408d2a44f44b80e882cfcc8d8f4b7869

Yonsei operation:
`/scratch2/whwjdqls99/cosmos/operations/phase1_twotask200k_guidance2_20261029`

Candidate: completed `edge_phase1_T97_20fps_2gpu_bs16_lora_exclude5_fd_invdyn_no_text_p1_200k_yonsei_20261017/checkpoints/iter_000200000`.
Original: `edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905/checkpoints/iter_000200000`.
Full paths, contracts and metadata hashes are frozen in the operation configuration.

`compare_phase1_guided.py` provides compute-only preparation, scoring and rendering.
The operation's `run.sh` invokes official Cosmos inference, separately resolving each
checkpoint's saved native contract. No training, sampler or checkpoint changes.

## Protocol

- Same71 fixed test windows as `phase1_latest_vs_baseline_20261029`: one window
  per recording, not every test window. T97/20fps/256, EMA, UniPC30, shift10,
  seed0, one observed frame. Case IDs, inputs and data hashes are preserved.
- Forward: camera guidance2, native text guidance1, using the established
  `camera_cfg_inference_config.py`. The null removes camera tokens and retains
  the same first image and caption. Both diffusion caching and text-KV reuse
  are disabled. This is not a trained camera-dropout/null-embedding claim.
- Primary comparison supplies no caption to either checkpoint. Also report
  original200k with its usual caption as a separate input-rich reference.
- Inverse: unchanged native video→camera inference, guidance1, no caption.
  Camera guidance cannot be applied to this task because camera is the output.
- Reuse20 verified original guidance2 videos (10 per caption condition) from
  the earlier guidance sweep, plus all71 original inverse outputs. Existing
  guidance code is byte-identical; checkpoint/input/sampler/artifact hashes
  are checked. Generate193 missing guided videos and71 new inverse outputs.

## Metrics and qualitative outputs

PSNR/SSIM/LPIPS-Alex score generated frames1–96, with native GT resize/pad and
equal recording weights. GT self-metrics are measured through the same code;
PSNR retains the established finite100dB identity convention. Preserve per-frame,
horizon and per-case measurements.

Inverse metrics use the existing backward-framewise9D camera convention:
rotation degrees, direction cosine, predicted/GT scale and path-length ratios,
scale-matched step translation error and Sim(3)-aligned trajectory ATE.
The translation/ATE values are not raw metric trajectory errors. Recomputed
original/GT inverse metrics must match prior results; GT direction cosine can
be below1 because of the existing near-zero-motion regularization.

Paired differences bootstrap71 recordings5000 times with seed20261029. This
does not estimate generation-seed uncertainty or isolate training task mix from
caption policy. Both checkpoints are200k; guidance2 is fixed by the request,
not selected from these new test scores.

Qualitative indices0,7,14,28,42,63 are fixed in advance. Six primary videos show
GT / original200k(no text, cameraCFG2) / two-task200k(no text, cameraCFG2).
Two extra videos retain the original's caption, and two figures compare inverse
trajectories with GT. GT display is spatially aligned exactly as for scoring.
Each97-frame/20fps video marks observed frame0 green and generated frames1–96
red; rendered borders and frame counts are audited after decoding.

## Completion evidence

`METRICS_COMPLETE.json`, `MEDIA_COMPLETE.json` and `publication.json` record
separate completion stages. `COMPLETE.json` is created only after quantitative
tables, all10 media files and the quantitative ZIP are remotely verified.
These markers distinguish scoring, media generation and remote publication.
Inference, scoring and rendering execute on owned Yonsei compute allocations,
never login/cs-gpu-01/node37. The allocation is retained after the evaluation.

Final status: **complete**. `publication.json` is `complete_verified`, all 11
remote artifacts (eight videos, two figures, one quantitative ZIP) passed SHA256
verification, `COMPLETE.json` exists, and `publication_resume.exit_code` is `0`.

## Results: all 71 cases, no skips

Both checkpoints are at 200k. Forward dynamics uses camera CFG 2 and native
text guidance 1, with only frames 1–96 scored.

| Model/input | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| GT identity | 100.000000 | 1.000000 | 0.000000 |
| Original, no caption | 19.283648 | 0.597921 | 0.290324 |
| Original, with caption | 19.327673 | 0.601862 | 0.285381 |
| Two-task, no caption | 19.750820 | 0.618447 | 0.278854 |

Inverse dynamics retains native guidance 1 and no caption.

| Model | Rotation ° ↓ | Direction cosine ↑ | Scale ratio →1 | Scale-matched translation mm ↓ | Sim(3) ATE cm ↓ |
|---|---:|---:|---:|---:|---:|
| Measured GT | 0.010294 | 0.998700 | 1.000000 | 0.000000 | 0.000008 |
| Original | 0.360608 | 0.763071 | 1.113875 | 5.754159 | 3.622723 |
| Two-task | 0.314169 | 0.784360 | 1.079978 | 5.146429 | 3.164110 |

The GT rotation residual comes from the existing regularized 6D rotation
reconstruction; it was not overwritten with zero. All reused original inverse
metrics and measured GT metrics passed parity checks against the prior report.

For matched no-caption inputs, two-task minus original paired changes are
PSNR +0.467 dB [95% CI +0.297, +0.653], SSIM +0.02053
[+0.01429, +0.02739], and LPIPS −0.01147 [−0.01627, −0.00738].
Inverse differences are rotation −0.04644° [−0.05322, −0.04047],
translation −0.60773 mm [−0.70407, −0.50634], and ATE −0.45861 cm
[−0.65017, −0.26939]. Video intervals also favor the two-task checkpoint
against the captioned original reference. These are cohort-specific gains;
they do not isolate task removal from the training caption-policy change.

Inference completed with worker exit codes `0 0`. The guidance audit verifies
30 conditional and 30 camera-absent calls per newly generated video, all at
camera scale 2. All eight comparison MP4s passed frame-count and input/output
border checks. Representative forward previews and inverse plots were visually
inspected; final upload completion is established by the receipts above.

Publication handoff: compute-node downloads from Notion were slow, while an
identical remote SHA256 check through the login connection took 1.4 seconds.
Only the publisher was stopped and resumed through that connection; no model,
scoring or rendering work ran on login. The original coordinator therefore
records `exit_code=143`. Use `publication_resume.exit_code`, `publication.json`
and `COMPLETE.json` for the final publication status. `publication_handoff.json`
preserves the reason and stopped process ID. The Slurm allocation is retained.
