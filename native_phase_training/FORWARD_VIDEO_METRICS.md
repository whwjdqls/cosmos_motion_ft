# Forward Video Metrics

The canonical 71-sequence forward-dynamics evaluation now has two optional
advanced metrics in addition to PSNR, SSIM, and LPIPS.

## Metric contracts

- `DreamSim 0.2.1`: the official ensemble model. It measures aligned RGB-frame
  cosine distance over generated frames 1--96 (lower is better). The report
  includes full-suffix and early/middle/late means. DreamSim is an image metric;
  it does not independently measure temporal realism.
- `CD-FVD VideoMAE-v2-SSv2`: the CVPR 2024 content-debiased FVD toolkit at
  revision `a1e037ab7cb087debd2221d14ae4a001ec054201`. It replaces the legacy
  TensorFlow/I3D feature extractor with the toolkit's self-supervised
  VideoMAE-v2-SSv2 features. The full score uses every generated suffix frame
  (RGB frames 1--96); horizon scores use every frame in their 32-frame range.
  The model's temporal positional embeddings are interpolated by the official
  toolkit for these lengths. It is not numerically comparable to legacy I3D FVD.
  The evaluator also pins the official `vit_g_hybrid_pt_1200e_ssv2_ft.pth`
  checkpoint by SHA-256, so a changed download fails instead of changing scores.

Both evaluators exclude conditioned frame 0, require all 97 GT/generated
frames, and apply the same native aspect-preserving bicubic resize/pad used by
`evaluate_inverse_forward.py`. CD-FVD is a set metric. With only 71 held-out
sequences its absolute value has substantial finite-sample uncertainty; use it
for controlled comparisons on these exact 71 windows, not as a universal score.

## Setup

Run setup on the login node; run metric inference on a GPU node:

```bash
bash native_phase_training/setup_forward_video_metrics.sh
```

The setup pins the external CD-FVD checkout under
`~/.cache/cosmos_motion_ft/third_party/` and installs DreamSim into the `cosmos`
environment without replacing PyTorch or other dependencies.

## Evaluation

```bash
PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft \
  /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  native_phase_training/evaluate_forward_dreamsim.py \
  --inference-root RUN/eval_full71_inverse_forward/iter_STEP/inference \
  --eval-root /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2

PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft \
  /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  native_phase_training/evaluate_forward_cdfvd.py \
  --inference-root RUN/eval_full71_inverse_forward/iter_STEP/inference \
  --eval-root /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2
```

Outputs are written beside the existing analysis:

- `analysis/dreamsim_metrics.json`
- `analysis/cdfvd_videomae_metrics.json`

The CD-FVD evaluator caches GT feature statistics under the shared evaluation
input root. The cache contract includes sample names, frame indices, model
checkpoint hash, and toolkit revision, so it cannot be reused silently after a
metric-contract or feature-inference-precision change. FP32 is the official
toolkit default and is used for canonical reports; `--half-precision` is only a
non-canonical speed/memory option.
