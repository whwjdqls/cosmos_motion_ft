# Quick frozen-expert timestep probe

On 2026-11-05 the user requested timestep probing, then reduced its scope to
a quick fit check. A full sampling/MLP sweep was stopped at that request.
The completed check fits only small linear ridge probes to existing pooled
features; no extra GPU work or model training was performed afterward.

Normalized sigma is recoverable from both frozen experts at all seven bridge
layers in this small check. Held-out video MAE ranges0.0048–0.0127 and
R²0.9964–0.9995; motion MAE0.0178–0.0478 and R²0.9376–0.9927.

| Bridge layer | Video train R² | Video test MAE | Video test R² | Motion train R² | Motion test MAE | Motion test R² |
|---:|---:|---:|---:|---:|---:|---:|
|3|1.0000|0.0048|0.9995|0.9977|0.0178|0.9927|
|7|1.0000|0.0096|0.9981|0.9981|0.0340|0.9710|
|11|1.0000|0.0091|0.9978|1.0000|0.0478|0.9376|
|15|1.0000|0.0127|0.9964|1.0000|0.0383|0.9581|
|19|1.0000|0.0070|0.9987|0.9992|0.0322|0.9687|
|23|1.0000|0.0072|0.9988|1.0000|0.0412|0.9573|
|27|1.0000|0.0110|0.9974|1.0000|0.0460|0.9447|

## Protocol and scope

- Exact Phase1 video-I/O EMA200k and original selected Phase2 motion200k
  used by image-joint Phase3; both experts stayed frozen, all bridges bypassed.
- Capture immediately after layers3,7,11,15,19,23,27. Use each actual newly
  initialized MM-DiT `norm_g`/`norm_m`: FP32 LayerNorm, eps1e-5, gamma1,beta0.
  Mean pool1600 video tokens or97 temporal motion tokens, excluding shape.
- Joint task, blank captions, supplied neutral shape and clean first latent.
  Fourteen randomized quantile strata from native Waver/video and logit-normal/
  motion samplers, both training shift3, plus exact sigma0/1 anchors. Gaussian
  direction is fixed within a recording/modality sweep; recording seeds and
  modality sigma pairings are independent. Sixteen observations per recording.
-16 training,4 validation and8 held-out recordings, no recording overlap:
  256/64/128 observations. Standardization is training-only; validation MSE
  chooses ridge alpha from0.01/1/100. Predictions are not clipped.
- This is a small feasibility check, not a full benchmark or causal evidence
  that explicit timestep conditioning cannot help. Training has fewer examples
  than2048 feature dimensions. Global pooling is more informative than a local
  Wan attention group; no local-group time probe or Phase3 time-conditioning
  training ablation was run. The experts already receive native time embeddings.

Batch-one extraction temporarily shared GPU1 with the user's explicit approval,
peaking around9GB. Acceptance verified bit-exact bridge-bypass versus disabled
expert outputs, own-sigma feature sensitivity, opposite-modality sigma
independence, no gradients and unchanged expert parameter versions. The probe
process exited and released GPU1; existing training/checkpoints were untouched.

## Artifacts

Sep-1 node-local operation:
`/mnt/local/jinhyung.kim/cosmos/operations/timestep_probe_frozen_experts_20261105`.
`QUICK_REPORT.md`, `QUICK_COMPLETE.json`, `quick_fit/results.json`, selected
`quick_fit/cases.json`, probe weights and per-layer prediction-vs-GT PNG/PDF
panels are retained there. `plan.json`, `source_manifest.json`, inputs/features,
`feature_configuration.json`, `ACCEPTANCE.json` and the exact extraction/fit
scripts retain provenance. The raw quick results SHA256 is
`8ca5adfcaa520002b22c558e8cb3a05f79cd4236d09d29d7988faf855bb121cd`.
These node-local artifacts are not automatically available on sep-2 or Yonsei.
