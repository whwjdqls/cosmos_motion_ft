# Camera-only FiLM after frozen bidirectional EMA200k

Train seven zero-initialized FP32 camera18 + frozen motion-time FiLMs (15,663,872 parameters), while the full EMA200k base/specialists stay frozen. The task is100% camera→motion for50k NEW updates. No video/image/caption or native camera action tokens enter this branch. Empty-GEN video↔motion bridges are bypassed, so camera-only directly adapts the frozen motion specialist; retaining the base does not imply its video bridge participates in C2M.

Global32=2x16, no accumulation, AdamW2e-4/1000 warmup/cosine50k, original motion losses/noise, EMA.999. Camera features and train-only statistics reuse existing validated112503-window caches; differing cache metadata requires the exact array/row pairing receipt. Fixed71 EMA C2M monitoring and2 W&B videos at0/every5k. Full FiLM/EMA/optimizer/schedule/RNG/cursor checkpoints at5k/final and graceful stops; baseSHA required on restore.

The node-local deferred queue, sealed source, CPU receipts and actual post-200k two-rank acceptance are documented in migration/AFTER_BIDIR200K_FULLTEST_C2M_TRAJFILM_SEP2_20261114.md. Do not launch production without the matching preflight receipt or replace the frozen200k base with a different step/source.
