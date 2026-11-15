# Temporal-local cross-only camera Phase3

Matched ablation of the directional temporal-local camera/no-camera-loss run.
All private MMDiT norms/Q/K/V/output projections and zero-initialized gates,
seven layer positions, parameter count and frozen specialists are retained.
Only bridge connectivity changes: no video-video or motion-motion bridge keys.
Cross keys stay inside Wan groups `{0},{1..4},...,{93..96}`. Clean queries get
exact zero residuals; empty attention rows are handled without NaNs or self
edges. Shape bypasses the bridge. Native specialist attention is unchanged.

Fresh 200k, V2M50/M2V50/J0, no captions, videoW1, EMA0.999 from0, global32=
2x16/accum1, same standard Phase1 video-I/O EMA200k and selected Phase2 200k.
Camera training inputs are noisy GT tokens, predictions are detached, and
camera loss is zero. Validation samples cameras from independent noise, uses
EMA/native35-step UniPC/CFG1 on fixed71 every10k, and scores motion/video.

Launch with `python -m motion_expert_phase3_edge_camera_cross_only.launch` and
the existing directional launcher flags, including `--attention-scope
temporal_local`, local camera/cache paths and fixed71 validation configuration.
The guarded launcher requires two empty coordinated GPUs, freezes sources,
passes actual2-rank/micro16 native+camera preflight, then4→6 full-state reload
before production. Resume with this package and its own frozen sources/contract.
For external inference call `policy.configure_checkpoint(payload)` and this
package's `sample.sample`; a generic joint-attention loader is insufficient.

`bridge_attention_type=cross_only` is saved in the recipe and fingerprint.
The grouped rectangular kernel is equivalent to the dense cross-only mask;
tests independently check outputs/gradients, zero-gate identity, clean/shape
bypass, safe empty rows, BF16 checkpoint recomputation and recipe rejection.
