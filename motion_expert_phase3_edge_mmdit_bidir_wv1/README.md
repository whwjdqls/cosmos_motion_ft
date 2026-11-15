# Bidirectional two-task MMDiT, video weight one

Fresh 200k counterpart of the directional temporal-local camera variant, with
bidirectional bridge attention and camera prediction removed. V2M/M2V are sampled
50/50, captions remain empty in both specialists, and there is no joint task.
Video weight is 1; native motion reconstruction/physical weights are unchanged.
Use EMA 0.999 from step zero, grouped BF16 bridges, global batch 32 (2x16), and
the same frozen Phase1/Phase2 checkpoints and fixed71 validation assets.

Launch with `python -m motion_expert_phase3_edge_mmdit_bidir_wv1.launch`, the
standard local Phase1/Phase2/cache/output paths, `--ema-decay 0.999`,
`--validation-weights ema`, and `--validation-config`. The guarded launcher
requires empty GPUs or explicit coordinated owned PIDs, native two-rank preflight,
and a full-state 4-to-6 update reload probe before fresh production.
Use the saved source snapshot and contract for resume and evaluation. The native
trainer protects exact100k from rotation; arm the independent milestone watcher.
