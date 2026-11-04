# Original UniEgoMotion final350 metrics update

Completed with exit0. Root COMPLETE.json and VERIFIED_PUBLICATION.json establish
successful scoring/publication; exact remote table contents were rechecked at
06:56KST on November4. The final summary is linked by Notion heading
`40dd2a44-f44b-8148-ac8f-e3bdefb0ecd3`. New inference covers71 RGB-only and213
RGB+trajectory-model input-ablation predictions. No new videos were uploaded.

| Final350 model | MPJPE cm | Motion FID | R@3 % (mean±SD) |
|---|---:|---:|---:|
| GT |0.00|0.000000|70.83±1.80|
| Trajectory specialist |11.29|0.272554|45.83±2.39|
| RGB reconstruction |41.16|0.223031|39.58±0.90|
| RGB+trajectory reconstruction |6.55|0.121541|65.62±2.71|
| Multi-task RGB+trajectory |8.48|0.130458|60.42±2.39|

The user requested metrics only on the existing Video2motion comparison page:
https://app.notion.com/p/binne/Video2motion-comparison-3ffd2a44f44b806ca79ec5987d6fd1f3 .
No videos are rendered or uploaded by this operation.

All four original shape-predicting checkpoints are now verified at stored
epoch349/global_step555800, meaning350 completed epochs. These are distinct
from the newer supplied-shape baselines and mixed-conditioning training run.

| Variant | Final checkpoint SHA256 | Evaluation action |
|---|---|---|
| Trajectory-only | aa49a8b790d531be5825937742dcee2ff814036a850e53c569c27672878f9b4c | Reuse matching final71 predictions |
| All-task RGB+trajectory | 99e8fecb0cab9db65389b95b1e0e327c9c3d0f025f36d985c47902f08121e3d9 | Reuse matching final71x3 input-ablation predictions |
| RGB reconstruction | b676f59219bdb5ca9ca5201b7661897d7751a6a474b96da18d66090d63d6700d | New final71 RGB predictions |
| RGB+trajectory reconstruction | d6d8018319eb8bed9c2709e6d8f64f0b861ed3556ceca24bccf44f82ec2bcd11 | New final71x3 input-ablation predictions |

The old page's RGB-only/reconstruction rows used341420/339832 updates. Keep
them as historical results; the final section must explicitly supersede those
rows for final-checkpoint comparisons. No old artifacts or page sections are
overwritten.

Operation root:
`/scratch2/whwjdqls99/cosmos/operations/uem_final350_metrics_20261104`.
Tmux: `uem_final350_metrics_20261104`; retained one4090 allocation2671154,
node34. The two4090 mixed-conditioning trainer on node32 is unchanged.

Worker order: native `evaluate_uem_variants.py` inference for RGB_recon/recon;
Body23 reconstruction/physics report; Nymeria TMR scoring via
`evaluate_reconstruction_tmr.py`; metrics publication and exact remote table
readback by operation `verify_and_publish.py`. The `all` subdirectory links
the immutable prior final predictions and is never sampled/written.

Preserve the original71 windows, seeds, floor calibration, train normalization,
state-dict weights (no EMA substitution), native1000-step sampler and no supplied
GT shape/caption. Report measured GT. Reuse the exact train-only Body23 leaf
completion adapter; evaluator normalization stays separate from generator stats.
Retrieval uses32 candidates and grouping seeds0/1/2. FID is a71-case diagnostic.

The initial launch note predates completion. Root `COMPLETE.json` and
`VERIFIED_PUBLICATION.json` now verify the finished evaluation and publication;
`worker.log`, per-variant progress and `exit_status` retain execution evidence.
The allocation was retained after completion and subsequently assigned to the
separate full-test V2M sweep.
