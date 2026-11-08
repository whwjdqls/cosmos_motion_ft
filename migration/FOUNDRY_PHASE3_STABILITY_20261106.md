# Foundary sep-1 Phase3 stability audit — 2026-11-06

Report: https://app.notion.com/p/410d2a44f44b800aa692ec6abae46238

The user requested gradient, bridge-attention and gate checks on both current
caption-free50/50 runs, then asked to inspect the old caption-trained checkpoints
first. The old-model audit and figures are published. Current-run diagnostics
are queued at the native10k save: the user explicitly selected waiting instead
of checkpointing/restarting early. Do not stop or modify live training for this.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_stability_20261106/`.
Read `watch_10k_status.json` and actual per-probe COMPLETE receipts; this document
is not evidence that current10k diagnostics or publication have completed.

## Old checkpoints: completed and published

Directional18842 and bidirectional18878, matching immutable resume snapshot,
plus protected source-migrated old10k checkpoints. All14 gates/model are nonzero;
all saved bridge/Adam tensors finite. Gates are signed linear multipliers, not
sigmoids. Small video gates still produce substantial actual residual updates.

Four windows, V2M/M2V, sigma0.1/0.5/0.9,7 bridges,16 heads. Main old M2V probe
explicitly supplies cached captions; V2M is empty. Caption-empty controls also
exist. Native grouped BF16 attention outputs are reconstructed bit-exactly using
FP32 softmax; masking has exactly zero prohibited mass. Early attention is very
sharp and becomes sharper from10k to18.8k. Head selections remain diverse;
local four-frame motion-query similarity does not establish global collapse.

Two native-loss mixed-task batch2 backward probes/model show gate scalars
accounting for99.43–99.997% of squared combined gradient norm in the caption-
present audit. Weighted video loss dominates the sampled branch norms. These
are offline diagnostics, not decomposition of particular live global32 updates.
Do not label every attention matrix or core gradient as exploding.

Additional fixed-projected-QKV FP32 arithmetic comparisons and individual-head
plots are published. This does not test fully FP32 projection/backward behavior.
One-step bridge ablations affect M2V loss strongly but are not generation metrics.

Raw receipts: checkpoint_stats.json, old_summary.json,
old_{directional,bidirectional}_captioned/, old_{directional,bidirectional}_v2/,
old_10k_{directional,bidirectional}/, old_precision_{directional,bidirectional}/.
Source hashes and unchanged-weight receipts are recorded. No optimizer step is
performed by the diagnostic script. No live-source snapshots are modified.

## Current caption-free10k checks: complete and published

Both real10k checkpoints were preserved by the detached watcher and audited
using their exact immutable source snapshots. Both COMPLETE receipts exist.
All saved98 bridge/98 Adam states and two-rank RNG/cursor states were verified;
all inspected tensors, probabilities and logged histories are finite. Actual
inputs at BOTH specialists are empty across all checked cases/tasks. No probe
performs an optimizer step. No training stop/restart/settings change occurred.

Near10k, live logged pre-clip norm medians18.96 directional/14.78 bidirectional.
Offline mixed-task batch2 probes attribute99.988–99.997% of squared norm to gate
scalars. Early attention remains sharp. All scalar gates are nonzero, but BOTH
models' block23 motion branches have only~5–6e-7 target update/input RMS ratio:
they are effectively inactive at this checkpoint. Do not claim all gates are
healthy simply because their scalars are nonzero, or label the whole model as
collapsed solely from peaked attention.

Current sep-1 runs remain directional GPUs4/7 and bidirectional GPUs2/3,
50/50/no joint; W&B p3-1841eb77997bc0da105e and p3-8012b39423adff404c6e.
Native71-window checkpoint validation was still scoring when the current
report was assembled; consult its own COMPLETE/summary and current W&B.
The stability operation COMPLETE is NOT200k training completion or proof of
validation completion. A quantitative no-joint/joint comparison must wait for
actual matched validation results.

User clarified disappointing10k outputs are new sep-2 joint40/40/20 runs.
Remote W&B evaluator asset/sampling identity matches old/new joint models;
bidirectional V2M FID0.505→0.573, MPJPE32.9→36.4cm, skating13.3→17.8cm/s.
Directional changes and video metrics are mixed. Do not infer that joint
training alone caused this regression. No direct sep-2 weights, attention,
gates or visual samples were inspected. Existing validation on both nodes is
CFG1,35 UniPC steps, empty captions. No motion/shape CFG2 is used.

Receipts: new_10k_{directional,bidirectional}/diagnostics.json,
new_10k_checkpoints/, new_10k_summary.json, executive_report.md,
sep2_joint_10k_wandb_comparison.json, NOTION_NEW_10K_PUBLISHED.json, COMPLETE.json.
The read-only watcher finished; do not start duplicate probes or publishers.
The Notion report now contains old audits, current10k bridge audit, sep-2
matched quality tables and a consolidated assessment with hypotheses/proposed
controlled experiments. No proposal was applied to live runs.

## Final validation/report update

Both current sep-1 native71 validations are complete and both trainers continued
beyond10k. All report sections, current gate/attention plots, completed same-node
quality comparisons and proposed (unapplied) ablations are published in Notion.
Current no-joint V2M FID0.618/0.759 and MPJPE35.32/38.99cm are also worse than
same-node caption-trained10k baselines; quality issues are not confined to joint.

Cross-node evaluator weights/definitions match but cases and text-cache hashes
differ, as does node-specific config. Exact same-window/text-reference identity
has not been verified across nodes. Do not claim a causal joint/no-joint benefit
from raw cross-node metric gaps. Read CROSS_NODE_COMPARISON_LIMIT.json.
The initial strict cross-node assertion correctly failed before any purported
matched comparison was published; the final report states the limitation.

Final measured and proposed sections: executive_report.md, new_10k_report.md,
fresh_10k_quality_report.md. COMPLETE.json records actual post-validation running
states; NOTION_NEW_10K_PUBLISHED.json records publication. No future publisher or
stability probe is still required for this request. Training continues unchanged.
