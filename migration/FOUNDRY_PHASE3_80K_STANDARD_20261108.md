# Standardized80k Phase3 comparison, sep-1

Completed/published2026-11-08: all21 settings,1491 outputs (568 reused/923 new),
exact80k EMA weights. Full tables, chart, paired guidance comparisons, evidence
CSV/ZIP and9 fixed-case videos/contact sheets are on the destination page.
AUDIT/COMPLETE/PUBLISHED and REPORT_PUBLISHED.md record successful publication.
No model or default guidance is selected: user explicitly deferred selection.
This completes sep-1 only; other-server comparisons require their own receipts.

Destination: https://app.notion.com/p/412d2a44f44b80a596efe81457114ea2 .
All three active variants use their exact80000-step EMA checkpoint.
Stable IDs and exact EMA histories are recorded in the page registry.

Operation:
`/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_all_variants_80k_standard_20261108T021836Z`.
Read plan.json and RUNNING/ERROR/COMPLETE before launching anything. Three
isolated evaluators use existing own GPUs7/3/1 with30% caps; all training stays.

Seven settings/model: M2V CFG1/1.1/1.5, V2M CFG1.5/2, video+text→motion
CFG1.5/2. Whole-bridge off reference. Caption enters frozen motion expert only
for video+text, identical in both CFG branches; generator always empty. This
does not change caption-free Phase3 training. No inference camera tokens.

Native35 steps, motion/video shifts3/10, T97/20FPS/256square, supplied neutral
shape, seeds0–70. V2M64+7 contiguous; M2V virtual-rank32+4/32+3, decode4.
The canonical batch rule is shared across servers. PriorW1/80k V2M32 outputs
are kept historically, but rerun at64 rather than silently mixing batch rules.

Eight already-matching settings/568 predictions are verified and reused:
M2V CFG1 all3, W1M2V1.5, W10plain/textV2M2 both directions.13 remaining
settings require923 new outputs. Coverage is1491predictions/21settings total.
Initial verified rows/protocol and final complete matrix are both published.

Portable cohort SHA256:
`a4792d1aa52edc0de938cbc7a8dc87452dd17d7395bd4412013bc1f70930ce71`.
protocol.json and the Notion attachment contain per-case array/caption hashes,
all evaluator weights/stats and metric definitions. Compare path-independent
arrays, not node-local config-file paths. A text-cache superset requires71
embedding parity before ranking; no substitution of FVD426 with pooled71.

Keep independent server sections and node-qualified model IDs. This session
does not claim that another server's evaluations are launched or finished.
