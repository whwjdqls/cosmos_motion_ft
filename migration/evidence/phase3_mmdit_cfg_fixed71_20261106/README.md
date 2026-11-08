# Phase3 MMDiT fixed71 CFG experiment archive

Exact evaluation sources, pinned plans and quantitative results from the sep-1
caption-free, no-joint directional/bidirectional 10k experiment. This archive
contains no checkpoints, sampled latent arrays, datasets, environments or credentials.
The experiment report is [here](../../FOUNDRY_PHASE3_MMDIT_CFG_FIXED71_20261106.md).

The root runner/helper implement cross-edge CFG2. `whole_bridge/` contains the
whole-bridge-off CFG2 runner/helper and the three-policy comparison builder.
`verify_cfg1.py` reproduces native CFG1 against the saved first32 cases/task/model.
`metrics.csv` uses Git-friendly LF line endings with identical cells.
`SHA256.json` records the exact archived file bytes; these Python sources were
syntax checked without importing or launching any model.

These are archived operational scripts, not a portable production entrypoint.
Both plans retain the exact original sep-1 snapshot, specialist, checkpoint,
validation and output paths for provenance. Do not rerun against those completed
outputs. To reproduce elsewhere, first stage the same checksum-matched assets
on that node's NVMe, copy this directory into a fresh operation directory, and
update both plans' node-local input/output paths. Preserve the relative root /
`whole_bridge` layout and point each `runs[].out` to its corresponding new model
subdirectory. The comparison builder expects the native CFG1 validation under
each recorded training run, followed by completed cross-edge and whole-bridge
results. The runners verify the immutable source, training contract and evaluator
assets before loading models.

Use the site's project Python and framework environment. Launch each selected
model with `run_eval.py directional` or `run_eval.py bidirectional` and set
`CUDA_VISIBLE_DEVICES` only after coordinating GPU access. The original approval
was limited to sep-1 GPUs3/7 alongside the user's training, with a30% process
memory cap, sampling batch32 and RGB decode batch4. This archive is not approval
for a new GPU launch. See the report for full sampling and guidance conventions.
