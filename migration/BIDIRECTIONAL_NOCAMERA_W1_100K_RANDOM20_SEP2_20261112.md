# Bidirectional local no-camera W1 EMA100k random20 evaluation

User requested on2026-11-12: evaluate the current bidirectional temporal-local,
no-camera, videoW1 run when it reaches100k and publish quantitative results to
Final Variations. Training continues toward200k. The target is the fresh
sep2_bidir5050_w1_nocamera_EMA0 run, not an older joint or camera variant.

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_bidirectional_nocamera_wv1_EMA100k_random20_20261112`.

The detached controller waits for the independent milestone copy's VERIFIED.json
and checkpoint, checks full SHA256/raw+EMA/optimizer/scheduler/two-rank RNG/data
cursor and step100000, then pins EMA weights. Existing100k preservation watcher
remains independent and was verified alive. No trainer restart or mutation.

Cohort: frozen random20 v1, exactly1420 windows (20 in each of71 recordings),
manifest SHA47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df.
The same verified native inputs/captions/seeds and local evaluator assets are
reused. New100k model outputs are generated independently. All required random20
captions are covered; the separate full-test six-caption issue does not apply.

Tasks: M2V CFG1.1, V2M CFG1.5, video+text→motion CFG1.5, whole-seven-bridge-disabled
reference. Native UniPC35, shifts video10/motion3, supplied neutral shape, initial
image for M2V, T97/20FPS/256px. Generator captions remain empty. Explicit text
only reaches the frozen motion expert, identically in both guidance branches.
No camera stream is used or predicted. The exact checkpoint's own frozen policy
and implementation hashes are validated, including its custom no-camera package.

Batches32M2V/64motion/decode4, logged OOM backoff; four355-case virtual shards and
up to two independent GPU workers. Select only empty GPUs and respect existing
leases, preferring5/3 then0/1/2/4/6/7 as available. Do not share occupied GPUs or
interrupt the existing full-test workers or either trainer.

Metric protocol: model and measuredGT Body23 physical/reconstruction/UPE/LPE/sJPE,
matched91f9d8 TMR FID/retrieval, PSNR/SSIM/LPIPS-Alex/DreamSim/native I3D FVD.
Recompute all1420-case distribution metrics globally, never shard averages;
8520 FVD segments,1408 retrieval cases with12 documented drops per shuffle,
three seeds. Recording bootstrap intervals5000 draws/seed20261110. Per-case scores,
features/embeddings and precise case provenance retained. GPU nativeCFG1 parity,
EMA tensor/source checks and output audits precede completion. Four CPU tests
passed (cohort/input parity, frozen policy restoration/mismatch rejection, actual
80k checkpoint state validation, occupied-GPU rejection). All driver sources parse.

Notion: https://app.notion.com/p/binne/Final-Variations-412d2a44f44b80a2936aca27f96ac016
Queue section published. After auditing, the local controller writes COMPLETE.json
and NOTION_OUTBOX.json with exact report/CSV/page/marker. Publish the completed
metrics through the connected Notion tools when the connection is active. No
persistent local Notion credentials are configured; do not claim an expired chat
tool cell is a durable publisher. Use unique marker
sep2-bidirectional-nocamera-w1-EMA100k-random20-20261112 to prevent duplicates.

Authoritative status: MANAGER_STARTED.json plus PID/start identity, status.json,
MANAGER_ERROR.json, per-shard ERROR/COMPLETE, random20/AUDIT and COMPLETE,
NOTION_OUTBOX and publication receipts. The current watcher is armed and waiting
for the preserved100k, not yet sampling. Do not call startup/checkpoint readiness
an evaluation result. The full-test directional evaluation remains separate.
