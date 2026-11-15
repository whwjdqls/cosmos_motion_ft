# sep-2 listed MMDiT EMA100k random20 queue

The user requested a30-minute wait, then Git commit/merge/pull and evaluation
of only the models listed in Final Variations, permitting sharing our training
GPUs. Wait ended2026-11-10 01:33:44UTC. Local work was committed652c340 and
merged with origin/master d194609 asb3ddd34, keeping both sides of the only
AGENTS_ALL.md conflict. Local master includes the merge and evaluator guards.
No remote push is claimed.19 bridge/camera/preservation checks and3 explicit
checkpoint-eligibility checks passed.

Operation:

```text
/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_sep2_mmdit100k_random20_20261110
```

Finite native preprocessing and the queue manager are detached, with durable
status/error records. Manager2845512; report follower2917951. Always verify
current process identities. The first four workers started on1/2/3/4 and passed
exact EMA/source loading plus native CFG1 parity for M2V and V2M. Their first64
M2V predictions are saved; initial V2M sampling uses batch64. These are progress
receipts, not completed quantitative evaluation.

| Model key | Exact100k at setup | Training mix V2M/M2V/J | Video weight | EMA start |
|---|---|---|---:|---:|
| bidirectional | protected |40/40/20|10|50017|
| directional | protected |40/40/20|10|50819|
| directional_video1 | protected |40/40/20|1|0|
| camera_local | protected |50/50/0|1|0|
| camera_global | waiting |50/50/0|1|0|
| camera_cross_only | waiting |50/50/0|1|0|

Only these six retained sep-2 rows are eligible. The separate sep-1 queue owns
that node's rows. Deleted duplicates, FiLM and pre-MMDiT prototypes are excluded.
The last two models enter automatically when their independently protected100k
files appear. The manager waits without retaining an idle GPU model/allocation.
It checks resident UID/command and free memory before launching a new worker;
other researchers' processes must not be shared or signalled.

Protocol from the pulled EMA200k/100k evaluator: exact1420 random windows,
20/each of71 test recordings, manifest SHA256
47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df.
M2V whole-bridge CFG1.1, V2M and video+text1.5. EMA/native UniPC35, shifts10/3,
T97/20FPS/256px, supplied neutral shape. Generator captions stay empty; explicit
text is routed only to the frozen motion expert in both branches. Camera starts
from private Gaussian noise; its solver uses the conditional readout without
separate camera CFG. Cross-only must restore its own frozen policy/sample.

All1420 normalized motion/shape windows passed native/cache parity. Older video
latents failed a fresh native check and were not reused for evaluation. The
common_inputs folder is a CPU intermediate only; all final video inputs were
freshly re-encoded once into native_inputs, shared across every selected model.
No shared NAS transfer or live training changes occurred. During preparation
a short cache diagnostic ran onGPU6 after another user occupied it; it exited
and their process was untouched. The actual queue is restricted to our1/2/3/4.

Four355-window virtual shards/model,64-case chunks; requested sampling32M2V/
64V2M, decode4, with OOM backoff. Explicit saved training_tasks allow the listed
joint-trained MMDiT models while preserving the imported50/50 default guard.
Driver is frozen at8369d59 plus operation scripts/hashes in DRIVER_SOURCE_MANIFEST.
Each model uses its own original checkpoint source snapshot/contract.

Each completed model requires4260 outputs, per-window metrics/features/TMR
embeddings, global FID/FVD/retrieval and a passed output AUDIT. FVD uses8520
native I3D400 descriptors. Never average shard distribution scores or pool this
subset with fixed71/full-test results. UPE/LPE/sJPE and physical metrics remain
in the per-case/global reports. Reporter automatically builds combined report.md
and metrics.csv as models complete; root COMPLETE requires all six models.

Read RUNNING.json, MANAGER_ERROR.json, models/*/shards/*/{status,ERROR}.json,
per-model AUDIT/COMPLETE/summary/metrics/report and root REPORT_STATUS.json.
Do not restart a live manager or overwrite partial outputs. Successful shard
chunks are resumable, but inspect errors before any retry.

Notion progress/protocol section is published at:
https://app.notion.com/p/binne/Final-Variations-412d2a44f44b80a2936aca27f96ac016
Pending rows are not metrics. Publish completed reports there when available;
the CPU reporter writes local artifacts, not Notion. Preserve other-node rows.
