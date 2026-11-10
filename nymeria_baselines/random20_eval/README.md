# Random20 offline MMDiT evaluator

Use the frozen cohort and defaults in
`migration/artifacts/nymeria_random20_v1/`. Run `verify_selection.py` there after
pulling. The same1420 physical windows, captions, seeds and order apply to every
model/node. This package does not allocate GPUs or modify training.

`worker.py` is the checked-in version of sep-1's running EMA200k evaluator,
extended to arbitrary saved steps, late EMA histories, optional input-bundle
reuse and native camera-from-noise sampling. It uses each checkpoint's own
immutable source snapshot and verifies its original contract/source hashes.
The global aggregator recomputes1420-case FID/FVD/retrieval from retained
features/embeddings; it does not average shard distribution scores.

Run inside a coordinated allocation, with the existing site environment and
framework/Kimodo import paths configured. Four **virtual** shards are required,
regardless of physical GPU count; run them sequentially when using fewer GPUs:

```bash
python nymeria_baselines/random20_eval/worker.py 0 --operation "$EVAL_OPERATION"
python nymeria_baselines/random20_eval/worker.py 1 --operation "$EVAL_OPERATION"
python nymeria_baselines/random20_eval/worker.py 2 --operation "$EVAL_OPERATION"
python nymeria_baselines/random20_eval/worker.py 3 --operation "$EVAL_OPERATION"
python nymeria_baselines/random20_eval/aggregate.py --operation "$EVAL_OPERATION"
```

Set `CUDA_VISIBLE_DEVICES` to the coordinated GPU for each worker. Source the
node's `.site.local.sh`; `COSMOS_FRAMEWORK_EDGE_ROOT`, `REPO_ROOT`, source snapshot
and dependency paths must resolve locally. Stage datasets/caches on node-local
storage before running on Foundary.

The operation directory contains `plan.json`, a byte-identical `manifest.json`,
`phase2_contract.json`, and `native_rows.json` indexed by case ID. Its plan uses:

- `run`: `canonical_id`, `step`, `camera`, `run_dir`, `snapshot`, `phase1`, `phase2`,
  `path_map`; preserve each checkpoint's architecture/source history.
- `checkpoint`, `checkpoint_sha256`, `manifest`, `manifest_sha256`, `nymeria_root`,
  `native_rows`, `shards:4`, `sampling_batch:{motimg2video:32,video2motion:64}`,
  `decode_batch:4`, and three `tasks` dictionaries (`id`, `mode`, `guidance`, `text`).
- `source_old_plan`: a node-local native evaluator plan containing its
  `validation_config` path. Its pinned metric assets and original fixed71 case
  manifest are checked before substituting the separately pinned random20 cases.
  Cross-node relocations may set `allow_asset_path_relocation:true`: only the
  original path-dependent config/case-file identities are excluded, while all
  other weight/stat/adapter/preprocessing/text-cache identities remain checked.
- `parent_cohort` optionally supplies the original full-test cohort for an
  additional normalized-motion cache parity check. Native preprocessing,
  checkpoint statistics and bundle checks remain mandatory without that cache.
- `prepared_inputs_operation` optionally points to a completed operation with
  `cases.json`. UUID/start/caption/seed and native bundle/stat checks must match.
  This reuses immutable RGB/video/motion inputs; generated samples remain new.

Camera models use independent private camera noise, never test GT camera.
Whole-bridge CFG scales the motion/video target; camera integration uses its
conditional readout without separate camera CFG. Auxiliary camera outputs are
saved and globally audited. Video captions stay empty. Only the explicit
video+text→motion condition sends text to the frozen motion expert, identically
in both CFG branches.

Success requires every shard's `COMPLETE.json`, then `AUDIT.json` and the global
report. `ERROR.json` inside a shard records failures. All frame scores, I3D
features, motion embeddings and physical scores are retained for later rescoring.
Labels must distinguish raw/EMA, checkpoint step, node, camera prediction,
direction, video weight and EMA start. This queue covers recent MMDiT models;
older FiLM/image-joint prototypes are excluded at the user's request.
