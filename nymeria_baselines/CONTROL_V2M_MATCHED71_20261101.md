# Phase 3 control 150k: matched video-to-motion evaluation

Requested destination: https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2

Operation on sep-2:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_control_v2m_matched71_20261101T080800Z`.
The `operations/eval_control_v2m_current_ops.txt` pointer selects this operation.

The existing four-task/equal/no-caption control trainer continues on GPUs 1/7.
Its latest completed 150k checkpoint was hard-linked into the operation before
inference, preserving it against training retention. The run uses the original
Phase 1 EMA 200k and selected Phase 2 200k specialists, strict bridge loading and
the contracted historical execution checkout. The main working branch was merged
with current master separately; the pinned evaluation source did not change.

## Matched inputs and settings

- The same 71 fixed held-out T97 windows, one per recording, as the camera-FiLM
  charts; all local conditioning bundles match their published SHA256 hashes.
- Video only and video plus its saved caption, each at bridge scales 1 and 2:
  284 predictions. Same per-case seeds, supplied neutral shapes, 20 fps,
  UniPC 35 steps and shift 3. No measured camera poses in these V2M arms.
- Scale 2 retains the same caption in its whole-bridge-disabled reference.
  It is bridge extrapolation rather than a trained missing-video CFG null.
- Caption conditioning is an inference intervention in both specialist reasoners;
  every 35/70 reasoner-call count is checked. Native sampling with a zero GT motion
  template agrees bitwise with the scale-1 wrapper; captions change predictions.
- GPU 6 is shared with the existing video scorer under the user's explicit
  instruction. Both trainers and the earlier evaluation workers continue.

## Metrics and publication

Physical metrics use the unchanged Body23 implementation: MPJPE, PA-MPJPE,
root ADE/FDE, relative root ADE, height-based foot skate, penetration, airborne
fraction and acceleration error. Positions retain native trajectories; only
PA-MPJPE aligns. GT is recomputed and checked against the published reference.
Control-minus-FiLM differences and caption effects use 5,000 paired recording
bootstrap resamples, seed 20261029. These intervals do not cover training-replica
or generation-seed uncertainty. The snapshots have different training steps.

The user staged the exact Nymeria evaluator under shared
`cosmos/cosmos_models/evaluation/nymeria_matched_tmr`. It was copied once to local
`cosmos_storage/evaluation_models/nymeria_matched_tmr`; all nine files and byte
sizes were checked. Checkpoint SHA256:
`91f9d8adec6bcaa448571dade670da0f94916e448b875a0853eda29ef44a1e21`.
All eight normalization files match its saved contract. The older July C45
fixture is a different checkpoint and is not used.

Semantic scoring uses the same motion encoder constructor and raw weights,
unchanged Nymeria embedding path, train-only Body23-to-SOMA30 completion and
20-to-30 fps conversion. Exact final text embeddings are reused from the
published evidence; no live text encoder/cache is needed. Recomputed completed
and full-GT embeddings match the published arrays bitwise. GT retrieval and the
completion diagnostic match exactly. FID uses all 71 cases; R@1/2/3 uses 32
paired candidates, grouping seeds 0/1/2, 64 cases per seed with seven dropped,
reporting mean and sample SD.

The intended publication adds control rows to the existing video, caption,
semantic and physical tables while preserving all previous rows and media.
`INFERENCE_COMPLETE.json`, `SEMANTICS_COMPLETE.json` and
`PHYSICAL_METRICS_COMPLETE.json` identify completed stages. The final report,
numerical ZIP, remote file SHA256 and `publication.json` distinguish a prepared
result from verified publication. This setup record is not a completion claim.

## Completed measurements

All 284 predictions and their physical/semantic scores completed successfully;
both worker exit codes are zero. All 71 GT physical scores match the reference.

| Input | Bridge scale | MPJPE cm | PA-MPJPE cm | Root ADE cm | FID | R@3 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 1 | 20.97 | 9.74 | 19.12 | 0.225692 | 51.56 |
| Video | 2 | 17.33 | 7.96 | 15.92 | 0.173296 | 61.46 |
| Video + caption | 1 | 17.05 | 8.94 | 15.67 | 0.131431 | 66.67 |
| Video + caption | 2 | 15.30 | 7.77 | 14.03 | 0.132699 | 67.71 |

Control has lower MPJPE than the two evaluated FiLM snapshots at each matched
video/caption/scale setting; all eight paired MPJPE intervals exclude zero.
Native control minus hidden-state 150k is -5.21 cm [-8.02, -2.89]; versus
trajectory 130k it is -10.02 cm [-13.72, -6.87]. The checkpoints differ in
architecture and, for trajectory, training updates and recipe. Native control
is near original 200k on MPJPE (20.97 versus 20.56 cm); this is an interim
checkpoint comparison rather than a final equal-update ranking.

## Verified publication

Completed 2026-11-01 08:55 UTC. Sixteen control rows were added directly to
eight existing Notion tables. All 124 prior table lines and 26 existing file
references were preserved. The numerical archive was downloaded from Notion;
its 1,068,891 bytes and SHA256 matched the local file exactly:
`274ce286bd751209b62a8be456cb6942bd9730e2e8252ac581fbd3862e093a58`.
Operation `publication.json=complete_verified`, `remote_tables_verified.json`,
`remote_verification.json` and `COMPLETE.json` record the final evidence.

## Shared checkpoint backup

User requested transferring the evaluated control model to shared storage.
The full 150k bridge/optimizer/scheduler/RNG milestone and evaluation evidence
are verified at
`/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_data/phase3_control_150k_20261101`.
Checkpoint file: `step_000150000.pt`; SHA256:
`ff529a1be630e381016e6522b6577a7abf91eff2d79866350bfe53ce3d24d3fc`.
Seven payload files total 1,410,838,534 bytes; `READY.json`, `manifest.json` and
`SHA256SUMS` establish readiness. All shared copies passed checksum readback.
The README identifies the separate, existing shared frozen-specialist backups.
Local recovery/evaluation copies remain available for the active control run.
Transfer evidence: local operation `share_control150k_20261101T085837Z`.
