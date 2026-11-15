# GA-MPJPE across saved full-test reconstruction evaluations

The user requested GA-MPJPE for **all saved motion-reconstruction evaluations**,
then explicitly narrowed coverage to **full evaluation, not random20**.

Completed/readback-verified central report:
https://www.notion.so/40dd2a44f44b8080aad3cda6b5fea7c8#419d2a44f44b8168be4bd04940f960e8 .

Operation on Yonsei:
`/scratch2/whwjdqls99/cosmos/operations/ga_all_fulltest_20261116`.
Driver: `nymeria_baselines/ga_fulltest_report.py`.

## Coverage

- **25 local rows**, including GT references and different input/cohort/joint
  protocols, not25 different checkpoints. These include all8 completed original
  Phase3/UEM full-test ablations; final555800 supplied-shape UEM with video,
  trajectory and both; final EgoEgo/EgoAllo measured-camera runs; DA3 raw and
  user-approved exclude19 cohorts; and the separate common16 ReViV comparison.
- **18 published Foundry rows**, including3 actually published GT references:
  directional/bidirectional trajectory FiLM EMA100k camera/camera+video/
  camera+video+text; directional/bidirectional EMA200k V2M and V2M+text;
  directional-camera EMA200k; Naeive EMA200k V2M; bidirectional no-camera EMA200k.
- No1420-window/random20, fixed71 previews, incomplete/user-excluded models,
  video-only outputs, camera-pose-only extraction or unpaired Phase2 generation.
  Naeive full-test T2M/TI2M columns are generation, so are outside the requested
  reconstruction scope. Fixed71 FiLM50k monitoring is not a full-test result.

Counts remain explicit per row. Local full test uses11911 annotation cases /
11677 physical windows /71 recordings. The specifically authorized DA3 exclude19
report has11892 annotation cases /11658 physical windows. Published Foundry
full-test results use11677 unique physical windows /71 recordings. These are
not interchangeable denominators or fully matched model/input comparisons.

## Alignment definition and verification

GA-MPJPE fits **one proper rotation, translation and uniform scale per entire
T97 sequence**, jointly over all evaluated frame-joint points, and then averages
Euclidean joint error. Reflections are forbidden. There is no per-frame fit.

Common Body23 results already computed this exact operation under the historical
`pa_mpjpe_m` name. This report reaggregates ALL saved per-case errors, verifies
the score completion/summary hashes, original case order and cohort size, and
independently recomputes first/middle/last clip alignment in float64 for each
local run. Differences must stay below1e-4m. Synthetic global-transform,
time-varying-translation and reflection tests passed. The immediately preceding
GT-camera report independently recomputed all11911 cases/model, not just probes.

Source chain:
`motion_expert_phase3_edge_joint.validation_metrics.physical` →
`nymeria_baselines.evaluate_uem_phase3.measure` →
`motion_expert_joint_attention.eval_motion_recon.joint_metrics`.
The last flattens `(T,J,3)` to `(T*J,3)` before one Umeyama fit.

ReViV is different: its existing `PA_SIM` aligns each frame independently.
We explicitly use its saved **`GA_SIM_MPJPE_mm`**, whose source/configuration and
scorer SHA verify one whole-clip rotation/translation/scale. GT/ReViV/two UEM
comparators use their original conservative16-joint mapping and remain in a
separate group. Never relabel ReViV's11.5971cm per-frame PA as GA; its whole-clip
GA is16.3515cm. Joint sets are not directly interchangeable.

Foundry predictions are not mounted on Yonsei. Their completed FULL-test tables
were fetched from Final Variations page412d2a44f44b80a2936aca27f96ac016, using the
same shared whole-clip validation convention. These values are explicitly
**published source values, not fresh independent local re-fits**. The separate
table records the original heading/table IDs. GT is included only where the
source actually published the physical value, rather than invented as zero.

## Selected full-test results, GA-MPJPE cm

| Local Body23 run / input | Cases | GA-MPJPE cm |
| --- | ---: | ---: |
| GT self |11911|approximately0|
| Phase3 original baseline / video |11911|10.6707|
| Phase3 four-task control / video |11911|11.6754|
| Phase3 hidden-state FiLM / video |11911|12.9681|
| Phase3 floor-aware18D FiLM / video |11911|18.3360|
| Final555800 UEM / video |11911|18.3626|
| Final555800 UEM / trajectory |11911|8.9396|
| Final555800 UEM / video + trajectory |11911|7.7479|
| EgoEgo EMA555800 / measured camera |11911|11.8597|
| EgoAllo200k / measured camera |11911|9.4395|
| EgoEgo / DA3, exclude19 |11892|16.8652|
| EgoAllo / DA3, exclude19 |11892|15.5442|
| UEM / DA3, exclude19 |11892|15.3947|

The Notion report and local CSVs contain all rows, including earlier UEM RGB
variants, unfiltered DA3 results, Common16 and published Foundry conditions.
DA3's assumed initial gravity/height prior and raw UEM instabilities remain
documented. A low aligned error does not establish stable raw trajectories,
correct floor height or good foot contact.

## Artifacts and completion

- `summary.json`: local results, source hashes, exact cohorts, checkpoint
  identities where provided, independent geometry probes, remote source IDs.
- `comparison.csv`: all25 local rows; `remote_reported.csv`:18 remote rows.
- `per_case/`: all cached local sequence-global GA values with original IDs.
- `inventory.json`: scanned saved full-test summaries and eligibility metadata.
- `final_variations_page.json`, `remote_fulltest_sections.json`: source readback
  retained only in Scratch2; not committed to Git.
- `GEOMETRY_SELFTEST.json`, `NOTION_VERIFIED.json`, `COMPLETE.json`: acceptance
  and actual remote table readback. Completion explicitly marks remote values
  `remote_independently_recomputed=false`.

Existing predictions, original score tables, frozen statistics, samplers,
checkpoints and training were not modified by this reporting task. No new model
inference or random20 evaluation was launched.
