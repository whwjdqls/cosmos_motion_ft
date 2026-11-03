# Shared FiLM checkpoints for sep-2

Staged from sep-1 on 2026-11-01 and SHA256 verified after copying. No GPU
tests or sep-2 jobs were launched. The trajectory trainer continues on sep-1.

Shared parent: `/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/phase3`.

| Model | Saved step | Package directory | Checkpoint bytes |
| --- | ---: | --- | ---: |
| Hidden-state camera FiLM | 200000, final | `hidden_state_film_200k_20261101` | 1,635,911,779 |
| Floor-aware trajectory FiLM | 180000, latest saved at handoff | `trajectory_film_180k_20261101` | 1,597,758,323 |

The hidden checkpoint is `checkpoints/step_000200000.pt`, SHA256
`6258b69ac92fcd8e144adc30dc8d97b34467927bddaa25eb55e6a5e40502ed88`.
The trajectory checkpoint is `checkpoints/step_000180000.pt`, SHA256
`5f0a970139c102b6f1c32853910f54c5248de34ffaa93f204020eafb213d844b`.
The trajectory run had passed182k but had not saved190k or completed200k.

Each package includes `README_SEP2.md`, `SHA256SUMS`, `READY.json`,
`HANDOFF.json`, native contract/specialist metadata, checkpoint verification,
the actual compatible scientific source archive and sep-1 execution provenance.
The trajectory package additionally includes its original18D camera
normalization. `FILM_CHECKPOINTS_20261101.json` in the shared parent indexes both.

## Copy on sep-2

Check hostname and local capacity, then copy each package to
`/mnt/local/jinhyung.kim/cosmos/models/phase3/` using the exact rsync command in
its README. Run `sha256sum -c SHA256SUMS` from the copied package. Load weights
and datasets from sep-2 local NVMe, not the shared volume.

Both checkpoints store bridge/FiLM and full training state, requiring the
original frozen four-task Phase1 EMA200k and original selected Phase2
cross-caption200k specialists. Existing matching sep-2 copies can be reused.
`DEPENDENCIES.json` references their already-shared archives under
`cosmos_data/yonsei_training_20261027/archives`, plus Edge assets, native DCP,
Wan VAE, normalization and optional motion/camera/trajectory caches. Those
large dependencies were not duplicated during this transfer. Do not substitute
the newer two-task Phase1 or the current no-shape Phase2 experiment.

Source is pinned to268fe8d8f57d7d824cf63948bb434323f0d10104 with the historical
shared-data overlay required by the contract. Preserve scientific hashes and
translate reads to equivalent local artifacts. Sep-1 launcher admission is
host-specific and cannot serve as sep-2's owned GPU admission.

Local transfer receipts:
`/mnt/local/jinhyung.kim/cosmos/operations/stage_hidden_film_200k_share_20261101/SHARED_VERIFIED.json`
and
`/mnt/local/jinhyung.kim/cosmos/operations/stage_trajectory_film_latest_share_20261101/SHARED_VERIFIED.json`.
Previously published matched evaluations used hidden150k/trajectory130k; no
new200k/180k evaluation result is claimed by this checkpoint transfer.
