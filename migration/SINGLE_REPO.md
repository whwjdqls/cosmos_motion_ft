# Running the current code on another server

The repository contains our Phase1/2/3 models, modality bridges and FiLM
variants, EgoControl-Edge adaptation, POC, evaluator, renderers, and baseline
training/evaluation adapters. Third-party source is restored into `.deps/`
from `dependencies.lock.json`; our changes to it are stored in `patches/`.
This is one entry repository, not a claim that CUDA, datasets or pretrained
weights are embedded in Git.

## Code captured on 2026-10-27

| Dependency | Pinned revision | Local changes |
| --- | --- | --- |
| Cosmos Edge | `d4599e2e43fbd06168e9884205b9b66c3902d8f6` | Clean upstream checkout |
| Cosmos Nano reference | `82f82293ffd8983651cd51d8191287da3973f534` | Clean upstream checkout |
| Kimodo | `5e3daacf09887a6c8581a8f496629b008b6ba4d5` | Clean user fork |
| UniEgoMotion | `580c92c6d70a91672c4106bab30ca82cdd80f379` | Four patched/new files, including the Nymeria-SOMA adapter and configuration |
| EgoAllo | `0c20ae8c8d48be4ff682803ea523bcb00a6a708d` | Six patched/new files, including the body-only adaptation, corrected data adapter and training schedule helper |

The baseline changes previously existed only in Yonsei Scratch2. Patch and
resulting-file SHA256 checks now protect their restore. Dependency checkouts
retain their own `.git` metadata: Edge's existing framework-revision checks
continue to see the real upstream revision. Upstream licenses remain in those
checkouts. EgoControl-Edge is implemented in `egocontrol_edge/`; it does not
import the reference EgoControl repository at runtime.

## Transfer the current source

The current experiment packages and migration setup are included in this
repository. On the destination server, start with:

```bash
git clone https://github.com/whwjdqls/cosmos_motion_ft.git
cd cosmos_motion_ft
```

For a snapshot that also includes subsequent local, unpublished changes,
export the tracked and nonignored untracked files:

```bash
python3 migration/export_code_bundle.py /path/outside/repo/cosmos_source.tar.gz
```

This excludes `.git`, `.deps`, ignored local configuration, logs and video
outputs. It verifies every included file against the embedded SHA256 manifest.
It does not serialize the shell environment, credentials, or scheduler state.
After transferring, unpack into an empty location and work inside
`cosmos_motion_ft/`. Historical notes retain their original absolute paths.

## Restore source and environments

```bash
python3 migration/bootstrap_sources.py
python3 migration/bootstrap_sources.py --check
cp migration/new_server.example.sh .site.local.sh
# Edit .site.local.sh: storage, runs, environment and optionally CUDA paths.
source .site.local.sh
```

The source installer verifies existing destinations and refuses source drift;
it does not reset or overwrite an existing modified checkout. Individual
components can be named, e.g. `cosmos-framework-edge kimodo_open`. Network
access and any required Git credentials are needed for first-time fetching.

Install on an owned compute allocation with `uv` available:

```bash
bash migration/bootstrap_environment.sh cosmos "$COSMOS_ENV_ROOT"
# Optional baselines need separate Python/Torch environments:
bash migration/bootstrap_environment.sh uniegomotion /path/to/envs/uem
bash migration/bootstrap_environment.sh egoallo /path/to/envs/egoallo
```

The recorded Linux x86_64 environments are Cosmos Python3.13/Torch2.10/cu128,
UniEgoMotion Python3.10/Torch2.5.1/cu118, and EgoAllo
Python3.12/Torch2.7.1/cu126. Installed-version snapshots are in `environment/`;
they are not wheel archives or a guarantee that every package remains available.
The installer refuses an existing environment, preserves the recorded package
versions, installs editable sources separately and runs `uv pip check`.
CUDA drivers/toolkit, ffmpeg and other system libraries remain server setup.
Do not copy the old environment directory or its compiled extensions to a
different server. A fresh environment/GPU acceptance run is still required.

## Paths and entrypoints

`COSMOS_SITE=custom` is explicit. It requires absolute `COSMOS_STORAGE_ROOT`
and `COSMOS_ENV_ROOT`; it detects this repo and defaults dependencies to
`REPO/.deps`. `WEKA_ROOT`, `COSMOS_RUNS_ROOT`, model paths and framework roots
can be overridden. `restored_env.sh` and `runtime_paths.py` use the same custom
resolver. Existing `yonsei`/`grasp` behavior is unchanged.

Use owned Slurm compute steps, with the target server's account, partition,
QoS, CPUs and GPUs. Do not replay Yonsei job IDs or tmux sessions.

| Work | Entry point |
| --- | --- |
| Edge Phase1 two tasks, prefix1/17 | `native_phase_training/train_phase1_edge_two_tasks_no_text.sh` |
| Edge Phase2 Python entrypoints | `motion_expert_t2m_edge/run.sh` |
| Projection A / self-cross DDP ablations | `train_kv_projection_ddp.sh`, `train_self_cross_knorm_ddp.sh` in that directory |
| Phase3 and variants | Respective `motion_expert_phase3_edge*/run.sh` with the trainer's explicit checkpoint/cache/output arguments |
| Phase3 mask-cache optimization | `scripts/train_phase3_performance.py --training-module MODULE` through the same wrapper |
| EgoControl Edge | `egocontrol_edge/run.sh`; explicitly supply cache, initial checkpoint and validation config |
| Nymeria evaluator | `shape_aware_TMR/nymeria_run.sh` |
| POC | `motion_expert/train_bs_full_contact_site.sh` |
| UniEgoMotion | `nymeria_baselines.train_uem_clean` or `train_uem_variants`, with `--upstream "$UNIEGOMOTION_ROOT"` and explicit cache/output arguments |
| EgoAllo | `nymeria_baselines.train_egoallo_clean --upstream "$EGOALLO_ROOT" --cache ... --experiment ...` |

Legacy one-off Yonsei/H200 shell scripts and publication scripts remain
historical records, not portable entrypoints. The new server does not need
`cosmos_restart_20261023` process snapshots; use the repository entrypoints
and explicit run configuration. W&B/Notion/Hugging Face authentication must
be configured separately. W&B uses the selected team or launcher default.

## Transfer artifacts separately

Required subsets depend on the experiment:

- Exact pretrained Edge/Nano weights and converted DCP, Wan VAE, and tokenizer.
- Nymeria/SOMA data, splits, floor calibration, matching motion statistics,
  T97 latents and accepted Phase3/camera/trajectory caches.
- Phase1 and Phase2 initialization checkpoints for Phase3; resume state for
  a continuation. Preserve optimizer, scheduler, RNG, contracts and pointers.
- BONES data for mixed/BONES runs, evaluator weights, evaluator-specific stats
  and LLM2Vec/text caches for scoring. Training and evaluator stats differ.

See `SITE_STORAGE.md` for original locations and `SERVER_MIGRATION.md` for the
older archive restore tools. Those July/September archive inventories do not
cover every newer October run or cache. Keep source statistics and split
manifests byte-identical. Some cache manifests and checkpoint contracts contain
absolute paths: remap/rebuild them deliberately on the destination and run the
existing acceptance gate. Do not suppress strict fingerprint/hash checks to
force a resume. This code consolidation does not claim an existing checkpoint
can resume on arbitrary paths without that migration check.

## Verification performed

- All five source checkouts restored at the locked commits; repeated
  `bootstrap_sources.py --check` passed, including every patched-file hash.
- Six focused tests passed: existing Yonsei/Grasp behavior, custom shell/Python
  agreement, explicit missing-root failure, and Edge wrapper path propagation.
- No current training job, allocation, dataset, checkpoint, or installed
  environment was changed. Full installation and GPU training on the new
  server have not been tested; that server's hardware/paths are not yet known.
