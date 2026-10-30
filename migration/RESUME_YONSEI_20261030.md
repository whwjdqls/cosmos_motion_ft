# Resume the unfinished Yonsei experiments on another server

Prepared 2026-10-30. Repository base: `5f76bb9334bb79431440162c8291dbbfc58d57eb`.
The accompanying source bundle also records any newer working-tree files.

**Use the final guide in `resume_addendum/RESUME.md` on Drive.** The initial
`handoff/RESUME.md` predates the source-identity audit below. `READY.json` marks
the final audited handoff; the worker's `COMPLETE.json` marks checkpoint uploads.

## Runs to continue

| Run | Saved step | Target | Training entrypoint |
|---|---:|---:|---|
| Phase3 four-task control | 107793 | 200000 | `motion_expert_phase3_edge_fourtask.train` |
| Phase3 hidden-state camera FiLM | 92879 | 200000 | `motion_expert_phase3_edge_camfilm.train` |
| Phase3 floor-aware trajectory FiLM | 63864 | 200000 | `motion_expert_phase3_edge_camtrajfilm.train` |
| EgoControl-Edge, seven conditioning blocks | 91345 | 200000 | `egocontrol_edge/run.sh` |

These are completed checkpoint saves, not the last number printed in a log.
EgoControl logged 91509 before interruption but its last complete published save
is 91345. Phase3 uses `checkpoints/recovery_latest.pt` and `latest.json`.
EgoControl needs the entire `iter_000091345` DCP directory: model/EMA, optimizer,
scheduler and trainer. A model-only LoRA delta cannot resume this training.

Phase1 two-task is finished at 200k. Both UniEgoMotion RGB-only and RGB+trajectory
runs finished 350 epochs. They do not need continuation. The deliberately stopped
action-only ablation is outside this four-run handoff.

## Drive contents

- Shared artifact snapshot: `gggdrive:cosmos_data/yonsei_training_20261027`.
  Its 38 archives include base Edge/DCP/VAE, frozen Phase1 EMA200k and Phase2
  Self→cross/K-norm/caption-only B256 200k, statistics, metadata, source camera
  files, accepted motion/camera/18D-trajectory caches, latents, validation inputs.
- New handoff: `gggdrive:cosmos_data/yonsei_resume_20261030`.
  `handoff/` contains this guide, source bundle, exact run metadata/launch
  arguments and the old artifact receipts. `archives/` and `receipts/` contain
  the four newer complete run snapshots. Read `COMPLETE.json` and
  `remote_verification.json` before claiming the new checkpoint upload finished.
- The old run checkpoints in the October27 backup are substantially older.
  Extract the shared snapshot first and the new run snapshots last.

The handoff deliberately excludes credentials, Conda environments, Slurm state
and generated videos. Authenticate W&B/Hugging Face/Drive separately. Preserve
the saved W&B ID and project for a continuation; do not run two writers to the
same run. All four source trainers are stopped at preparation time.

## Restore artifacts and source

Run downloads/extraction on a permitted compute or data-transfer node. Configure
your own rclone remote pointing to the same Drive; its name can differ.

```bash
rclone copy YOUR_REMOTE:cosmos_data/yonsei_training_20261027 shared_backup --checksum
rclone copy YOUR_REMOTE:cosmos_data/yonsei_resume_20261030 resume_backup --checksum
```

Verify each archive's bytes and MD5/SHA256 against its JSON receipt before
extracting. Archives are PAX tar with absolute source paths stored without the
leading slash. Use `tar -tf FILE.tar` to inspect, then `tar -xpf FILE.tar -C STAGE`
into a staging directory. Never blindly extract into `/` on a shared server.
PAX nanosecond mtimes must survive, especially for frozen Phase2 and dataset
files: the resume identities use size/mtime as well as hashes.

Create mappings for the old roots documented in `run_specs.json`:

| Original root | Contents |
|---|---|
| `/lustre/whwjdqls99/cosmos/weka` | Base models, frozen Phase1, statistics, Nymeria sources/latents |
| `/scratch2/whwjdqls99/cosmos` | Caches, frozen Phase2 and current run directories |
| `/home/whwjdqls99/cosmos_motion_ft` | Repository source |

The most reliable strict-resume layout exposes restored artifacts at their old
absolute paths through approved mounts/aliases. If that is impossible, set new
paths explicitly and perform an audited identity migration before launching.
Do not rewrite cached fingerprints or remove checkpoint compatibility checks.
The destination server/path layout is not supplied yet, so arbitrary-path resume
is not claimed to have passed a loading test.

Clone the repository and select the recorded revision, or unpack the verified
`cosmos_source.tar.gz` in `handoff/` (it includes its own SHA256 manifest).
In a **dedicated resume checkout**, overlay the supplied original training file:

```bash
cp resume_backup/resume_addendum/compatibility_overlay/motion_expert_phase3_edge/data.py \
   cosmos_motion_ft/motion_expert_phase3_edge/data.py
sha256sum cosmos_motion_ft/motion_expert_phase3_edge/data.py
# c21a7bb57e2276f668ea7fed5bfba35ea46756f3e9443003c6efce82ac985262
```

All three saved Phase3 contracts expect this exact historical file, found at
Git commit `087ecbe2e2ed8d73208a4487425b404d511443a2`. Current master has a newer
version. The supplied overlay restores the original implementation rather than
altering the contract. The source audit checks all saved implementation hashes,
shared bridge and baseline dependencies against the resulting checkout.
The regular Yonsei checkout was not changed. Do not use this dedicated resume
checkout for a new experiment intended to use the newer loader.
Follow `migration/SINGLE_REPO.md` to restore pinned dependencies and build a new
Cosmos Python3.13/Torch2.10/cu128 environment. Keep the real Edge Git revision.
Use `COSMOS_SITE=custom` and a destination `.site.local.sh`; do not reuse Yonsei
job IDs, CUDA paths or an old environment directory.

## Phase3 continuation

All three runs use T97/20fps/256, Nymeria/SOMA only, global batch32,
two GPUs × microbatch16 × accumulation1, frozen original Phase1 EMA200k and
the same frozen Phase2 200k. Preserve the saved recipe, AdamW state and
200k-step cosine schedule. Do not reset the optimizer or warmup.

The control and hidden-state FiLM have four equally sampled tasks:
first image+motion→video, RGB→motion, camera→motion, camera+RGB→motion.
Trajectory FiLM uses its separate balanced task policy, floor-aware 18D camera
representation and motion-noise policy; its full `contract.json` is authoritative.
Do not interchange native camera actions and floor-aware trajectory caches.

`run_specs.json` contains original argv and exact output/checkpoint paths.
`metadata/` includes each run's contract, source identities, latest pointer,
acceptance record and W&B IDs when present. Explicitly pass the transferred
Phase1, Phase2, motion cache, camera cache and output paths; old defaults can
select the earlier camera cache. Use the revalidated camera cache ending
`revalidated_20261021`; trajectory FiLM additionally uses
`phase3_edge_camera_trajectory18_T97_camfloor_v1_20261023`.

Inside a destination allocation, after site setup, the command form is:

```bash
bash motion_expert_phase3_edge/run.sh \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  /ABS/REPO/scripts/train_phase3_performance.py \
  --training-module RUN_MODULE \
  --phase1 PHASE1_200K --phase2 PHASE2_200K \
  --cache MOTION_CACHE --camera-cache REVALIDATED_CAMERA_CACHE \
  --out ORIGINAL_RUN_DIRECTORY --microbatch 16 --workers 4 \
  --resume auto --acceptance MATCHING_ACCEPTANCE_JSON
```

For trajectory FiLM also supply its recorded `--reference-run` and
`--trajectory-cache`, set `PHASE3_RUN_DIR` to the output directory, and use
`motion_expert_phase3_edge_camtrajfilm/run.sh` for integrated W&B. For control
and hidden-state FiLM run `scripts/stream_phase3_wandb.py` with the original
W&B ID/project/state as in the captured metadata. Preserve the validated
FP32 mask-cache execution policy; do not enable TF32 as part of this resume.

Restore the supplemental `logging_history/` metrics JSONL to the original run
directories before starting the logger. It uses source-record indices for W&B
history. The present logger derives its ID from the **resolved absolute run
path**; a symlink to a different canonical path can produce a new ID despite
copying status.json. To continue the same W&B run, preserve that canonical path
(for example with an approved bind mount), or explicitly adapt and review the
logger to use the captured run ID. Merely copying status.json is insufficient.

The repository wrappers require an owned Slurm compute allocation. A server
without Slurm needs an explicit reviewed compute-access adaptation; do not
fabricate `SLURM_JOB_ID` to bypass access checks.

Before production, confirm source/cache/stat fingerprints and the saved
acceptance fingerprint. On the destination GPU, verify checkpoint step, full
optimizer/scheduler/RNG load, both ranks, finite loss and a fresh checkpoint
reload in an isolated output copy. Preserve strict guards and original files.
The existing acceptance files are provenance, not proof a new server works.

## EgoControl continuation

Keep first image+motion→video only, seven conditioning blocks, original Phase1
EMA200k warm-start lineage, T97/global32=2×16, accumulation1, original losses,
200k schedule and validation previews in the training process.

Use `COSMOS_SITE=custom` and explicitly set `EGOCONTROL_CACHE`,
`EGOCONTROL_INITIAL_CHECKPOINT`, `EGOCONTROL_VALIDATION_CONFIG`,
`EGOCONTROL_MOTION_MEAN`, `EGOCONTROL_MOTION_STD`, `IMAGINAIRE_OUTPUT_ROOT`.
Keep the exact original `EGOCONTROL_RUN_NAME` from `run_specs.json` and set:

```bash
export EGOCONTROL_GPUS=2 EGOCONTROL_MICROBATCH=16
export EGOCONTROL_CONDITIONING_LAYOUT=phase3_seven
export EGOCONTROL_RESUME="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_camera/egocontrol_edge/$EGOCONTROL_RUN_NAME/checkpoints/iter_000091345"
bash egocontrol_edge/run.sh train
```

The launcher selects full strict training-state resume and does not load EMA
into the regular weights during continuation. Preserve `wandb_id.txt`.
Check startup says `resume=true`, restores step91345 and scheduler state, and
advances from there. Restore complete run metadata/configuration alongside DCP.

## Limits

This is a checkpoint/data/source handoff, not a destination deployment. No new
server was supplied or accessed. Exact future performance, device memory and
fingerprint compatibility after path changes still require the destination
acceptance check. Training and evaluator statistics must remain distinct.
