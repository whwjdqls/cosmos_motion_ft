# Trajectory-FiLM W&B logging

Production launches through this package's `run.sh` automatically supervise the
existing `scripts/stream_phase3_wandb.py` sidecar. Set `PHASE3_RUN_DIR` to the
same output directory passed to the trainer. The launcher recognizes the module
`motion_expert_phase3_edge_camtrajfilm.train`, including when launched through
`-m torch.distributed.run`. Preflight launches bypass W&B.

The default destination is `jungbinc-upenn/cosmos-motion-ft`, under comparison
group `edge-phase3-fourtask-camfilm-20261021`. The run name defaults to the output
directory name; its variant is `balanced5050-flooraware18d-motionnoise-camtrajfilm`.
`WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_RUN_GROUP`, and `WANDB_NAME` can override
these labels without changing the training recipe. Use the existing authorized
W&B credentials; do not put API keys in repository files or launch commands.
Yonsei's inherited API key was found to select an account without access to the
target project. The operations worker now unsets it, allowing the existing
authorized `~/.netrc` credentials to be used. A run-local marker named
`wandb_stream/use_netrc_credentials` makes sidecar retries do the same without
restarting training. This marker contains no credentials and does not affect
other runs.

The actual `contract.json` supplies configuration and provenance. Every numeric
field written by the trainer to `metrics.jsonl` is uploaded, including losses,
learning rate, task counts, step time, and memory when present. Charts use
`train/step`, the optimizer step. W&B's internal history step is the source
record number, preserving append-only history after checkpoint resumes.

The sidecar uses a deterministic run ID derived from the resolved output path.
Keep that path and the W&B entity/project unchanged when resuming on another
allocation. Repeated launches resume the same W&B run and skip already-uploaded
records. Replaced or truncated histories are rejected rather than merged
silently. Only one logger may hold the sidecar lock.

Local logging state lives in `<output>/wandb_stream/`:

- `stream.log`: sidecar stdout/stderr, including authentication or network errors.
- `status.json`: W&B run URL, upload progress, and latest optimizer step.
- `supervisor.json`: logger lifecycle and final-sync status.

The supervisor starts logging after both source files exist, retries an exited
logger after 60 seconds, and attempts a bounded final backfill on trainer exit.
Network failures do not stop training; local metrics remain available. Logger
launch is not proof of successful upload: use the sidecar's `status.json` and
W&B URL for that. No checkpoints, datasets, source code, or media are uploaded.
GPU telemetry is not collected by the sidecar; trainer-emitted memory and timing
metrics are retained. All processes run inside the training compute allocation.

This wiring does not start training or approve its preflight. Live W&B upload
has not yet been tested for this new run.
