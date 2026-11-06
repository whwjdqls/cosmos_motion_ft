# User-selected W&B destination on Yonsei

On2026-11-06 the user supplied a W&B credential and selected that account as
the destination. Verified identity: **jungbinc**, team/default entity
**jungbinc-upenn**, project **cosmos-motion-ft**. No API key is recorded in
this repository or the operation receipts. The supplied key was entered
through a hidden TTY and saved by the official `wandb login`; credential files
remain mode600. The unchanged base training SDK0.25 also authenticates the
saved login asjungbinc.

## Current links

- Phase2 no shape:
  https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/cfj632ue
- Phase3 bidirectional+joint/video1:
  https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-eed7a56cdcf72f3f4569
- Phase3 validation dashboard (explicit optimizer-step point plots, latest
  cards and matching GT references):
  https://wandb.ai/jungbinc-upenn/cosmos-motion-ft?nw=xikr5ab126f

Existing `scene_motion` cloud runs/views are retained as historical copies.
This is a backfill/new-destination stream, not deletion or transfer of the
source records. Both model processes, checkpoint/optimizer states, losses,
data, sampling and training steps continued without restart.

## Active logging

- **Phase3:** only the old logging sidecar was stopped withSIGTERM and flushed.
  New sidecar on owned2616204compute CPUs uses the exact existing immutable
  JSONL streamer and a separate `launch/wandb_state_jungbinc_upenn` directory.
  It backfilled the complete scalar history and10k validation, then follows
  incoming training/validation events in the new team. It shares the original
  logical run ID; namespace differs by entity. No GPU work was interrupted.
- **Phase2:** its logger lives inside the trainer, so it was left running.
  Official W&B `beta_sync.sync(live=True)` on owned2622173compute CPUs forwards
  both resumed local transaction fragments, existing metrics/media, and new
  appended records into the same logical ID underjungbinc-upenn. The old
  embedded online logger can continue its historical destination; the new
  team receives the live copy. No lossy text-log parsing or model restart.
  The sync finishes when the original transaction stream closes. At a future
  deliberate trainer resume, use the new team directly and retire this old
  session's live mirror after its flush, avoiding concurrent target writers.

The new loggers use an existing isolated UI SDK environment (W&B0.31), without
upgrading the active training environment. API verification checks both target
runs are running with current steps, no-shape contract, video weight1 and the
actual10k validation metrics. New workspace sections/axes/points/GT references
were read back and verified, and the new run notes contain its dashboard link.

## Persistent defaults and future resumes

- An old literal API-key export in `~/.bash_profile` was overriding saved login.
  It was removed without echoing/copying the secret. Login shells now unset
  `WANDB_API_KEY` and select `WANDB_ENTITY=jungbinc-upenn`.
- Existing Slurm allocations may retain their original environment even if
  the submitting shell unsets a variable. **Unset `WANDB_API_KEY` inside the
  actual compute process**, then use the saved login; do not copy a key into
  launchers, docs, command arguments or output.
- The two run-local `env.sh` files now selectjungbinc-upenn and unset inherited
  API keys for future launches. Running Python environments were not altered.
- For this Phase3 run's future recovery, use the original snapshot and tested
  VAE-sync bootstrap, with the destination-specific wrapper:
  `PHASE3_ALLOCATION=<owned_job> bash /scratch2/whwjdqls99/cosmos/operations/wandb_login_switch_20261106/future_resume.sh`.
  It retains global32 and the same checkpoint/data/acceptance identities.

Operation: `/scratch2/whwjdqls99/cosmos/operations/wandb_login_switch_20261106`.
Key-free receipts: `IDENTITY.json`, `OLD_PHASE3_LOGGER_STOPPED.json`,
`PHASE2_LIVE_SYNC_STARTED.json`, `DESTINATION_VERIFIED.json` and
`dashboard/WORKSPACE_VERIFIED.json`. Tmux sessions:
`wandb_phase2_jungbinc_sync_20261106`, `wandb_phase3_jungbinc_20261106`.
Current logging logs are `phase2_sync.log` / `phase3_logger.log` there;
older `scene_motion` sidecar status files are historical.
