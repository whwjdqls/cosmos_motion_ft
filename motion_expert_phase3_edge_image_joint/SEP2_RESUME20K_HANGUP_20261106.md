# sep-2 strict20k resume and terminal-hangup protection

Both caption-free40/40/20 experiments stopped around2026-11-06 06:51UTC.
Directional's TorchElastic agent explicitly recorded SIGHUP and forwarded it
to both workers. Bidirectional ended at the same time with no recorded model
or OOM traceback; the signal sender/session cause was not established.

Both full20k checkpoints and20k validations were complete. Last logged updates
were20590/20930, so the unsaved tail must repeat. Resumes retain the original
run directories, W&B identities, frozen359-file snapshots and native source
fingerprints. Each checkpoint contains98 bridge tensors,98 Adam states, two
rank RNG states and next global draw640000. No recipe/fingerprint migration,
optimizer reset, data-cursor change or repeated20k validation was needed.

| Model | Current GPUs | W&B run |
| --- | --- | --- |
| Bidirectional + joint | 0,4 | [p3-d47749076808b4bfecc2](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d47749076808b4bfecc2) |
| Directional + joint | 1,7 | [p3-79f9ba1f501245ec7b18](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-79f9ba1f501245ec7b18) |

GPU3 was occupied by another process after the interruption. Under the user's
standing authorization to use unoccupied GPUs, bidirectional rank1 moved to
GPU4. The logical two-rank layout remains unchanged, with RNG restored by rank.
At the07:44UTC verification snapshot the jobs reached20340/20330, with finite
losses/gradients and both original W&B runs online. This is a historical
snapshot; read live status rather than treating it as a current step promise.

Training remains caption-free, global32=2×16, accumulation1,40% V2M/40% M2V/
20% joint, groupedBF16 MMDiT, same P1/P2 specialists, video weight10 and native
motion objectives, no additional Phase3 LoRA/EMA. Restored optimizer/RNG/cursor
state does not imply bit-identical GPU numerical trajectories. Histories remain
append-only, including repeated optimizer-step values after recovery.

Local operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_no_text_resume20k_20261106T072751Z`.
Pointer: `cosmos_storage/operations/phase3_mmdit_no_text_current_ops.txt`.
Read `plan.json`, `PREFLIGHT_AUDIT.json`, `HANGUP_ACCEPTANCE.json`, per-arm
`STRICT_RESUME_VERIFIED.json`, `RUNNING_VERIFIED.json`,
`REMOTE_WANDB_VERIFIED.json`, `COMPLETE.json` and `notion_publication.json`.
Checkpoint pins are local hardlinks. GPU lease/writer locks remain active.

The deployed operation uses external launch wrappers around the original
frozen trainer. This repository now provides the same protection through
`scripts/phase3_agent_entry.py` and `scripts/phase3_worker_entry.py`, included
in future launcher snapshots. The shared image-joint launcher uses these for
preflight/probes, production and W&B, with detached sessions and closed stdin.
This also covers its two-task and video-weight-one entrypoints.

TorchElastic overwrites inherited SIGHUP ignore dispositions, so nohup alone
does not fix its agent. The agent wrapper handles only SIGHUP specially and
delegates TERM/INT/QUIT to the native handler. The worker wrapper ignores HUP
before executing the unchanged trainer/logger and preserves DataLoader spawn
bootstrapping. Native training stop handlers remain intact. For a checkpointed
stop, signal the owned rank0 worker with SIGUSR1; the detached launcher PID is
not the training-stop interface.

Do not blindly rerun the historical20k operation: a later checkpoint may exist,
or a trainer may already own its writer/GPU locks. Future resumes require the
latest complete checkpoint, matching immutable sources, acceptance and exact
saved recipe, plus newly checked GPU ownership. Updating master does not patch
the live snapshots or restart any job.

[Training/recovery report](https://app.notion.com/p/40fd2a44f44b814fb925c3d86c250284).
