# Caption-free Phase3 replacements

Handover verified at2026-11-05 22:41UTC. Both new runs reached80 updates with
finite losses/gradients after all native acceptance and full-state reload gates
passed. Bidirectional GPUs0/3 averaged about1.37s/update and directional1/7
about1.31s/update after old workers exited. Only experiment identity, caption
policy and dropout differ between the new and old recipes; all359 frozen Python
files per new run were checksum-verified. Code fix merged/pushed asb0a8356.

New W&B: [bidirectional](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d47749076808b4bfecc2),
[directional](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-79f9ba1f501245ec7b18).
Remote recipes verify all-task empty captions and dropout0. Both old captioned
trainers stopped only after BOTH replacements were healthy: bidirectional18651,
directional18635. Their final recovery states contain98 bridge tensors,98 Adam
states, both rank RNGs and exact data cursors; prior10k and final checkpoints
are hardlinked under the operation's `preserved/{attention}/` directories.
All old workers and loggers exited. New training continues toward200k.

Phase3 must not receive captions for any task. The prior sep-2 directional
and bidirectional40/40/20 runs accidentally retained dataset captions for
motion+image→video with10% dropout. Their fixed71 validation already used
empty captions. The replacements start fresh bridges from the same frozen
Phase1 video-I/O EMA200k and Phase2 200k specialists; they do not continue the
captioned bridge checkpoints.

Phase3 datasets now emit empty captions, objective input preparation clears
injected text, and both specialist model entrances clear captions again.
Image-joint sampling also clears captions and rejects caption CFG. Contracts
record all-task empty captions, dropout0 and new no-text experiment identities.
Legacy model checkpoints and source snapshots are preserved for reproducible
historical evaluation; use their matching snapshots when reproducing old text
conditioning. The frozen Phase2 specialist and its training are unchanged.

All other active-run settings are retained: bidirectional GPUs0/3, directional
GPUs1/7, fresh200k, global32=2×16/accumulation1, task probabilities40% V2M,
40% M2V,20% image-only joint; seven MMDiT bridges with grouped BF16 execution,
clean motion E(0), video weight10, original specialist losses/noise/schedule,
no additional Phase3 LoRA or EMA, same local training cache and fixed71
validation every10k (sampling32, decode4, native35/CFG1).

The handover retains both captioned trainers until BOTH replacements pass:

- native two-rank zero-gate specialist parity and frozen-weight checks;
- bit-exact outputs with blank versus poisoned captions for all three tasks,
  with both actual specialist caption inputs observed empty;
- nonzero gradients from each joint target to real bridge parameters;
- fresh4-update save and strict full-state reload through6;
- finite production updates, all three tasks, correct recipe and remote W&B.

Only then signal the old rank0 trainers with SIGUSR1 and verify complete
optimizer/RNG/data-cursor recovery checkpoints and clean process exit.

Local operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_mmdit_no_text_handover_20261105`.
Its `plan.json` contains exact input/output/GPU/PID paths; `status.json` reports
live handover stage. `REPLACEMENTS_HEALTHY.json`, `OLD_RUNS_PRESERVED.json` and
`COMPLETE.json` establish actual cutover. Do not infer completion from this
launch document. Each new run preserves its own immutable source snapshot.
