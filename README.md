# cosmos_motion_ft

This is a research checkout for Cosmos-3 Nano motion and egocentric world-model experiments.

The canonical, current context for agents is [`AGENTS_ALL.md`](AGENTS_ALL.md).
Read it first. It consolidates the active joint-attention design, data contracts, run history, launch rules,
and stale-doc notes.

Current primary work area:

`motion_expert_joint_attention/`

For a fresh-server restore, including environments, cloud paths, data,
normalization arrays, evaluator weights, and checkpoints, use
[`migration/SERVER_MIGRATION.md`](migration/SERVER_MIGRATION.md).

Important constraints:

- Real runs require `/home/jungbin_cho/cosmos-framework`, the `cosmos` env, Slurm GPU nodes, and `/weka/jungbin/...`.
- `nymeria_world/` is older native-camera work and is not the current source of truth.
- `motion_expert/` is a POC/reference area and should not be mixed with the joint-attention trainer.
- Generated logs, Slurm outputs, cached latents, stats, mp4s, and checkpoints are artifacts; do not edit them unless explicitly requested.
