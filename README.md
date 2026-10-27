# cosmos_motion_ft

This is a research checkout for Cosmos-3 Edge and Nano motion and egocentric world-model experiments.

The canonical, current context for agents is [`AGENTS_ALL.md`](AGENTS_ALL.md).
Read it first. It consolidates the active joint-attention design, data contracts, run history, launch rules,
and stale-doc notes.

Current code includes `native_phase_training/`, `motion_expert_t2m_edge/`,
`motion_expert_phase3_edge*/`, `egocontrol_edge/`, `nymeria_baselines/`, and
`shape_aware_TMR/`. Nano reference code is in `motion_expert_joint_attention/`.

For the current code on a new server, start with
[`migration/SINGLE_REPO.md`](migration/SINGLE_REPO.md): pinned dependencies,
our upstream patches, environment snapshots, and explicit custom server paths.

For a fresh-server restore, including environments, cloud paths, data,
normalization arrays, evaluator weights, and checkpoints, use
[`migration/SERVER_MIGRATION.md`](migration/SERVER_MIGRATION.md).

Important constraints:

- Real runs require the pinned Cosmos framework, a compatible environment,
  owned compute nodes, and external model/data artifacts. Paths are site-specific.
- `nymeria_world/` is older native-camera work and is not the current source of truth.
- `motion_expert/` is a POC/reference area and should not be mixed with the joint-attention trainer.
- Generated logs, Slurm outputs, cached latents, stats, mp4s, and checkpoints are artifacts; do not edit them unless explicitly requested.
