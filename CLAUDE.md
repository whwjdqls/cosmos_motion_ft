# Claude Bootstrap

This checkout is shared by Claude and Codex. The canonical repository context is:

`/home/jungbin_cho/cosmos_motion_ft/AGENTS_ALL.md`

For every external data path (/weka/...), its contents, the script that generated it,
and how to regenerate after the a3ultra cluster is gone, read `PROVENANCE.md`. The
external preprocessing pipeline is vendored under `external/`.

Read that file before non-trivial edits. For the current main work area, also inspect the relevant code in
`motion_expert_joint_attention/`, especially:

- `task_plan.py`
- `joint_motion_model.py`
- `mot_joint_attention.py`
- `mot_joint_layer.py`
- `train.py`
- `sample.py`

Do not rely on older root-level design notes or `nymeria_world/` planning docs for current joint-attention work.
