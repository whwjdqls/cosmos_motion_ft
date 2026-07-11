# motion_expert_joint_attention

Current joint-attention context is consolidated in:

`/home/jungbin_cho/cosmos_motion_ft/AGENTS_ALL.md`

This directory is the active main work area. The implementation wraps frozen Cosmos-3 Nano reasoner/generator
and adds a trainable motion pathway with shared joint attention.

Read code in this order when changing behavior:

1. `task_plan.py`
2. `nymeria_joint_dataset.py`
3. `joint_motion_model.py`
4. `mot_joint_attention.py`
5. `mot_joint_layer.py`
6. `train.py`
7. `sample.py`

Use `AGENTS_ALL.md` for the task table, data contracts, run history, environment rules, and known gaps.
