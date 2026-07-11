# Codex Bootstrap

This repository uses one shared agent context file because both Codex and Claude work in this checkout.

Before non-trivial edits, read:

`/home/jungbin_cho/cosmos_motion_ft/AGENTS_ALL.md`

For current joint-attention work, verify claims against code in:

- `motion_expert_joint_attention/task_plan.py`
- `motion_expert_joint_attention/joint_motion_model.py`
- `motion_expert_joint_attention/mot_joint_attention.py`
- `motion_expert_joint_attention/mot_joint_layer.py`
- `motion_expert_joint_attention/train.py`
- `motion_expert_joint_attention/sample.py`

For native official-compatible Phase 1 camera/video training, read:

- `native_phase_training/README.md`
- `native_phase_training/latent_omni_model.py`
- `native_phase_training/latent_nymeria_dataset.py`
- `native_phase_training/experiment.py`
- `native_phase_training/prep_test_eval.py`
- `native_phase_training/visualize_checkpoint.py`
- `native_phase_training/checkpoint_eval_callback.py`
- `native_phase_training/sbatch_phase1_native_camera.sh`
- `native_phase_training/sbatch_checkpoint_eval.sh`

Key reminders:

- Primary work area is `motion_expert_joint_attention/`.
- Active isolated native-Cosmos Phase 1 work is in `native_phase_training/`.
- This is not a standalone package; real runs require `/home/jungbin_cho/cosmos-framework`, the `cosmos` env, Slurm GPU nodes, and `/weka/jungbin/...`.
- Do not edit generated logs, Slurm outputs, cached latents, stats, mp4s, or checkpoints unless explicitly requested.
- Use the smallest relevant verification after edits.
