#!/usr/bin/env bash
# =============================================================================
# run.sh — invariant launch wrapper for the motion_expert_joint_attention repo
# =============================================================================
#
# This encodes the one-and-only correct way to invoke ANY python script in this
# repo (train.py / sample.py / a --smoke self-test / build_bones_pairs.py, ...).
# It activates the `cosmos` conda env, scrubs LD_LIBRARY_PATH (a stale value
# breaks the cosmos CUDA stack), sets the allocator + tokenizer env, cd's into
# the cosmos-framework root (so the relative QWEN_JSON / asset paths resolve),
# fixes PYTHONPATH to see the framework + both motion-expert repos, then exec's
# the requested script through the env's python.
#
# Everything after the wrapper is passed straight to the python script:
#
#     bash run.sh <script.py> [args...]
#
# -----------------------------------------------------------------------------
# HOW TO LAUNCH
# -----------------------------------------------------------------------------
#
# 1) Single-GPU debug (interactive). srun/sbatch locks an entire 8xGPU node, so
#    for debugging just ssh into the node and run directly:
#
#        ssh a3ultravis-a3ultranodeset-0
#        bash run.sh /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py \
#            --data_mix both --objective velocity --batch_size 8 --steps 2000 \
#            --out /weka/jungbin/cosmos_motion_ft_runs/ja_debug
#
# 2) Multi-GPU training. Wrap the SCRIPT (not run.sh) in torchrun by passing the
#    torchrun module + its target as the args to run.sh — i.e. let the wrapper
#    set up env, then exec python -m torch.distributed.run. The simplest form is
#    to invoke torchrun *through* this wrapper via the python interpreter:
#
#        bash run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
#            /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py \
#            --data_mix both --objective velocity --batch_size 64 --steps 200000 \
#            --out /weka/jungbin/cosmos_motion_ft_runs/ja_v1
#
#    (FSDP wraps each MoTJointLayer; only _moe_motion + the motion heads carry
#    grad, so the frozen reasoner/generator weights are sharded read-only.)
#
#    srun CAVEAT: if you launch under srun, use exactly ONE task and let
#    torchrun fan out the per-GPU ranks itself. Do NOT let srun spawn one task
#    per GPU AND torchrun spawn nproc_per_node procs — that double-forks and the
#    ranks collide on the rendezvous port. Always:
#
#        srun --ntasks=1 --gpus-per-node=8 bash run.sh -m torch.distributed.run \
#            --standalone --nproc_per_node=8 \
#            /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py ...
#
# -----------------------------------------------------------------------------
# EXAMPLE USAGES
# -----------------------------------------------------------------------------
#
#   # Full training run (single process, single/visible GPUs):
#   bash run.sh /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py \
#       --data_mix both --objective velocity --batch_size 64 --steps 200000 \
#       --out /weka/jungbin/cosmos_motion_ft_runs/ja_v1
#
#   # Fast self-test: builds the network, runs one joint forward+backward, and
#   # asserts finite loss, nonzero grad on _moe_motion/heads, exactly-zero grad
#   # on the reasoner and _moe_gen pathways:
#   bash run.sh /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py --smoke
#
# =============================================================================

set -euo pipefail

# --- conda: activate the `cosmos` env (cu128) ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos

# --- env hygiene (load-bearing for the cosmos CUDA stack) ---
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# --- cwd: cosmos-framework root (relative QWEN_JSON / asset paths resolve here) ---
cd /home/jungbin_cho/cosmos-framework

# --- import paths: framework + both motion-expert repos ---
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/motion_expert:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention

# --- exec the requested script through the env's python ---
PY=~/miniforge3/envs/cosmos/bin/python
exec "$PY" "$@"
