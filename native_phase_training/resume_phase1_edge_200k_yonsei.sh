#!/usr/bin/env bash
# Continue the existing filtered Yonsei Phase 1 run from 100k to 200k.
set -euo pipefail

[[ "${SLURM_JOB_ID:-}" == 2168300 ]] || {
    printf 'Run this launcher inside retained four-PRO6000 allocation 2168300.\n' >&2
    exit 1
}

export REPO_ROOT=/home/whwjdqls99/cosmos_motion_ft
export COSMOS_SITE=yonsei
export COSMOS_STORAGE_ROOT=/lustre/whwjdqls99/cosmos
export WEKA_ROOT=${COSMOS_STORAGE_ROOT}/weka
export RUN_ROOT=${WEKA_ROOT}/cosmos_motion_ft_runs
export COSMOS_ENV_ROOT=${COSMOS_STORAGE_ROOT}/envs/cosmos
export COSMOS_FRAMEWORK_EDGE_ROOT=/home/whwjdqls99/cosmos-framework-edge
export COSMOS_FRAMEWORK_ROOT=/home/whwjdqls99/cosmos-framework
export EDGE_MODEL_ROOT=${WEKA_ROOT}/Cosmos3-Edge
export EDGE_DCP_ROOT=${WEKA_ROOT}/cosmos3_edge_dcp
export BASE_CHECKPOINT_PATH=${EDGE_DCP_ROOT}
export WAN_VAE_PATH=${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth
export NYMERIA_LATENT_ROOT=${WEKA_ROOT}/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256
export IMAGINAIRE_OUTPUT_ROOT=${RUN_ROOT}
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export CUDA_PATH=${CUDA_HOME}
export PATH=${COSMOS_ENV_ROOT}/bin:${PATH}
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${COSMOS_FRAMEWORK_EDGE_ROOT}:${PYTHONPATH:-}
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

export NATIVEP1_MODEL_FAMILY=edge
export NYMERIA_NUM_FRAMES=97
export NYMERIA_RESOLUTION=256
export NYMERIA_MODE=mixture
export NYMERIA_DROP_MODES=
export NYMERIA_REPLACE_STANDALONE_C=1
export NYMERIA_STANDALONE_C_SUBJECT=camera_wearer
export NYMERIA_QUALITY_FILTER=
export NATIVEP1_ADAPTATION_MODE=global_lora
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_WORLD_SIZE=4
export NATIVEP1_CLIPS_PER_GPU=8
export NATIVEP1_EXPECTED_LATENT_HW=16
export NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_MAX_ITER=200000
export NATIVEP1_SAVE_ITER=5000
export NATIVEP1_AUTO_EVAL=1
export NATIVEP1_AUTO_EVAL_EVERY=5000
export NATIVEP1_VIZ_N=5
export NATIVEP1_EVAL_PREFIX_LENGTHS=1
export NATIVEP1_AUTO_EVAL_FULL71=0
export NATIVEP1_EVAL_INPUT_DIR=${RUN_ROOT}/native_phase1_eval_inputs_viz5_edge_256_T97_s10_camera_wearer_v1
export NATIVEP1_FULL71_EVAL_INPUT_DIR=${RUN_ROOT}/native_phase1_eval_inputs_full71_256_T97_v2
export NATIVEP1_WANDB_MODE=online
export WANDB_PROJECT=cosmos3_camera
export WANDB_ENTITY=scene_motion
export WANDB_RUN_ID=jzek4tqt
export WANDB_RESUME=must
export NATIVEP1_RUN_NAME=edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905

RUN=${RUN_ROOT}/cosmos3_camera/camera_world_edge/${NATIVEP1_RUN_NAME}
CHECKPOINT=${RUN}/checkpoints/iter_000100000
LOG=${COSMOS_STORAGE_ROOT}/logs/edge_phase1_exclude5_yonsei_2168300_resume_100000_to_200000_20260908.log
for component in model optim scheduler trainer; do
    [[ -s "${CHECKPOINT}/${component}/.metadata" ]] || {
        printf 'Missing resume checkpoint component: %s\n' "${component}" >&2
        exit 1
    }
done

exec 9>/home/whwjdqls99/cosmos_phase1_edge_resume_2168300.lock
flock -n 9 || { printf 'A resume process already holds the training lock.\n' >&2; exit 1; }
ARCHIVE=${RUN}/resume_200k_provenance_20260908
mkdir -p "${ARCHIVE}"
for name in config.yaml config.pkl job_env.yaml launch_info.yaml native_phase1_contract.json; do
    if [[ -f "${RUN}/${name}" && ! -e "${ARCHIVE}/${name}" ]]; then
        cp -- "${RUN}/${name}" "${ARCHIVE}/${name}"
    fi
done
exec > >(tee -a "${LOG}") 2>&1
printf '[phase1-200k] %s node=%s allocation=%s\n' "$(date '+%F %T %Z')" "$(hostname)" "${SLURM_JOB_ID}"
printf '[phase1-200k] checkpoint=%s; restoring model/EMA, optimizer, scheduler step, and trainer\n' "${CHECKPOINT}"
printf '[phase1-200k] GPUs=4 clips/GPU=8 global_batch=32; five-person exclusion unchanged\n'
printf '[phase1-200k] LambdaLinear cycle=200000 warmup=500 f_start=f_max=0.4 f_min=0\n'
printf '[phase1-200k] At step 100000: LoRA LR=1.0025063e-5; action LR=4.0100251e-5; decay to zero at 200000\n'
printf '[phase1-200k] This is an extended-schedule continuation, not a from-scratch 200k reproduction.\n'

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK DEVICE_MEMORY_FRACTION
cd "${COSMOS_FRAMEWORK_EDGE_ROOT}"
exec "${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=4 \
    "${REPO_ROOT}/native_phase_training/run_latent_train.py" \
    --sft-toml "${REPO_ROOT}/native_phase_training/world_camera_nymeria_latent_edge.toml" \
    job.name="${NATIVEP1_RUN_NAME}" \
    job.wandb_mode=online \
    trainer.max_iter=200000 \
    trainer.grad_accum_iter=1 \
    trainer.logging_iter=50 \
    'scheduler.cycle_lengths=[200000]' \
    'scheduler.warm_up_steps=[500]' \
    'scheduler.f_start=[0.4]' \
    'scheduler.f_max=[0.4]' \
    'scheduler.f_min=[0.0]' \
    checkpoint.save_iter=5000 \
    checkpoint.load_path="${CHECKPOINT}" \
    checkpoint.load_training_state=true \
    'checkpoint.keys_to_skip_loading=[]' \
    'checkpoint.keys_not_to_resume=[]' \
    checkpoint.strict_resume=true \
    model.config.compile.enabled=false \
    model.config.ema.enabled=true \
    model.config.parallelism.data_parallel_shard_degree=1 \
    model.config.parallelism.data_parallel_replicate_degree=4 \
    model.config.parallelism.context_parallel_shard_degree=1 \
    model.config.parallelism.cfg_parallel_shard_degree=1
