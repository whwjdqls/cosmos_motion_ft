#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --job-name=p2c45eval
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p2c45eval-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
BUNDLE=/weka/jungbin/shape_aware_motion_eval_c45_20260715
RUN=/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d
CKPT=${PHASE2_C45_CKPT:-${RUN}/ckpt_step200000.pt}
OUT=${PHASE2_C45_OUT:-${RUN}/eval_shape_tmr_c45_native_unipc35}
MANIFEST_DIR=${OUT}/manifests
TEXT_CACHE=${OUT}/benchmark_plus_evaluation_llm2vec.pt
SMOKE_OUT=${OUT}/smoke_gate
BATCH_SIZE=${PHASE2_C45_BATCH_SIZE:-4}
VIZ_LIMIT=${PHASE2_C45_VIZ_LIMIT:-1}

echo "[p2c45eval] node=$(hostname) date=$(date)"
echo "[p2c45eval] checkpoint=${CKPT}"
echo "[p2c45eval] output=${OUT}"
echo "[p2c45eval] T2M=BONES six suites + all floor/guard-valid Nymeria test windows"
echo "[p2c45eval] TI2M=all aligned floor/guard-valid Nymeria windows, text CFG2 and no-CFG1"
echo "[p2c45eval] native x0 shift3, official UniPC-35, per-case fixed noise"
echo "[p2c45eval] C45 stats are evaluator-only; Phase-2 stats unnormalize generator output"
test -f "${CKPT}"
test -f "${BUNDLE}/artifacts/evaluator/c45_step_00005000.pt"
test -f "${BUNDLE}/artifacts/text/benchmark_llm2vec.pt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

mkdir -p "${OUT}"
export SHAPE_TMR_BUNDLE=${BUNDLE}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash "${D}/run.sh" "${D}/prepare_shape_tmr_eval.py" \
  --out-dir "${MANIFEST_DIR}" \
  --bundle-root "${BUNDLE}"

bash "${D}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${D}/precompute_nymeria_tmr_text.py" \
  --captions-json "${MANIFEST_DIR}/evaluator_captions.json" \
  --base-cache "${BUNDLE}/artifacts/text/benchmark_llm2vec.pt" \
  --bundle-root "${BUNDLE}" \
  --out "${TEXT_CACHE}"

echo "[p2c45eval] running one-GPU end-to-end smoke gate"
CUDA_VISIBLE_DEVICES=0 bash "${D}/run.sh" "${D}/eval_phase2_shape_tmr.py" \
  --checkpoint "${CKPT}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --text-cache "${TEXT_CACHE}" \
  --tmr-ckpt "${BUNDLE}/artifacts/evaluator/c45_step_00005000.pt" \
  --tmr-stats "${BUNDLE}/artifacts/evaluator/stats/motion" \
  --out-dir "${SMOKE_OUT}" \
  --cohorts bones_content_overview,nymeria_t2m,nymeria_ti2m \
  --max-cases 2 \
  --batch-size 1 \
  --steps 2 \
  --viz-limit 1
bash "${D}/run.sh" "${D}/_verify_shape_tmr_smoke_outputs.py" \
  --out-dir "${SMOKE_OUT}" \
  --cases-per-cohort 2

bash "${D}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${D}/eval_phase2_shape_tmr.py" \
  --checkpoint "${CKPT}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --text-cache "${TEXT_CACHE}" \
  --tmr-ckpt "${BUNDLE}/artifacts/evaluator/c45_step_00005000.pt" \
  --tmr-stats "${BUNDLE}/artifacts/evaluator/stats/motion" \
  --out-dir "${OUT}" \
  --batch-size "${BATCH_SIZE}" \
  --steps 35 \
  --viz-limit "${VIZ_LIMIT}"

echo "[p2c45eval] completed date=$(date) summary=${OUT}/summary.json"
