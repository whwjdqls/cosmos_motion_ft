#!/usr/bin/env bash
set -Eeuo pipefail
umask 0027
family=${1:?a100 or pro6000}
number=${2:?1 or 2}
[[ $number == 1 || $number == 2 ]] || exit 2
root=/scratch2/minsu1206/cosmos/operations/gpu_pair_requests_20261105
op=$root/${family}_${number}
mkdir -p "$op"
exec 9>"$op/request.lock"; flock -n 9 || exit 3
exec >>"$op/request.log" 2>&1
case $family in
 a100) partition=suma_a100; qos=a100_qos; gres=gpu:a100:2;;
 pro6000) partition=asus_pro6000,gigabyte_pro6000; qos=pro6000_qos; gres=gpu:RTXPRO6000:2;;
 *) exit 2;;
esac
printf '%s requested %s GPUs=2 CPUs=16 time=3days qos=%s\n' "$(date -Is)" "$family" "$qos"
export MINSU_GPU_PAIR_OPERATION=$op
srun --job-name="minsu_${family}_pair${number}" --partition="$partition" --qos="$qos" \
 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres="$gres" --mem=128G --time=3-00:00:00 \
 bash -c '
 set -Eeuo pipefail
 python3 - <<"PY"
import json,os,socket,time
from pathlib import Path
p=Path(os.environ["MINSU_GPU_PAIR_OPERATION"])
d={"job":os.environ["SLURM_JOB_ID"],"node":socket.gethostname(),"state":"allocated_idle", "time":time.time(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"cpus":os.environ.get("SLURM_CPUS_PER_TASK")}
(p/"allocation.json").write_text(json.dumps(d,indent=2)+"\n");print(json.dumps(d),flush=True)
PY
 nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$MINSU_GPU_PAIR_OPERATION/gpus.txt"
 exec sleep infinity
 '
