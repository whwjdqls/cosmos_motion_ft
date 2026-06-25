#!/usr/bin/env bash
# Wait for ckpt_step<STEP>.pt in the bigLR run, then sample 5 prompts on node 0 and
# measure root/pose jitter through the bit-exact decode pipeline. Writes a summary.
# Usage: eval_at_ckpt.sh <STEP>   e.g. eval_at_ckpt.sh 100000
set -u
STEP="$1"
RUN=/weka/jungbin/cosmos_motion_ft_runs/full_full_lr5e5_proj3x_drop10_20260618_112947
SCR=/home/jungbin_cho/cosmos_motion_ft
CKPT="$RUN/ckpt_step$(printf '%06d' "$STEP").pt"
OUT="$RUN/eval_step$((STEP/1000))k_s5_cfg2p5"
SUMMARY="$SCR/eval_milestone_${STEP}.summary"
NODE=a3ultravis-a3ultranodeset-0
GPU=0
CONDA=/home/jungbin_cho/miniforge3
COSMOS_PY=$CONDA/envs/cosmos/bin/python
KIMODO_PY=$CONDA/envs/kimodo/bin/python

echo "[wait] for $CKPT ..."
while [ ! -f "$CKPT" ]; do sleep 120; done
sleep 30   # let the save flush

echo "[sample] $CKPT -> $OUT"
ssh "$NODE" "cd /home/jungbin_cho/cosmos-framework; \
  CUDA_VISIBLE_DEVICES=$GPU EVAL_OUT='$OUT' $COSMOS_PY $SCR/run_sample_validation.py \
  --ckpt '$CKPT' --shift 5 --cfg 2.5 --steps 50 --frames 120 > $SCR/eval_milestone_${STEP}.log 2>&1"

echo "[measure] jitter"
$KIMODO_PY - "$OUT" "$SUMMARY" "$STEP" <<'PY'
import sys, numpy as np, glob, os
OUT, SUMMARY, STEP = sys.argv[1], sys.argv[2], sys.argv[3]
lines=[f"=== step-{int(STEP)//1000}k jitter (cfg2.5 shift5 50steps) ==="]
rs=[]; ps=[]
for f in sorted(glob.glob(OUT+"/*_joints.npy")):
    j=np.load(f); root=j[:,0,:]
    r=np.linalg.norm(np.diff(root,axis=0),axis=1).mean()
    rel=j-j[:,0:1,:]; p=np.linalg.norm(np.diff(rel,axis=0),axis=2).mean()
    rs.append(r); ps.append(p)
    lines.append(f"  {os.path.basename(f).replace('_joints.npy',''):>10}: root={r:.3f}  pose={p:.3f}  totaldisp={np.linalg.norm(root[-1]-root[0]):.2f}")
if rs:
    lines.append(f"  AVG root-step={np.mean(rs):.3f} m  pose-jitter={np.mean(ps):.3f} m   (real: root 0.010 / pose 0.009; step-50k: root 0.28 / pose 0.11)")
open(SUMMARY,"w").write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
echo "[done] summary -> $SUMMARY"
