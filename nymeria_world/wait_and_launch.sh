#!/usr/bin/env bash
# Wait for camera preprocessing to finish, then launch the real Phase-2 training on node 0.
CAM=/weka/jungbin/nymeriaplus_kimodo_proportional/camera_rgb
# wait until all 5 preprocessing workers have finished
while [ "$(ssh a3ultravis-a3ultranodeset-1 'pgrep -fc preprocess_camera_rgb' 2>/dev/null)" -gt 0 ]; do
  sleep 60
done
N=$(ls $CAM/S*/*.npz 2>/dev/null | wc -l)
echo "preprocessing done: $N sequences" > $CAM/_launch_decision.log
RUN=/weka/jungbin/cosmos_motion_ft_runs/world_camera_nymeria_$(date +%Y%m%d_%H%M%S).log
ssh a3ultravis-a3ultranodeset-0 "bash /home/jungbin_cho/cosmos_motion_ft/nymeria_world/launch_camera_phase2.sh 8 100000 $RUN checkpoint.save_iter=5000 dataloader_train.max_samples_per_batch=16" &
echo "launched training -> $RUN" >> $CAM/_launch_decision.log
