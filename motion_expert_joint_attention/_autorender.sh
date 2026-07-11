#!/bin/bash
# Auto-render watcher: renders every viz_step*/ (that has a manifest but no mp4) to skeleton mp4.
unset LD_LIBRARY_PATH
RENDER=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/render_viz.py
PY=/home/jungbin_cho/miniforge3/envs/kimodo/bin/python
echo "[autorender] started $(date)"
while true; do
  for vdir in /weka/jungbin/cosmos_motion_ft_runs/*/viz_step*/; do
    [ -f "$vdir/manifest.json" ] || continue
    [ -f "$vdir/.rendered" ] && [ ! "$vdir/manifest.json" -nt "$vdir/.rendered" ] && continue
    # count only generated npys (do_viz also saves *_gt.npy GT companions -- not rendered directly)
    n=$(ls "$vdir"/*.mp4 2>/dev/null | wc -l); m=$(ls "$vdir"/*.npy 2>/dev/null | grep -cv '_gt\.npy$')
    [ "$n" -ge "$m" ] && [ "$m" -gt 0 ] && { touch "$vdir/.rendered"; continue; }  # already done
    echo "[autorender] $(date) rendering $vdir ($m clips)"
    $PY $RENDER "$vdir" > /dev/null 2>&1 && touch "$vdir/.rendered" && echo "  done -> $(ls "$vdir"/*.mp4 2>/dev/null | wc -l) mp4"
  done
  sleep 120
done
