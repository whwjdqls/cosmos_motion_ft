#!/usr/bin/env bash
# Opt in to the camera-aligned Head representation and its matching train statistics.
# Source this file before launching a NEW training/evaluation process:
#
#   source motion_expert_joint_attention/use_camera_head_v1.sh
#
# Historical checkpoints were trained with the original representation/statistics and
# must not be loaded under this environment except for an explicitly designed migration
# experiment.  A camhead-v1 model should initially use --bones_frac 0 because BONES has
# no synchronized egocentric camera from which to construct the same Head semantics.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead of executing it:" >&2
    echo "  source motion_expert_joint_attention/use_camera_head_v1.sh" >&2
    exit 1
fi

_camhead_data_root="${WEKA_ROOT:-/mnt/projects/ll/jungbinc/weka}/nymeriaplus_kimodo_proportional"
_camhead_run_root="${RUN_ROOT:-${WEKA_ROOT:-/mnt/projects/ll/jungbinc/weka}/cosmos_motion_ft_runs}"
export NYMERIA_UNIEGO_ROOT="${_camhead_data_root}/uniego_rep_camhead_v1"
export MOTION_STATS_MEAN="${_camhead_run_root}/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy"
export MOTION_STATS_STD="${_camhead_run_root}/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy"
export NYMERIA_MOTION_REPRESENTATION="camera_head_recanonicalization_v1"

for _camhead_required in "${NYMERIA_UNIEGO_ROOT}" "${MOTION_STATS_MEAN}" "${MOTION_STATS_STD}"; do
    if [[ ! -e "${_camhead_required}" ]]; then
        echo "[camhead-v1] missing required artifact: ${_camhead_required}" >&2
        unset _camhead_data_root _camhead_run_root _camhead_required
        return 1
    fi
done

echo "[camhead-v1] NYMERIA_UNIEGO_ROOT=${NYMERIA_UNIEGO_ROOT}"
echo "[camhead-v1] MOTION_STATS_MEAN=${MOTION_STATS_MEAN}"
echo "[camhead-v1] MOTION_STATS_STD=${MOTION_STATS_STD}"
unset _camhead_data_root _camhead_run_root _camhead_required
