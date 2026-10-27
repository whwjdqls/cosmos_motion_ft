#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$HERE/../motion_expert_phase3_edge/run.sh" "$@"
