#!/usr/bin/env bash
# Run inside an owned GPU allocation. The web UI calls the official rollout.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$ROOT/applications/habitat_gs/run.sh" interactive "$@"
