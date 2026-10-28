#!/usr/bin/env bash
# Recreate the Cosmos environment on a Foundary node without Slurm or root.
# Install Miniforge separately under /mnt/local and initialize conda first.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
destination=${1:?Usage: bootstrap_conda_environment.sh /mnt/local/USER/cosmos/envs/cosmos}
case "$destination" in
  /mnt/local/*) ;;
  *) printf 'Use a node-local /mnt/local environment path.\n' >&2; exit 2 ;;
esac
[[ ! -e "$destination" ]] || { printf 'Refusing existing environment: %s\n' "$destination" >&2; exit 2; }
command -v conda >/dev/null
command -v uv >/dev/null
for key in CONDA_PKGS_DIRS PIP_CACHE_DIR UV_CACHE_DIR HF_HOME TORCH_HOME TMPDIR; do
  case "${!key:-}" in
    /mnt/local/*) ;;
    *) printf 'Set %s to a user-owned /mnt/local path first.\n' "$key" >&2; exit 2 ;;
  esac
done
deps=${COSMOS_DEPS_ROOT:-$repo/.deps}
python3 "$repo/migration/bootstrap_sources.py" cosmos-framework-edge --destination "$deps" --check
# Explicit pins include Python and FFmpeg; use cosmos-conda.yml only when
# deliberately refreshing the Conda lock for a different platform or release.
hostname
df -h /mnt/local
conda create --yes --prefix "$destination" \
  --file "$repo/migration/environment/cosmos-conda-linux-64.lock.txt"
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
export UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS:-2}
export MAX_JOBS=${MAX_JOBS:-4}
# Installation and CPU import checks must not consume unallocated GPUs.
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=
# The historical optional notebook widget declares Python <3.13. Nothing in
# the training code imports it; omit it instead of changing training packages.
requirements=$(mktemp "${TMPDIR%/}/cosmos-requirements.XXXXXX")
trap 'rm -f "$requirements"' EXIT
sed '/^jupyter-compare-view==0\.2\.4$/d' \
  "$repo/migration/environment/cosmos-cu128.txt" > "$requirements"
uv pip install --python "$destination/bin/python" --no-deps \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  -r "$requirements"
uv pip install --python "$destination/bin/python" --no-deps -e "$deps/cosmos-framework-edge"
# TorchCodec needs the environment's FFmpeg and the pinned NPP wheel. Keep
# this environment-specific so Conda restores the previous value on deactivate.
site_packages=$("$destination/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
conda env config vars set --prefix "$destination" \
  "LD_LIBRARY_PATH=$destination/lib:$site_packages/nvidia/npp/lib"
uv pip check --python "$destination/bin/python"
