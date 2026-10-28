#!/usr/bin/env bash
# Create the main Cosmos environment on a Foundary node without using a GPU.
# Miniforge must already be installed in the current user's node-local tree.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
local_root=$(realpath -m "${FOUNDRY_LOCAL_ROOT:-/mnt/local/$(id -un)}")
destination=$(realpath -m "${1:-$local_root/envs/cosmos}")
conda_bin=${CONDA_EXE:-$local_root/miniforge3/bin/conda}
case "$local_root/" in
    /mnt/local/*/) ;;
    *) printf 'Storage root must be under /mnt/local\n' >&2; exit 2;;
esac
case "$destination/" in
    "$local_root"/*/) ;;
    *) printf 'Environment must be inside %s\n' "$local_root" >&2; exit 2;;
esac
[[ -d "$local_root" && -O "$local_root" ]] || {
    printf 'Storage root must exist and belong to the current user\n' >&2; exit 2;
}
[[ ! -e "$destination" ]] || {
    printf 'Refusing existing environment: %s\n' "$destination" >&2; exit 2;
}
[[ -x "$conda_bin" ]] || { printf 'Install Miniforge first: %s\n' "$conda_bin" >&2; exit 2; }
case "$(realpath "$conda_bin")" in
    "$local_root"/*) ;;
    *) printf 'Use your own Miniforge installation under %s\n' "$local_root" >&2; exit 2;;
esac
[[ -O "$conda_bin" ]] || { printf 'Conda must belong to the current user\n' >&2; exit 2; }
command -v uv >/dev/null
export CONDA_ENVS_PATH="$local_root/envs"
export CONDA_PKGS_DIRS="$local_root/cache/conda/pkgs"
export XDG_CACHE_HOME="$local_root/cache"
export HF_HOME="$XDG_CACHE_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export TMPDIR="$local_root/tmp"
export CUDA_VISIBLE_DEVICES=""
export UV_CONCURRENT_BUILDS=2
export MAX_JOBS=2
export OMP_NUM_THREADS=2
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR" "$UV_CACHE_DIR"
hostname
df -h "$local_root"
python3 "$repo/migration/bootstrap_sources.py" --check
"$conda_bin" create --yes --prefix "$destination" \
    --file "$repo/migration/environment/foundary-cosmos-linux-64.explicit.txt"
uv pip install --python "$destination/bin/python" --no-deps \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
    -r "$repo/migration/environment/foundary-cosmos-cu128.txt"
uv pip install --python "$destination/bin/python" --no-deps \
    -e "${COSMOS_DEPS_ROOT:-$repo/.deps}/cosmos-framework-edge"
"$destination/bin/python" "$repo/migration/configure_foundary_runtime.py"
uv pip check --python "$destination/bin/python"
"$conda_bin" env config vars set --prefix "$destination" \
    PYTHONNOUSERSITE=1 \
    CONDA_ENVS_PATH="$CONDA_ENVS_PATH" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_HOME/datasets" \
    HF_XET_CACHE="$HF_HOME/xet" \
    TORCH_HOME="$TORCH_HOME" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions" \
    PIP_CACHE_DIR="$PIP_CACHE_DIR" UV_CACHE_DIR="$UV_CACHE_DIR" \
    UV_PYTHON_INSTALL_DIR="$local_root/python" \
    CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" \
    NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" \
    WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb" WANDB_DATA_DIR="$local_root/wandb/data" \
    WANDB_DIR="$local_root/wandb/runs" TMPDIR="$TMPDIR" GRADIO_TEMP_DIR="$TMPDIR/gradio"
printf 'Created %s; GPU acceptance requires coordinated GPU access.\n' "$destination"
