#!/usr/bin/env bash
# Separate simulator environment; run inside an owned compute allocation.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
: "${APP_STORAGE:?Set APP_STORAGE to a bulk-storage directory}"
: "${SLURM_JOB_ID:?Run setup in an owned compute allocation}"
case "$(hostname -s)" in *login*|cs-gpu-01|node37) exit 2;; esac
ENV_ROOT=${HABITAT_ENV_ROOT:-$APP_STORAGE/env}
SOURCE=$REPO_ROOT/.deps/habitat-gs
REV=cb01b2fdf850ce3cab28ad00461a68ac7719f23f
mkdir -p "$APP_STORAGE"
exec 9>"$APP_STORAGE/setup.lock"
flock -n 9 || exit 3
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone https://github.com/zju3dv/habitat-gs.git "$SOURCE"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$REV" ]] || {
  printf 'Expected Habitat-GS revision %s; use a separate pinned checkout.\n' "$REV" >&2; exit 2;
}
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  if [[ -n "${HABITAT_BOOTSTRAP_PYTHON:-}" ]]; then
    "$HABITAT_BOOTSTRAP_PYTHON" -m venv --symlinks "$ENV_ROOT"
  else
    "${CONDA_EXE:-/home/whwjdqls99/miniconda3/bin/conda}" create -y -p "$ENV_ROOT" \
      --override-channels -c conda-forge python=3.12 cmake=3.27 pip ninja
  fi
fi
export PATH="$ENV_ROOT/bin:${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}/bin:$PATH"
export CUDACXX=${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}/bin/nvcc
export CMAKE_BUILD_PARALLEL_LEVEL=${APP_BUILD_CPUS:-4}
python -m pip install --upgrade pip
python -m pip install 'cmake==3.27.9' ninja scikit-build-core pybind11 'numpy>=2,<2.4' wheel setuptools opencv-python-headless
if [[ "${HABITAT_INSTALL_AVATAR_DEPS:-0}" == 1 ]]; then
  python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
  python -m pip install smplx chumpy
fi
# Native static-GS rendering supports optional torch=None. Do not pull the
# avatar/body-training stack into this scene-only environment by default.
python -m pip install attrs gitpython imageio imageio-ffmpeg matplotlib 'numba>=0.60' \
  numpy-quaternion 'pillow==10.4.0' 'scipy>=1.13' tqdm
# An interrupted initial clone can leave a submodule repository without HEAD.
# Git's ordinary submodule update cannot recover that state. Restore only
# these unborn checkouts to the exact parent-pinned commit; preserve all
# existing valid checkouts and let ordinary update validate them afterward.
while read -r mode type revision path; do
  [[ "$mode" == 160000 ]] || continue
  if [[ -e "$SOURCE/$path/.git" ]] && \
      ! git -C "$SOURCE/$path" rev-parse --verify HEAD >/dev/null 2>&1; then
    printf 'Repairing incomplete submodule %s at %s\n' "$path" "$revision"
    git -C "$SOURCE/$path" fetch --depth 1 origin "$revision"
    git -C "$SOURCE/$path" checkout --detach "$revision"
  fi
done < <(git -C "$SOURCE" ls-tree -r HEAD)
git -C "$SOURCE" submodule update --init --recursive --depth 1 --jobs 4
# Static Gaussian CUDA kernels include GLM, which upstream does not vendor.
# Keep the header-only dependency isolated instead of requiring system packages.
GLM_ROOT=$APP_STORAGE/deps/glm
GLM_REV=0af55ccecd98d4e5a8d1fad7de25ba429d60e863
if [[ ! -d "$GLM_ROOT/.git" ]]; then
  mkdir -p "$APP_STORAGE/deps"
  git clone --depth 1 --branch 1.0.1 https://github.com/g-truc/glm.git "$GLM_ROOT"
fi
[[ "$(git -C "$GLM_ROOT" rev-parse HEAD)" == "$GLM_REV" ]]
test -f "$GLM_ROOT/glm/glm.hpp"
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF HABITAT_BUILD_GUI_VIEWERS=OFF \
  python -m pip install --no-deps --no-build-isolation -v "$SOURCE" \
  --config-settings="cmake.define.CMAKE_CUDA_FLAGS=-I$GLM_ROOT" \
  --config-settings="cmake.define.CMAKE_CXX_FLAGS=-I$GLM_ROOT" \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89 \
  --config-settings=build-dir="$APP_STORAGE/build"
python -m pip freeze > "$APP_STORAGE/habitat_environment.txt"
printf '%s\n' "$REV" > "$APP_STORAGE/habitat_revision.txt"
printf 'Simulator environment installed at %s\n' "$ENV_ROOT"
