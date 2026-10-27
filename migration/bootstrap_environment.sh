#!/usr/bin/env bash
# Recreate a separate environment from recorded installed versions. Never
# modify an existing environment or the running Yonsei installations.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:?Usage: bootstrap_environment.sh cosmos|uniegomotion|egoallo NEW_ENV_PATH}
destination=${2:?Supply a NEW environment path}
[[ ! -e "$destination" ]] || { printf 'Refusing existing environment: %s\n' "$destination" >&2; exit 2; }
[[ "$destination" == /* ]] || { printf 'Environment path must be absolute\n' >&2; exit 2; }
host=$(hostname -s)
[[ -n "${SLURM_JOB_ID:-}" && "$host" != *login* && "$host" != cs-gpu-01 && "$host" != node37 ]] || {
    printf 'Install on an owned compute allocation.\n' >&2; exit 2;
}
command -v uv >/dev/null
deps=${COSMOS_DEPS_ROOT:-$repo/.deps}
extra=()
case "$profile" in
  cosmos)
    py=3.13; lock=cosmos-cu128; cuda=cu128; source_dir=$deps/cosmos-framework-edge
    extra=(--extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0)
    ;;
  uniegomotion) py=3.10; lock=uniegomotion-cu118; cuda=cu118; source_dir=;;
  egoallo) py=3.12; lock=egoallo-cu126; cuda=cu126; source_dir=$deps/egoallo;;
  *) printf 'Unknown profile: %s\n' "$profile" >&2; exit 2;;
esac
if [[ -n "$source_dir" && ! -f "$source_dir/pyproject.toml" ]]; then
    printf 'Restore sources first: %s\n' "$source_dir" >&2; exit 2
fi
uv venv --python "$py" "$destination"
# No dependency re-resolution: this is an installed-version snapshot. Editable
# sources are installed separately without upgrading the captured dependencies.
uv pip install --python "$destination/bin/python" --no-deps \
    --index-strategy unsafe-best-match --extra-index-url "https://download.pytorch.org/whl/$cuda" \
    "${extra[@]}" -r "$repo/migration/environment/$lock.txt"
if [[ -n "$source_dir" ]]; then
    uv pip install --python "$destination/bin/python" --no-deps -e "$source_dir"
fi
printf 'Installed snapshot. Check CUDA/imports and the experiment acceptance gate before training.\n'
uv pip check --python "$destination/bin/python"
