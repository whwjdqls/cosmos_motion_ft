"""Environment-driven path resolution for the restored Cosmos motion server."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/mnt/projects/ll/jungbinc"))
WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", PROJECT_ROOT / "weka"))
RUN_ROOT = Path(os.environ.get("RUN_ROOT", WEKA_ROOT / "cosmos_motion_ft_runs"))
HF_HOME = Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface"))
COSMOS_FRAMEWORK_ROOT = Path(
    os.environ.get("COSMOS_FRAMEWORK_ROOT", PROJECT_ROOT / "cosmos-framework")
)
COSMOS_ENV_ROOT = Path(
    os.environ.get("COSMOS_ENV_ROOT", PROJECT_ROOT / "miniconda3" / "envs" / "cosmos")
)
WAN_VAE_PATH = Path(os.environ.get("WAN_VAE_PATH", WEKA_ROOT / "wan22_vae" / "Wan2.2_VAE.pth"))
PHASE1_GEN_INIT = Path(
    os.environ.get(
        "COSMOS_PHASE1_GEN_INIT",
        RUN_ROOT
        / "portable"
        / "native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt",
    )
)


_LEGACY_PREFIXES = (
    ("/weka/jungbin/cosmos_motion_ft_runs", RUN_ROOT),
    ("/weka/jungbin", WEKA_ROOT),
    ("/mnt/shared/jungbin_cho/cosmos_motion_ft_runs", RUN_ROOT),
    ("/home/jungbin_cho/cosmos_motion_ft", REPO_ROOT),
    ("/home/jungbin_cho/cosmos-framework", COSMOS_FRAMEWORK_ROOT),
    ("/home/jungbin_cho/miniforge3/envs/cosmos", COSMOS_ENV_ROOT),
)


def resolve_legacy_path(value: str | os.PathLike[str] | None) -> str | None:
    """Map a checkpoint-recorded old-server path onto the restored roots."""
    if value is None:
        return None
    original = os.fspath(value)
    if os.path.exists(original):
        return original
    for old, new in _LEGACY_PREFIXES:
        if original == old or original.startswith(old + os.sep):
            suffix = original[len(old) :].lstrip(os.sep)
            return str(new / suffix)
    return original
