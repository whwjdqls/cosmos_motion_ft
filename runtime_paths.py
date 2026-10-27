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

# BEGIN EXPLICIT COSMOS SITE SELECTOR
# Keep this block last so these values override historical machine defaults.
from dataclasses import dataclass as _site_dataclass
from pathlib import Path as _SitePath
import os as _site_os


@_site_dataclass(frozen=True)
class CosmosSitePaths:
    site: str
    repo_root: _SitePath
    storage_root: _SitePath
    weka_root: _SitePath
    runs_root: _SitePath
    hf_home: _SitePath
    torch_home: _SitePath
    wan_vae_path: _SitePath
    cosmos3_edge_root: _SitePath
    cosmos3_nano_root: _SitePath
    cosmos3_edge_dcp_root: _SitePath
    base_checkpoint_path: _SitePath


def get_cosmos_site_paths(site: str | None = None) -> CosmosSitePaths:
    selected = site or _site_os.environ.get("COSMOS_SITE")
    if selected == "custom":
        from migration.custom_paths import environment
        paths = environment()
        return CosmosSitePaths(
            site="custom", repo_root=_SitePath(paths["REPO_ROOT"]),
            storage_root=_SitePath(paths["PROJECT_ROOT"]), weka_root=_SitePath(paths["WEKA_ROOT"]),
            runs_root=_SitePath(paths["RUN_ROOT"]), hf_home=_SitePath(paths["HF_HOME"]),
            torch_home=_SitePath(paths["TORCH_HOME"]), wan_vae_path=_SitePath(paths["WAN_VAE_PATH"]),
            cosmos3_edge_root=_SitePath(paths["COSMOS3_EDGE_ROOT"]),
            cosmos3_nano_root=_SitePath(paths["COSMOS3_NANO_ROOT"]),
            cosmos3_edge_dcp_root=_SitePath(paths["COSMOS3_EDGE_DCP_ROOT"]),
            base_checkpoint_path=_SitePath(paths["BASE_CHECKPOINT_PATH"]),
        )
    if selected == "yonsei":
        repo_root = _SitePath("/home/whwjdqls99/cosmos_motion_ft")
        storage_root = _SitePath("/lustre/whwjdqls99/cosmos")
        weka_root = storage_root / "weka"
    elif selected == "grasp":
        repo_root = _SitePath("/home/jungbinc/cosmos_motion_ft")
        storage_root = _SitePath("/mnt/projects/ll/jungbinc")
        weka_root = storage_root / "weka"
    elif selected is None:
        raise RuntimeError(
            "COSMOS_SITE must be set to 'yonsei' or 'grasp' before importing runtime_paths"
        )
    else:
        raise RuntimeError(
            f"Unsupported COSMOS_SITE={selected!r}; expected 'yonsei' or 'grasp'"
        )

    return CosmosSitePaths(
        site=selected,
        repo_root=repo_root,
        storage_root=storage_root,
        weka_root=weka_root,
        runs_root=weka_root / "cosmos_motion_ft_runs",
        hf_home=storage_root / ".cache" / "huggingface",
        torch_home=storage_root / ".cache" / "torch",
        wan_vae_path=weka_root / "wan22_vae" / "Wan2.2_VAE.pth",
        cosmos3_edge_root=weka_root / "Cosmos3-Edge",
        cosmos3_nano_root=weka_root / "Cosmos3-Nano",
        cosmos3_edge_dcp_root=weka_root / "cosmos3_edge_dcp",
        base_checkpoint_path=weka_root / "cosmos3_nano_dcp",
    )


SITE_PATHS = get_cosmos_site_paths()
COSMOS_SITE = SITE_PATHS.site
COSMOS_REPO_ROOT = SITE_PATHS.repo_root
COSMOS_STORAGE_ROOT = SITE_PATHS.storage_root
COSMOS_RUNS_ROOT = SITE_PATHS.runs_root
COSMOS3_EDGE_ROOT = SITE_PATHS.cosmos3_edge_root
COSMOS3_NANO_ROOT = SITE_PATHS.cosmos3_nano_root
COSMOS3_EDGE_DCP_ROOT = SITE_PATHS.cosmos3_edge_dcp_root

_site_exports = {
    "COSMOS_SITE": SITE_PATHS.site,
    "COSMOS_REPO_ROOT": SITE_PATHS.repo_root,
    "COSMOS_STORAGE_ROOT": SITE_PATHS.storage_root,
    "PROJECT_ROOT": SITE_PATHS.storage_root,
    "REPO_ROOT": SITE_PATHS.repo_root,
    "WEKA_ROOT": SITE_PATHS.weka_root,
    "COSMOS_RUNS_ROOT": SITE_PATHS.runs_root,
    "RUN_ROOT": SITE_PATHS.runs_root,
    "HF_HOME": SITE_PATHS.hf_home,
    "TORCH_HOME": SITE_PATHS.torch_home,
    "WAN_VAE_PATH": SITE_PATHS.wan_vae_path,
    "COSMOS3_EDGE_ROOT": SITE_PATHS.cosmos3_edge_root,
    "COSMOS3_NANO_ROOT": SITE_PATHS.cosmos3_nano_root,
    "COSMOS3_EDGE_DCP_ROOT": SITE_PATHS.cosmos3_edge_dcp_root,
    "BASE_CHECKPOINT_PATH": SITE_PATHS.base_checkpoint_path,
    "COSMOS_PHASE1_GEN_INIT": SITE_PATHS.runs_root
    / "portable"
    / "native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt",
}
for _site_name, _site_value in _site_exports.items():
    _site_os.environ.setdefault(_site_name, str(_site_value))


def _site_coerce_like_existing(name: str, value: _SitePath):
    existing = globals().get(name)
    return str(value) if isinstance(existing, str) else value


# Preserve the pre-existing public constant types where possible.
PROJECT_ROOT = _site_coerce_like_existing("PROJECT_ROOT", SITE_PATHS.storage_root)
REPO_ROOT = _site_coerce_like_existing("REPO_ROOT", SITE_PATHS.repo_root)
WEKA_ROOT = _site_coerce_like_existing("WEKA_ROOT", SITE_PATHS.weka_root)
RUN_ROOT = _site_coerce_like_existing("RUN_ROOT", SITE_PATHS.runs_root)
RUNS_ROOT = RUN_ROOT
HF_HOME = _site_coerce_like_existing("HF_HOME", SITE_PATHS.hf_home)
TORCH_HOME = _site_coerce_like_existing("TORCH_HOME", SITE_PATHS.torch_home)
WAN_VAE_PATH = _site_coerce_like_existing("WAN_VAE_PATH", SITE_PATHS.wan_vae_path)
BASE_CHECKPOINT_PATH = _site_coerce_like_existing(
    "BASE_CHECKPOINT_PATH", SITE_PATHS.base_checkpoint_path
)
PHASE1_GEN_INIT = _site_coerce_like_existing(
    "PHASE1_GEN_INIT", _SitePath(_site_exports["COSMOS_PHASE1_GEN_INIT"])
)

# These mappings must be built from the selected site, not from the historical
# defaults that were initialized before the explicit selector.
if COSMOS_SITE == "custom":
    from migration.custom_paths import environment as _custom_environment
    _custom_paths = _custom_environment()
    COSMOS_FRAMEWORK_ROOT = Path(_custom_paths["COSMOS_FRAMEWORK_ROOT"])
    COSMOS_ENV_ROOT = Path(_custom_paths["COSMOS_ENV_ROOT"])
    PHASE1_GEN_INIT = Path(_custom_paths["COSMOS_PHASE1_GEN_INIT"])

_LEGACY_PREFIXES = (
    ("/weka/jungbin/cosmos_motion_ft_runs", RUN_ROOT),
    ("/weka/jungbin", WEKA_ROOT),
    ("/mnt/shared/jungbin_cho/cosmos_motion_ft_runs", RUN_ROOT),
    ("/home/jungbin_cho/cosmos_motion_ft", REPO_ROOT),
    ("/home/jungbin_cho/cosmos-framework", COSMOS_FRAMEWORK_ROOT),
    ("/home/jungbin_cho/miniforge3/envs/cosmos", COSMOS_ENV_ROOT),
)
if COSMOS_SITE == "custom":
    _LEGACY_PREFIXES += (
        ("/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs", RUN_ROOT),
        ("/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs", RUN_ROOT),
        ("/lustre/whwjdqls99/cosmos/weka", WEKA_ROOT),
        ("/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs", RUN_ROOT),
        ("/mnt/projects/ll/jungbinc/weka", WEKA_ROOT),
        ("/home/whwjdqls99/cosmos_motion_ft", REPO_ROOT),
        ("/home/jungbinc/cosmos_motion_ft", REPO_ROOT),
        ("/home/whwjdqls99/cosmos-framework-edge", Path(_custom_paths["COSMOS_FRAMEWORK_EDGE_ROOT"])),
        ("/home/whwjdqls99/cosmos-framework", COSMOS_FRAMEWORK_ROOT),
    )
# END EXPLICIT COSMOS SITE SELECTOR
