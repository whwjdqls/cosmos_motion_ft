"""One path resolver for a new server; standard library only, no model imports."""
from __future__ import annotations

import os
from pathlib import Path
import shlex


def environment() -> dict[str, str]:
    env = os.environ
    for key in ("COSMOS_STORAGE_ROOT", "COSMOS_ENV_ROOT"):
        if not env.get(key):
            raise RuntimeError(f"COSMOS_SITE=custom requires {key}")
        if not Path(env[key]).is_absolute():
            raise RuntimeError(f"{key} must be absolute")
    repo = Path(env.get("COSMOS_REPO_ROOT", Path(__file__).resolve().parents[1]))
    storage = Path(env["COSMOS_STORAGE_ROOT"])
    weka = Path(env.get("WEKA_ROOT", storage / "weka"))
    runs = Path(env.get("COSMOS_RUNS_ROOT", env.get("RUN_ROOT", weka / "cosmos_motion_ft_runs")))
    deps = Path(env.get("COSMOS_DEPS_ROOT", repo / ".deps"))
    values = {
        "COSMOS_SITE": "custom", "COSMOS_REPO_ROOT": repo, "REPO_ROOT": repo,
        "COSMOS_STORAGE_ROOT": storage, "PROJECT_ROOT": storage, "WEKA_ROOT": weka,
        "COSMOS_RUNS_ROOT": runs, "RUN_ROOT": runs,
        "COSMOS_ENV_ROOT": env["COSMOS_ENV_ROOT"], "COSMOS_DEPS_ROOT": deps,
        "COSMOS_PYTHON": Path(env["COSMOS_ENV_ROOT"]) / "bin/python",
        "COSMOS_FRAMEWORK_ROOT": deps / "cosmos-framework",
        "COSMOS_FRAMEWORK_EDGE_ROOT": deps / "cosmos-framework-edge",
        "KIMODO_ROOT": deps / "kimodo_open",
        "UNIEGOMOTION_ROOT": deps / "UniEgoMotion", "EGOALLO_ROOT": deps / "egoallo",
        "HF_HOME": storage / ".cache/huggingface", "TORCH_HOME": storage / ".cache/torch",
        "WAN_VAE_PATH": weka / "wan22_vae/Wan2.2_VAE.pth",
        "COSMOS3_EDGE_ROOT": weka / "Cosmos3-Edge", "EDGE_MODEL_ROOT": weka / "Cosmos3-Edge",
        "COSMOS3_NANO_ROOT": weka / "Cosmos3-Nano",
        "COSMOS3_EDGE_DCP_ROOT": weka / "cosmos3_edge_dcp",
        "BASE_CHECKPOINT_PATH": weka / "cosmos3_nano_dcp",
        "COSMOS_PHASE1_GEN_INIT": runs / "portable/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt",
        "IMAGINAIRE_OUTPUT_ROOT": runs,
    }
    # Alias roots above have a single explicit precedence. Other paths may be
    # overridden individually, e.g. model weights on a read-only shared volume.
    aliases = {"COSMOS_SITE", "COSMOS_REPO_ROOT", "REPO_ROOT", "COSMOS_STORAGE_ROOT", "PROJECT_ROOT", "WEKA_ROOT", "COSMOS_RUNS_ROOT", "RUN_ROOT"}
    result = {key: str(value if key in aliases else env.get(key, value)) for key, value in values.items()}
    result["HUGGINGFACE_HUB_CACHE"] = env.get("HUGGINGFACE_HUB_CACHE", result["HF_HOME"] + "/hub")
    result["COSMOS3_NANO_SNAPSHOT"] = env.get("COSMOS3_NANO_SNAPSHOT", result["COSMOS3_NANO_ROOT"])
    result["COSMOS_TEXT_TOKENIZER_PATH"] = env.get("COSMOS_TEXT_TOKENIZER_PATH", result["COSMOS3_NANO_SNAPSHOT"] + "/text_tokenizer")
    return result


if __name__ == "__main__":
    for key, value in environment().items():
        print(f"export {key}={shlex.quote(value)}")
