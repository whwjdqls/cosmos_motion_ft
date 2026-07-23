#!/usr/bin/env python
"""Train entrypoint that registers the local cached-latent experiment first."""

from __future__ import annotations

import argparse
import os
import traceback

import torch
from loguru import logger as logging

# Register experiment/world_camera_nymeria_latent_nano before TOML resolution.
import native_phase_training.experiment  # noqa: F401

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.scripts.train import (
    _apply_deterministic_config_overrides,
    _setup_deterministic_env_and_backends,
    launch,
)
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.lazy_config import LazyConfig
from cosmos_framework.utils.serialization import to_yaml
from native_phase_training.run_contract import persist_run_contract


def _configure_tensorboard_log_dir(config: Config) -> None:
    """Give the native TensorBoard callback a discoverable per-run event path.

    The Cosmos default callback falls back to ``$IMAGINAIRE_OUTPUT_ROOT/tensorboard``
    when ``log_dir`` is null, which mixes unrelated runs in one directory. Keep the
    callback itself, but make this run's event file live under the run directory
    unless the user explicitly sets ``TB_LOG_DIR``.
    """
    try:
        callbacks = config.trainer.callbacks
        tensorboard = callbacks.get("tensorboard") if hasattr(callbacks, "get") else callbacks.tensorboard
    except Exception:
        return
    if tensorboard is None:
        return
    if getattr(tensorboard, "log_dir", None):
        logging.info(f"TensorBoard log_dir already set: {tensorboard.log_dir}")
        return

    log_dir = os.environ.get("TB_LOG_DIR")
    if not log_dir:
        log_dir = os.path.join(config.job.path_local, "tensorboard")
    tensorboard.log_dir = log_dir
    logging.info(f"TensorBoard log_dir: {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached-latent native Cosmos SFT")
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attach_vscode_debugger", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if args.deterministic:
        _setup_deterministic_env_and_backends()

    config: Config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    _configure_tensorboard_log_dir(config)
    if args.deterministic:
        _apply_deterministic_config_overrides(config)
    args.config = args.sft_toml

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        contract_path = persist_run_contract(config)
        logging.info(f"Native Phase-1 run contract: {contract_path}")

    if args.dryrun:
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
        return

    if args.attach_vscode_debugger:
        import debugpy

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            debugpy.listen(("0.0.0.0", 3002))
            debugpy.wait_for_client()

    launch(config, args)


if __name__ == "__main__":
    main()
