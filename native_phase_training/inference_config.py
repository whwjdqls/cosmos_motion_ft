"""Config shim for official Cosmos inference with local native-phase experiments."""

from __future__ import annotations

# Registers experiment/world_camera_nymeria_latent_nano before Hydra compose.
import native_phase_training.experiment  # noqa: F401

from cosmos_framework.configs.base.config import make_config as _make_base_config


def make_config():
    return _make_base_config()
