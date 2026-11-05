"""Contracted bridge execution settings shared by training and sampling."""
from __future__ import annotations

import argparse
from functools import wraps

from .settings import RECIPE


def add_execution_options(parser):
    parser.add_argument("--bridge-attention-backend", choices=("dense", "grouped"))
    parser.add_argument("--bridge-precision", choices=("fp32", "bf16_attention", "bf16_bridge"))
    return parser


def configure_execution(backend=None, precision=None):
    backend, precision = backend or "dense", precision or "fp32"
    if backend not in {"dense", "grouped"} or precision not in {"fp32", "bf16_attention", "bf16_bridge"}:
        raise ValueError("Unsupported bridge execution policy")
    if precision != "fp32" and backend != "grouped":
        raise ValueError("BF16 bridge execution requires grouped attention")
    RECIPE.pop("bridge_execution", None)
    if (backend, precision) != ("dense", "fp32"):
        RECIPE["bridge_execution"] = dict(backend=backend, precision=precision)


def execution_context(configuration):
    from scripts.phase3_performance import execution_policy
    return execution_policy(cache_bridge_masks=True, checkpoint_motion=True,
        grouped_attention=configuration.get("backend", "dense") == "grouped",
        precision=configuration.get("precision", "fp32"))


def training_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        parser = add_execution_options(argparse.ArgumentParser(add_help=False))
        options, _ = parser.parse_known_args()
        configure_execution(options.bridge_attention_backend, options.bridge_precision)
        # Keep this context alive through backward and checkpoint recomputation.
        with execution_context(RECIPE.get("bridge_execution", {})):
            return function(*args, **kwargs)
    return wrapped


def sampling_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with execution_context(RECIPE.get("bridge_execution", {})):
            return function(*args, **kwargs)
    return wrapped
