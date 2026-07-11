"""Train the BONES-SEED x0 MotionExpert with the native Cosmos sigma schedule.

This is a controlled entrypoint over ``bs_train.py``. It changes the parser
defaults to the native schedule while preserving the legacy entrypoint and all
other model, data, optimization, loss, and visualization behavior.
"""
from __future__ import annotations

import bs_native_flow
from bs_train import main as train_main


if __name__ == "__main__":
    train_main(
        parser_defaults={
            "schedule": "native",
            "pred": "x0",
            "native_shift": bs_native_flow.DEFAULT_SHIFT,
            "native_num_train_timesteps": bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS,
        }
    )
