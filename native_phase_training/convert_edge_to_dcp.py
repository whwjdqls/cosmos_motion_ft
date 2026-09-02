#!/usr/bin/env python
"""Convert local Cosmos3-Edge weights without any Hub processor lookup.

NVIDIA's converter correctly accepts a local checkpoint path, but the model
configuration embedded in that artifact still names the processor as
``repository=nvidia/Cosmos3-Edge, revision=main``.  Model construction therefore
tries to run the framework's Hub downloader.  Patch only that lazy processor
factory for the duration of conversion and source it from ``EDGE_MODEL_ROOT``.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

from cosmos_framework.scripts import convert_model_to_dcp as converter

import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.data.generator import processors
from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel


def main() -> None:
    edge_root = Path(os.environ.get("EDGE_MODEL_ROOT", "/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge"))
    for name in ("modular_model_index.json", "tokenizer.json", "preprocessor_config.json"):
        if not (edge_root / name).is_file():
            raise FileNotFoundError(f"local Edge processor artifact is missing {edge_root / name}")

    original = processors.build_processor_lazy

    def build_processor_local(*args, repository=None, revision=None, subdir="", **kwargs):
        if repository is not None and "Cosmos3-Edge" in repository:
            local_path = edge_root / subdir if subdir else edge_root
            return processors.build_processor(str(local_path), **kwargs)
        return original(
            *args,
            repository=repository,
            revision=revision,
            subdir=subdir,
            **kwargs,
        )

    processors.build_processor_lazy = build_processor_local

    args = tyro.cli(converter.Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    converter._redirect_avae_to_local(hf_path)
    model_dict = checkpoint_config.load_model_config_dict()

    # The public Edge config also references the registered Wan VAE artifact.
    # Conversion only needs the tokenizer to construct model geometry, and this
    # checkout already has the byte-identical Wan2.2 file used for training.
    vae_path = Path(
        os.environ.get(
            "WAN_VAE_PATH",
            "/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth",
        )
    )
    if not vae_path.is_file():
        raise FileNotFoundError(f"local Wan2.2 VAE is missing: {vae_path}")
    runtime_config = model_dict.get("config", model_dict)
    if "tokenizer" not in runtime_config:
        raise KeyError(f"Edge model config has no tokenizer block; top-level keys={sorted(model_dict)}")
    runtime_config["tokenizer"]["bucket_name"] = ""
    runtime_config["tokenizer"]["object_store_credential_path_pretrained"] = ""
    runtime_config["tokenizer"]["vae_path"] = str(vae_path)

    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
    state_dict = get_model_state_dict(hf_model.model)
    model_size = sum(
        parameter.numel() * parameter.element_size()
        for parameter in state_dict.values()
        if isinstance(parameter, torch.Tensor)
    )
    thread_count = max(1, math.ceil(model_size / (5 * 1024**3)))
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    source_checkpoint_json = hf_path / "checkpoint.json"
    if source_checkpoint_json.exists():
        shutil.copy(source_checkpoint_json, args.output_path / "checkpoint.json")
    hf_config.save_pretrained(args.output_path / "model")
    print(f"Saved local Edge DCP to {args.output_path}")


if __name__ == "__main__":
    main()
