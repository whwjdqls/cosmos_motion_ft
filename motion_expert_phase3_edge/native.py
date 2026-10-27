"""Full frozen Edge loader and native generator packing, with no framework patches."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import yaml

import config as phase2_config
from checkpoint import configure_attention_from_checkpoint
from edge_loader import EdgeGeometry, FrozenEdgeReasoner
from motion_expert_phase3_edge.settings import RECIPE, sha256


def dcp_directory(path):
    path = Path(path)
    if (path / "model/.metadata").is_file():
        return path / "model"
    if (path / ".metadata").is_file():
        return path
    raise FileNotFoundError(f"No DCP metadata under {path}")


def load_selected_dcp(net, path, prefix, select):
    from torch.distributed.checkpoint import FileSystemReader, load_state_dict

    reader = FileSystemReader(str(dcp_directory(path)))
    metadata = reader.read_metadata().state_dict_metadata
    destination = net.state_dict()
    selected = {}
    for name, tensor in destination.items():
        if not select(name):
            continue
        key = prefix + name
        if key not in metadata or tuple(metadata[key].size) != tuple(tensor.shape):
            raise RuntimeError(f"Missing or incompatible specialist tensor: {key}")
        selected[key] = tensor
    if not selected:
        raise RuntimeError(f"Empty checkpoint selection from {path}")
    load_state_dict(selected, storage_reader=reader, no_dist=True)
    print(f"[phase3/load] {path}: {len(selected)} tensors ({prefix})", flush=True)
    return sorted(selected)


class FullFrozenEdge(FrozenEdgeReasoner):
    """Reuse Phase-2 text/RoPE methods but retain native video generation."""

    def __init__(self, phase1, phase2, device):
        from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
        from cosmos_framework.data.generator.processors import build_processor
        from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
        from cosmos_framework.utils.generator.lora import inject_lora_pre_fsdp
        from cosmos_framework.utils.lazy_config import instantiate

        self.device, self.dtype, self.verbose = torch.device(device), torch.bfloat16, True
        self.motion_contract = configure_attention_from_checkpoint(phase2)
        expected = {
            "model_family": "cosmos3_edge_nemotron_2b_dense_vl",
            "hidden_size": 2048, "num_layers": 28, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "motion_dim": 283,
            "motion_intermediate_size": 3072, "motion_layer_indices": RECIPE["bridge_layers"],
            "motion_attention_mode": "self_cross_final_hidden",
            "cross_attention_key_normalization": "rmsnorm",
            "cross_attention_value_normalization": "none", "cross_attention_reasoner_rope": True,
            "text_packing": "nano_caption_only", "motion_io_dtype": "float32",
            "motion_norm_dtype": "bfloat16", "motion_self_attention_per_gap": 0,
            "motion_self_attention_before_joint": False,
            "motion_representation": "camera_head_recanonicalization_v1",
        }
        for key, value in expected.items():
            if self.motion_contract.get(key, False if isinstance(value, bool) else None) != value:
                raise RuntimeError(f"Wrong phase-2 contract {key}: {self.motion_contract.get(key)!r} != {value!r}")
        for key, path in (("motion_mean", phase2_config.MOTION_STATS_MEAN),
                          ("motion_std", phase2_config.MOTION_STATS_STD)):
            if sha256(path) != self.motion_contract[key]:
                raise RuntimeError(f"Phase-2 normalization mismatch: {key}")
        base = dcp_directory(phase2_config.EDGE_DCP_ROOT)
        if sha256(base / ".metadata") != self.motion_contract["edge_dcp_metadata"]:
            raise RuntimeError("Phase-2 pretrained Edge DCP identity mismatch")

        run = Path(phase1).parent.parent
        contract = json.loads((run / "native_phase1_contract.json").read_text())
        saved = yaml.safe_load((run / "config.yaml").read_text())["model"]["config"]
        rf = saved["rectified_flow_training_config"]
        if (float(rf["loss_scale"]) != 10 or rf["normalize_loss_by_active"]
                or rf.get("sample_level_loss_averaging", False)
                or rf["train_time_weight"] != "uniform"):
            raise RuntimeError("Phase-1 video objective differs from the approved phase-3 recipe")
        if saved["lora_rank"] != 16 or saved["lora_alpha"] != 32:
            raise RuntimeError("Unexpected phase-1 LoRA geometry")
        diffusion = saved["diffusion_expert_config"]
        cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
        with torch.device("meta"):
            language = instantiate(cfg["vlm_config"]["model_instance"])
            nc = Cosmos3VFMNetworkConfig(
                vlm_config=language.config, latent_patch_size=2, latent_downsample_factor=16,
                latent_channel_size=48, max_latent_h=20, max_latent_w=20,
                max_latent_t=saved["state_t"], enable_fps_modulation=True,
                enable_vision_modality_embeddings=diffusion["enable_vision_modality_embeddings"],
                enable_media_modality_embedding=diffusion["enable_media_modality_embedding"],
                enable_action_modality_embedding=diffusion["enable_action_modality_embedding"],
                base_fps=24, vision_gen=True, action_gen=True, sound_gen=False,
                joint_attn_implementation="two_way", timestep_scale=0.001,
                action_dim=64, num_embodiment_domains=32,
                action_io_projector_type=saved["action_io_projector_type"],
                temporal_compression_factor_vision=4, video_temporal_causal=False,
                enable_input_bias=saved["enable_input_bias"],
            )
            nc._attn_implementation_internal = "eager"
            net = Cosmos3VFMNetwork(language_model=language, config=nc)
            inject_lora_pre_fsdp(net, lora_rank=16, lora_alpha=32,
                                lora_target_modules=saved["lora_target_modules"])
        net.to(dtype=self.dtype).to_empty(device=self.device)
        net.language_model.model.rotary_emb.init_weights(buffer_device=self.device)
        net.time_embedder.float()
        net.time_embedder._init_weights(buffer_device=self.device)
        base_keys = load_selected_dcp(net, base, "net.", lambda name: "lora_" not in name)
        select_delta = lambda name: "lora_" in name or name.startswith(
            ("action2llm.", "llm2action.", "action_modality_embed"))
        delta_keys = load_selected_dcp(net, phase1, "net_ema.", select_delta)
        if sum("lora_" in name for name in delta_keys) != 224:
            raise RuntimeError("Expected all 224 phase-1 EMA LoRA tensors")
        net.eval().requires_grad_(False)
        self.net, self.backbone = net, net.language_model.model
        self.geometry = EdgeGeometry(2048, 28, 16, 8, 128)
        self.config = getattr(language.config, "text_config", language.config)
        self.visual = nn.Identity().eval()  # These two tasks have no SigLIP image branch.
        self.multimodal_lm = SimpleNamespace(model=self.backbone, visual=self.visual, config=language.config)
        self.processor = build_processor(str(phase2_config.EDGE_MODEL_ROOT))
        self.tokenizer, special = add_special_tokens(self.processor.tokenizer)
        special["eos_token_id"] = int(self.tokenizer.eos_token_id)
        if self.tokenizer.bos_token_id is not None:
            special["bos_token_id"] = int(self.tokenizer.bos_token_id)
        self.special_tokens = {key: int(value) for key, value in special.items()}
        self.provenance = {
            "phase1": str(phase1), "phase1_ema": True,
            "phase1_metadata_sha256": sha256(dcp_directory(phase1) / ".metadata"),
            "phase1_contract": contract, "base_tensors": len(base_keys),
            "phase1_delta_tensors": len(delta_keys), "phase2": str(phase2),
            "phase2_contract": self.motion_contract,
        }

    def pack_video(self, captions, modes, latents, sigma):
        from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
        from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
        from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
        from native_phase_training.latent_nymeria_dataset import _format_prompt_for_mode

        text, plans = [], []
        for caption, mode in zip(captions, modes, strict=True):
            if mode == "video2motion":
                caption, clean = "", list(range(25))
            elif mode == "motimg2video":
                sample = {"ai_caption": caption, "conditioning_fps": 20.0,
                          "video": torch.empty(3, 97, 1, 1), "image_size": (256, 256)}
                caption = _format_prompt_for_mode(sample, "image2video", None)["ai_caption"]
                clean = [0]
            else:
                raise ValueError(mode)
            text.append(list(self.tokenizer(caption, add_special_tokens=False)["input_ids"]))
            plans.append(SequencePlan(has_text=True, has_vision=True,
                                      condition_frame_indexes_vision=clean))
        data = GenerationDataClean(batch_size=len(modes), is_image_batch=False,
                                   x0_tokens_vision=list(latents), fps_vision=torch.full((len(modes),), 20.0))
        packed = pack_input_sequence(
            sequence_plans=plans, input_text_indexes=text, gen_data_clean=data,
            input_timesteps=sigma.detach().float().cpu() * 1000.0,
            special_tokens=self.special_tokens, latent_patch_size=2,
            include_end_of_generation_token=False,
            unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=15000,
            enable_fps_modulation=True, base_fps=24.0, temporal_compression_factor=4,
            vision_temporal_position_mode="latent_index", action_dim=64,
        )
        packed.to_cuda()
        return packed
