"""Load only the frozen Cosmos-3 Edge Nemotron backbone needed by Phase 2."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import config


@dataclass(frozen=True)
class EdgeGeometry:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


def _strip_generation_pathway(text_model: torch.nn.Module) -> None:
    """Remove unused video-generation weights after the reasoner DCP load.

    Phase 2 never creates generator tokens. Keeping `_moe_gen` tensors would
    waste several GiB per replicated L40 process and would make it too easy to
        accidentally use Phase-1/video parameters in the Phase-2 run.
    """

    for layer in text_model.layers:
        for name in (
            "input_layernorm_moe_gen",
            "post_attention_layernorm_moe_gen",
            "mlp_moe_gen",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        for name in (
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
            "o_proj_moe_gen",
            "q_norm_moe_gen",
            "k_norm_moe_gen",
        ):
            if hasattr(layer.self_attn, name):
                delattr(layer.self_attn, name)
    if hasattr(text_model, "norm_moe_gen"):
        delattr(text_model, "norm_moe_gen")


class FrozenEdgeReasoner:
    """Pinned Edge backbone, processor, token contract, and rotary helper."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ) -> None:
        from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import (
            EDGE_MODEL_CONFIG,
        )
        from cosmos_framework.data.generator.processors import build_processor
        from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
            Cosmos3VFMNetwork,
            Cosmos3VFMNetworkConfig,
        )
        from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = bool(verbose)
        config.validate_artifacts()

        edge_cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
        with torch.device("meta"):
            language_model = lazy_instantiate(edge_cfg["vlm_config"]["model_instance"])
            network_cfg = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                vision_gen=False,
                action_gen=False,
                sound_gen=False,
                joint_attn_implementation="two_way",
                timestep_scale=1.0 / 1000.0,
            )
            net = Cosmos3VFMNetwork(language_model=language_model, config=network_cfg)

        # Cast on meta, materialize once, initialize only the non-persistent RoPE
        # buffers, then read the exact reasoner subset directly into its storage.
        net = net.to(dtype=dtype)
        net.to_empty(device=self.device)
        text_model = net.language_model.model
        text_model.rotary_emb.init_weights(buffer_device=self.device)
        self._load_reasoner_dcp(net, config.EDGE_DCP_ROOT / "model")

        cfg = net.language_model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.geometry = EdgeGeometry(
            hidden_size=int(text_cfg.hidden_size),
            num_layers=len(text_model.layers),
            num_attention_heads=int(text_cfg.num_attention_heads),
            num_key_value_heads=int(text_cfg.num_key_value_heads),
            head_dim=int(text_cfg.head_dim),
        )
        expected = EdgeGeometry(
            config.HIDDEN_SIZE,
            config.NUM_LAYERS,
            config.NUM_ATTENTION_HEADS,
            config.NUM_KEY_VALUE_HEADS,
            config.HEAD_DIM,
        )
        if self.geometry != expected:
            raise RuntimeError(f"Edge geometry drift: live={self.geometry} expected={expected}")
        if any(layer.self_attn.k_norm_und_for_gen is None for layer in text_model.layers):
            raise RuntimeError("Edge backbone is missing k_norm_und_for_gen on one or more layers")

        # Edge's understanding vision tower is not part of the DCP. Load the
        # checkpoint-local SigLIP2 + projector bundle through the framework's
        # native loader, which also installs the multimodal token/rope config.
        causal_lm = net.language_model
        causal_lm._local_checkpoint_dir = str(config.EDGE_MODEL_ROOT)
        causal_lm._ensure_vision_tower()
        visual = causal_lm.visual

        _strip_generation_pathway(text_model)
        text_model.requires_grad_(False)
        text_model.eval()
        visual.requires_grad_(False)
        visual.eval()
        self.backbone = text_model
        self.visual = visual
        # The multimodal prefill helper needs only these three fields. Do not
        # retain the uninitialized LM head or any generator-owned outer module.
        self.multimodal_lm = SimpleNamespace(
            model=text_model,
            visual=visual,
            config=causal_lm.config,
        )
        self.config = text_cfg

        # The outer CausalLM, LM head, and top-level generator modules are not
        # retained. `text_model` and the frozen visual tower survive by the
        # explicit references above.
        del net, language_model

        processor = build_processor(str(config.EDGE_MODEL_ROOT))
        tokenizer, special_tokens = add_special_tokens(processor.tokenizer)
        special_tokens["eos_token_id"] = int(tokenizer.eos_token_id)
        if tokenizer.bos_token_id is not None:
            special_tokens["bos_token_id"] = int(tokenizer.bos_token_id)
        self.processor = processor
        self.tokenizer = tokenizer
        self.special_tokens = {key: int(value) for key, value in special_tokens.items()}
        required_tokens = {"bos_token_id", "eos_token_id", "start_of_generation", "end_of_generation"}
        missing_tokens = sorted(required_tokens - self.special_tokens.keys())
        if missing_tokens:
            raise RuntimeError(f"Edge processor is missing special tokens: {missing_tokens}")
        if max(self.special_tokens.values()) >= int(text_cfg.vocab_size):
            raise RuntimeError(
                f"special token exceeds Edge vocab {text_cfg.vocab_size}: {self.special_tokens}"
            )

        if verbose:
            n_reasoner = sum(parameter.numel() for parameter in self.backbone.parameters())
            n_visual = sum(parameter.numel() for parameter in self.visual.parameters())
            print(
                "[FrozenEdgeReasoner] "
                f"layers={self.geometry.num_layers} hidden={self.geometry.hidden_size} "
                f"heads={self.geometry.num_attention_heads}/{self.geometry.num_key_value_heads} "
                f"head_dim={self.geometry.head_dim} frozen_reasoner={n_reasoner/1e9:.3f}B "
                f"frozen_visual={n_visual/1e6:.1f}M "
                f"tokens={self.special_tokens}",
                flush=True,
            )

    @staticmethod
    def _is_reasoner_key(name: str) -> bool:
        if not name.startswith("language_model.model."):
            return False
        if "_moe_gen" in name or name.endswith("norm_moe_gen.weight"):
            return False
        return True

    def _load_reasoner_dcp(self, net: torch.nn.Module, model_dir: Path) -> None:
        from torch.distributed.checkpoint import FileSystemReader, load_state_dict

        if not (model_dir / ".metadata").is_file():
            raise FileNotFoundError(f"Edge DCP metadata not found: {model_dir / '.metadata'}")
        reader = FileSystemReader(str(model_dir))
        metadata = reader.read_metadata()
        source_keys = set(metadata.state_dict_metadata)
        target_state = net.state_dict()
        selected: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        for target_name, tensor in target_state.items():
            if not self._is_reasoner_key(target_name):
                continue
            source_name = target_name if target_name in source_keys else f"net.{target_name}"
            if source_name not in source_keys:
                missing.append(target_name)
                continue
            selected[source_name] = tensor
        if missing:
            raise RuntimeError(f"Edge DCP is missing {len(missing)} reasoner tensors: {missing[:12]}")
        if not selected:
            raise RuntimeError(f"Edge DCP {model_dir} selected no reasoner tensors")
        load_state_dict(selected, storage_reader=reader, no_dist=True)
        sentinels = [
            target_state["language_model.model.embed_tokens.weight"],
            target_state[f"language_model.model.layers.{config.NUM_LAYERS - 1}.mlp.down_proj.weight"],
            target_state["language_model.model.norm.weight"],
        ]
        if not all(torch.isfinite(tensor).all().item() for tensor in sentinels):
            raise RuntimeError("non-finite values found after Edge reasoner DCP load")
        if self.verbose:
            total = sum(tensor.numel() for tensor in selected.values())
            print(
                f"[FrozenEdgeReasoner] loaded {len(selected)} reasoner tensors "
                f"({total/1e9:.3f}B parameters) from {model_dir}",
                flush=True,
            )

    def tokenize_generation(self, text: str | None) -> torch.LongTensor:
        """Use Edge's native diffusion-conditioning text prefix.

        Motion is the generated/full-attention modality in Phase 2, so text is
        packed exactly as `[BOS, raw text, EOS, start_of_generation]`. The CFG
        null caption remains structurally valid as `[BOS, EOS, SOG]`.
        """

        encoded = self.tokenizer("" if text is None else str(text), add_special_tokens=False)
        raw_ids = list(encoded.get("input_ids", []))
        ids = [self.special_tokens["bos_token_id"]]
        ids.extend(int(value) for value in raw_ids)
        ids.extend(
            [self.special_tokens["eos_token_id"], self.special_tokens["start_of_generation"]]
        )
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def encode_text(self, text: str | None) -> dict[str, torch.Tensor | int]:
        ids = self.tokenize_generation(text)
        embeds = self.backbone.embed_tokens(ids).to(self.dtype)
        length = int(ids.numel())
        positions = torch.arange(length, device=self.device, dtype=torch.long)
        return {
            "input_ids": ids.detach(),
            "inputs_embeds": embeds.detach(),
            "position_ids": positions.unsqueeze(0).expand(3, -1).detach(),
            "next_position_id": length,
        }

    @torch.no_grad()
    def encode_reasoner_image_text(
        self,
        text: str | None,
        image_chw: torch.Tensor,
        *,
        image_size: int = config.REASONER_IMAGE_SIZE,
    ) -> dict[str, torch.Tensor | int]:
        """Encode one frame and caption into the frozen Edge reasoner stream."""

        from PIL import Image
        from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.reasoner_multimodal_utils import (
            prepare_multimodal_reasoner_inputs,
        )

        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        image = image_chw.detach().cpu()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image_chw must be [3,H,W], got {tuple(image.shape)}")
        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).to(torch.uint8)
        if tuple(image.shape[-2:]) != (image_size, image_size):
            image = torch.nn.functional.interpolate(
                image.float().unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0).round().clamp_(0, 255).to(torch.uint8)
        pil = Image.fromarray(image.permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        pixel_budget = image_size * image_size
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil,
                    "min_pixels": pixel_budget,
                    "max_pixels": pixel_budget,
                },
                {"type": "text", "text": "" if text is None else str(text)},
            ],
        }]
        processor_output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = processor_output["input_ids"].to(self.device)
        attention_mask = processor_output.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if input_ids.ndim == 1:
            input_ids_b = input_ids.unsqueeze(0)
            attention_mask_b = (
                attention_mask.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim == 1
                else attention_mask
            )
        else:
            input_ids_b = input_ids
            attention_mask_b = attention_mask
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            _mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            self.multimodal_lm,
            input_ids=input_ids_b,
            pixel_values=processor_output["pixel_values"].to(self.device),
            image_grid_thw=processor_output["image_grid_thw"].to(self.device),
            attention_mask=attention_mask_b,
        )
        if deepstack_visual_embeds:
            raise RuntimeError("Edge Nemotron unexpectedly returned deepstack visual embeddings")
        sample_positions = position_ids[:, 0, :].detach()
        return {
            "input_ids": input_ids_b.squeeze(0).detach(),
            "inputs_embeds": inputs_embeds.squeeze(0).to(self.dtype).detach(),
            "position_ids": sample_positions,
            "visual_pos_mask": visual_pos_masks.squeeze(0).detach(),
            "next_position_id": int(sample_positions.max().item()) + 1,
        }

    def rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.to(self.device)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(1)
        dummy = torch.empty(0, device=self.device, dtype=self.dtype)
        cos, sin = self.backbone.rotary_emb(dummy, position_ids=position_ids)
        return cos.squeeze(0).to(self.dtype), sin.squeeze(0).to(self.dtype)


__all__ = ["EdgeGeometry", "FrozenEdgeReasoner"]
