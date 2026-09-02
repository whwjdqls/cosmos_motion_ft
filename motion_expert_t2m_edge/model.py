"""Cosmos-3 Edge reasoner plus the original Phase-2 T2M/TI2M expert."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

import config
from attention import build_layout
from edge_loader import FrozenEdgeReasoner
from layer import EdgeTextMotionLayer, _fresh_norm_like
from motion_heads import MotionHeads


def _motion_positions(num_frames: int, text_length: int, device: torch.device) -> torch.Tensor:
    """Build the same shape-token + T x 1 x 1 mRoPE layout as Nano Phase 2."""

    from cosmos_framework.data.generator.sequence_packing.mrope import (
        get_3d_mrope_ids_vae_tokens,
    )

    shape_t = max(0, int(text_length) - 1)
    shape = torch.tensor(
        [[shape_t], [text_length], [text_length]], dtype=torch.long, device=device
    )
    frame, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=num_frames,
        grid_h=1,
        grid_w=1,
        temporal_offset=text_length,
        reset_spatial_indices=True,
        fps=None,
        temporal_compression_factor=1,
        base_temporal_compression_factor=4,
        start_frame_offset=0,
    )
    return torch.cat((shape, frame.to(device=device, dtype=torch.long)), dim=1)


class EdgePhase2MotionExpert(nn.Module):
    """Frozen Edge multimodal reasoner with a fresh Nano-style motion pathway.

    The motion pathway preserves the original Phase-2 topology.  Its residual
    width and Q/K/V head geometry are set by Edge because those tensors enter a
    shared attention operation with reasoner tokens.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.compute_dtype = dtype
        edge = FrozenEdgeReasoner(device=self.device, dtype=dtype, verbose=verbose)
        # Keep the 2B frozen reasoner outside this module's registered parameter
        # tree.  The per-layer wrappers hold non-registering references too, so
        # DDP synchronizes and checkpoints only the motion expert.
        object.__setattr__(self, "backbone", edge.backbone)
        object.__setattr__(self, "_edge", edge)

        geometry = edge.geometry
        self.motion_layer_indices = tuple(config.MOTION_LAYER_INDICES)
        motion_set = set(self.motion_layer_indices)
        self.layers = nn.ModuleList(
            [
                EdgeTextMotionLayer(
                    base_layer=base_layer,
                    hidden_size=geometry.hidden_size,
                    num_heads=geometry.num_attention_heads,
                    num_kv_heads=geometry.num_key_value_heads,
                    head_dim=geometry.head_dim,
                    motion_intermediate_size=config.MOTION_INTERMEDIATE_SIZE,
                    has_motion=index in motion_set,
                )
                for index, base_layer in enumerate(self.backbone.layers)
            ]
        )
        self.heads = MotionHeads(geometry.hidden_size)
        self.norm_moe_motion = _fresh_norm_like(self.backbone.norm, geometry.hidden_size)
        self.layers.to(device=self.device, dtype=dtype)
        self.heads.to(device=self.device, dtype=dtype)
        # Match the original Phase-2 contract: sinusoidal time features and
        # their small MLP are evaluated in fp32, then cast into motion tokens.
        self.heads.time_embedder.to(device=self.device, dtype=torch.float32)
        self.norm_moe_motion.to(device=self.device, dtype=dtype)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        if verbose:
            trainable = sum(parameter.numel() for parameter in self.trainable_parameters())
            print(
                f"[EdgePhase2MotionExpert] motion_layers={list(self.motion_layer_indices)} "
                f"motion_mlp=swiglu/{config.MOTION_INTERMEDIATE_SIZE} "
                f"trainable={trainable/1e6:.2f}M",
                flush=True,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self._edge.visual.eval()
        return self

    def tokenize_captions(self, captions: Sequence[str | None]) -> list[torch.LongTensor]:
        return [self._edge.tokenize_generation(caption) for caption in captions]

    def prepare_conditions(
        self,
        captions: Sequence[str | None],
        *,
        modes: Sequence[str] | None = None,
        reasoner_images: Sequence[torch.Tensor | None] | None = None,
        image_size: int = config.REASONER_IMAGE_SIZE,
    ) -> list[dict]:
        """Build frozen reasoner embeddings for text-only or image+text inputs."""

        if modes is None:
            modes = ["text2motion"] * len(captions)
        if reasoner_images is None:
            reasoner_images = [None] * len(captions)
        if not (len(captions) == len(modes) == len(reasoner_images)):
            raise ValueError("captions, modes, and reasoner_images must have equal length")
        conditions: list[dict] = []
        for caption, mode, image in zip(captions, modes, reasoner_images, strict=True):
            if mode == "text2motion":
                if image is not None:
                    raise ValueError("text2motion must not carry a reasoner image")
                conditions.append(self._edge.encode_text(caption))
            elif mode == "textimg2motion":
                if image is None:
                    raise ValueError("textimg2motion requires a reasoner image")
                conditions.append(
                    self._edge.encode_reasoner_image_text(
                        caption, image, image_size=image_size
                    )
                )
            else:
                raise ValueError(f"unsupported Edge Phase-2 mode: {mode!r}")
        return conditions

    def _validate_inputs(
        self,
        reasoner_inputs: Sequence[dict],
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        neutral_joints: torch.Tensor,
        motion_pad_mask: torch.Tensor,
    ) -> None:
        batch, frames, features = x_sigma.shape
        if features != config.MOTION_DIM:
            raise ValueError(f"motion feature width must be {config.MOTION_DIM}, got {features}")
        if len(reasoner_inputs) != batch or tuple(sigma.shape) != (batch,):
            raise ValueError("reasoner-condition and sigma batch sizes must match motion")
        if tuple(neutral_joints.shape) != (batch, config.NUM_JOINTS, 3):
            raise ValueError(
                f"neutral_joints must be {(batch, config.NUM_JOINTS, 3)}, "
                f"got {tuple(neutral_joints.shape)}"
            )
        if tuple(motion_pad_mask.shape) != (batch, frames):
            raise ValueError(
                f"motion_pad_mask must be {(batch, frames)}, got {tuple(motion_pad_mask.shape)}"
            )
        if any(int((~motion_pad_mask[row]).sum()) <= 0 for row in range(batch)):
            raise ValueError("every sample needs at least one valid motion frame")

    def forward(
        self,
        *,
        reasoner_inputs: Sequence[dict],
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        neutral_joints: torch.Tensor,
        motion_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(reasoner_inputs, x_sigma, sigma, neutral_joints, motion_pad_mask)
        device = self.device
        batch, frames, _ = x_sigma.shape
        pad = motion_pad_mask.to(device=device, dtype=torch.bool)
        valid = ~pad
        x_sigma = x_sigma.to(device=device)
        sigma = sigma.to(device=device, dtype=torch.float32)
        neutral_joints = neutral_joints.to(device=device)

        shape_hidden = self.heads.encode_shape(neutral_joints)
        frame_hidden = self.heads.encode_motion(x_sigma, sigma, valid)
        packed_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        text_lengths: list[int] = []
        motion_lengths: list[int] = []
        valid_indices: list[torch.Tensor] = []

        for sample in range(batch):
            condition = reasoner_inputs[sample]
            text = condition["inputs_embeds"].to(device=device, dtype=self.compute_dtype).detach()
            text_positions = condition["position_ids"].to(device=device, dtype=torch.long)
            if text.ndim != 2 or text.shape[1] != config.HIDDEN_SIZE:
                raise ValueError(
                    f"reasoner inputs_embeds must be [L,{config.HIDDEN_SIZE}], got {tuple(text.shape)}"
                )
            if tuple(text_positions.shape) != (3, text.shape[0]):
                raise ValueError(
                    f"reasoner position_ids must be [3,L], got {tuple(text_positions.shape)}"
                )
            frame_index = torch.nonzero(valid[sample], as_tuple=False).flatten()
            motion = torch.cat(
                (shape_hidden[sample], frame_hidden[sample].index_select(0, frame_index)), dim=0
            )
            packed_parts.extend((text, motion))
            text_length = int(text.shape[0])
            motion_length = int(motion.shape[0])
            text_lengths.append(text_length)
            motion_lengths.append(motion_length)
            valid_indices.append(frame_index)

            next_position = int(
                condition.get("next_position_id", int(text_positions.max().item()) + 1)
            )
            position_parts.extend(
                (text_positions, _motion_positions(motion_length - 1, next_position, device))
            )

        packed = torch.cat(packed_parts, dim=0).to(self.compute_dtype)
        positions = torch.cat(position_parts, dim=1)
        layout = build_layout(text_lengths, motion_lengths, device)
        cos, sin = self._edge.rope(positions)
        for layer in self.layers:
            packed = layer(packed, layout, cos, sin)

        motion_rows = self.norm_moe_motion(packed.index_select(0, layout.motion_idx))
        prediction = torch.zeros(
            batch, frames, config.MOTION_DIM, device=device, dtype=torch.float32
        )
        offset = 0
        for sample, motion_length in enumerate(motion_lengths):
            sample_rows = motion_rows[offset : offset + motion_length]
            offset += motion_length
            decoded = self.heads.decode(sample_rows[1:])
            prediction[sample].index_copy_(0, valid_indices[sample], decoded)
        return prediction

    def named_trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for _, parameter in self.named_trainable_parameters())

    def frozen_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.backbone.parameters()
        yield from self._edge.visual.parameters()

    def motion_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the fresh motion pathway, never the frozen Edge base."""

        state = self.state_dict()
        return {name: value for name, value in state.items() if not name.startswith("backbone.")}

    def load_motion_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        expected = set(self.motion_state_dict())
        supplied = set(state)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"motion checkpoint key mismatch: missing={missing[:12]} "
                f"unexpected={unexpected[:12]}"
            )
        incompatible = self.load_state_dict(state, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith("backbone.")]
        if strict and (bad_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                f"motion checkpoint load mismatch: missing={bad_missing[:12]} "
                f"unexpected={incompatible.unexpected_keys[:12]}"
            )

# Backward-compatible import name for the first Edge wiring prototype.
EdgeT2MMotionExpert = EdgePhase2MotionExpert


__all__ = ["EdgePhase2MotionExpert", "EdgeT2MMotionExpert"]
