"""Camera-token-only LoRA for native Cosmos packed generation attention.

The native generator routes text through the understanding stream and video /
action tokens through the generation stream.  Standard LoRA on
``k_proj_moe_gen`` or ``v_proj_moe_gen`` therefore adapts both video and action
tokens.  This module keeps the ordinary DCP-compatible ``lora_A`` / ``lora_B``
parameters but applies their residual only at packed action-token rows.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.model.vfm.mot.unified_mot import PackedAttentionMoT
from cosmos_framework.utils import log
from cosmos_framework.utils.vfm.lora import LoraInjectedLinear


_ACTIVE_CAMERA_TOKEN_MASK: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar(
    "native_phase1_camera_token_mask",
    default=None,
)


@contextmanager
def camera_token_mask_context(camera_mask: torch.Tensor):
    """Expose the projection mask context for focused contract tests."""
    token = _ACTIVE_CAMERA_TOKEN_MASK.set(camera_mask)
    try:
        yield
    finally:
        _ACTIVE_CAMERA_TOKEN_MASK.reset(token)


class CameraTokenLoraLinear(LoraInjectedLinear):
    """LoRA linear whose residual is evaluated only for masked token rows."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"camera-token LoRA expects [tokens,channels], got {tuple(x.shape)}")
        camera_mask = _ACTIVE_CAMERA_TOKEN_MASK.get()
        if camera_mask is None:
            raise RuntimeError("camera-token LoRA was called without a packed camera-token mask")
        if camera_mask.dtype is not torch.bool:
            raise TypeError(f"camera-token mask must be bool, got {camera_mask.dtype}")
        if camera_mask.ndim != 1 or camera_mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"camera-token mask shape {tuple(camera_mask.shape)} does not match {tuple(x.shape)}"
            )
        if camera_mask.device != x.device:
            raise ValueError(f"camera-token mask is on {camera_mask.device}, input is on {x.device}")

        base_out = F.linear(x, self.weight, self.bias)
        camera_indexes = camera_mask.nonzero(as_tuple=False).flatten()
        if camera_indexes.numel() == 0:
            return base_out

        camera_x = x.index_select(0, camera_indexes)
        camera_residual = self.lora_B(self.lora_A(camera_x)) * self._lora_scale
        residual = torch.zeros_like(base_out)
        residual.index_copy_(0, camera_indexes, camera_residual)
        return base_out + residual


class CameraMaskedPackedAttention(PackedAttentionMoT):
    """Install the pack's camera mask while generation K/V are projected."""

    def forward(self, pack, *args, **kwargs):
        camera_mask = pack.get("_camera_token_mask_gen")
        if camera_mask is None:
            raise RuntimeError("packed sequence is missing _camera_token_mask_gen")
        with camera_token_mask_context(camera_mask):
            return super().forward(pack, *args, **kwargs)


def build_camera_token_mask(
    packed_sequence: Any,
    factored_pack: dict[str, Any],
    parallel_dims: Any | None = None,
) -> torch.Tensor:
    """Map joint packed action indices into the local generation-stream rows."""
    if "_causal_indices" not in factored_pack or "_full_indices" not in factored_pack:
        raise KeyError("factored pack is missing causal/full index metadata")
    if "full_only_seq" not in factored_pack:
        raise KeyError("camera-token LoRA requires a FactoredSequencePack")

    full_indices = factored_pack["_full_indices"].long()
    causal_indices = factored_pack["_causal_indices"]
    joint_length = int(full_indices.numel() + causal_indices.numel())
    device = factored_pack["full_only_seq"].device
    joint_mask = torch.zeros(joint_length, dtype=torch.bool, device=device)

    action = getattr(packed_sequence, "action", None)
    action_indexes = None if action is None else action.sequence_indexes
    if action_indexes is not None:
        if not isinstance(action_indexes, torch.Tensor):
            raise TypeError("finalized action sequence_indexes must be a tensor")
        action_indexes = action_indexes.to(device=device, dtype=torch.long)
        if action_indexes.ndim != 1:
            raise ValueError(f"action sequence indexes must be 1-D, got {tuple(action_indexes.shape)}")
        if action_indexes.numel():
            if int(action_indexes.min()) < 0 or int(action_indexes.max()) >= joint_length:
                raise ValueError(
                    f"action sequence indexes fall outside packed length {joint_length}: "
                    f"[{int(action_indexes.min())},{int(action_indexes.max())}]"
                )
            if action_indexes.unique().numel() != action_indexes.numel():
                raise ValueError("action sequence indexes contain duplicates")
            joint_mask[action_indexes] = True

    global_gen_mask = joint_mask.index_select(0, full_indices.to(device=device))
    expected_actions = 0 if action_indexes is None else int(action_indexes.numel())
    if int(global_gen_mask.sum()) != expected_actions:
        raise ValueError("not every action token belongs to the generation/full-attention stream")

    local_length = int(factored_pack["full_only_seq"].shape[0])
    if factored_pack.get("is_sharded", False):
        if parallel_dims is None or not parallel_dims.cp_enabled:
            raise RuntimeError("pack is context-sharded but context parallelism is unavailable")
        cp_group = parallel_dims.cp_mesh.get_group()
        rank = torch.distributed.get_rank(cp_group)
        world_size = torch.distributed.get_world_size(cp_group)
        padded_global_length = local_length * world_size
        if global_gen_mask.numel() > padded_global_length:
            raise ValueError("global generation mask exceeds the context-parallel padded length")
        padded = torch.zeros(padded_global_length, dtype=torch.bool, device=device)
        padded[: global_gen_mask.numel()] = global_gen_mask
        return padded.narrow(0, rank * local_length, local_length)

    if global_gen_mask.numel() > local_length:
        raise ValueError("generation mask exceeds the factored generation stream")
    local_mask = torch.zeros(local_length, dtype=torch.bool, device=device)
    local_mask[: global_gen_mask.numel()] = global_gen_mask
    return local_mask


def _extract_packed_sequence(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    packed_sequence = kwargs.get("packed_seq")
    if packed_sequence is None and args:
        packed_sequence = args[0]
    if packed_sequence is None:
        raise RuntimeError("camera-token LoRA network hook did not receive packed_seq")
    return packed_sequence


def install_camera_token_lora(network: torch.nn.Module) -> torch.nn.Module:
    """Convert injected K/V LoRA and attach packed-mask propagation hooks."""
    converted_linears = 0
    converted_attentions = 0
    for module_name, module in network.named_modules():
        child_name = module_name.rsplit(".", 1)[-1]
        if child_name in {"k_proj_moe_gen", "v_proj_moe_gen"}:
            if not isinstance(module, LoraInjectedLinear):
                raise TypeError(f"{module_name} is not an injected LoRA linear: {type(module).__name__}")
            module.__class__ = CameraTokenLoraLinear
            converted_linears += 1

    for module in network.modules():
        if not isinstance(module, PackedAttentionMoT):
            continue
        if not isinstance(module.k_proj_moe_gen, CameraTokenLoraLinear):
            continue
        if not isinstance(module.v_proj_moe_gen, CameraTokenLoraLinear):
            continue
        module.__class__ = CameraMaskedPackedAttention
        converted_attentions += 1

    if converted_linears == 0 or converted_attentions == 0:
        raise RuntimeError(
            f"camera-token LoRA installation found linears={converted_linears}, "
            f"attentions={converted_attentions}"
        )
    if converted_linears != 2 * converted_attentions:
        raise RuntimeError(
            f"camera-token LoRA expected two K/V linears per attention: "
            f"linears={converted_linears}, attentions={converted_attentions}"
        )

    def capture_packed_sequence(module, args, kwargs) -> None:
        module._camera_token_source_pack = _extract_packed_sequence(args, kwargs)

    def attach_camera_mask(_module, args, kwargs) -> None:
        pack = kwargs.get("pack")
        if pack is None and args:
            pack = args[0]
        if pack is None or not isinstance(pack, dict):
            raise RuntimeError("camera-token LoRA language-model hook did not receive a sequence pack")
        source_pack = getattr(network, "_camera_token_source_pack", None)
        if source_pack is None:
            raise RuntimeError("camera-token LoRA has no source PackedSequence")
        pack["_camera_token_mask_gen"] = build_camera_token_mask(
            source_pack,
            pack,
            getattr(network, "parallel_dims", None),
        )

    network.register_forward_pre_hook(capture_packed_sequence, with_kwargs=True)
    network.language_model.register_forward_pre_hook(attach_camera_mask, with_kwargs=True)
    log.info(
        f"Installed camera-token K/V LoRA on {converted_attentions} attention layers "
        f"({converted_linears} projections)"
    )
    return network
