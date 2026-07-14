"""Top-level trainable model: Motion Expert via 3-way SHARED (joint) attention.

This wraps a FROZEN Cosmos-3 Nano (`FrozenCosmos` from `cosmos_loader.py`) and ADDS a
trainable MOTION pathway (`_moe_motion`) that participates in the real MoT joint attention
alongside the frozen reasoner (`*`) and generator (`*_moe_gen`) pathways. Unlike the
cross-attention PoC (`motion_expert/motion_expert.py`), motion tokens here live in the SAME
packed sequence as the reasoner tokens and share ONE joint attention per layer:

  - REASONER tokens : causal self-attention over reasoner only (FROZEN).
  - GENERATOR tokens: full attention over {reasoner, generator, motion} (FROZEN by default).
  - MOTION tokens   : full attention over {reasoner, generator, motion} (TRAINABLE _moe_motion).
  => generator<->motion is bidirectional; reasoner is a pure causal context encoder.

In the text->motion regime there are NO generator tokens (`gen_idx` is empty); motion attends
over {reasoner, motion}. The generator pathway is still present and frozen.

7-task extension (see DESIGN_7TASK.md)
--------------------------------------
The ONE load-bearing change vs the text->motion baseline is flipping `gen_idx` from the
hardcoded `torch.empty(0)` to a REAL generator segment built by `gen_heads.GenHeads.
build_gen_segment(...)` whenever the task packs a generator-carried modality (image / video /
camera). The packed per-sample order stays `[ reasoner(und) | generator(gen) | motion(mot) ]`;
we build `und_idx` / `gen_idx` / `mot_idx`, `full_idx = sort(cat(gen_idx, mot_idx))`, the
matching rotary positions (3D-mRoPE for the gen-present case via the gen segment's mrope ids,
falling back to the 1-D continuation for text->motion), and the per-token `condition_mask`
(clean vs noised) from the per-sample `task_plan.ResolvedPlan`. Gen tokens route through
`_moe_gen` (already wired in `MoTJointLayer`); motion routes through `_moe_motion`. The two-call
joint attention (`mot_joint_attention.py`) is UNCHANGED. `forward()` returns a dict of
per-modality predictions (motion / video / camera) so `flow.py` / the trainer apply the per-task
losses; the text->motion call path (no gen, single motion target) is preserved exactly.

This file assembles the network:
  reasoner embed (frozen)  ->  36 x MoTJointLayer (shared attention)  ->  motion final norm
  ->  MotionHeads.decode  ->  pred[B,Tm,283]    (+ optional gen video/camera decode)

and owns:
  * the motion-expert size (`motion_intermediate_size`, handed to each MoTJointLayer; the expert
    is always freshly/randomly initialized, NEVER warm-started from the generator),
  * the freeze rule (train iff motion-always OR (reasoner_lora/gen_lora and `lora_`) OR
    (gen_full and a `_moe_gen` / gen I/O head)),
  * the optional LoRA-on-generator / LoRA-on-reasoner / full-generator-FT toggles,
  * the packed-sequence assembly + rotary cos/sin construction,
  * `forward(...)` (one denoiser call) and `predict_closure(...)` used by `flow.py`.

Sibling contracts this file depends on (defined in their own modules):
  cosmos_loader.FrozenCosmos       — frozen net handle: .net, .tm, .num_heads, .num_kv_heads,
                                      .head_dim, .hidden, .n_layers, .rope(positions)->(cos,sin)
  motion_heads.MotionHeads         — encode_shape / encode_motion / decode + the I/O params
  gen_heads.GenHeads               — encode/decode video|image|camera + build_gen_segment (NEW)
  task_plan.resolve_sample         — per-sample CLEAN/NOISED layout + loss spec (NEW)
  mot_joint_layer.MoTJointLayer    — one 3-pathway joint-attention decoder layer
  mot_joint_attention.build_offsets — varlen offsets/maxlens for the two-call mask (built once)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import task_plan as TP
from cosmos_loader import FrozenCosmos
from gen_heads import GenHeads
from modality_bridge import BridgeMeta, LocalModalityBridge
from mot_joint_attention import build_offsets
from mot_joint_layer import MoTJointLayer
from motion_heads import MotionHeads
from uniego_layout import FEAT_DIM  # 283
from config import TIMESTEP_SCALE  # 1e-3; embedder sees (t * timestep_scale), so we pass t / scale
from cosmos_framework.model.vfm.mot.unified_3dmrope_utils import get_3d_mrope_ids_vae_tokens


# ---- RMSNorm helper: a FRESH RMSNorm matching a base's type/eps (NO weight copy) ----------
def _fresh_rmsnorm_like(base: nn.Module, hidden: int) -> nn.Module:
    """Build a fresh RMSNorm with the same class/eps/shape as `base` — weight stays at default 1.0.

    We match only the *type* and `eps` of the frozen `norm_moe_gen` (so the motion final norm is
    the right RMSNorm variant), but DO NOT copy its weight: the motion expert is never warm-started
    from the generator. Falls back to `nn.RMSNorm` if the base type can't be reconstructed.
    """
    eps = getattr(base, "variance_epsilon", getattr(base, "eps", 1e-6))
    try:
        norm = base.__class__(hidden, eps=eps)
    except TypeError:
        try:
            norm = base.__class__(hidden)
            if hasattr(norm, "variance_epsilon"):
                norm.variance_epsilon = eps
            elif hasattr(norm, "eps"):
                norm.eps = eps
        except Exception:
            norm = nn.RMSNorm(hidden, eps=eps)
    return norm  # weight = 1.0 (fresh); never copied from the generator


class JointMotionModel(nn.Module):
    """Frozen Cosmos + trainable 3-way motion expert (shared joint attention).

    Args:
        cosmos: a `FrozenCosmos` wrapping the materialized + weight-loaded Cosmos3VFMNetwork.
        motion_dim: motion feature dim (283-d uniego).
        objective: the MOTION-pathway flow objective ('velocity' | 'x0'); the head always
            emits 283-d and train.py / flow.py interpret it as v-hat or x0-hat. Drives
            `self.sample`'s sampler dispatch. ``motion_schedule='native'`` keeps x0 prediction
            but uses Cosmos's shifted logit-normal training sigma and native inference ladder.
            PER-MODALITY
            design: vision/camera are ALWAYS velocity regardless (Cosmos-native rectified
            flow); this knob never touches the gen pathway.
        motion_intermediate_size: FFN width of the motion expert (the only size knob; the head
            geometry + 4096 residual are fixed by the shared joint attention). The expert is
            ALWAYS randomly initialized -- never warm-started from the generator.
        gen_lora: if True, inject LoRA on q/k/v/o_proj_moe_gen of the base layers and mark the
            LoRA params (+ motion params) trainable; the generator base stays frozen.
        reasoner_lora: if True, inject LoRA on the reasoner q/k/v/o_proj and mark those LoRA
            params trainable too (else the reasoner stays fully FROZEN). Independent of the gen
            choice. (DESIGN_7TASK.md section 5.)
        gen_full: if True, FULL generator finetune -- every `_moe_gen` param AND the generator
            I/O heads (vae2llm / llm2vae / action2llm / llm2action / action_modality_embed /
            patch/unpatch / the generator time_embedder) become trainable. Mutually exclusive
            with gen_lora (one of {frozen, lora, full} for the gen pathway).
        freeze_gen: if True, generator LoRA/full/action-head params may still be present for
            warm-start compatibility but are excluded from optimization. Used for bridge-only
            Phase 3 runs that load Phase-1 generator LoRA and keep it frozen.
        freeze_motion: PHASE-1 curriculum. If True, the motion pathway (`_moe_motion` + motion
            heads + `norm_moe_motion`) is EXCLUDED from the trainable set entirely: it is still
            BUILT (architecture unchanged) but carries no grad and is never stepped, so Phase 1
            trains only the gen-LoRA on the camera tasks (no motion tokens ever appear). Default
            False keeps today's behavior (motion always trains).
        motion_layer_stride: SPARSE-DEPTH knob. The motion expert (the 3-way joint attention) fires
            only every Nth backbone layer; the frozen reasoner+generator still run all `n_layers`.
            The motion-layer set is `{i for i in range(n_layers) if (i+1) % stride == 0}`, so
            stride=3 -> {2,5,...,35} = 12 motion blocks; stride=6 -> {5,11,...,35} = 6 blocks; the
            LAST layer (n_layers-1) is always included (since n_layers=36 is a multiple of both 3 and
            6). Plain (non-motion) layers carry ZERO motion params and run only the frozen 2-way
            reasoner+generator path; at those layers gen/reasoner NEVER attend motion, and motion
            rows pass through unchanged.

    Train scope (see DESIGN_7TASK.md section 5): motion is ALWAYS fully trained; the reasoner
    and generator pathways each pick EXACTLY ONE of {frozen, lora, full}. Defaults (all toggles
    False) reproduce today's text->motion behavior (reasoner + generator frozen).
    """

    # Substrings that name the FROZEN generator I/O heads on `cosmos.net`. When `gen_full`,
    # these (plus any `_moe_gen` param) are made trainable. Kept here so freeze() and
    # assert_frozen_grads_zero() share ONE definition of "is a generator I/O head".
    _GEN_IO_HEAD_NAMES = (
        "vae2llm",
        "llm2vae",
        "action2llm",
        "llm2action",
        "action_modality_embed",
    )

    def __init__(
        self,
        cosmos: FrozenCosmos,
        motion_dim: int = FEAT_DIM,
        objective: str = "velocity",
        motion_schedule: str = "legacy",
        motion_shift: float = 3.0,
        motion_num_train_timesteps: int = 1000,
        motion_native_solver: str = "euler",
        gen_schedule: str = "legacy",
        gen_shift: float = 3.0,
        gen_num_train_timesteps: int = 1000,
        gen_native_solver: str = "unipc",
        gen_packing: str = "legacy",
        gen_fps: float = 20.0,
        gen_temporal_margin: float = 15000.0,
        motion_intermediate_size: int = 3072,
        gen_lora: bool = False,
        gen_lora_rank: int = 16,
        gen_lora_alpha: int = 16,
        reasoner_lora: bool = False,
        gen_full: bool = False,
        freeze_gen: bool = False,
        freeze_motion: bool = False,
        motion_layer_stride: int = 3,
        motion_mrope: str = "legacy",
        coupling: str = "joint",
        textimg_condition: str = "reasoner",
        reasoner_image_size: int = 256,
    ):
        super().__init__()
        if gen_lora and gen_full:
            raise ValueError(
                "gen_lora and gen_full are mutually exclusive (pick frozen / lora / full "
                "for the generator pathway)"
            )
        self.cosmos = cosmos
        self.motion_dim = motion_dim
        if objective not in ("velocity", "x0"):
            raise ValueError(f"objective must be 'velocity' or 'x0', got {objective!r}")
        if motion_schedule not in ("legacy", "native"):
            raise ValueError(
                f"motion_schedule must be 'legacy' or 'native', got {motion_schedule!r}"
            )
        if motion_schedule == "native" and objective != "x0":
            raise ValueError("motion_schedule='native' requires objective='x0'")
        if float(motion_shift) <= 0.0:
            raise ValueError(f"motion_shift must be positive, got {motion_shift}")
        if int(motion_num_train_timesteps) <= 1:
            raise ValueError(
                "motion_num_train_timesteps must be greater than one, got "
                f"{motion_num_train_timesteps}"
            )
        if motion_native_solver not in ("euler", "unipc"):
            raise ValueError(
                f"motion_native_solver must be 'euler' or 'unipc', got {motion_native_solver!r}"
            )
        if gen_schedule not in ("legacy", "native"):
            raise ValueError(f"gen_schedule must be 'legacy' or 'native', got {gen_schedule!r}")
        if float(gen_shift) <= 0.0:
            raise ValueError(f"gen_shift must be positive, got {gen_shift}")
        if int(gen_num_train_timesteps) <= 1:
            raise ValueError(
                "gen_num_train_timesteps must be greater than one, got "
                f"{gen_num_train_timesteps}"
            )
        if gen_native_solver not in ("euler", "unipc"):
            raise ValueError(
                f"gen_native_solver must be 'euler' or 'unipc', got {gen_native_solver!r}"
            )
        if gen_packing not in ("legacy", "native"):
            raise ValueError(f"gen_packing must be 'legacy' or 'native', got {gen_packing!r}")
        if float(gen_fps) <= 0.0:
            raise ValueError(f"gen_fps must be positive, got {gen_fps}")
        if float(gen_temporal_margin) < 0.0:
            raise ValueError(
                f"gen_temporal_margin must be non-negative, got {gen_temporal_margin}"
            )
        if int(gen_lora_rank) <= 0 or int(gen_lora_alpha) <= 0:
            raise ValueError(
                f"gen_lora_rank/alpha must be positive, got {gen_lora_rank}/{gen_lora_alpha}"
            )
        self.objective = objective
        self.motion_schedule = motion_schedule
        self.motion_shift = float(motion_shift)
        self.motion_num_train_timesteps = int(motion_num_train_timesteps)
        self.motion_native_solver = motion_native_solver
        self.gen_schedule = gen_schedule
        self.gen_shift = float(gen_shift)
        self.gen_num_train_timesteps = int(gen_num_train_timesteps)
        self.gen_native_solver = gen_native_solver
        self.gen_packing = gen_packing
        self.gen_fps = float(gen_fps)
        self.gen_temporal_margin = float(gen_temporal_margin)
        self.gen_lora = gen_lora
        self.gen_lora_rank = int(gen_lora_rank)
        self.gen_lora_alpha = int(gen_lora_alpha)
        self.reasoner_lora = reasoner_lora
        self.gen_full = gen_full
        self.freeze_gen = freeze_gen
        if coupling not in ("joint", "bridge_local"):
            raise ValueError(f"coupling must be 'joint' or 'bridge_local', got {coupling!r}")
        if textimg_condition not in ("generator", "reasoner"):
            raise ValueError(
                f"textimg_condition must be 'generator' or 'reasoner', got {textimg_condition!r}"
            )
        self.coupling = coupling
        self.textimg_condition = textimg_condition
        if int(reasoner_image_size) <= 0:
            raise ValueError(
                f"reasoner_image_size must be positive, got {reasoner_image_size}"
            )
        self.reasoner_image_size = int(reasoner_image_size)
        if motion_mrope not in ("legacy", "cosmos3d"):
            raise ValueError(f"motion_mrope must be 'legacy' or 'cosmos3d', got {motion_mrope!r}")
        self.motion_mrope = motion_mrope
        # PHASE-1 curriculum: when True, the motion pathway (_moe_motion + motion heads +
        # norm_moe_motion) is EXCLUDED from the trainable set / optimizer / grad-clip / all-reduce.
        # The motion expert is still BUILT (so the architecture is unchanged) but gets no grad and
        # is never stepped -- Phase 1 trains only the gen-LoRA on the camera tasks (no motion tokens
        # ever appear). Default False preserves today's behavior (motion always trains).
        self.freeze_motion = freeze_motion
        self.hidden = cosmos.hidden
        self.head_dim = cosmos.head_dim
        self.n_layers = cosmos.n_layers

        tm = cosmos.tm  # the transformer module that owns .layers / embed_tokens / norm / rotary

        # ---- frozen references read directly from the base net (NOT re-registered as ours) --
        # Held as attributes (not submodules) so they don't double-count params or get grads
        # flipped by our freeze(); they belong to `cosmos.net` which we freeze wholesale.
        self._embed_tokens = tm.embed_tokens          # reasoner token embedding (frozen)
        self._reasoner_norm = tm.norm                 # reasoner final norm (only if we read R out)
        self._rotary_emb = getattr(tm, "rotary_emb", None)
        # generator final norm: needed to norm the gen rows before the gen decode heads.
        self._gen_norm = getattr(tm, "norm_moe_gen", tm.norm)

        # ---- SPARSE-DEPTH motion-layer set: the 3-way joint attention fires only every Nth layer.
        # MOTION_LAYERS = {i | (i+1) % stride == 0}; stride=3 -> {2,5,...,35}=12 blocks, stride=6 ->
        # {5,11,...,35}=6 blocks. Always includes the last layer (n_layers is a multiple of 3 & 6).
        # The frozen reasoner+generator still run ALL n_layers; only the motion expert is sparse.
        if motion_layer_stride < 1:
            raise ValueError(f"motion_layer_stride must be >= 1, got {motion_layer_stride}")
        self.motion_layer_stride = motion_layer_stride
        self.motion_layers = {
            i for i in range(self.n_layers) if (i + 1) % motion_layer_stride == 0
        }

        # ---- optional LoRA on the generator and/or reasoner pathway --------------------------
        # INJECT BEFORE building the MoTJointLayers: each MoTJointLayer captures direct references
        # to the base layer's `_moe_gen` / reasoner projections (into a SimpleNamespace) at
        # construction time. `inject_lora_pre_fsdp` REPLACES those submodules on the base layer with
        # LoRA wrappers; if we injected AFTER the layers were built, the captured references would
        # still point at the pre-LoRA `nn.Linear` and the gen tokens would BYPASS the LoRA entirely
        # (grad would never reach the gen-LoRA -- fatal for the camera-only Phase-1 regime, where
        # there is no motion pathway to mask the disconnection). Injecting first makes the captured
        # `q_proj_moe_gen` etc. the LoRA-wrapped modules, so the gen forward is on the trainable path.
        if gen_lora:
            self._inject_gen_lora()
        if reasoner_lora:
            self._inject_reasoner_lora()

        # ---- n_layers joint layers: motion layers own a trainable _moe_motion pathway; plain
        # layers carry NO motion params and run only the frozen 2-way reasoner+generator path. ----
        scaling = getattr(cosmos, "scaling", self.head_dim ** -0.5)
        self.layers = nn.ModuleList([
            MoTJointLayer(
                base_layer=tm.layers[i],
                apply_rotary_fn=tm.layers[i].self_attn._apply_rotary_pos_emb,
                num_heads=cosmos.num_heads,
                num_kv_heads=cosmos.num_kv_heads,
                head_dim=self.head_dim,
                scaling=scaling,
                hidden=self.hidden,
                motion_intermediate_size=motion_intermediate_size,
                has_motion=(i in self.motion_layers),
            )
            for i in range(self.n_layers)
        ])
        self.motion_intermediate_size = motion_intermediate_size
        if self.coupling == "bridge_local":
            self.bridges = nn.ModuleDict({
                str(i): LocalModalityBridge(self.hidden, cosmos.num_heads, self.head_dim)
                for i in self.motion_layers
            })
        else:
            self.bridges = nn.ModuleDict()

        # ---- motion I/O heads (motion2llm / shape2llm / llm2motion / embeds / time_embedder) --
        self.heads = MotionHeads(hidden=self.hidden, motion_dim=motion_dim)

        # ---- generator I/O adapter (NO trainable params of its own; calls frozen net heads) ---
        # Held as a plain attribute (not an nn.Module) so it never registers params; it is a
        # pure call wrapper over the frozen cosmos.net generator heads (LoRA'd / full-FT'd at the
        # net level, never here).
        object.__setattr__(self, "gen", GenHeads(cosmos, dtype=torch.bfloat16, device=cosmos.device))

        # ---- motion final norm: FRESH trainable RMSNorm (matches norm_moe_gen's type/eps only) -
        base_gen_norm = getattr(tm, "norm_moe_gen", tm.norm)
        self.norm_moe_motion = _fresh_rmsnorm_like(base_gen_norm, self.hidden)

        # ---- freeze: motion (unless freeze_motion); gen/reasoner per the toggles (LoRA already
        # injected above, before the layers captured their `_moe_gen` references) --------------
        self.freeze()

    def _build_motion_positions(
        self,
        *,
        n_motion_frames: int,
        temporal_offset: int,
        device,
        include_shape: bool = True,
    ) -> tuple[torch.Tensor, int]:
        """Official-style 3D-mRoPE IDs for the motion segment.

        Motion frames are a temporal ``T x 1 x 1`` grid with temporal compression 1, matching
        Cosmos action/camera tokens rather than being appended after all video patches. This makes
        frame ``k`` live at the same physical temporal coordinate family as video/camera frame
        ``k`` when gen tokens are present.

        The shape token is non-temporal conditioning. It uses the last text temporal plane with
        distinct spatial axes, so it does not collide with either text ``(i,i,i)`` tokens or
        video/motion frame-0 ``(temporal_offset,*,*)`` coordinates.
        """
        parts = []
        if include_shape:
            shape_t = max(0, int(temporal_offset) - 1)
            shape_pos = torch.tensor(
                [[shape_t], [temporal_offset], [temporal_offset]],
                dtype=torch.long,
                device=device,
            )
            parts.append(shape_pos)
        if n_motion_frames > 0:
            frame_pos, next_off = get_3d_mrope_ids_vae_tokens(
                grid_t=n_motion_frames,
                grid_h=1,
                grid_w=1,
                temporal_offset=temporal_offset,
                reset_spatial_indices=True,
                fps=None,
                base_fps=getattr(self.gen, "base_fps", 24.0),
                temporal_compression_factor=1,
                base_temporal_compression_factor=getattr(self.gen, "tcf_vision", 4),
                start_frame_offset=0,
            )
            parts.append(frame_pos.to(device).long())
        else:
            next_off = temporal_offset + 1 if include_shape else temporal_offset
        return torch.cat(parts, dim=1), int(next_off)

    # -------------------------------------------------------------------------------------
    # LoRA injection
    # -------------------------------------------------------------------------------------
    def _materialize_injected_lora(self):
        """Move the just-injected LoRA params OFF the meta device and initialize them.

        ``inject_lora_pre_fsdp`` leaves ``lora_A``/``lora_B`` as UNINITIALIZED meta-device
        parameters (so they can be sharded before FSDP). Because ``self.cosmos.net`` is a plain
        attribute (NOT a submodule of this ``nn.Module``), a later ``model.to(device)`` does NOT
        reach these params -- they would stay on meta and the FIRST backward through a LoRA'd gen
        projection trips "expected device meta but got cuda" (this surfaced once the gen-LoRA
        actually joined the forward path in the camera-only Phase-1 regime). So we materialize +
        init here, in-place, on the frozen net's device. Idempotent: `to_empty` on already-real
        params is a no-op-ish re-alloc, so we only touch modules that still have meta LoRA weights.
        """
        from cosmos_framework.utils.vfm.lora import (
            LoraInjectedLinear,
            init_lora_weights_post_materialization,
        )

        dev = getattr(self.cosmos, "device", None) or next(self.cosmos.net.parameters()).device
        touched = 0
        for module in self.cosmos.net.modules():
            if not isinstance(module, LoraInjectedLinear):
                continue
            if module.lora_A.weight.is_meta or module.lora_B.weight.is_meta:
                module.lora_A.to_empty(device=dev)
                module.lora_B.to_empty(device=dev)
                touched += 1
        if touched:
            init_lora_weights_post_materialization(self.cosmos.net)

    def _inject_gen_lora(self):
        """Inject LoRA on q/k/v/o_proj_moe_gen of the frozen base layers (generator pathway)."""
        from cosmos_framework.utils.vfm.lora import inject_lora_pre_fsdp

        inject_lora_pre_fsdp(
            self.cosmos.net,
            lora_rank=self.gen_lora_rank,
            lora_alpha=self.gen_lora_alpha,
            lora_target_modules="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
        )
        self._materialize_injected_lora()

    def _inject_reasoner_lora(self):
        """Inject LoRA on the reasoner q/k/v/o_proj (the plain, non-`_moe_gen` projections).

        Mirrors `_inject_gen_lora` but targets the understanding/causal pathway so the reasoner
        can be lightly adapted instead of fully frozen (DESIGN_7TASK.md section 5). The injected
        LoRA params carry `lora_` in their names and are picked up by `freeze()`.
        """
        from cosmos_framework.utils.vfm.lora import inject_lora_pre_fsdp

        inject_lora_pre_fsdp(
            self.cosmos.net,
            lora_rank=getattr(self.cosmos, "lora_rank", 16),
            lora_alpha=getattr(self.cosmos, "lora_alpha", 16),
            lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        )
        self._materialize_injected_lora()

    # -------------------------------------------------------------------------------------
    # Freeze logic
    # -------------------------------------------------------------------------------------
    def _is_gen_io_head(self, name: str) -> bool:
        """True if `name` (a model-level param name) belongs to a frozen generator I/O head."""
        return any(h in name for h in self._GEN_IO_HEAD_NAMES)

    def _is_motion_name(self, name: str) -> bool:
        """True if `name` belongs to the MOTION pathway (_moe_motion + motion heads + final norm).

        This is the single definition of "motion-owned param" shared by the trainable predicate,
        the `--freeze_motion` exclusion, and the `--init_motion` subset loader. It covers the new
        per-layer `_moe_motion` weights, the motion I/O heads (attached under `heads.` here, which
        include `motion2llm`/`llm2motion`/`shape2llm`/`motion_modality_embed`/`shape_type_embed`/
        the motion-owned `time_embedder`), and the motion final norm `norm_moe_motion`.
        """
        return (
            "_moe_motion" in name
            or name.startswith("heads.")
            or "heads." in name
            or "norm_moe_motion" in name
        )

    def _is_trainable_name(self, name: str) -> bool:
        """The single predicate for "this param trains" under the active toggles.

        trainable iff:
          * MOTION (unless `freeze_motion`): `_moe_motion` / a motion head / `norm_moe_motion`, OR
          * (gen_lora OR reasoner_lora) AND the param is a LoRA adapter (`lora_`), OR
          * gen_full AND the param is a `_moe_gen` weight or a generator I/O head.

        When `freeze_motion` is set (Phase-1 curriculum) the motion pathway is EXCLUDED entirely,
        so only the active gen/reasoner adapters train (e.g. gen-LoRA on the camera tasks).
        When `freeze_gen` is set, generator LoRA/full/action-head params are EXCLUDED even if
        `gen_lora` or `gen_full` was requested to instantiate/load them.

        Note: when ONLY gen_lora is on, `lora_`-named params on the reasoner projections do not
        exist (we never injected them), so the `lora_` rule cannot accidentally train the
        reasoner; the same holds the other way for reasoner_lora vs the gen projections.
        """
        n = name.lower()
        if name.startswith("bridges.") or ".bridges." in name:
            return self.coupling == "bridge_local"
        # motion trains unless the Phase-1 freeze is active.
        if self._is_motion_name(name):
            return not self.freeze_motion
        # LoRA adapters (gen and/or reasoner) when their toggle is on.
        if (self.gen_lora or self.reasoner_lora) and "lora_" in n:
            if self.freeze_gen and "_moe_gen" in name:
                return False
            return True
        # gen-LoRA ALSO adapts the camera (action) I/O heads -- matching nymeria_world's winning
        # LoRA config (keys_to_select included action2llm/llm2action/action_modality_embed). The
        # ego-camera pseudo-action is a domain-specific modality whose base projection heads are
        # generic (loaded from the base's zero-shot camera path), so they MUST adapt or the LoRA is
        # forced to compensate for a frozen projection. vae2llm/llm2vae stay frozen (base Cosmos
        # video decode is already strong), exactly as nymeria_world's LoRA did.
        # HISTORY (2026-07-02): "loaded from the base" is only TRUE since the
        # train_motion_ft._diffusers_to_net_key fix -- before it, the base-weight loader silently
        # SKIPPED the action_proj_in/out (action2llm/llm2action) and the reasoner attention
        # (to_q/k/v/out + norm_q/k) keys, leaving them at RANDOM init. Any checkpoint trained
        # before that date adapted a random action head / random reasoner attention, NOT the
        # pretrained prior. (Loader verified loaded=808 skipped=6 post-fix.)
        if (
            self.gen_lora
            and not self.freeze_gen
            and any(h in name for h in ("action2llm", "llm2action", "action_modality_embed"))
        ):
            return True
        # full generator finetune.
        if self.gen_full and not self.freeze_gen and ("_moe_gen" in name or self._is_gen_io_head(name)):
            return True
        return False

    # -------------------------------------------------------------------------------------
    # Unified parameter view: `self` (motion pathway + heads + norm_moe_motion) PLUS the
    # generator/reasoner LoRA / gen_full params, which live under `self.cosmos.net` and are
    # NOT registered as submodules of this model (the per-layer MoTJointLayer holds the frozen
    # base in a plain SimpleNamespace, and `self.cosmos` is a plain attribute). Without this,
    # `model.parameters()` / `model.named_parameters()` would MISS every LoRA / gen_full param,
    # so the optimizer, grad-clip, all-reduce, freeze, and the frozen-grad check would all
    # silently ignore the generator pathway. `cosmos.net` params are namespaced with a
    # ``cosmos.net.`` prefix so names never collide with the model's own params.
    # -------------------------------------------------------------------------------------
    _NET_PREFIX = "cosmos.net."

    def named_all_parameters(self):
        """Yield (name, param) over BOTH the model's own params and `cosmos.net`'s params.

        The net's names are prefixed with ``cosmos.net.`` so `_is_trainable_name` (which keys on
        substrings like ``_moe_gen`` / ``lora_`` / the gen I/O head names) still matches, and the
        two namespaces never collide. De-dups by python id so a param registered in both views
        (e.g. `_embed_tokens`, which is also under `cosmos.net`) is yielded once.
        """
        seen: set[int] = set()
        for n, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield n, p
        for n, p in self.cosmos.net.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield self._NET_PREFIX + n, p

    def freeze(self):
        """Apply the train-scope toggles by setting requires_grad per `_is_trainable_name`.

        Motion (`_moe_motion` + heads + `norm_moe_motion`) always trains; the reasoner and
        generator pathways train only under their LoRA / full toggles. Everything else in
        `cosmos.net` is frozen.
        """
        # 1) Freeze the ENTIRE frozen Cosmos net first.
        for p in self.cosmos.net.parameters():
            p.requires_grad_(False)

        # 2) Re-enable grad per the active toggles (matched by parameter NAME) over BOTH the
        #    model's own params AND the net's (LoRA / gen_full live under cosmos.net).
        for name, p in self.named_all_parameters():
            p.requires_grad_(self._is_trainable_name(name))

        # The frozen references (embed_tokens / reasoner norm) are not our submodules, but enforce
        # frozen anyway unless reasoner_lora has injected LoRA into them (idempotent; they live
        # under cosmos.net already and are re-checked by name above when reasoner_lora is on).
        if not self.reasoner_lora:
            for mod in (self._embed_tokens, self._reasoner_norm):
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad_(False)

    def trainable_parameters(self):
        """All params that train under the active toggles (own + cosmos.net LoRA/gen_full)."""
        out, seen = [], set()
        for _n, p in self.named_all_parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def is_motion_head(self, name: str) -> bool:
        """True if `name` (a model-level param name) belongs to the motion I/O heads.

        Thin wrapper over `MotionHeads.is_motion_head_name` so train.py / freeze logic can
        ask the model directly. The heads live under the `heads.` attribute here, so we accept
        both the bare head-relative name and the `heads.`-prefixed model-level name.
        """
        return name.startswith("heads.") or MotionHeads.is_motion_head_name(name)

    def trainable_state_dict(self):
        """State dict filtered to the TRAINABLE set only.

        Returns a CPU-detached dict containing exactly the params that carry grad under the
        active toggles — the new `_moe_motion` per-layer pathway, the motion final norm
        (`norm_moe_motion`), the motion I/O heads, plus (when enabled) the gen/reasoner LoRA
        adapters or the full `_moe_gen` + generator I/O heads. The frozen weights are excluded so
        checkpoints store only the trainable delta (overlaid by name at load time).
        """
        sd = {}
        for n, p in self.named_all_parameters():
            if p.requires_grad:
                sd[n] = p.detach().cpu()
        return sd

    def assert_frozen_grads_zero(self):
        """--smoke sanity: every param NOT in the active trainable set must carry zero grad.

        Call AFTER a backward. Trips if any frozen param accumulated grad, which would mean the
        freeze routing leaked. Respects the active {reasoner_lora, gen_lora, gen_full} toggles via
        the shared `_is_trainable_name` predicate.
        """
        bad = []
        for name, p in self.named_all_parameters():
            if self._is_trainable_name(name):
                continue
            if p.requires_grad:
                bad.append(f"{name} requires_grad=True (should be frozen)")
            if p.grad is not None and p.grad.abs().sum().item() != 0.0:
                bad.append(f"{name} has nonzero grad (frozen leak)")
        assert not bad, "frozen-grad check FAILED:\n  " + "\n  ".join(bad[:20])
        return True

    def assert_motion_frozen(self):
        """PHASE-1 sanity: under `--freeze_motion` every motion param must have requires_grad=False.

        The motion expert is still BUILT (so the architecture matches Phase 3), but with
        `freeze_motion` it must carry NO grad and never be stepped -- only the gen-LoRA trains.
        Trips if any `_moe_motion` / motion-head / `norm_moe_motion` param is still trainable.
        """
        assert self.freeze_motion, "assert_motion_frozen called without freeze_motion set"
        bad = [name for name, p in self.named_all_parameters()
               if self._is_motion_name(name) and p.requires_grad]
        assert not bad, (
            "freeze_motion FAILED: motion params still require grad:\n  "
            + "\n  ".join(bad[:20])
        )
        return True

    # -------------------------------------------------------------------------------------
    # Subset warm-start: load a PRIOR checkpoint's gen or motion params by name (strict=False)
    # -------------------------------------------------------------------------------------
    def _load_subset(self, ckpt_sd: dict, select) -> tuple[int, int, int]:
        """Overlay the params of `ckpt_sd` whose NAME passes `select(name)` into this model.

        `ckpt_sd` is a name->tensor dict from a prior run's checkpoint (the trainer saves
        `trainable_state_dict()`, keyed by `named_all_parameters()` names -- own params keep their
        plain names; `cosmos.net` params carry the ``cosmos.net.`` prefix). We match by name into
        this fresh model's `named_all_parameters()`, copy where the name is present AND `select`
        accepts it AND shapes match, and SKIP everything else (strict=False). Returns
        `(loaded, skipped_missing, skipped_shape)` counts.
        """
        own = {name: p for name, p in self.named_all_parameters()}
        loaded = skipped_missing = skipped_shape = 0
        with torch.no_grad():
            for name, tensor in ckpt_sd.items():
                if not select(name):
                    continue
                p = own.get(name)
                if p is None:
                    skipped_missing += 1
                    continue
                if tuple(p.shape) != tuple(tensor.shape):
                    skipped_shape += 1
                    continue
                p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
                loaded += 1
        return loaded, skipped_missing, skipped_shape

    def load_gen_subset(self, ckpt_sd: dict) -> tuple[int, int, int]:
        """Warm-start ONLY the generator/gen-LoRA params from a (Phase-1) checkpoint.

        Selects the gen keys under `cosmos.net.` -- the LoRA adapters (`lora_`), the `_moe_gen`
        weights, and the generator I/O heads (vae2llm/llm2vae/action2llm/llm2action/
        action_modality_embed). Motion + reasoner params are ignored. Returns
        `(loaded, skipped_missing, skipped_shape)`.
        """
        def is_gen(name: str) -> bool:
            if not name.startswith(self._NET_PREFIX):
                return False
            return "lora_" in name.lower() or "_moe_gen" in name or self._is_gen_io_head(name)

        return self._load_subset(ckpt_sd, is_gen)

    def load_motion_subset(self, ckpt_sd: dict) -> tuple[int, int, int]:
        """Warm-start ONLY the motion pathway (_moe_motion + heads + norm_moe_motion) from a
        (Phase-2) checkpoint. Gen + reasoner params are ignored. Returns
        `(loaded, skipped_missing, skipped_shape)`."""
        return self._load_subset(ckpt_sd, self._is_motion_name)

    # -------------------------------------------------------------------------------------
    # Rotary cos/sin for the packed sequence
    # -------------------------------------------------------------------------------------
    def _build_cos_sin(self, positions: torch.Tensor, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [N_total] (1-D) OR [3, N_total] (3D-mRoPE) -> (cos, sin) [N_total, head_dim].

        Uses the base net's rotary path (`cosmos.rope` / `rotary_emb`) so motion / generator
        frames get temporal (and, for the gen case, spatial) rotary consistent with the reasoner
        sequence. Reasoner rows use causal positions 0..T_text-1; gen + motion rows continue from
        T_text onward (built by the caller). A `[3, N]` form carries the full T/H/W mRoPE axes for
        the gen-present case; a plain 1-D `[N]` form is the text->motion fast path (replicated
        across the 3 axes by `cosmos.rope`). Falls back to a self-contained RoPE table if no rope
        handle is available (1-D positions only).
        """
        rope = getattr(self.cosmos, "rope", None)
        if rope is not None:
            cos, sin = rope(positions)  # accepts 1-D [N] or 2-D [3, N]; -> [N, head_dim] each
            return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

        # Self-contained fallback (standard RoPE table over head_dim, base 1e6 like Qwen3).
        # Only supports 1-D positions; the gen-present path requires the real rope handle.
        if positions.dim() != 1:
            positions = positions[0]
        hd = self.head_dim
        half = hd // 2
        base = float(getattr(self.cosmos, "rope_theta", 1.0e6))
        inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        ang = positions.float()[:, None] * inv_freq[None, :]      # [N, half]
        ang = torch.cat([ang, ang], dim=-1)                       # [N, head_dim]
        return ang.cos().to(dtype), ang.sin().to(dtype)

    # -------------------------------------------------------------------------------------
    # Forward (one denoiser call)
    # -------------------------------------------------------------------------------------
    def forward(
        self,
        input_ids_list: list[torch.Tensor],   # per-sample reasoner token ids [T_text_s] (long)
        x_t: Optional[torch.Tensor] = None,    # [B, Tm, 283] noised motion (padded to batch-max)
        t_or_sigma: torch.Tensor = None,       # [B] flow time / sigma (shared across modalities)
        neutral_joints: Optional[torch.Tensor] = None,   # [B, 30, 3] actor skeleton (motion tasks)
        motion_pad_mask: Optional[torch.Tensor] = None,  # [B, Tm] True = padded motion frame
        noisy_frame_mask: Optional[torch.Tensor] = None, # [B, Tm] True = noised motion frame
        *,
        modes: Optional[list[str]] = None,     # per-sample task mode (one of task_plan.TASKS)
        video_latents: Optional[list[Optional[torch.Tensor]]] = None,  # per-sample [C,T_lat,h,w]
        camera_action: Optional[list[Optional[torch.Tensor]]] = None,  # per-sample [T-1,9]
        reasoner_inputs: Optional[list[Optional[dict]]] = None,
        return_dict: Optional[bool] = None,
    ):
        """Run the joint forward over a (possibly mixed-mode) batch.

        Two call regimes, both supported:

          (A) text->motion fast path (BACKWARD COMPATIBLE): `modes is None`. Every sample is a
              motion target with NO generator tokens (`gen_idx` empty). Returns `pred[B, Tm, 283]`
              (a bare tensor) exactly as before, unless `return_dict=True`.

          (B) 7-task path: `modes` is a per-sample list of task names. For each sample we resolve
              its `task_plan.ResolvedPlan` (CLEAN/NOISED per token + loss spec), build the real
              generator segment via `self.gen.build_gen_segment(...)` (image/video/camera) and the
              motion segment (when present), pack `[ und | gen | mot ]`, run the joint layers, and
              decode each SUPERVISED modality. Returns a dict
              `{motion_pred, video_pred, camera_pred, resolved}` so the trainer applies the
              per-task flow losses; condition-only / absent modalities are omitted.

        `t_or_sigma[s]` is the single flow time/sigma for sample `s`, shared by every NOISED
        token of that sample (motion frames AND any noised gen frames), matching the rectified-flow
        contract used across modalities.
        """
        if return_dict is None:
            return_dict = modes is not None

        # Resolve device/dtype from whichever modality tensor is present.
        if x_t is not None:
            device = x_t.device
            B = x_t.shape[0]
            Tm = x_t.shape[1]
        else:
            device = self.cosmos.device
            B = len(input_ids_list)
            Tm = 0
        dtype = torch.bfloat16

        if modes is None:
            modes = ["text2motion"] * B
        if video_latents is None:
            video_latents = [None] * B
        if camera_action is None:
            camera_action = [None] * B
        if reasoner_inputs is None:
            reasoner_inputs = [None] * B
        if t_or_sigma is None:
            t_or_sigma = torch.zeros(B, device=device)
        # ---- TIMESTEP PRE-SCALE (the fix for "loss drops but samples are noise") ---------------
        # `t_or_sigma` is the true flow time in [0,1] (used for NOISING upstream, outside forward).
        # Both the motion head (motion_heads.encode_motion) and the gen path (gen_heads/Cosmos)
        # embed `t * timestep_scale` internally (Cosmos convention, timestep_scale=1e-3). Feeding raw
        # t makes the embedder see t*1e-3 -> a ~constant, non-discriminative signal (t=0 vs t=1
        # indistinguishable), so the model learns only the t-averaged velocity. Pre-divide by
        # TIMESTEP_SCALE so `t/scale * scale == t` reaches the embedder. Mirrors train_motion_ft.py.
        # Noising already happened upstream with the true t, so this is embedding-only.
        t_or_sigma = t_or_sigma / TIMESTEP_SCALE

        # Whether ANY sample uses [3, N] mRoPE positions -> decides 1-D vs 3D rotary build.
        any_3d_positions = False

        # --- (1) per-sample segment assembly: reasoner | generator | motion -------------------
        und_rows: list[torch.Tensor] = []          # [T_text_s, hidden]
        gen_rows: list[torch.Tensor] = []          # [N_gen_s, hidden] (may be 0)
        mot_rows: list[torch.Tensor] = []          # [n_mot_s, hidden] (may be 0)
        gen_segments: list[Optional[object]] = []  # the GenSegment (decode bookkeeping) per sample
        valid_frame_idx: list[Optional[torch.Tensor]] = []  # original motion rows kept, per sample
        positions_list: list[torch.Tensor] = []    # rotary positions ([N_s] or [3, N_s]) per sample
        resolved_list: list[Optional[TP.ResolvedPlan]] = []
        gen_frame_list: list[torch.Tensor] = []
        gen_clean_list: list[torch.Tensor] = []
        motion_frame_list: list[torch.Tensor] = []
        reasoner_visual_masks: list[Optional[torch.Tensor]] = []
        reasoner_deepstacks: list[Optional[list[torch.Tensor]]] = []

        for s in range(B):
            r_in = reasoner_inputs[s]
            if r_in is not None:
                und_emb = r_in["inputs_embeds"].to(device=device, dtype=dtype)
                if und_emb.dim() == 3:
                    und_emb = und_emb.squeeze(0)
                reasoner_pos = r_in["position_ids"].to(device=device).long()
                reasoner_visual_masks.append(r_in.get("visual_pos_mask", None))
                reasoner_deepstacks.append(r_in.get("deepstack_visual_embeds", None))
            else:
                ids = input_ids_list[s].to(device).long().view(-1)
                und_emb = self._embed_tokens(ids)              # [T_text_s, hidden]
                reasoner_pos = None
                reasoner_visual_masks.append(None)
                reasoner_deepstacks.append(None)
            T_text_s = und_emb.shape[0]
            und_rows.append(und_emb.to(dtype))

            mode = modes[s]
            plan = TP.build_task_plan(mode)
            reasoner_textimg = (
                self.textimg_condition == "reasoner" and mode == "textimg2motion"
            )

            # ---- motion segment (present iff the task carries motion) ------------------------
            if plan.motion.present:
                assert x_t is not None and neutral_joints is not None, \
                    f"task {mode!r} packs motion but x_t/neutral_joints are None"
                valid = (~motion_pad_mask[s]) if motion_pad_mask is not None \
                    else torch.ones(Tm, dtype=torch.bool, device=device)
                vidx = torch.nonzero(valid, as_tuple=False).view(-1)           # [n_mot_s]
                valid_frame_idx.append(vidx)
                # motion clean/noised policy: "all" -> motion is a CLEAN condition (motimg2video);
                # "shape_only" -> all valid frames noised. Build the per-frame noisy mask the
                # encoder expects (shape token handled separately, always clean).
                if plan.motion.clean_policy == "all":
                    nfm_s = torch.zeros(1, Tm, dtype=torch.bool, device=device)  # all clean
                else:
                    nfm_s = (noisy_frame_mask[s:s + 1] if noisy_frame_mask is not None
                             else valid.view(1, Tm))
                shape_tok = self.heads.encode_shape(neutral_joints[s:s + 1])[0]   # [1, hidden]
                mtoks = self.heads.encode_motion(
                    x_t[s:s + 1], t_or_sigma[s:s + 1], nfm_s
                )[0]                                                             # [Tm, hidden]
                mtoks_valid = mtoks[vidx]                                       # [n_mot_s, hidden]
                mrow = torch.cat([shape_tok.to(dtype), mtoks_valid.to(dtype)], dim=0)  # [1+n_mot, h]
                motion_frame_list.append(torch.cat([
                    torch.full((1,), -1, device=device, dtype=torch.long),
                    vidx.to(device=device, dtype=torch.long),
                ], dim=0))
            else:
                valid_frame_idx.append(None)
                mrow = und_emb.new_zeros(0, self.hidden)
                motion_frame_list.append(torch.empty(0, device=device, dtype=torch.long))
            mot_rows.append(mrow)

            # ---- generator segment (present iff the task carries image/video/camera) ---------
            resolved = None
            seg = None
            has_gen_for_sample = plan.has_gen and not reasoner_textimg
            if has_gen_for_sample:
                t_lat = 0
                vlat = video_latents[s]
                if (plan.video.present or plan.image.present) and vlat is not None:
                    # video: full latent stack; image-only: a single frame 0 (resolver -> t_lat=1).
                    t_lat = 1 if (plan.image.present and not plan.video.present) else int(vlat.shape[1])
                n_camera = 0
                cam = camera_action[s]
                if plan.camera.present and cam is not None:
                    n_camera = int(cam.shape[0])
                mvm = None
                if plan.motion.present:
                    # motion frame count (excluding shape token) for the resolver's bookkeeping.
                    mvm = [True] * int(valid_frame_idx[s].numel())
                resolved = TP.resolve_sample(
                    mode, t_lat=t_lat, n_camera=n_camera,
                    motion_valid_mask=mvm, has_shape_token=plan.motion.present,
                )
                native_gen_pack = self.gen_packing == "native"
                gen_temporal_offset = (
                    float(T_text_s) + self.gen_temporal_margin
                    if native_gen_pack
                    else T_text_s
                )
                seg = self.gen.build_gen_segment(
                    resolved,
                    video_latents=vlat,
                    camera_action=cam,
                    sigma=t_or_sigma[s:s + 1].reshape(1),
                    temporal_offset=gen_temporal_offset,
                    fps=self.gen_fps if native_gen_pack else None,
                )
            gen_segments.append(seg)
            resolved_list.append(resolved)

            if seg is not None:
                gen_rows.append(seg.tokens.to(dtype))
                any_3d_positions = True
                gen_mrope = seg.mrope_ids.to(device)                           # [3, N_gen]
                next_off = seg.next_temporal_offset
                g_frames = torch.full((seg.tokens.shape[0],), -1, device=device, dtype=torch.long)
                for pname, (rs, re) in seg.offsets.items():
                    part = seg.parts[pname]
                    if pname in ("video", "image"):
                        spatial = int(part.grid[1] * part.grid[2])
                        frames = torch.arange(part.grid[0], device=device, dtype=torch.long)
                        g_frames[rs:re] = frames.repeat_interleave(spatial)
                gen_frame_list.append(g_frames)
                gen_clean_list.append(seg.condition_mask.to(device).bool())
            else:
                gen_rows.append(und_emb.new_zeros(0, self.hidden))
                gen_mrope = None
                next_off = T_text_s
                gen_frame_list.append(torch.empty(0, device=device, dtype=torch.long))
                gen_clean_list.append(torch.empty(0, device=device, dtype=torch.bool))

            # ---- rotary positions for this sample's packed rows ------------------------------
            # Reasoner: causal 0..T_text_s-1. Gen: the segment's 3D-mRoPE ids (already offset by
            # T_text_s). Motion has two modes:
            #   legacy   : continue sequentially after the gen segment, replicated across axes.
            #   cosmos3d : official-style T x 1 x 1 frame ids starting at T_text_s, so motion
            #              frame k shares the same physical time origin as video/camera frame k.
            n_mot = mrow.shape[0]
            if self.motion_mrope == "cosmos3d" and n_mot > 0:
                mot_mrope, _mot_next = self._build_motion_positions(
                    n_motion_frames=n_mot - 1,
                    temporal_offset=T_text_s,
                    device=device,
                    include_shape=True,
                )
                any_3d_positions = True
                if reasoner_pos is not None:
                    parts3 = [reasoner_pos if reasoner_pos.dim() == 2
                              else reasoner_pos.view(1, -1).expand(3, -1)]
                else:
                    und_pos = torch.arange(T_text_s, device=device, dtype=torch.long)
                    parts3 = [und_pos.view(1, -1).expand(3, -1)]
                if gen_mrope is not None:
                    parts3.append(gen_mrope)
                parts3.append(mot_mrope)
                positions_list.append(torch.cat(parts3, dim=1))
            elif gen_mrope is not None:
                # gen-present sample: full [3, N_s] mRoPE (reasoner causal, gen segment ids, then
                # motion continuing temporally after the gen segment).
                if reasoner_pos is not None:
                    parts3 = [reasoner_pos if reasoner_pos.dim() == 2
                              else reasoner_pos.view(1, -1).expand(3, -1)]
                else:
                    und_pos = torch.arange(T_text_s, device=device, dtype=torch.long)
                    parts3 = [und_pos.view(1, -1).expand(3, -1)]               # [3, T_text_s]
                parts3.append(gen_mrope)                                       # [3, N_gen]
                if n_mot > 0:
                    mot_pos = torch.arange(next_off, next_off + n_mot, device=device, dtype=torch.long)
                    parts3.append(mot_pos.view(1, -1).expand(3, -1))          # [3, n_mot]
                positions_list.append(torch.cat(parts3, dim=1))               # [3, N_s]
            else:
                # no-gen sample (text->motion): 1-D contiguous positions (reasoner then motion).
                # Normalized to [3, N] in step (3) only if some OTHER sample in the batch has gen.
                if reasoner_pos is not None:
                    parts3 = [reasoner_pos if reasoner_pos.dim() == 2
                              else reasoner_pos.view(1, -1).expand(3, -1)]
                    if n_mot > 0:
                        mot_pos = torch.arange(T_text_s, T_text_s + n_mot, device=device, dtype=torch.long)
                        parts3.append(mot_pos.view(1, -1).expand(3, -1))
                    positions_list.append(torch.cat(parts3, dim=1))
                    any_3d_positions = True
                else:
                    pos = torch.arange(T_text_s + n_mot, device=device, dtype=torch.long)
                    positions_list.append(pos)                                # [N_s]

        # --- (2) assemble the packed sequence + role-index tensors + offsets -------------------
        packed_parts: list[torch.Tensor] = []
        und_idx_parts: list[torch.Tensor] = []
        gen_idx_parts: list[torch.Tensor] = []
        mot_idx_parts: list[torch.Tensor] = []
        und_lens: list[int] = []
        gen_lens: list[int] = []    # generator token count per sample (for the 2-way bundle)
        full_lens: list[int] = []   # gen+motion token count per sample
        total_lens: list[int] = []
        gen_offsets: list[tuple[int, int]] = []   # (start,end) absolute rows of gen, per sample
        mot_offsets: list[tuple[int, int]] = []   # (start,end) absolute rows of motion, per sample
        und_offsets_abs: list[tuple[int, int]] = []
        cursor = 0
        for s in range(B):
            u = und_rows[s]
            g = gen_rows[s]
            m = mot_rows[s]
            nu, ng, nm = u.shape[0], g.shape[0], m.shape[0]
            packed_parts.append(u)
            packed_parts.append(g)
            packed_parts.append(m)
            und_idx_parts.append(torch.arange(cursor, cursor + nu, device=device, dtype=torch.long))
            und_offsets_abs.append((cursor, cursor + nu))
            gstart = cursor + nu
            gen_idx_parts.append(torch.arange(gstart, gstart + ng, device=device, dtype=torch.long))
            mstart = gstart + ng
            mot_idx_parts.append(torch.arange(mstart, mstart + nm, device=device, dtype=torch.long))
            gen_offsets.append((gstart, gstart + ng))
            mot_offsets.append((mstart, mstart + nm))
            und_lens.append(nu)
            gen_lens.append(ng)
            full_lens.append(ng + nm)             # full block = gen UNION motion (packed order)
            total_lens.append(nu + ng + nm)
            cursor += nu + ng + nm

        packed_sequence = torch.cat(packed_parts, dim=0).to(dtype)              # [N_total, hidden]
        und_idx = torch.cat(und_idx_parts, dim=0) if und_idx_parts else \
            torch.empty(0, device=device, dtype=torch.long)
        gen_idx = torch.cat(gen_idx_parts, dim=0) if gen_idx_parts else \
            torch.empty(0, device=device, dtype=torch.long)
        mot_idx = torch.cat(mot_idx_parts, dim=0) if mot_idx_parts else \
            torch.empty(0, device=device, dtype=torch.long)

        # full query set = generator UNION motion rows, in packed (ascending row) order. With gen
        # present this couples gen<->motion bidirectionally; with gen empty (text->motion) it
        # reduces to mot_idx exactly as before.
        full_idx = torch.sort(torch.cat([gen_idx, mot_idx], dim=0)).values

        # build_offsets returns the varlen cu_seqlens + maxlens for the two-call mask. We rebuild
        # und_idx/full_idx ourselves above (same packed contract: und block, then gen, then mot).
        # offsets_3way: the FULL 3-way bundle (reasoner causal; gen u motion full over all rows),
        # used at MOTION layers exactly as before.
        (
            _bo_und_idx,
            _bo_full_idx,
            und_offsets,
            full_offsets,
            sample_offsets,
            (max_und_len, max_full_len, max_sample_len),
        ) = build_offsets(und_lens, full_lens, total_lens, device)
        offsets_3way = {
            "full_idx": full_idx,
            "und_offsets": und_offsets,
            "full_offsets": full_offsets,
            "sample_offsets": sample_offsets,
            "max_und_len": max_und_len,
            "max_full_len": max_full_len,
            "max_sample_len": max_sample_len,
        }

        # --- (2b) SPARSE-DEPTH: 2-way (reasoner+generator) bundle, built ONCE -----------------
        # At a PLAIN layer the motion expert does NOT run; we gather ONLY the reasoner+gen rows
        # into a contiguous buffer (rg = und u gen, per-sample contiguous since the packed layout
        # is [und | gen | mot] -> und block then gen block are adjacent) and run the SAME two-call
        # mask on that smaller buffer. With no motion rows in the buffer, gen/reasoner cannot attend
        # motion; motion rows in `packed` are simply left untouched. The 2-way "full" block = gen
        # only; the KV = und+gen only. This bundle is layout-only (no values) so it is built once.
        rg_idx_parts: list[torch.Tensor] = []          # absolute rows of und u gen, per sample
        und_idx_local_parts: list[torch.Tensor] = []   # und rows WITHIN the gathered rg buffer
        gen_idx_local_parts: list[torch.Tensor] = []    # gen rows WITHIN the gathered rg buffer
        rg_total_lens: list[int] = []                   # nu+ng per sample (the 2-way sample length)
        rg_cursor = 0                                   # running pointer INTO the gathered rg buffer
        for s in range(B):
            nu = und_lens[s]
            ng = gen_lens[s]
            gstart, gend = gen_offsets[s]
            ustart = gstart - nu                        # und block precedes gen block
            rg_idx_parts.append(
                torch.arange(ustart, ustart + nu, device=device, dtype=torch.long))
            rg_idx_parts.append(
                torch.arange(gstart, gend, device=device, dtype=torch.long))
            und_idx_local_parts.append(
                torch.arange(rg_cursor, rg_cursor + nu, device=device, dtype=torch.long))
            gen_idx_local_parts.append(
                torch.arange(rg_cursor + nu, rg_cursor + nu + ng, device=device, dtype=torch.long))
            rg_total_lens.append(nu + ng)
            rg_cursor += nu + ng

        rg_idx = (torch.cat(rg_idx_parts, dim=0) if rg_idx_parts
                  else torch.empty(0, device=device, dtype=torch.long))
        und_idx_local = (torch.cat(und_idx_local_parts, dim=0) if und_idx_local_parts
                         else torch.empty(0, device=device, dtype=torch.long))
        gen_idx_local = (torch.cat(gen_idx_local_parts, dim=0) if gen_idx_local_parts
                         else torch.empty(0, device=device, dtype=torch.long))
        empty_mot = torch.empty(0, device=device, dtype=torch.long)
        empty_gen = torch.empty(0, device=device, dtype=torch.long)

        # 2-way offsets over the gathered rg buffer: full block = gen only, KV = und+gen.
        (
            _bo2_und_idx,
            _bo2_full_idx,
            und_offsets_2way,
            full_offsets_2way,
            sample_offsets_2way,
            (max_und_len_2, max_full_len_2, max_sample_len_2),
        ) = build_offsets(und_lens, gen_lens, rg_total_lens, device)
        offsets_2way = {
            "full_idx": gen_idx_local,            # full queries within rg = gen rows only
            "und_offsets": und_offsets_2way,
            "full_offsets": full_offsets_2way,
            "sample_offsets": sample_offsets_2way,
            "max_und_len": max_und_len_2,
            "max_full_len": max_full_len_2,
            "max_sample_len": max_sample_len_2,
        }
        # rotary cos/sin gathered to the rg buffer order (built once below, after cos/sin exist).

        # --- (2c) bridge mode: reasoner+motion bundle for motion expert without gen K/V -------
        rm_idx_parts: list[torch.Tensor] = []
        und_idx_rm_parts: list[torch.Tensor] = []
        mot_idx_rm_parts: list[torch.Tensor] = []
        rm_total_lens: list[int] = []
        mot_lens: list[int] = []
        rm_cursor = 0
        for s in range(B):
            nu = und_lens[s]
            mstart, mend = mot_offsets[s]
            nm = mend - mstart
            ustart, uend = und_offsets_abs[s]
            rm_idx_parts.append(torch.arange(ustart, uend, device=device, dtype=torch.long))
            rm_idx_parts.append(torch.arange(mstart, mend, device=device, dtype=torch.long))
            und_idx_rm_parts.append(torch.arange(rm_cursor, rm_cursor + nu, device=device, dtype=torch.long))
            mot_idx_rm_parts.append(
                torch.arange(rm_cursor + nu, rm_cursor + nu + nm, device=device, dtype=torch.long)
            )
            rm_total_lens.append(nu + nm)
            mot_lens.append(nm)
            rm_cursor += nu + nm
        rm_idx = (torch.cat(rm_idx_parts, dim=0) if rm_idx_parts
                  else torch.empty(0, device=device, dtype=torch.long))
        und_idx_rm = (torch.cat(und_idx_rm_parts, dim=0) if und_idx_rm_parts
                      else torch.empty(0, device=device, dtype=torch.long))
        mot_idx_rm = (torch.cat(mot_idx_rm_parts, dim=0) if mot_idx_rm_parts
                      else torch.empty(0, device=device, dtype=torch.long))
        (
            _bo_rm_und_idx,
            _bo_rm_full_idx,
            und_offsets_rm,
            full_offsets_rm,
            sample_offsets_rm,
            (max_und_len_rm, max_full_len_rm, max_sample_len_rm),
        ) = build_offsets(und_lens, mot_lens, rm_total_lens, device)
        offsets_rm = {
            "full_idx": mot_idx_rm,
            "und_offsets": und_offsets_rm,
            "full_offsets": full_offsets_rm,
            "sample_offsets": sample_offsets_rm,
            "max_und_len": max_und_len_rm,
            "max_full_len": max_full_len_rm,
            "max_sample_len": max_sample_len_rm,
        }

        # --- (3) rotary cos/sin for ALL packed rows -------------------------------------------
        if any_3d_positions:
            # mixed 1-D / 3-D per sample -> normalize every sample's positions to [3, N_s] then cat.
            pos3_parts = []
            for p in positions_list:
                pos3_parts.append(p if p.dim() == 2 else p.view(1, -1).expand(3, -1))
            if any(p.dtype.is_floating_point for p in pos3_parts):
                pos3_parts = [p.float() for p in pos3_parts]
            positions = torch.cat(pos3_parts, dim=1)                          # [3, N_total]
        else:
            positions = torch.cat(positions_list, dim=0)                      # [N_total]
        cos, sin = self._build_cos_sin(positions, device=device, dtype=dtype)  # each [N_total, hd]
        # rotary gathered to the rg (reasoner+gen) buffer order, for the 2-way plain-layer calls.
        cos_rg = cos.index_select(0, rg_idx)
        sin_rg = sin.index_select(0, rg_idx)
        cos_rm = cos.index_select(0, rm_idx)
        sin_rm = sin.index_select(0, rm_idx)

        # --- (4) run the joint layers (SPARSE-DEPTH: 3-way at motion layers, 2-way elsewhere) ---
        # MOTION layers run the full [und | gen | mot] 3-way joint attention. PLAIN layers gather
        # ONLY the reasoner+gen rows (rg), run the frozen 2-way reasoner+gen path on that buffer,
        # and scatter the result back -- motion rows are left untouched, and gen/reasoner never
        # attend motion at a plain layer (there are no motion rows in the gathered buffer). The
        # frozen reasoner+generator thus run ALL layers; only the motion expert is sparse.
        packed = packed_sequence
        def _apply_deepstack(layer_idx: int, cur: torch.Tensor) -> torch.Tensor:
            for s in range(B):
                ds = reasoner_deepstacks[s]
                vm = reasoner_visual_masks[s]
                if ds is None or vm is None or layer_idx >= len(ds):
                    continue
                ustart, uend = und_offsets_abs[s]
                rows = cur[ustart:uend].clone()
                mask = vm.to(device=rows.device, dtype=torch.bool)
                rows[mask] = rows[mask] + ds[layer_idx].to(device=rows.device, dtype=rows.dtype)
                cur = cur.index_copy(0, torch.arange(ustart, uend, device=device), rows)
            return cur

        def _apply_bridge(layer_idx: int, cur: torch.Tensor) -> torch.Tensor:
            key = str(layer_idx)
            if key not in self.bridges:
                return cur
            bridge = self.bridges[key]
            for s in range(B):
                gs, ge = gen_offsets[s]
                ms, me = mot_offsets[s]
                if ge <= gs or me <= ms:
                    continue
                mode_s = modes[s]
                if mode_s not in ("video2motion", "motimg2video"):
                    continue
                g = cur[gs:ge]
                m = cur[ms:me]
                meta = BridgeMeta(
                    mode=mode_s,
                    gen_frame=gen_frame_list[s].to(device),
                    gen_clean=gen_clean_list[s].to(device),
                    motion_frame=motion_frame_list[s].to(device),
                )
                g2, m2 = bridge(g, m, meta)
                cur = cur.index_copy(0, torch.arange(gs, ge, device=device), g2)
                cur = cur.index_copy(0, torch.arange(ms, me, device=device), m2)
            return cur

        for i, layer in enumerate(self.layers):
            if i in self.motion_layers and self.coupling == "joint":
                packed = layer(packed, und_idx, gen_idx, mot_idx, cos, sin, offsets_3way)
                packed = _apply_deepstack(i, packed)
            elif i in self.motion_layers and self.coupling == "bridge_local":
                rg = packed.index_select(0, rg_idx)
                rg_out = layer(
                    rg, und_idx_local, gen_idx_local, empty_mot,
                    cos_rg, sin_rg, offsets_2way,
                )
                packed = packed.index_copy(0, rg_idx, rg_out.to(packed.dtype))
                if mot_idx.numel() > 0:
                    rm = packed.index_select(0, rm_idx)
                    rm_out = layer(
                        rm, und_idx_rm, empty_gen, mot_idx_rm,
                        cos_rm, sin_rm, offsets_rm,
                    )
                    packed = packed.index_copy(
                        0, mot_idx, rm_out.index_select(0, mot_idx_rm).to(packed.dtype)
                    )
                packed = _apply_bridge(i, packed)
                packed = _apply_deepstack(i, packed)
            else:
                rg = packed.index_select(0, rg_idx)              # gather und u gen rows
                rg_out = layer(
                    rg, und_idx_local, gen_idx_local, empty_mot,
                    cos_rg, sin_rg, offsets_2way,
                )
                packed = packed.index_copy(0, rg_idx, rg_out.to(packed.dtype))  # scatter back
                packed = _apply_deepstack(i, packed)

        # --- (5) decode each supervised modality ----------------------------------------------
        out: dict = {"resolved": resolved_list}

        # motion decode: norm the motion rows, drop shape token, decode -> pred[B,Tm,283].
        motion_pred = None
        if mot_idx.numel() > 0:
            mrows = self.norm_moe_motion(packed[mot_idx])                      # [sum n_mot, hidden]
            motion_pred = x_t.new_zeros(B, Tm, self.motion_dim) if x_t is not None else None
            off = 0
            for s in range(B):
                start, end = mot_offsets[s]
                nm = end - start
                if nm == 0:
                    continue
                sample_rows = mrows[off:off + nm]                             # [1+n_mot_s, hidden]
                off += nm
                frame_rows = sample_rows[1:]                                  # drop shape token
                if frame_rows.shape[0] == 0:
                    continue
                decoded = self.heads.decode(frame_rows.unsqueeze(0))[0]       # [n_mot_s, 283]
                if valid_frame_idx[s] is not None and motion_pred is not None:
                    motion_pred[s].index_copy_(0, valid_frame_idx[s], decoded.to(motion_pred.dtype))
        out["motion_pred"] = motion_pred

        # generator decode: per-sample, only the SUPERVISED gen modalities (video / camera). The
        # gen rows are normed with the generator final norm before the decode heads.
        video_pred: list[Optional[torch.Tensor]] = [None] * B
        camera_pred: list[Optional[torch.Tensor]] = [None] * B
        if gen_idx.numel() > 0:
            for s in range(B):
                seg = gen_segments[s]
                resolved = resolved_list[s]
                if seg is None or resolved is None:
                    continue
                gstart, gend = gen_offsets[s]
                if gend == gstart:
                    continue
                gen_hidden = self._gen_norm(packed[gstart:gend])             # [N_gen_s, hidden]
                for name, (rs, re) in seg.offsets.items():
                    mr = resolved.modalities.get(name)
                    if mr is None or not mr.supervised:
                        continue
                    part = seg.parts[name]
                    part_hidden = gen_hidden[rs:re]                          # [n_tok, hidden]
                    if name == "video":
                        nfi = part.noisy_frame_idx.to(device)
                        # supervised tokens = the patches of the noised latent frames.
                        spatial = part.grid[1] * part.grid[2]
                        # rows within this modality of the noised frames (T-major patch order).
                        noisy_rows = (nfi.view(-1, 1) * spatial
                                      + torch.arange(spatial, device=device).view(1, -1)).reshape(-1)
                        gen_hidden_noisy = part_hidden.index_select(0, noisy_rows)
                        T_lat, gh, gw = part.grid
                        # original latent (T_lat, h, w): recover h,w from grid * patch.
                        p = self.gen.patch
                        orig = (T_lat, gh * p, gw * p)
                        video_pred[s] = self.gen.decode_video(
                            gen_hidden_noisy, grid=part.grid,
                            noisy_frame_idx=nfi, original_latent_shape=orig,
                        )
                    elif name == "camera":
                        noisy = ~part.condition_mask.to(device).bool()        # [T-1]
                        gen_hidden_noisy = part_hidden[noisy]
                        camera_pred[s] = self.gen.decode_camera(gen_hidden_noisy)
        out["video_pred"] = video_pred
        out["camera_pred"] = camera_pred

        if return_dict:
            return out
        # backward-compatible: text->motion returns the bare motion tensor.
        return motion_pred

    # -------------------------------------------------------------------------------------
    # Sampling closure used by flow.py (text->motion)
    # -------------------------------------------------------------------------------------
    def predict_closure(
        self,
        input_ids_list: list[torch.Tensor],
        neutral_joints: torch.Tensor,
        motion_pad_mask: torch.Tensor,
        noisy_frame_mask: torch.Tensor,
        *,
        modes: Optional[list[str]] = None,
        video_latents: Optional[list[Optional[torch.Tensor]]] = None,
        camera_action: Optional[list[Optional[torch.Tensor]]] = None,
        reasoner_inputs: Optional[list[Optional[dict]]] = None,
    ):
        """Return a callable `fn(x, t_b) -> motion_pred[B,Tm,283]` for the samplers in `flow.py`.

        `flow.py` calls this with a fixed conditioning context (the captured `input_ids_list` and
        any clean generator modalities): once with the conditional ids, once with null '' ids for
        CFG. `x` is the current noised motion, `t_b` the per-sample flow time / sigma. The motion
        prediction is returned as a bare tensor (the closure targets the MOTION modality), so it
        plugs straight into the existing motion samplers. When `modes`/`video_latents`/
        `camera_action` are given, the clean conditioning modalities (image/video) are packed
        identically in both the cond and null passes (only the text content differs).
        """
        def fn(x: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
            out = self.forward(
                input_ids_list=input_ids_list,
                x_t=x,
                t_or_sigma=t_b,
                neutral_joints=neutral_joints,
                motion_pad_mask=motion_pad_mask,
                noisy_frame_mask=noisy_frame_mask,
                modes=modes,
                video_latents=video_latents,
                camera_action=camera_action,
                reasoner_inputs=reasoner_inputs,
                return_dict=True,
            )
            return out["motion_pred"]

        return fn

    @torch.no_grad()
    def sample(
        self,
        caption: str,
        neutral_joints: torch.Tensor,   # [1, 30, 3]
        T: int,
        steps: int = 50,
        guidance: float = 2.0,
        tokenizer=None,
        device=None,
        objective: str | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample one motion clip for a single caption (used by train.py's in-train viz).

        Tokenizes `caption` (and the empty/null prompt for CFG), builds the per-sample masks,
        and runs the matching rectified-flow sampler from `flow.py` against `predict_closure`.
        Returns normalized motion `[1, T, 283]`. (Text->motion only; the per-task multi-modality
        sampler lives in `sample.py`.)
        """
        import flow

        if tokenizer is None:
            tokenizer = self.cosmos.tokenize
        if device is None:
            device = neutral_joints.device
        obj = objective or self.objective
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        motion_pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        noisy_frame_mask = torch.ones(1, T, dtype=torch.bool, device=device)

        predict_cond = self.predict_closure(
            input_ids_list=[tokenizer(caption)],
            neutral_joints=neutral_joints,
            motion_pad_mask=motion_pad_mask,
            noisy_frame_mask=noisy_frame_mask,
        )
        predict_null = self.predict_closure(
            input_ids_list=[tokenizer("")],
            neutral_joints=neutral_joints,
            motion_pad_mask=motion_pad_mask,
            noisy_frame_mask=noisy_frame_mask,
        )

        sampler = flow.motion_sampler(
            obj,
            schedule=self.motion_schedule,
            native_solver=self.motion_native_solver,
        )
        native_kwargs = {}
        if self.motion_schedule == "native":
            native_kwargs = {
                "native_shift": self.motion_shift,
                "native_num_train_timesteps": self.motion_num_train_timesteps,
            }
        x = sampler(
            predict_cond, T=T, motion_dim=self.motion_dim, steps=steps,
            guidance=guidance, predict_null=predict_null,
            batch=1, device=device, dtype=torch.float32,
            generator=generator,
            **native_kwargs,
        )
        return x


# -----------------------------------------------------------------------------------------
# Param-count sanity (informational; printed by train.py).
#
# Trainable params (motion ALWAYS) ~= n_layers * (_moe_motion q/k/v/o + qk-norms + 2 layernorms
#                     + MLP) + heads (motion2llm / shape2llm / llm2motion / time_embedder / embeds)
#                     + norm_moe_motion. PLUS, per the toggles: gen/reasoner LoRA adapters
#                     (gen_lora/reasoner_lora) or the full _moe_gen + generator I/O heads (gen_full).
# In --smoke, train.py calls `assert_frozen_grads_zero()` after a backward to prove the routing
# (trainable grad > 0; everything outside the active trainable set grad == 0).
# -----------------------------------------------------------------------------------------
