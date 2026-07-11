"""Trainable motion I/O heads for the 3-way joint-attention motion expert.

Mirrors Cosmos's `action2llm`/`llm2action` + `action_modality_embed` + `time_embedder`
pattern (`cosmos3_vfm_network.py:_encode_action`/`_decode_action`), but motion-specific
and vision-decoupled (no `vision_gen` assert). These heads are the I/O of the new
`_moe_motion` weight pathway: they map the 283-D normalized uniego motion rep <-> the
4096-D Cosmos hidden, inject the rectified-flow timestep as a Cosmos-style ADDITIVE token
bias on noisy frames, and provide the per-actor shape token from `neutral_joints`.

Token layout (per sample, motion segment):
    [ shape_tok ]  [ frame_1 .. frame_T ]
  - frame tokens : motion2llm(x_t : 283->4096) + motion_modality_embed + time_bias(noisy)
  - shape_tok    : shape2llm(neutral_joints flat : 90->4096) + shape_type_embed

Frame rotary (3D-mRoPE) is applied LATER inside the MoT layer (the same rotary path the
generator uses), NOT here. These heads only produce the additive content of each token.

Cosmos-faithful default: the flow-time signal is an ADDITIVE token bias on the noisy
motion frames (computed in fp32), exactly like the generator's `condition_mask` path --
there is NO AdaLN/modulation. An AdaLN variant is provided behind `adaln=True` purely as a
documented ablation (the PoC used AdaLN-zero); the default path stays additive.

All public shapes are batch-first [B, T, *]; the model flattens to packed rows downstream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from uniego_layout import FEAT_DIM, N_JOINTS  # 283, 30

HIDDEN = 4096


# ---------------------------------------------------------------------------
# Timestep embedder (Cosmos-style; reused from the framework if available).
# ---------------------------------------------------------------------------
def _load_cosmos_timestep_embedder():
    """Return the framework `TimestepEmbedder` class, or None if unavailable.

    Prefer the real Cosmos `TimestepEmbedder` (sinusoidal freq -> SiLU MLP, fp32) so the
    motion timestep bias is bit-identical to the generator's. Import is best-effort: in the
    self-contained / no-framework setting we fall back to the local copy below.
    """
    try:  # framework path (cosmos env, cwd=cosmos-framework)
        from cosmos_framework.model.vfm.mot.modeling_utils import TimestepEmbedder

        return TimestepEmbedder
    except Exception:
        try:  # via the repo's loader shim, if present
            from cosmos_loader import TimestepEmbedder  # type: ignore

            return TimestepEmbedder
        except Exception:
            return None


class LocalTimestepEmbedder(nn.Module):
    """Self-contained copy of Cosmos's `TimestepEmbedder` (sinusoidal -> SiLU MLP, fp32).

    Kept byte-for-byte compatible with `modeling_utils.TimestepEmbedder` so the framework
    weights can be copied in either direction and the additive bias matches the generator.
    Output: [N, hidden].
    """

    def __init__(self, hidden_size: int = HIDDEN, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        nn.init.trunc_normal_(self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.mlp[0].bias)
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.trunc_normal_(self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ---------------------------------------------------------------------------
# Motion heads.
# ---------------------------------------------------------------------------
class MotionHeads(nn.Module):
    """Trainable motion I/O for the `_moe_motion` pathway.

    Submodules
    ----------
    motion2llm : Linear(283 -> 4096)
        Per-frame normalized motion vector -> hidden. Plain `nn.Linear` (single embodiment,
        so no `DomainAwareLinear`).
    shape2llm : MLP(90 -> 4096)
        `neutral_joints (30,3)` flattened -> the shape-token hidden.
    llm2motion : Linear(4096 -> 283)  [zero-init weight + bias]
        Final motion-pathway hidden at frame positions -> predicted rectified-flow target.
        Zero-init so the warm-started generator-like pathway starts as a near-identity.
    motion_modality_embed : Parameter(4096)
        Added to every motion (frame + shape) token; distinguishes the modality.
    shape_type_embed : Parameter(4096)
        Added to the shape token only.
    time_embedder : Cosmos `TimestepEmbedder` (or local fallback)
        Rectified-flow timestep -> additive bias on noisy frames (computed in fp32).

    Conventions
    -----------
    * `timestep_scale` (default 1e-3) matches the generator: the embedder sees
      `t * timestep_scale` (Cosmos scales raw timesteps before the embedder).
    * Default path is additive timestep bias (Cosmos-faithful). `adaln=True` swaps in an
      AdaLN-zero modulation generator purely as a documented ablation.
    """

    def __init__(
        self,
        hidden: int = HIDDEN,
        motion_dim: int = FEAT_DIM,
        n_joints: int = N_JOINTS,
        timestep_scale: float = 0.001,
        adaln: bool = False,
        init_std: float | None = None,
    ):
        super().__init__()
        self.hidden = hidden
        self.motion_dim = motion_dim
        self.n_joints = n_joints
        self.shape_dim = n_joints * 3  # 90
        self.timestep_scale = float(timestep_scale)
        self.adaln = bool(adaln)

        # --- I/O projections ---
        self.motion2llm = nn.Linear(motion_dim, hidden)
        self.shape2llm = nn.Sequential(
            nn.Linear(self.shape_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.llm2motion = nn.Linear(hidden, motion_dim)

        # --- learned additive embeds (added to motion tokens) ---
        self.motion_modality_embed = nn.Parameter(torch.zeros(hidden))
        self.shape_type_embed = nn.Parameter(torch.zeros(hidden))

        # --- rectified-flow timestep embedder (Cosmos-style additive bias) ---
        TE = _load_cosmos_timestep_embedder()
        if TE is not None:
            self.time_embedder = TE(hidden)
            if hasattr(self.time_embedder, "_init_weights"):
                self.time_embedder._init_weights()
        else:
            self.time_embedder = LocalTimestepEmbedder(hidden)

        # --- optional AdaLN ablation (default OFF) ---
        # Produces (shift, scale, gate) x 2 (attn + ffn) per noisy frame from the time embed.
        # NOT used on the Cosmos-faithful default path; the model only consults `self.adaln`.
        if self.adaln:
            self.adaln_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))
            nn.init.zeros_(self.adaln_proj[1].weight)
            nn.init.zeros_(self.adaln_proj[1].bias)
        else:
            self.adaln_proj = None

        self._init_weights(init_std)

    def _init_weights(self, init_std: float | None):
        std = (1.0 / math.sqrt(self.hidden)) if init_std is None else float(init_std)
        # motion2llm: small truncated-normal, zero bias (matches PoC head init).
        nn.init.trunc_normal_(self.motion2llm.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.motion2llm.bias)
        # shape2llm MLP: default Linear/LayerNorm init is fine; just zero the biases.
        for m in self.shape2llm:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                nn.init.zeros_(m.bias)
        # llm2motion: ZERO-INIT (weight + bias) so the warm-started pathway starts neutral.
        nn.init.zeros_(self.llm2motion.weight)
        nn.init.zeros_(self.llm2motion.bias)
        # modality / shape-type embeds: small truncated-normal (zeros also valid; PoC uses tn).
        nn.init.trunc_normal_(self.motion_modality_embed, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.shape_type_embed, std=std, a=-3 * std, b=3 * std)

    # ------------------------------------------------------------------ encode
    def encode_motion(
        self,
        x_t: torch.Tensor,            # [B, T, 283]  noised motion at timestep t
        t_or_sigma: torch.Tensor,     # [B]          per-sample rectified-flow timestep
        noisy_frame_mask: torch.Tensor,  # [B, T] bool  True where the frame is noised (loss target)
    ) -> torch.Tensor:
        """Encode motion frames -> hidden tokens [B, T, 4096].

        h = motion2llm(x_t) + motion_modality_embed, then for every NOISY frame add the
        timestep bias `time_embedder(t * timestep_scale)` broadcast over that sample's noisy
        frames. The timestep embed is computed in fp32 (autocast disabled) then cast back to
        the hidden dtype, matching Cosmos. Frame rotary is applied later in the model.

        Clean (conditioning) frames receive NO timestep bias -- they keep the clean signal,
        exactly like the generator's `condition_mask` path.
        """
        B, T, _ = x_t.shape
        dtype = self.motion2llm.weight.dtype
        h = self.motion2llm(x_t.to(dtype))
        h = h + self.motion_modality_embed.view(1, 1, -1)

        # Per-sample timestep bias, computed in fp32, broadcast over that sample's frames.
        t = (t_or_sigma.reshape(B).float() * self.timestep_scale)
        with torch.autocast(device_type=h.device.type, enabled=False):
            t_emb = self.time_embedder(t.float())          # [B, 4096] fp32
        t_emb = t_emb.to(h.dtype)
        mask = noisy_frame_mask.to(h.dtype).unsqueeze(-1)  # [B, T, 1]
        h = h + mask * t_emb.unsqueeze(1)                  # add only on noisy frames
        return h

    # --------------------------------------------------------------- encode shape
    def encode_shape(self, neutral_joints: torch.Tensor) -> torch.Tensor:
        """Encode per-actor neutral skeleton -> the shape token [B, 1, 4096].

        shape2llm(flatten(neutral_joints)) + shape_type_embed + motion_modality_embed.
        The shape token is a motion-pathway token (it carries the modality embed too) and is
        dropped at decode.
        """
        B = neutral_joints.shape[0]
        dtype = self.shape2llm[0].weight.dtype
        flat = neutral_joints.reshape(B, self.shape_dim).to(dtype)
        s = self.shape2llm(flat)
        s = s + self.shape_type_embed.view(1, -1) + self.motion_modality_embed.view(1, -1)
        return s.unsqueeze(1)  # [B, 1, 4096]

    # ------------------------------------------------------------------ decode
    def decode(self, motion_hidden: torch.Tensor) -> torch.Tensor:
        """Decode final motion-pathway hidden at frame positions -> predicted target.

        motion_hidden : [B, T, 4096]  (T = number of motion FRAME tokens, shape token already
                        dropped by the caller). Returns [B, T, 283] in fp32.
        """
        dtype = self.llm2motion.weight.dtype
        out = self.llm2motion(motion_hidden.to(dtype))
        return out.to(torch.float32)

    # ------------------------------------------------------- AdaLN ablation helper
    def adaln_modulation(self, t_or_sigma: torch.Tensor):
        """ABLATION ONLY. Return 6 modulation tensors (shift/scale/gate x attn/ffn) [B, 4096].

        Only valid when constructed with `adaln=True`. The Cosmos-faithful default path does
        NOT call this; it is exposed so the MoT layer can optionally FiLM-modulate the motion
        pathway instead of adding the timestep bias.
        """
        if self.adaln_proj is None:
            raise RuntimeError("adaln_modulation requires MotionHeads(adaln=True)")
        B = t_or_sigma.reshape(-1).shape[0]
        t = t_or_sigma.reshape(B).float() * self.timestep_scale
        with torch.autocast(device_type=t_or_sigma.device.type, enabled=False):
            t_emb = self.time_embedder(t.float())
        t_emb = t_emb.to(self.adaln_proj[1].weight.dtype)
        mod = self.adaln_proj(t_emb)  # [B, 6*hidden]
        return mod.chunk(6, dim=-1)   # shift_a, scale_a, gate_a, shift_f, scale_f, gate_f

    # ----------------------------------------------------------- freeze helpers
    def param_names(self) -> list[str]:
        """All parameter names owned by these heads (for the freeze logic)."""
        return [n for n, _ in self.named_parameters()]

    def trainable_parameters(self):
        """Iterator over the params that must be TRAINABLE (all of them)."""
        return (p for _, p in self.named_parameters())

    @staticmethod
    def is_motion_head_name(name: str) -> bool:
        """True if `name` (a network-level param name) belongs to a motion head.

        Used by `joint_motion_model.py` freeze logic:
            requires_grad = ("_moe_motion" in name) or is_motion_head_name(name)
        Matches whether the heads are attached at the top level or under a `motion_heads.`
        attribute.
        """
        n = name.split("motion_heads.")[-1]
        return (
            n.startswith("motion2llm")
            or n.startswith("shape2llm")
            or n.startswith("llm2motion")
            or n.startswith("motion_modality_embed")
            or n.startswith("shape_type_embed")
            or n.startswith("time_embedder")
            or n.startswith("adaln_proj")
        )


__all__ = ["MotionHeads", "LocalTimestepEmbedder", "HIDDEN"]
