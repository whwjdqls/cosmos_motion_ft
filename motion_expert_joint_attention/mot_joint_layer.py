"""The 3-pathway MoT decoder layer for the motion-expert joint-attention repo.

This wraps ONE real Cosmos ``MoTDecoderLayer`` and adds a THIRD weight pathway
(``_moe_motion``) alongside its two frozen pathways:

    * REASONER (understanding)  : plain names (``q_proj``, ``mlp``, ``input_layernorm``, ...)
    * GENERATOR (vision/video)  : ``*_moe_gen`` names
    * MOTION    (NEW, trainable): ``*_moe_motion`` names (this file owns these)

All three token groups live in ONE packed sequence and share ONE joint attention
at every layer (Mixture-of-Transformers). A token's *role* selects only WHICH
weights it flows through; the attention *mask* is keyed on attn-mode
(causal vs full) and is realized by ``mot_joint_attention.joint_attention``
(the two-call und/full primitive, mirroring ``two_way_attention``).

Role -> attends-to (enforced by the joint attention, not by this file):
    reasoner : reasoner only, causally (lower-triangular self-attn).
    generator: reasoner u generator u motion (dense).
    motion   : reasoner u generator u motion (dense).
=> generator<->motion bidirectional; both read the reasoner conditioning;
   reasoner never sees gen or motion (stays a pure causal context encoder).

Ownership / FSDP (load-bearing, read before refactoring)
--------------------------------------------------------
This module *owns* (registers as ``nn.Module`` children, hence trainable and
FSDP-wrapped) ONLY the fresh ``_moe_motion`` submodules. The frozen reasoner and
generator submodules belong to the ``base_layer`` (a real ``MoTDecoderLayer`` that
lives in ``joint_motion_model``); we keep *references* to them in a plain Python
container (``self._frozen`` -- a SimpleNamespace, NOT an ``nn.Module`` /
``nn.ModuleDict``). Storing them in an ``nn.`` container here would re-register
them as children of THIS module, double-counting their parameters and causing a
second FSDP wrap of weights another module already owns. Using
``object.__setattr__`` for the namespace bypasses ``nn.Module.__setattr__`` so the
reference is never promoted into ``_modules``. The frozen refs already have
``requires_grad=False`` (set by the model's freeze pass); we never re-register
them, so they are never wrapped or stepped here.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

# joint_attention is the two-call (causal + full) 3-way mask primitive that lives
# in the sibling file. Imported lazily-tolerant so this module can be imported in
# isolation (e.g. for static inspection) before that file exists.
try:  # pragma: no cover - import wiring
    from mot_joint_attention import joint_attention
except Exception:  # pragma: no cover
    joint_attention = None  # resolved at call time; see forward().


def _like_norm(ref_norm: nn.Module, dim: int) -> nn.Module:
    """Build a norm with the SAME class/eps as ``ref_norm`` (or Identity).

    Used to give the motion pathway qk-norms / layernorms that match the base
    reasoner's norm family (``Qwen3VLTextRMSNorm`` / ``Nemotron3DenseVLRMSNorm``
    / ``nn.Identity``) without importing the variant class directly.
    """
    if isinstance(ref_norm, nn.Identity):
        return nn.Identity()
    cls = type(ref_norm)
    eps = getattr(ref_norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(ref_norm, "eps", 1e-6)
    try:
        return cls(dim, eps=eps)
    except TypeError:
        # Some RMSNorm variants take (dim) positionally without an eps kwarg.
        return cls(dim)


def _run_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an MLP block, normalizing dense vs MoE return shapes.

    Mirrors ``unified_mot._run_mlp``: ``Qwen3VLMoeTextSparseMoeBlock`` returns
    ``(Tensor, LBLMetadata)``; dense MLPs return just the tensor. We discard the
    LBL metadata here (the motion pathway is dense in the default recipe; this
    keeps the layer correct even if a MoE base is used).
    """
    out = mlp(x)
    if isinstance(out, tuple):
        return out[0]
    return out


class _MotionMLP(nn.Module):
    """Fresh dense SwiGLU FFN for the motion pathway — deliberately SMALLER than the generator.

    ``hidden`` (the residual width, 4096) is FIXED: motion tokens share the packed residual
    stream with the reasoner/generator, and the attention q/k/v/o keep the base head geometry so
    the shared joint attention works. The only free width is ``intermediate`` (the FFN hidden),
    which we set well below the generator's so the motion expert is light. SiLU/SwiGLU matches the
    Qwen3 backbone family. Always randomly initialized; never copied from the generator.
    """

    def __init__(self, hidden: int, intermediate: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoTJointLayer(nn.Module):
    """One MoT decoder layer with reasoner + generator (frozen) + motion (trainable).

    Args:
        base_layer: a real ``MoTDecoderLayer`` (already materialized & frozen).
            We reference its reasoner and ``_moe_gen`` submodules; we do NOT take
            ownership of them (see module docstring).
        apply_rotary_fn: ``base_layer.self_attn._apply_rotary_pos_emb`` -- the
            variant-correct ``apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)``.
        num_heads, num_kv_heads, head_dim, scaling: from ``base_layer.self_attn`` — these are
            FIXED to the base geometry (the joint attention stacks all roles into one q/k/v, so
            cross-role Q·Kᵀ needs identical head count / head_dim across reasoner/gen/motion).
        hidden: model hidden size (also fixed — the shared residual width).
        motion_intermediate_size: the motion FFN hidden width. The ONLY size knob; set below the
            generator's so the motion expert is lighter. The motion pathway is ALWAYS freshly
            (randomly) initialized — never warm-started from the generator (a video prior is a
            poor init for 283-d motion).
        has_motion: when True (default) this layer owns a trainable ``_moe_motion`` pathway and the
            packed sequence runs the FULL 3-way joint attention (reasoner + generator + motion).
            When False this is a "sparse-depth" PLAIN layer: it constructs NO ``_moe_motion``
            submodules at all (ZERO motion params) and runs only the frozen 2-way reasoner+generator
            path. The model interleaves the two: the 3-way attention fires only every Nth backbone
            layer, while the frozen reasoner+generator still run all ``n_layers``. A has_motion=False
            layer MUST be called with ``mot_idx`` empty (no motion rows in its gathered buffer).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        apply_rotary_fn,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        hidden: int,
        motion_intermediate_size: int = 3072,
        has_motion: bool = True,
    ):
        super().__init__()
        attn = base_layer.self_attn

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.hidden = hidden
        self.has_motion = has_motion
        self._apply_rotary_pos_emb = apply_rotary_fn

        # ---- Frozen references (reasoner + generator). NOT registered as children. ----
        # Stored in a plain SimpleNamespace via object.__setattr__ so nn.Module's
        # __setattr__ does not sweep them into self._modules (which would double
        # count / double-FSDP-wrap weights base_layer already owns).
        frozen = SimpleNamespace(
            # reasoner (understanding) pathway
            q_proj=attn.q_proj,
            k_proj=attn.k_proj,
            v_proj=attn.v_proj,
            o_proj=attn.o_proj,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            input_layernorm=base_layer.input_layernorm,
            post_attention_layernorm=base_layer.post_attention_layernorm,
            mlp=base_layer.mlp,
            # generator pathway (_moe_gen)
            q_proj_moe_gen=attn.q_proj_moe_gen,
            k_proj_moe_gen=attn.k_proj_moe_gen,
            v_proj_moe_gen=attn.v_proj_moe_gen,
            o_proj_moe_gen=attn.o_proj_moe_gen,
            q_norm_moe_gen=attn.q_norm_moe_gen,
            k_norm_moe_gen=attn.k_norm_moe_gen,
            input_layernorm_moe_gen=base_layer.input_layernorm_moe_gen,
            post_attention_layernorm_moe_gen=base_layer.post_attention_layernorm_moe_gen,
            mlp_moe_gen=base_layer.mlp_moe_gen,
        )
        object.__setattr__(self, "_frozen", frozen)

        # ---- Trainable motion pathway (_moe_motion): identical shapes to _moe_gen. ----
        # SPARSE-DEPTH: when has_motion=False this layer owns NO motion submodules at all (zero
        # motion params); it is a pure frozen 2-way reasoner+generator layer. Guard the whole
        # construction (and _reset_motion_params) so no q/k/v/o_proj_moe_motion, motion norms, or
        # mlp_moe_motion are ever registered. forward() guards every *_moe_motion access to match.
        if has_motion:
            bias = attn.q_proj_moe_gen.bias is not None
            q_dim = num_heads * head_dim
            kv_dim = num_kv_heads * head_dim

            self.q_proj_moe_motion = nn.Linear(hidden, q_dim, bias=bias)
            self.k_proj_moe_motion = nn.Linear(hidden, kv_dim, bias=bias)
            self.v_proj_moe_motion = nn.Linear(hidden, kv_dim, bias=bias)
            self.o_proj_moe_motion = nn.Linear(q_dim, hidden, bias=bias)

            # qk-norm matches base.self_attn.q_norm_moe_gen (RMSNorm(head_dim) or Identity).
            self.q_norm_moe_motion = _like_norm(attn.q_norm_moe_gen, head_dim)
            self.k_norm_moe_motion = _like_norm(attn.k_norm_moe_gen, head_dim)

            # two layernorms (RMSNorm(hidden)) matching the gen layernorm family.
            self.input_layernorm_moe_motion = _like_norm(base_layer.input_layernorm_moe_gen, hidden)
            self.post_attention_layernorm_moe_motion = _like_norm(
                base_layer.post_attention_layernorm_moe_gen, hidden
            )

            # MLP: a FRESH, SMALLER dense SwiGLU FFN — NEVER copied from the generator.
            # intermediate_size < the generator's, so the motion expert is deliberately lighter than
            # the video generator (motion is 283-d, far simpler than video latents).
            self.mlp_moe_motion = _MotionMLP(hidden, motion_intermediate_size, bias=bias)

            # Match the base dtype/device of the gen pathway (bf16 on the model device).
            ref_param = attn.q_proj_moe_gen.weight
            self.to(device=ref_param.device, dtype=ref_param.dtype)

            # ---- Fresh, generator-INDEPENDENT init (motion is NEVER warm-started from gen). ----
            self._reset_motion_params()

    # -------------------------------------------------------------------------
    # init helpers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _reset_motion_params(self) -> None:
        """Fresh, generator-INDEPENDENT init for every motion param.

        The motion expert is NEVER warm-started from the generator: the generator's weights are a
        *video* prior and a poor init for 283-d motion. All motion Linears get a small normal init
        (std 0.02); biases 0; RMSNorm gains stay at their default (1.0). qk-norms / layernorms are
        already fresh (built by ``_like_norm``), so nothing is inherited from ``_moe_gen``.
        """
        for mod in (
            self.q_proj_moe_motion, self.k_proj_moe_motion, self.v_proj_moe_motion,
            self.o_proj_moe_motion, self.mlp_moe_motion,
        ):
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.normal_(sub.weight, mean=0.0, std=0.02)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        packed: torch.Tensor,          # [N, hidden]
        und_idx: torch.Tensor,         # long [N_und] rows of the reasoner tokens
        gen_idx: torch.Tensor,         # long [N_gen] rows of the generator tokens
        mot_idx: torch.Tensor,         # long [N_mot] rows of the motion tokens
        cos: torch.Tensor,             # [N, head_dim] per-row rotary cos (packed order)
        sin: torch.Tensor,             # [N, head_dim] per-row rotary sin (packed order)
        attn_offsets: dict,            # bundle of varlen offsets (see joint_attention)
    ) -> torch.Tensor:
        """3-pathway MoT layer forward over one packed sequence.

        ``attn_offsets`` carries the precomputed varlen metadata for
        ``joint_attention``, including ``full_idx = sort(cat(gen_idx, mot_idx))``
        (the gen u motion rows in packed order). This file does not recompute it;
        the model builds it once per step.

        Steps (mirror PackedAttentionMoT.forward + MoTDecoderLayer.forward, 3-way):
          (A) pre-attn norm per role
          (B) q/k/v per role -> [n, heads, hd] -> qk-norm -> rotary -> scatter
          (C) ONE joint attention (causal und + full gen u motion)
          (D) o_proj per role + residual
          (E) post-attn norm + MLP per role + residual
        """
        if joint_attention is None:  # resolve at call time if import was deferred.
            from mot_joint_attention import joint_attention as _ja
        else:
            _ja = joint_attention

        f = self._frozen
        N = packed.shape[0]
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # SPARSE-DEPTH guard: a PLAIN (has_motion=False) layer owns no _moe_motion submodules, so
        # it must NEVER touch a motion row. The model guarantees mot_idx is empty for such a layer
        # (it is only ever called on a gathered reasoner+gen buffer with no motion rows); we assert
        # that contract here so a misuse fails loudly instead of an AttributeError deep in a branch.
        has_motion = self.has_motion
        if not has_motion:
            assert mot_idx.numel() == 0, (
                "MoTJointLayer(has_motion=False) called with motion rows present; a plain layer "
                "has no _moe_motion params and must be run on a reasoner+gen-only buffer."
            )

        full_idx = attn_offsets["full_idx"]  # sort(cat(gen_idx, mot_idx)) in packed order

        # ----- (A) pre-attention norm, per role -----
        h = packed.clone()
        if und_idx.numel():
            h.index_copy_(0, und_idx, f.input_layernorm(packed.index_select(0, und_idx)))
        if gen_idx.numel():
            h.index_copy_(0, gen_idx, f.input_layernorm_moe_gen(packed.index_select(0, gen_idx)))
        if has_motion and mot_idx.numel():
            h.index_copy_(
                0, mot_idx, self.input_layernorm_moe_motion(packed.index_select(0, mot_idx))
            )

        # ----- (B) q/k/v per role, qk-norm, rotary, scatter into packed q/k/v -----
        q_all = packed.new_zeros(N, H, D)
        k_all = packed.new_zeros(N, H_kv, D)
        v_all = packed.new_zeros(N, H_kv, D)

        def _qkv(idx, q_proj, k_proj, v_proj, q_norm, k_norm):
            if idx.numel() == 0:
                return
            hr = h.index_select(0, idx)  # [n, hidden]
            q = q_proj(hr).view(-1, H, D)
            k = k_proj(hr).view(-1, H_kv, D)
            v = v_proj(hr).view(-1, H_kv, D)
            q = q_norm(q)
            k = k_norm(k)
            c = cos.index_select(0, idx)  # [n, head_dim]
            s = sin.index_select(0, idx)  # [n, head_dim]
            q, k = self._apply_rotary_pos_emb(q, k, c, s, unsqueeze_dim=1)
            q_all.index_copy_(0, idx, q.to(q_all.dtype))
            k_all.index_copy_(0, idx, k.to(k_all.dtype))
            v_all.index_copy_(0, idx, v.to(v_all.dtype))

        _qkv(und_idx, f.q_proj, f.k_proj, f.v_proj, f.q_norm, f.k_norm)
        _qkv(
            gen_idx,
            f.q_proj_moe_gen, f.k_proj_moe_gen, f.v_proj_moe_gen,
            f.q_norm_moe_gen, f.k_norm_moe_gen,
        )
        if has_motion:
            _qkv(
                mot_idx,
                self.q_proj_moe_motion, self.k_proj_moe_motion, self.v_proj_moe_motion,
                self.q_norm_moe_motion, self.k_norm_moe_motion,
            )

        # ----- (C) ONE joint attention: causal(und) + full(gen u motion over all) -----
        # joint_attention's softmax scaling is the framework `attention()` default
        # (1/sqrt(head_dim)), which equals self.scaling for the base Cosmos layer; there is no
        # `scale` kwarg to thread. We pass exactly the varlen bundle joint_attention expects.
        attn_out = _ja(
            q_all, k_all, v_all,
            und_idx=und_idx,
            full_idx=full_idx,
            sample_offsets=attn_offsets["sample_offsets"],
            max_und_len=attn_offsets["max_und_len"],
            max_full_len=attn_offsets["max_full_len"],
            max_sample_len=attn_offsets["max_sample_len"],
            und_offsets=attn_offsets["und_offsets"],
            full_offsets=attn_offsets["full_offsets"],
        )  # [N, heads*head_dim]

        # ----- (D) o_proj per role + residual -----
        o = packed.new_zeros(N, self.hidden)
        if und_idx.numel():
            o.index_copy_(0, und_idx, f.o_proj(attn_out.index_select(0, und_idx)))
        if gen_idx.numel():
            o.index_copy_(0, gen_idx, f.o_proj_moe_gen(attn_out.index_select(0, gen_idx)))
        if has_motion and mot_idx.numel():
            o.index_copy_(0, mot_idx, self.o_proj_moe_motion(attn_out.index_select(0, mot_idx)))
        packed = packed + o

        # ----- (E) post-attention norm + MLP per role + residual -----
        def _mlp_block(idx, post_norm, mlp):
            if idx.numel() == 0:
                return
            x = packed.index_select(0, idx)
            ln = post_norm(x)
            packed.index_copy_(0, idx, x + _run_mlp(mlp, ln))

        _mlp_block(und_idx, f.post_attention_layernorm, f.mlp)
        _mlp_block(gen_idx, f.post_attention_layernorm_moe_gen, f.mlp_moe_gen)
        if has_motion:
            _mlp_block(
                mot_idx, self.post_attention_layernorm_moe_motion, self.mlp_moe_motion
            )

        return packed


# ====================================================================== #
# Self-test: CPU, tiny dims, pure-pytorch SDPA reference for `attention`. #
#                                                                        #
# Proves the SPARSE-DEPTH interleaving masking WITHOUT a GPU. We build a  #
# motion layer (has_motion=True) and a plain layer (has_motion=False),    #
# run them with the EXACT 3-way / 2-way bundles `joint_motion_model.py`   #
# builds, and assert:                                                     #
#   (a) at a PLAIN layer, gen+reasoner output rows are INVARIANT to       #
#       perturbing the motion VALUE rows (motion excluded), and the       #
#       motion rows are returned UNCHANGED;                               #
#   (b) at a MOTION layer, gen output rows DO change when motion values   #
#       change (gen reads motion).                                        #
# ====================================================================== #
def _selftest() -> None:  # pragma: no cover - exercised via __main__
    import math
    import sys
    import types

    import mot_joint_attention as MJA

    # ---- pure-pytorch varlen SDPA reference for `attention` (CPU) -------------------------
    def _ref_attention(query, key, value, is_causal=False, causal_type=None, scale=None,
                       cumulative_seqlen_Q=None, cumulative_seqlen_KV=None,
                       max_seqlen_Q=None, max_seqlen_KV=None, **kwargs):
        q = query[0]; k = key[0]; v = value[0]
        s_q, hh, dd = q.shape
        s_kv, h_kv, _ = k.shape
        if scale is None:
            scale = 1.0 / math.sqrt(dd)
        if h_kv != hh:
            rep = hh // h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = q.new_zeros(s_q, hh, dd)
        cq = cumulative_seqlen_Q.tolist(); ckv = cumulative_seqlen_KV.tolist()
        for i in range(len(cq) - 1):
            qs, qe = cq[i], cq[i + 1]; ks, ke = ckv[i], ckv[i + 1]
            if qe == qs:
                continue
            qb = q[qs:qe].transpose(0, 1); kb = k[ks:ke].transpose(0, 1); vb = v[ks:ke].transpose(0, 1)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * scale
            if is_causal:
                lq, lk = qe - qs, ke - ks
                qi = torch.arange(lq).view(lq, 1); ki = torch.arange(lk).view(1, lk)
                scores = scores.masked_fill(qi < ki, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.matmul(attn, vb).transpose(0, 1)
        return out.unsqueeze(0)

    # Patch BOTH the symbol the layer resolves (mot_joint_attention.joint_attention closes over
    # `attention`) and our local module reference, so no framework kernel is needed.
    MJA.attention = _ref_attention
    globals()["joint_attention"] = MJA.joint_attention

    torch.manual_seed(0)
    hidden, H, H_kv, D = 8, 2, 2, 4   # tiny dims; head geometry shared across roles

    # ---- minimal fake "base layer" with the attribute surface __init__ reads --------------
    def lin(i, o):
        m = nn.Linear(i, o, bias=False)
        nn.init.normal_(m.weight, std=0.3)
        return m

    q_dim, kv_dim = H * D, H_kv * D
    attn_ns = SimpleNamespace(
        q_proj=lin(hidden, q_dim), k_proj=lin(hidden, kv_dim),
        v_proj=lin(hidden, kv_dim), o_proj=lin(q_dim, hidden),
        q_norm=nn.Identity(), k_norm=nn.Identity(),
        q_proj_moe_gen=lin(hidden, q_dim), k_proj_moe_gen=lin(hidden, kv_dim),
        v_proj_moe_gen=lin(hidden, kv_dim), o_proj_moe_gen=lin(q_dim, hidden),
        q_norm_moe_gen=nn.Identity(), k_norm_moe_gen=nn.Identity(),
    )
    base = SimpleNamespace(
        self_attn=attn_ns,
        input_layernorm=nn.Identity(), post_attention_layernorm=nn.Identity(), mlp=lin(hidden, hidden),
        input_layernorm_moe_gen=nn.Identity(), post_attention_layernorm_moe_gen=nn.Identity(),
        mlp_moe_gen=lin(hidden, hidden),
    )

    def rotary(q, k, c, s, unsqueeze_dim=1):  # identity rotary (cos=1, sin=0) for the test
        return q, k

    motion_layer = MoTJointLayer(base, rotary, H, H_kv, D, D ** -0.5, hidden,
                                 motion_intermediate_size=16, has_motion=True)
    plain_layer = MoTJointLayer(base, rotary, H, H_kv, D, D ** -0.5, hidden,
                                motion_intermediate_size=16, has_motion=False)

    # plain layer must own ZERO motion params.
    n_mot_plain = sum(1 for n, _ in plain_layer.named_parameters() if "_moe_motion" in n)
    assert n_mot_plain == 0, f"plain layer has {n_mot_plain} motion params (must be 0)"
    assert not hasattr(plain_layer, "mlp_moe_motion"), "plain layer must not build mlp_moe_motion"
    n_mot_motion = sum(1 for n, _ in motion_layer.named_parameters() if "_moe_motion" in n)
    assert n_mot_motion > 0, "motion layer must own _moe_motion params"

    # ---- 1-sample packed sequence: 2 und + 1 gen + 2 mot = 5 rows -------------------------
    nu, ng, nm = 2, 1, 2
    dev = "cpu"
    packed = torch.randn(nu + ng + nm, hidden)
    und_idx = torch.arange(0, nu)
    gen_idx = torch.arange(nu, nu + ng)
    mot_idx = torch.arange(nu + ng, nu + ng + nm)
    cos = torch.ones(packed.shape[0], D); sin = torch.zeros(packed.shape[0], D)

    # 3-way bundle (motion layer) -----------------------------------------------------------
    full_idx = torch.sort(torch.cat([gen_idx, mot_idx])).values
    (_u, _f, und_off, full_off, samp_off, (mu, mf, ms)) = MJA.build_offsets(
        [nu], [ng + nm], [nu + ng + nm], dev)
    off3 = {"full_idx": full_idx, "und_offsets": und_off, "full_offsets": full_off,
            "sample_offsets": samp_off, "max_und_len": mu, "max_full_len": mf, "max_sample_len": ms}

    # 2-way bundle (plain layer): rg = und u gen rows; full block = gen only ----------------
    rg_idx = torch.cat([und_idx, gen_idx])
    und_idx_local = torch.arange(0, nu)
    gen_idx_local = torch.arange(nu, nu + ng)
    empty_mot = torch.empty(0, dtype=torch.long)
    (_u2, _f2, und_off2, full_off2, samp_off2, (mu2, mf2, ms2)) = MJA.build_offsets(
        [nu], [ng], [nu + ng], dev)
    off2 = {"full_idx": gen_idx_local, "und_offsets": und_off2, "full_offsets": full_off2,
            "sample_offsets": samp_off2, "max_und_len": mu2, "max_full_len": mf2, "max_sample_len": ms2}

    def run_plain(p):
        rg = p.index_select(0, rg_idx)
        rg_out = plain_layer(rg, und_idx_local, gen_idx_local, empty_mot,
                             cos.index_select(0, rg_idx), sin.index_select(0, rg_idx), off2)
        return p.index_copy(0, rg_idx, rg_out.to(p.dtype))

    # (a) PLAIN layer: perturb the motion VALUE rows; gen+reasoner outputs must be invariant,
    #     and motion rows must be returned UNCHANGED (passed through, never read/written).
    out_p = run_plain(packed)
    packed_pert = packed.clone()
    packed_pert[mot_idx] += 100.0
    out_p2 = run_plain(packed_pert)
    assert torch.allclose(out_p[und_idx], out_p2[und_idx], atol=1e-5), \
        "PLAIN: reasoner output changed when motion perturbed (motion leaked in)"
    assert torch.allclose(out_p[gen_idx], out_p2[gen_idx], atol=1e-5), \
        "PLAIN: gen output changed when motion perturbed (gen read motion at a plain layer!)"
    assert torch.allclose(out_p[mot_idx], packed[mot_idx], atol=1e-6), \
        "PLAIN: motion rows not returned UNCHANGED"
    assert torch.allclose(out_p2[mot_idx], packed_pert[mot_idx], atol=1e-6), \
        "PLAIN: perturbed motion rows not passed through unchanged"

    # (b) MOTION layer: perturb the motion VALUE rows; gen output rows MUST change (gen reads mot).
    out_m = motion_layer(packed, und_idx, gen_idx, mot_idx, cos, sin, off3)
    out_m2 = motion_layer(packed_pert, und_idx, gen_idx, mot_idx, cos, sin, off3)
    assert not torch.allclose(out_m[gen_idx], out_m2[gen_idx], atol=1e-4), \
        "MOTION: gen output did NOT change when motion perturbed (gen failed to read motion)"
    # reasoner stays a pure causal encoder even at a motion layer (never reads gen/mot).
    assert torch.allclose(out_m[und_idx], out_m2[und_idx], atol=1e-5), \
        "MOTION: reasoner output changed when motion perturbed (reasoner leaked!)"

    # (c) text->motion sanity: gen empty -> plain layer = reasoner-only, motion layer = [und|mot].
    nu_t, nm_t = 2, 2
    packed_t = torch.randn(nu_t + nm_t, hidden)
    und_t = torch.arange(0, nu_t); mot_t = torch.arange(nu_t, nu_t + nm_t)
    empty_gen = torch.empty(0, dtype=torch.long)
    cos_t = torch.ones(packed_t.shape[0], D); sin_t = torch.zeros(packed_t.shape[0], D)
    full_idx_t = mot_t.clone()
    (_ut, _ft, und_offt, full_offt, samp_offt, (mut, mft, mst)) = MJA.build_offsets(
        [nu_t], [nm_t], [nu_t + nm_t], dev)
    off3_t = {"full_idx": full_idx_t, "und_offsets": und_offt, "full_offsets": full_offt,
              "sample_offsets": samp_offt, "max_und_len": mut, "max_full_len": mft, "max_sample_len": mst}
    # plain layer in text->motion: rg = und rows only, gen empty.
    rg_idx_t = und_t.clone()
    (_u2t, _f2t, und_off2t, full_off2t, samp_off2t, (mu2t, mf2t, ms2t)) = MJA.build_offsets(
        [nu_t], [0], [nu_t], dev)
    off2_t = {"full_idx": empty_gen, "und_offsets": und_off2t, "full_offsets": full_off2t,
              "sample_offsets": samp_off2t, "max_und_len": mu2t, "max_full_len": mf2t, "max_sample_len": ms2t}
    rg_t = packed_t.index_select(0, rg_idx_t)
    rg_out_t = plain_layer(rg_t, torch.arange(0, nu_t), empty_gen, empty_mot,
                           cos_t.index_select(0, rg_idx_t), sin_t.index_select(0, rg_idx_t), off2_t)
    out_t_plain = packed_t.index_copy(0, rg_idx_t, rg_out_t.to(packed_t.dtype))
    assert torch.allclose(out_t_plain[mot_t], packed_t[mot_t], atol=1e-6), \
        "text->motion PLAIN: motion rows not passed through unchanged"
    out_t_mot = motion_layer(packed_t, und_t, empty_gen, mot_t, cos_t, sin_t, off3_t)
    assert out_t_mot.shape == packed_t.shape, "text->motion MOTION layer shape mismatch"

    print("mot_joint_layer sparse-depth self-test PASSED: "
          "plain layer = frozen 2-way (motion excluded + unchanged); motion layer = 3-way "
          "(gen reads motion, reasoner stays causal); text->motion path runs.")


if __name__ == "__main__":
    _selftest()
