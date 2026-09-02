# SPDX-License-Identifier: OpenMDW-1.1
"""
Frozen Cosmos-3 Nano loader for the 3-way joint-attention motion expert.

This is the ONLY module in `motion_expert_joint_attention/` that touches the
Cosmos framework's network object directly. Everything downstream
(`mot_joint_layer.py`, `joint_motion_model.py`, ...) consumes:

  (a) the 36 `MoTDecoderLayer`s' frozen submodules (reasoner + `_moe_gen`
      pathways, grabbed via `layer_submodules`), and
  (b) `embed_tokens` / `rotary_emb` / the final norms (`norm`, `norm_moe_gen`),

so that the new trainable `_moe_motion` pathway can be cloned from the frozen
`_moe_gen` pathway and motion tokens get the SAME rotary / norm convention as
the generator.

It reuses the proven build/load helpers from the root experiment's
`train_motion_ft.py` (build a real `Cosmos3VFMNetwork` on meta, materialize on
CUDA, load the diffusers->native `_moe_gen` weights, freeze everything, eval).

NO trainable parameters are created here.

Network layout (Cosmos3-Nano, MoT):
    net (Cosmos3VFMNetwork)
      .language_model (Qwen3VLTextForCausalLM)
        .model (Qwen3VL[Moe]TextModel == "the TextModel")
          .embed_tokens   nn.Embedding(vocab, 4096)
          .layers         ModuleList[MoTDecoderLayer] (len == num_hidden_layers == 36)
          .norm           reasoner-pathway final RMSNorm
          .norm_moe_gen   generator-pathway final RMSNorm
          .rotary_emb     Qwen3VL[Moe]TextRotaryEmbedding (3D-mRoPE)
          .reasoner_forward(...)  AR reasoner helper (KV-cache)

Each `MoTDecoderLayer`:
    .self_attn (PackedAttentionMoT): q/k/v/o_proj (reasoner) + *_moe_gen (gen)
        + q_norm/k_norm (reasoner) + q_norm_moe_gen/k_norm_moe_gen (gen)
        + _apply_rotary_pos_emb, num_attention_heads, num_key_value_heads,
          head_dim, scaling
    .mlp / .mlp_moe_gen
    .input_layernorm / .input_layernorm_moe_gen
    .post_attention_layernorm / .post_attention_layernorm_moe_gen
"""

import os
import sys
from glob import glob

import torch

# The proven helpers (build/materialize/load/freeze/processor/special-tokens)
# live in the root experiment's trainer. Import by absolute path so this module
# is usable from anywhere (e.g. cwd == cosmos-framework for the relative
# QWEN_JSON used by build_network).
from runtime_paths import COSMOS_FRAMEWORK_ROOT, HF_HOME, REPO_ROOT  # noqa: E402

_ROOT = str(REPO_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train_motion_ft as _root_train  # noqa: E402
from train_motion_ft import (  # noqa: E402
    SPECIAL_TOKENS,
    build_network,
    build_text_processor,
    freeze_all,
    load_gen_weights,
    materialize,
)

__all__ = ["FrozenCosmos", "SPECIAL_TOKENS"]

_COSMOS_FRAMEWORK = str(COSMOS_FRAMEWORK_ROOT)
_ABS_QWEN_JSON = os.path.join(
    _COSMOS_FRAMEWORK,
    "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json",
)
if os.path.exists(_ABS_QWEN_JSON):
    _root_train.QWEN_JSON = _ABS_QWEN_JSON


_NANO_GLOB = os.path.join(
    os.environ.get("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub")),
    "models--nvidia--Cosmos3-Nano/snapshots/*",
)


def _local_nano_snapshot() -> str | None:
    snaps = sorted(glob(_NANO_GLOB))
    return snaps[0] if snaps else None


class FrozenCosmos:
    """Build + materialize + load + freeze a real Cosmos-3 Nano network and
    expose the handful of pieces the joint-attention motion expert needs.

    After construction:
        self.net          Cosmos3VFMNetwork (frozen, .eval())
        self.tm           net.language_model.model  (the TextModel)
        self.cfg          net.language_model.config
        self.hidden       hidden_size (4096)
        self.n_layers     len(self.tm.layers) (36)
        self.num_heads    config.num_attention_heads
        self.num_kv_heads config.num_key_value_heads
        self.head_dim     getattr(cfg,'head_dim', hidden//num_attention_heads)
        self.proc         the Cosmos text processor

    The reasoner, generator, and embed/rotary/norm are ALL frozen here; the
    trainable motion pathway is built downstream by cloning the `_moe_gen`
    submodules returned by `layer_submodules`.
    """

    def __init__(self, dtype=torch.bfloat16, device="cuda", verbose=True):
        self.dtype = dtype
        self.device = device
        self.verbose = verbose

        # 1. Build the real network on meta, then materialize on CUDA and load
        #    the diffusers->native generator (`_moe_gen`) weights.
        net, _base_config = build_network(tiny=False, dtype=dtype, action_gen=True)
        net = materialize(net, dtype=dtype)
        load_gen_weights(net, verbose=verbose)

        # 2. Freeze everything and switch to eval (no dropout / deterministic
        #    norms). The motion pathway + heads (trainable) live downstream.
        freeze_all(net)
        net.eval()
        self.net = net

        # 3. Reach the TextModel and its submodules.
        self.tm = net.language_model.model  # Qwen3VL[Moe]TextModel
        self.cfg = net.language_model.config

        # Sanity: the 36 MoT decoder layers.
        assert hasattr(self.tm, "layers"), "TextModel has no .layers"
        self.n_layers = len(self.tm.layers)

        # 4. Per-layer attention dims. Prefer the live PackedAttentionMoT (it
        #    already resolved head_dim robustly); fall back to the config.
        attn0 = self.tm.layers[0].self_attn  # PackedAttentionMoT
        self.num_heads = int(getattr(attn0, "num_attention_heads", None)
                             or self._cfg_get("num_attention_heads"))
        self.num_kv_heads = int(getattr(attn0, "num_key_value_heads", None)
                               or self._cfg_get("num_key_value_heads"))
        self.hidden = int(getattr(attn0, "hidden_size", None)
                         or self._cfg_get("hidden_size")
                         or getattr(net, "hidden_size", 4096))
        self.head_dim = int(getattr(attn0, "head_dim", None)
                           or self._cfg_get("head_dim")
                           or (self.hidden // self.num_heads))

        # 5. The text processor (tokenizer + chat template), same as inference.
        self.proc = build_text_processor()
        self.visual_tower_loaded = hasattr(self.net.language_model, "visual")

        if verbose:
            n_total = sum(p.numel() for p in net.parameters())
            print(
                f"[FrozenCosmos] layers={self.n_layers} hidden={self.hidden} "
                f"heads={self.num_heads} kv_heads={self.num_kv_heads} "
                f"head_dim={self.head_dim} params={n_total/1e9:.2f}B (all frozen)"
            )

    # -- config helper ---------------------------------------------------------
    def _cfg_get(self, name, default=None):
        """Robustly read a field from the (possibly nested) config wrapper.

        `net.language_model.config` is a `Qwen3VLMoTConfig` wrapper whose flat
        fields are exposed on `.text_config`. Try the wrapper, then its
        materialized text_config.
        """
        cfg = self.cfg
        if hasattr(cfg, name):
            return getattr(cfg, name)
        tcfg = getattr(cfg, "text_config", None)
        if tcfg is not None and hasattr(tcfg, name):
            return getattr(tcfg, name)
        return default

    # -- text tokenization -----------------------------------------------------
    def tokenize(self, text: str) -> torch.LongTensor:
        """text -> LongTensor[1, T] of reasoner input ids on self.device.

        Mirrors how the root trainer feeds raw strings through the frozen
        understanding tower. An empty string maps to a single [eos] token so
        the CFG-null branch always has at least one reasoner token to attend
        to causally.
        """
        text = "" if text is None else str(text)
        if len(text.strip()) == 0:
            ids = [SPECIAL_TOKENS["eos_token_id"]]
            return torch.tensor([ids], dtype=torch.long, device=self.device)

        enc = self.proc.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) == 0:
            ids = [SPECIAL_TOKENS["eos_token_id"]]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def tokenize_generation(self, text: str) -> torch.LongTensor:
        """Tokenize the native Cosmos generator reasoner prefix.

        ``sequence_packing._pack_text_tokens`` wraps raw tokenizer IDs as
        ``[BOS, raw text, EOS, start_of_generation]`` whenever generation
        tokens follow.  The historical joint path passed only raw IDs (and a
        lone EOS for empty text), which changes both the reasoner states and
        every downstream mRoPE offset.  Keep ``tokenize`` untouched for old
        motion checkpoints and expose the exact native generation form here.
        """
        text = "" if text is None else str(text)
        enc = self.proc.tokenizer(text, add_special_tokens=False)
        raw_ids = list(enc.get("input_ids", []))
        ids = []
        if "bos_token_id" in SPECIAL_TOKENS:
            ids.append(SPECIAL_TOKENS["bos_token_id"])
        ids.extend(raw_ids)
        ids.append(SPECIAL_TOKENS["eos_token_id"])
        ids.append(SPECIAL_TOKENS["start_of_generation"])
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def load_standalone_visual_tower(self, snapshot: str | None = None):
        """Attach the separately shipped Cosmos Nano Qwen3-VL vision tower.

        The local Nano release used by this repo materializes the MoT language
        model as ``Qwen3VLTextForCausalLM``. It has image token ids and a
        processor, but no ``language_model.visual`` module. NVIDIA ships the
        vision tower as ``vision_encoder/model.safetensors`` with unprefixed
        keys (``blocks.*``, ``patch_embed.*``, ``merger.*``). This helper loads
        that module lazily and attaches it as ``net.language_model.visual`` so
        the official ``prepare_multimodal_reasoner_inputs`` helper can be used
        without changing normal text-only or generator-latent paths.
        """
        lm = self.net.language_model
        if hasattr(lm, "visual"):
            self.visual_tower_loaded = True
            return lm.visual

        snapshot = snapshot or _local_nano_snapshot()
        if not snapshot:
            raise FileNotFoundError(f"no local Cosmos3-Nano snapshot found under {_NANO_GLOB}")
        vision_dir = os.path.join(snapshot, "vision_encoder")
        config_path = os.path.join(vision_dir, "config.json")
        weights_path = os.path.join(vision_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(
                "standalone visual tower files are missing: "
                f"config={config_path} exists={os.path.exists(config_path)}, "
                f"weights={weights_path} exists={os.path.exists(weights_path)}"
            )

        from safetensors.torch import load_file
        from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl import Qwen3VLVisionModel

        cfg = Qwen3VLVisionConfig.from_json_file(config_path)
        with torch.device("meta"):
            visual = Qwen3VLVisionModel(cfg)
        visual.to_empty(device=self.device)
        state = load_file(weights_path, device=str(torch.device(self.device)))
        missing, unexpected = visual.load_state_dict(state, strict=True, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"visual tower state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        visual.to(device=self.device, dtype=self.dtype)
        visual.eval()
        for p in visual.parameters():
            p.requires_grad_(False)

        lm.visual = visual
        self.visual_tower_loaded = True
        if self.verbose:
            n_params = sum(p.numel() for p in visual.parameters())
            print(
                f"[FrozenCosmos] loaded standalone visual tower from {vision_dir} "
                f"params={n_params/1e6:.1f}M"
            )
        return visual

    @torch.no_grad()
    def encode_reasoner_image_text(
        self,
        text: str,
        image_chw: torch.Tensor,
        *,
        image_size: int | None = None,
    ) -> dict:
        """Prepare one image+text prompt for the frozen Qwen3-VL reasoner path.

        ``image_chw`` is a uint8 or float tensor in ``[3,H,W]`` layout. The returned dict is
        intentionally tensor-only so ``JointMotionModel.forward`` can consume it without touching
        PIL/processor state. The visual tower and token embeddings are frozen, so this runs under
        ``no_grad``; any reasoner LoRA still applies later inside the decoder layers. When
        ``image_size`` is set, resize to that square before the processor. Cosmos Nano's released
        processor accepts 256x256 as its minimum image area; that produces 64 merged visual tokens
        instead of the 400 produced by a 640x640 Nymeria frame.
        """
        if not callable(getattr(self.proc, "apply_chat_template", None)):
            raise RuntimeError(
                "reasoner image conditioning requires a multimodal Cosmos/Qwen processor with "
                "apply_chat_template; the loaded processor is text-only."
            )
        if not hasattr(self.net.language_model, "visual"):
            self.load_standalone_visual_tower()

        from PIL import Image
        from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import (
            prepare_multimodal_reasoner_inputs,
        )

        img = image_chw.detach().cpu()
        if img.dtype != torch.uint8:
            img = img.clamp(0, 255).to(torch.uint8)
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"image_chw must be [3,H,W], got {tuple(img.shape)}")
        if image_size is not None:
            image_size = int(image_size)
            if image_size <= 0:
                raise ValueError(f"image_size must be positive or None, got {image_size}")
            if tuple(img.shape[-2:]) != (image_size, image_size):
                img = torch.nn.functional.interpolate(
                    img.float().unsqueeze(0),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0).round().clamp_(0, 255).to(torch.uint8)
        pil = Image.fromarray(img.permute(1, 2, 0).contiguous().numpy(), mode="RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": "" if text is None else str(text)},
            ],
        }]
        proc_in = self.proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = proc_in["input_ids"].to(self.device)
        attention_mask = proc_in.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if input_ids.dim() == 1:
            input_ids_b = input_ids.unsqueeze(0)
            attn_b = attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.dim() == 1 else attention_mask
        else:
            input_ids_b = input_ids
            attn_b = attention_mask
        pixel_values = proc_in["pixel_values"].to(self.device)
        image_grid_thw = proc_in["image_grid_thw"].to(self.device)
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            _mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            self.net.language_model,
            input_ids=input_ids_b,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attn_b,
        )
        return {
            "input_ids": input_ids_b.squeeze(0).detach(),
            "inputs_embeds": inputs_embeds.squeeze(0).to(self.dtype).detach(),
            "position_ids": position_ids[:, 0, :].detach(),
            "visual_pos_mask": visual_pos_masks.squeeze(0).detach(),
            "deepstack_visual_embeds": [x.to(self.dtype).detach() for x in deepstack_visual_embeds],
        }

    # -- reasoner embedding ----------------------------------------------------
    @torch.no_grad()
    def embed_text(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """input_ids[1, T] -> reasoner embeddings [1, T, 4096] (frozen
        embed_tokens, in self.dtype)."""
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        emb = self.tm.embed_tokens(input_ids)
        return emb.to(self.dtype)

    # -- rotary embeddings -----------------------------------------------------
    def rope(self, position_ids):
        """Thin wrapper over `self.tm.rotary_emb` returning per-token (cos, sin).

        Motion tokens MUST get the same rotary convention as the generator, so
        we route them through the network's own 3D-mRoPE module.

        Args:
            position_ids: either
                - an int seq_len (builds a plain 0..seq_len-1 1-D range), or
                - a 1-D LongTensor [N] (single rope axis, broadcast to 3 axes
                  by the rotary module), or
                - a 2-D LongTensor [3, N] (full unified-3d-mrope: T/H/W axes).
        Returns:
            (cos, sin), each [N, head_dim], on self.device in self.dtype.
        """
        if isinstance(position_ids, int):
            position_ids = torch.arange(
                position_ids, dtype=torch.long, device=self.device
            )
        position_ids = position_ids.to(self.device)

        # Qwen3VL[Moe]TextRotaryEmbedding.forward(x, position_ids) uses x only
        # for dtype/device. It expects position_ids of ndim==2 -> (B, N) which
        # it broadcasts to (3, B, N), or ndim==3 -> (3, B, N) directly. Match
        # the model-level convention in unified_mot._impl_forward: a 1-D [N] is
        # unsqueezed to [1, N]; a 2-D [3, N] is unsqueezed (mid) to [3, 1, N].
        _x = torch.tensor([], dtype=self.dtype, device=self.device)
        if position_ids.ndim == 1:
            pids = position_ids.unsqueeze(0)            # [1, N]
        elif position_ids.ndim == 2:
            pids = position_ids.unsqueeze(1)            # [3, 1, N]
        else:
            pids = position_ids
        cos, sin = self.tm.rotary_emb(_x, position_ids=pids)
        # rotary always collapses the T/H/W axis -> [1, N, head_dim].
        cos = cos.squeeze(0)  # [N, head_dim]
        sin = sin.squeeze(0)  # [N, head_dim]
        return cos.to(self.dtype), sin.to(self.dtype)

    def build_position_ids(self, n_und: int, n_motion: int, n_gen: int = 0):
        """Build per-token 3-axis (T==H==W) position ids for the no-video text->motion case.

        COHERENCE NOTE: `joint_motion_model._build_cos_sin` currently feeds `rope()` a plain 1-D
        `[N_total]` positions tensor it assembles per-sample inline (reasoner 0..T_text-1, motion
        continuing) — `rope()` accepts that 1-D form directly (it broadcasts to the 3 mRoPE axes),
        so the inline path is the working one. This helper produces the explicit `[3, N]` form for
        callers that want the full unified-3d-mrope layout (e.g. a future gen-present packing); it
        is kept as the documented contract, not dead code, and returns the SAME positions the 1-D
        path would, replicated across T/H/W.


        Layout per sample (packed):
            [ reasoner (n_und, causal) ][ generator (n_gen) ][ motion (n_motion) ]

        Reasoner tokens get a plain causal range 0..n_und-1. The motion (and
        any generator) tokens CONTINUE that range so they get temporal
        positions consistent with the rest of the sequence — i.e. motion frame
        f sits at absolute position n_und + n_gen + f. We replicate this scalar
        position across all 3 mRoPE axes (T == H == W) since text->motion has a
        1x1 spatial grid; the temporal axis is what carries frame order. This
        matches the unified_3d_mrope convention used by `build_pack_from_batch`
        in the root trainer for the no-video case.

        Returns:
            position_ids[3, N] LongTensor on self.device, N = n_und+n_gen+n_motion.
        """
        n = n_und + n_gen + n_motion
        rng = torch.arange(n, dtype=torch.long, device=self.device)  # [N]
        return rng.unsqueeze(0).expand(3, -1).contiguous()           # [3, N]

    # -- frozen per-layer submodules ------------------------------------------
    @staticmethod
    def layer_submodules(layer) -> dict:
        """Map a `MoTDecoderLayer`'s role -> frozen submodules + attn meta.

        Used by `mot_joint_layer.py` to (a) grab the frozen reasoner and
        generator submodules and (b) clone the generator submodules into a
        fresh trainable `_moe_motion` pathway (warm start as a generator-like
        denoiser).

        Returns dict:
          {
            "reasoner": {q_proj,k_proj,v_proj,o_proj, q_norm,k_norm,
                         input_layernorm, post_attention_layernorm, mlp},
            "gen":      {q_proj,k_proj,v_proj,o_proj, q_norm,k_norm,
                         input_layernorm, post_attention_layernorm, mlp}
                        (the *_moe_gen names),
            "apply_rotary_pos_emb": layer.self_attn._apply_rotary_pos_emb,
            "num_attention_heads": int,
            "num_key_value_heads": int,
            "head_dim": int,
            "scaling": float,
          }
        """
        attn = layer.self_attn  # PackedAttentionMoT

        reasoner = {
            "q_proj": attn.q_proj,
            "k_proj": attn.k_proj,
            "v_proj": attn.v_proj,
            "o_proj": attn.o_proj,
            "q_norm": attn.q_norm,
            "k_norm": attn.k_norm,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "mlp": layer.mlp,
        }
        gen = {
            "q_proj": attn.q_proj_moe_gen,
            "k_proj": attn.k_proj_moe_gen,
            "v_proj": attn.v_proj_moe_gen,
            "o_proj": attn.o_proj_moe_gen,
            "q_norm": attn.q_norm_moe_gen,
            "k_norm": attn.k_norm_moe_gen,
            "input_layernorm": layer.input_layernorm_moe_gen,
            "post_attention_layernorm": layer.post_attention_layernorm_moe_gen,
            "mlp": layer.mlp_moe_gen,
        }
        return {
            "reasoner": reasoner,
            "gen": gen,
            "apply_rotary_pos_emb": attn._apply_rotary_pos_emb,
            "num_attention_heads": int(attn.num_attention_heads),
            "num_key_value_heads": int(attn.num_key_value_heads),
            "head_dim": int(attn.head_dim),
            "scaling": float(attn.scaling),
        }


if __name__ == "__main__":
    # Minimal smoke test: build, tokenize, embed, rope, inspect a layer.
    # Run from cosmos-framework (relative QWEN_JSON):
    #   cd /home/jungbin_cho/cosmos-framework
    #   PYTHONPATH=.:/home/jungbin_cho/cosmos_motion_ft:\
    #     /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention \
    #     python -m cosmos_loader
    fc = FrozenCosmos(dtype=torch.bfloat16, device="cuda", verbose=True)

    ids = fc.tokenize("a person walks forward")
    print("tokenize ->", ids.shape, ids.dtype)
    emb = fc.embed_text(ids)
    print("embed_text ->", tuple(emb.shape), emb.dtype)

    T_text = ids.shape[1]
    T_motion = 8
    pids = fc.build_position_ids(T_text, T_motion)
    print("position_ids ->", tuple(pids.shape))
    cos, sin = fc.rope(pids)
    print("rope ->", tuple(cos.shape), tuple(sin.shape))

    sub = FrozenCosmos.layer_submodules(fc.tm.layers[0])
    print("layer_submodules keys:", list(sub.keys()))
    print("  reasoner:", list(sub["reasoner"].keys()))
    print("  gen     :", list(sub["gen"].keys()))
    print(
        "  heads/kv/head_dim/scaling:",
        sub["num_attention_heads"], sub["num_key_value_heads"],
        sub["head_dim"], sub["scaling"],
    )

    # All frozen.
    assert all(not p.requires_grad for p in fc.net.parameters()), "net not frozen"
    print("[smoke] OK — all params frozen, helpers return expected shapes.")
