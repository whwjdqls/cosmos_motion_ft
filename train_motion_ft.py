# SPDX-License-Identifier: OpenMDW-1.1
"""
REAL text -> motion finetune of the Cosmos3-Nano GENERATOR (reasoner frozen).

This extends the proven PoC (cosmos_motion_poc/motion_ft_poc.py): same real
Cosmos3VFMNetwork build, same diffusers->native _moe_gen weight load, same motion
I/O heads (motion2llm / llm2motion / motion_modality_embed), same two-way packed
forward, same rectified-flow loss v = noise - x0. The new pieces here are:

  1. A real Dataset over the exported (text, motion[T,369]) pairs
     (features.npy mmap + index.json), per the legacy root contract summarized in AGENTS_ALL.md. Falls back
     to a tiny synthetic dataset in the SAME format if the export is not ready.
  2. Real text tokenization via the Cosmos3-Nano processor (build_processor_lazy,
     repository="nvidia/Cosmos3-Nano"); token ids flow through the FROZEN
     understanding/causal (reasoner) pathway via net._encode_text.
  3. Batch_size > 1: multiple (text, motion) samples packed into ONE packed
     sequence (ragged T handled natively by the packer's action layout); the loss
     is masked to each sample's valid motion frames via the per-sample frame split.
  4. Default config = FULL GENERATOR finetune (all _moe_gen + time_embedder + the
     3 motion heads; reasoner frozen) with FSDP2 across N GPUs (--fsdp). --lora is
     the fallback (LoRA on the _moe_gen q/k/v/o projections + heads).
  5. Periodic checkpointing (trainable params + optimizer + step) to --out, rank-0
     only, and step/loss/lr/peak_mem logging to stdout + a logfile.

Run from cosmos-framework (so `import cosmos_framework` resolves):
    python train_motion_ft.py --data <dir> --out <run dir> [--fsdp] [--lora] ...
or distributed:
    torchrun --standalone --nproc_per_node=8 train_motion_ft.py --fsdp --data ... --out ...
"""

import argparse
import json
import math
import os
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn

# motion_decode.py (pure-torch kimodo FK/decode port) lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_decode as md  # noqa: E402

from cosmos_framework.data.vfm.sequence_packing import (
    SequencePlan,
    pack_input_sequence,
)
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.vfm.mot.attention import build_packed_sequence
from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)
from cosmos_framework.model.vfm.mot.unified_mot import (
    Qwen3VLMoTConfig,
    Qwen3VLTextForCausalLM,
)
from cosmos_framework.model.vfm.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
)
from cosmos_framework.utils.vfm.lora import (
    inject_lora_pre_fsdp,
    init_lora_weights_post_materialization,
)

# --------------------------------------------------------------------------------------
# constants (mirror the PoC)
# --------------------------------------------------------------------------------------
MOTION_DIM = 369
HIDDEN = 4096
_NANO_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/*"
)
NANO_SNAPSHOT = sorted(glob(_NANO_GLOB))[0] if glob(_NANO_GLOB) else None
QWEN_JSON = "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
SPECIAL_TOKENS = {
    "eos_token_id": 1,
    "start_of_generation": 2,
    "end_of_generation": 3,
}
LORA_TARGETS = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
TIMESTEP_SCALE = 0.001

# Fixed eval prompts for in-training visualization (held-out-style captions; mix of
# simple + BONES-SEED natural-caption style). Override with --viz_prompts_file (one
# "prompt|name" per line).
VIZ_PROMPTS = [
    ("a person walks forward", "walk"),
    ("a person waves their right hand", "wave"),
    ("a person sits down on a chair", "sit"),
    ("character picks up an object from the floor and then stands up straight", "pickup"),
    ("a person turns around and walks back the other way", "turn"),
]


def lr_factor(step, warmup, total, min_ratio, schedule):
    """Schedule multiplier in [min_ratio, 1]: linear warmup to 1 then cosine decay
    to min_ratio. Applied to each param-group's base_lr. schedule='constant' -> 1."""
    if schedule == "constant":
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    prog = float(step - warmup) / float(max(1, total - warmup))
    prog = min(max(prog, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))


def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


# --------------------------------------------------------------------------------------
# Dataset: mmap features.npy + index.json  ->  (text:str, motion:[T,369] float32)
# --------------------------------------------------------------------------------------
class TextMotionDataset(torch.utils.data.Dataset):
    """Reads the exported legacy root pair format, single-dir OR multi-shard.

    Single-dir (subset export):
      <data_dir>/features.npy : float32 memmap [total_frames, 369] (normalized)
      <data_dir>/index.json   : {offsets:int64[N+1], texts:[str]*N, lengths:[int]*N,
                                 filenames, sources, meta:{fps,dim,normalized,layout}}
      sample i = features[offsets[i]:offsets[i+1]]   (length lengths[i])

    Multi-shard (full export): <data_dir> contains NO features.npy/index.json of
    its own but one or more ``shard_*/`` subdirs, each itself a single-dir export.
    Each shard's features.npy is mmap'd independently; a global sample index maps
    sample -> (shard, local_offset_a, local_offset_b). Behaviour, return type, and
    crop logic are IDENTICAL to the single-dir path.
    """

    def __init__(self, data_dir: str, max_frames: int = 200):
        self.data_dir = data_dir
        self.max_frames = max_frames

        single = os.path.exists(os.path.join(data_dir, "index.json"))
        if single:
            shard_dirs = [data_dir]
        else:
            shard_dirs = sorted(
                d for d in glob(os.path.join(data_dir, "shard_*"))
                if os.path.exists(os.path.join(d, "index.json"))
            )
            if not shard_dirs:
                raise FileNotFoundError(
                    f"{data_dir} has no index.json and no shard_*/ with index.json"
                )

        # Per-shard: texts list, offsets array, feature path (lazy-mmap'd).
        self._shard_feat_paths: list[str] = []
        self._shard_offsets: list[np.ndarray] = []
        self._shard_feats: list = []          # lazy mmap, one per shard
        # Global flat index -> (shard_idx, local_sample_idx).
        self._shard_of: list[int] = []
        self._local_of: list[int] = []
        self.texts: list[str] = []
        lengths: list[int] = []

        for si, sd in enumerate(shard_dirs):
            with open(os.path.join(sd, "index.json")) as f:
                idx = json.load(f)
            dim = idx.get("meta", {}).get("dim", MOTION_DIM)
            assert dim == MOTION_DIM, f"{sd} index meta dim {dim} != {MOTION_DIM}"
            offs = np.asarray(idx["offsets"], dtype=np.int64)
            n_s = len(idx["texts"])
            self._shard_feat_paths.append(os.path.join(sd, "features.npy"))
            self._shard_offsets.append(offs)
            self._shard_feats.append(None)
            self.texts.extend(idx["texts"])
            lengths.extend(idx["lengths"])
            self._shard_of.extend([si] * n_s)
            self._local_of.extend(range(n_s))

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.n = len(self.texts)
        self.num_shards = len(shard_dirs)

    def _features(self, si: int):
        # lazy per-worker mmap (np.memmap is not fork-safe to pre-open and share)
        if self._shard_feats[si] is None:
            self._shard_feats[si] = np.load(self._shard_feat_paths[si], mmap_mode="r")
        return self._shard_feats[si]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        si = self._shard_of[i]
        li = self._local_of[i]
        offs = self._shard_offsets[si]
        feat = self._features(si)
        a, b = int(offs[li]), int(offs[li + 1])
        m = np.array(feat[a:b], dtype=np.float32)  # copy the mmap slice (writable)
        if m.shape[0] > self.max_frames:
            # random crop to max_frames (keeps a contiguous window)
            start = np.random.randint(0, m.shape[0] - self.max_frames + 1)
            m = m[start:start + self.max_frames]
        text = self.texts[i]
        return text, torch.from_numpy(m)  # (str, float32 [T,369])


def make_synthetic_dataset(out_dir: str, n: int = 64, seed: int = 0):
    """Write a tiny synthetic dataset in the exact legacy root export format."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    lengths = rng.integers(40, 201, size=n).astype(np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    feats = rng.standard_normal((total, MOTION_DIM)).astype(np.float32)
    np.save(os.path.join(out_dir, "features.npy"), feats)
    verbs = ["walks", "runs", "jumps", "waves", "sits down", "turns left",
             "kicks", "dances", "crouches", "raises both arms"]
    texts = [f"a person {verbs[i % len(verbs)]} (synthetic sample {i})"
             for i in range(n)]
    index = {
        "offsets": offsets.tolist(),
        "texts": texts,
        "lengths": lengths.tolist(),
        "filenames": [f"synthetic_{i}.npz" for i in range(n)],
        "sources": ["synthetic"] * n,
        "meta": {
            "fps": 20, "dim": MOTION_DIM, "normalized": True,
            "layout": "smooth_root3|heading2|jpos90|rot6d180|vel90|footc4",
        },
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    return out_dir


def collate(batch):
    """batch: list of (text:str, motion:[T,369])  ->  (texts:list[str], motions:list[Tensor])."""
    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]
    return texts, motions


# --------------------------------------------------------------------------------------
# Text tokenizer: the REAL Cosmos3-Nano processor (raw text -> token ids)
# --------------------------------------------------------------------------------------
def build_text_processor():
    """Load the Cosmos3-Nano processor the same way inference does.

    inference (inference.py:1056-1064) sources the processor straight from the
    downloaded checkpoint dir; build_processor_lazy(repository="nvidia/Cosmos3-Nano",
    revision="main") resolves to that same local snapshot. If the snapshot is
    already in the HF cache we point build_processor at it directly to avoid any
    network access.
    """
    from cosmos_framework.data.vfm.processors import build_processor

    if NANO_SNAPSHOT is not None and os.path.isdir(NANO_SNAPSHOT):
        return build_processor(NANO_SNAPSHOT)
    from cosmos_framework.data.vfm.processors import build_processor_lazy
    return build_processor_lazy(repository="nvidia/Cosmos3-Nano", revision="main")


# --------------------------------------------------------------------------------------
# build the real network (verbatim from the PoC)
# --------------------------------------------------------------------------------------
def build_network(tiny: bool, dtype=torch.bfloat16, action_gen: bool = False):
    with torch.device("meta"):
        base_config = Qwen3VLMoTConfig.from_json_file(json_file=QWEN_JSON)
        base_config.freeze_und = False
        base_config.qk_norm_for_text = True
        base_config.qk_norm_for_diffusion = True
        base_config.tie_word_embeddings = True
        base_config.use_moe = True
        if tiny:
            base_config.text_config_overrides = {"num_hidden_layers": 2}

        language_model = Qwen3VLTextForCausalLM(config=base_config)

        net_config = Cosmos3VFMNetworkConfig(
            vlm_config=language_model.config,
            latent_patch_size=2,
            latent_downsample_factor=8,
            latent_channel_size=48,
            position_embedding_type="unified_3d_mrope",
            joint_attn_implementation="two_way",
            vision_gen=True,
            action_gen=action_gen,
            # Action-support fields (mirror NANO_MODEL_CONFIG's max_action_dim=64,
            # num_embodiment_domains=32). When action_gen=False these are inert:
            # the network only reads them inside `if config.action_gen:` blocks, so
            # the root experiment (action_gen=False) is byte-for-byte unchanged.
            action_dim=64,
            num_embodiment_domains=32,
            sound_gen=False,
            timestep_scale=TIMESTEP_SCALE,
        )
        net_config._attn_implementation_internal = "eager"
        net = Cosmos3VFMNetwork(language_model=language_model, config=net_config)

        net.motion_dim = MOTION_DIM
        net.motion2llm = nn.Linear(MOTION_DIM, net.hidden_size)
        net.llm2motion = nn.Linear(net.hidden_size, MOTION_DIM)
        net.motion_modality_embed = nn.Parameter(torch.zeros(net.hidden_size))

    return net, base_config


def materialize(net, dtype=torch.bfloat16):
    net = net.to(dtype=dtype)
    net.to_empty(device="cuda")
    net.init_weights(buffer_device="cuda")
    init_motion_heads(net, dtype)
    return net


def init_motion_heads(net, dtype):
    std = 1.0 / math.sqrt(MOTION_DIM)
    nn.init.trunc_normal_(net.motion2llm.weight, std=std, a=-3 * std, b=3 * std)
    nn.init.zeros_(net.motion2llm.bias)
    std = 1.0 / math.sqrt(net.hidden_size)
    nn.init.trunc_normal_(net.llm2motion.weight, std=std, a=-3 * std, b=3 * std)
    nn.init.zeros_(net.llm2motion.bias)
    nn.init.trunc_normal_(net.motion_modality_embed, std=std, a=-3 * std, b=3 * std)
    net.motion2llm.to(dtype=dtype)
    net.llm2motion.to(dtype=dtype)
    net.motion_modality_embed.data = net.motion_modality_embed.data.to(dtype)


# --------------------------------------------------------------------------------------
# diffusers -> native key map + weight load (verbatim from the PoC)
# --------------------------------------------------------------------------------------
def _diffusers_to_net_key(name: str):
    import re

    for pat in (r"^vae\.", r"^vision_encoder\.", r"\bvisual\b", r"^text_encoder\."):
        if re.search(pat, name):
            return None
    n = name
    n = re.sub(r"\.self_attn\.add_q_proj\.", ".self_attn.q_proj_moe_gen.", n)
    n = re.sub(r"\.self_attn\.add_k_proj\.", ".self_attn.k_proj_moe_gen.", n)
    n = re.sub(r"\.self_attn\.add_v_proj\.", ".self_attn.v_proj_moe_gen.", n)
    n = re.sub(r"\.self_attn\.to_add_out\.", ".self_attn.o_proj_moe_gen.", n)
    n = re.sub(r"\.self_attn\.norm_added_q\.", ".self_attn.q_norm_moe_gen.", n)
    n = re.sub(r"\.self_attn\.norm_added_k\.", ".self_attn.k_norm_moe_gen.", n)
    # REASONER (understanding-pathway) attention. The snapshot stores it in diffusers
    # naming (to_q/to_k/to_v/to_out + norm_q/norm_k, projecting the und sequence per
    # diffusers_cosmos3/transformer.py); the net names are q_proj/... (unified_mot.py:464).
    # These rules were MISSING originally, so all 36 layers' reasoner attention (216
    # tensors) silently ran at RANDOM init in every diffusers-loaded run ("skipped=226").
    # Order matters: the add_* / norm_added_* gen rules above must run first.
    n = re.sub(r"\.self_attn\.to_q\.", ".self_attn.q_proj.", n)
    n = re.sub(r"\.self_attn\.to_k\.", ".self_attn.k_proj.", n)
    n = re.sub(r"\.self_attn\.to_v\.", ".self_attn.v_proj.", n)
    n = re.sub(r"\.self_attn\.to_out\.", ".self_attn.o_proj.", n)
    n = re.sub(r"\.self_attn\.norm_q\.", ".self_attn.q_norm.", n)
    n = re.sub(r"\.self_attn\.norm_k\.", ".self_attn.k_norm.", n)
    # Pretrained camera ACTION I/O heads (DomainAwareLinear: .fc + .bias submodules).
    # Also missing originally -> action2llm/llm2action were fresh-init, not the base's
    # zero-shot camera heads. action_modality_embed matches by name already.
    n = re.sub(r"^action_proj_in\.", "action2llm.", n)
    n = re.sub(r"^action_proj_out\.", "llm2action.", n)
    n = re.sub(r"^proj_in\.", "vae2llm.", n)
    n = re.sub(r"^proj_out\.", "llm2vae.", n)
    n = re.sub(r"^time_embedder\.linear_1\.", "time_embedder.mlp.0.", n)
    n = re.sub(r"^time_embedder\.linear_2\.", "time_embedder.mlp.2.", n)
    n = re.sub(r"^model\.(layers\.|norm)", r"language_model.model.\1", n)
    n = re.sub(r"^(layers\.|embed_tokens\.|norm)", r"language_model.model.\1", n)
    return n


def load_gen_weights(net, verbose=True):
    from safetensors.torch import load_file

    index_path = os.path.join(NANO_SNAPSHOT, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    net_sd = dict(net.named_parameters())
    net_sd.update(dict(net.named_buffers()))

    remapped = {}
    for shard in sorted(set(weight_map.values())):
        if "transformer/" not in shard:
            continue
        path = os.path.join(NANO_SNAPSHOT, shard)
        if not os.path.exists(path):
            continue
        sd = load_file(path)
        for k, v in sd.items():
            tk = _diffusers_to_net_key(k)
            if tk is None:
                continue
            remapped[tk] = v

    loaded, skipped = 0, 0
    with torch.no_grad():
        for tk, v in remapped.items():
            if tk in net_sd and net_sd[tk].shape == v.shape:
                net_sd[tk].copy_(v.to(net_sd[tk].dtype).to(net_sd[tk].device))
                loaded += 1
            else:
                skipped += 1

    n_moe_gen = sum(1 for n in net_sd if "_moe_gen" in n)
    n_moe_gen_loaded = sum(1 for tk in remapped if "_moe_gen" in tk and tk in net_sd)
    if verbose:
        print(
            f"[load] remapped={len(remapped)} loaded={loaded} skipped(shape/miss)={skipped} "
            f"| net _moe_gen params={n_moe_gen} loaded _moe_gen={n_moe_gen_loaded}"
        )
    return loaded, len(remapped), n_moe_gen - n_moe_gen_loaded


# --------------------------------------------------------------------------------------
# pack a REAL batch of (text_ids, motion[T,369]) into one packed sequence
# --------------------------------------------------------------------------------------
def build_pack_from_batch(text_ids_list, motions, device):
    """Pack B samples (ragged T) using the packer's ACTION layout (action_dim=369)
    purely as a layout generator -- correct text-causal / motion-full split,
    unified_3d_mrope position ids, two_way SplitInfo. Ragged T is native: each
    sample contributes exactly its own T motion frames.

    Returns (PackedSequence on cuda, sample_timesteps[B] in [0,1], x0[N_motion,369]).
    """
    B = len(motions)
    x0_motion = [m.to(torch.float32) for m in motions]  # list of [T_i, 369]

    plans = [
        SequencePlan(
            has_text=True,
            has_action=True,
            condition_frame_indexes_action=[],  # all frames noised + supervised
        )
        for _ in range(B)
    ]
    gen_data = GenerationDataClean(
        batch_size=B,
        is_image_batch=False,
        x0_tokens_action=x0_motion,
    )
    # one timestep per sample (teacher forcing): all of a sample's frames share sigma
    input_timesteps = torch.rand(B)
    input_timesteps_packed = input_timesteps / TIMESTEP_SCALE  # network rescales by *scale

    ps = pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=text_ids_list,
        gen_data_clean=gen_data,
        input_timesteps=input_timesteps_packed,
        special_tokens=SPECIAL_TOKENS,
        latent_patch_size=1,
        include_end_of_generation_token=False,
        position_embedding_type="unified_3d_mrope",
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        action_dim=MOTION_DIM,
    )
    ps.to_cuda()
    return ps, input_timesteps


def forward_loss(net, ps, sample_timesteps, dtype=torch.bfloat16,
                 loss_mode="kimodo", skeleton=None, stats=None, weights=None):
    """Encode text (frozen und) + motion (gen heads), real MoT forward, decode
    velocity.

    loss_mode:
      "mse"    -> plain rectified-flow MSE(pred_v, noise - x0) (PoC behavior).
      "kimodo" -> reconstruct x0_hat = x_t - t*pred_v (since x_t = x0 + t*v) and
                  apply kimodo's bones_seed loss: per-369-block weighted smooth-L1
                  + FK consistency (weights from `weights`). Motion frames are laid
                  out flat/contiguous and all valid, so the flat-tensor reductions
                  with pad_mask=None reproduce kimodo's exact per-block/FK averages.
    Returns (loss_scalar, components_dict)."""
    device = "cuda"
    action = ps.action
    assert action is not None and action.tokens is not None

    packed_sequence, _ = net._encode_text(ps)  # [N_total, hidden] (frozen reasoner)

    x0 = torch.cat([t.to(device) for t in action.tokens], dim=0).to(torch.float32)
    noise = torch.randn_like(x0)
    # per-sample t -> per-token t (one t per sample, repeated over that sample's frames)
    t_per_sample = sample_timesteps.to(device).to(torch.float32)  # [B]
    T_each = [s[0] for s in action.token_shapes]
    t_tokens = torch.cat(
        [t_per_sample[i].repeat(T_each[i]) for i in range(len(T_each))], dim=0
    )
    tcol = t_tokens.view(-1, 1)
    x_t = (1.0 - tcol) * x0 + tcol * noise
    v_target = noise - x0

    h_motion = net.motion2llm(x_t.to(dtype))
    h_motion = h_motion + net.motion_modality_embed.view(1, -1)
    with torch.autocast("cuda", enabled=True, dtype=torch.float32):
        # network computes time_embedder(action.timesteps * timestep_scale); the packer
        # stored timesteps as t/scale, so that product == t. Pass t directly here.
        ts_emb = net.time_embedder(t_tokens)
    h_motion = h_motion + ts_emb.to(dtype)

    packed_sequence = packed_sequence.clone()
    packed_sequence[action.sequence_indexes] = h_motion.to(packed_sequence.dtype)

    input_pack, attention_meta, natten_md = build_packed_sequence(
        "two_way",
        packed_sequence=packed_sequence,
        attn_modes=ps.attn_modes,
        split_lens=ps.split_lens,
        sample_lens=ps.sample_lens,
        packed_und_token_indexes=ps.text_indexes,
        packed_gen_token_indexes=action.sequence_indexes,
        num_heads=net.num_heads,
        head_dim=net.head_dim,
        num_layers=net.num_hidden_layers,
        is_image_batch=ps.is_image_batch,
    )

    packed_outputs, _lbl = net.language_model(
        input_pack,
        attention_mask=attention_meta,
        position_ids=ps.position_ids,
        natten_metadata_list=natten_md,
        memory=None,
    )

    last_hidden = get_context_parallel_last_hidden_state(
        packed_outputs=packed_outputs, parallel_dims=None
    )

    pred_v = net.llm2motion(last_hidden[action.mse_loss_indexes].to(dtype)).to(torch.float32)

    if loss_mode == "mse":
        loss = nn.functional.mse_loss(pred_v, v_target)
        return loss, {"loss": loss.detach()}

    # kimodo bones_seed loss on reconstructed clean motion x0_hat (normalized 369-d).
    # x_t = (1-t)x0 + t*noise = x0 + t*v  ->  x0 = x_t - t*v.
    x0_hat = x_t - tcol * pred_v  # [N_motion, 369], float32, normalized
    ld = md.kimodo_weighted_loss(
        x0_hat, x0, skeleton, stats, pad_mask=None, weights=weights,
    )
    return ld["loss"], ld


# --------------------------------------------------------------------------------------
# param selection
# --------------------------------------------------------------------------------------
def freeze_all(net):
    for p in net.parameters():
        p.requires_grad_(False)


def is_motion_head(name):
    return (
        name.startswith("motion2llm")
        or name.startswith("llm2motion")
        or name == "motion_modality_embed"
    )


def setup_full_generator(net):
    """Default: train all _moe_gen + time_embedder + 3 motion heads; reasoner frozen."""
    freeze_all(net)
    for n, p in net.named_parameters():
        if "_moe_gen" in n or n.startswith("time_embedder") or is_motion_head(n):
            p.requires_grad_(True)


def setup_lora(net):
    """LoRA injected separately; just also enable the 3 motion heads."""
    for n, p in net.named_parameters():
        if is_motion_head(n):
            p.requires_grad_(True)


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return train, total


def clip_grads(params, max_norm):
    """Grad clip that tolerates FSDP. clip_grad_norm_'s foreach path can't mix
    DTensor (sharded decoder-layer grads) and plain Tensor (motion-head grads) in
    one call ('mixed torch.Tensor and DTensor' in _foreach_norm), so we clip the
    two groups separately. Each group's clip_grad_norm_ uses a consistent tensor
    type; the per-group norms are computed independently (a reasonable, slightly
    conservative approximation of a single global-norm clip)."""
    from torch.distributed.tensor import DTensor

    dtensor_params, plain_params = [], []
    for p in params:
        if p.grad is None:
            continue
        (dtensor_params if isinstance(p.grad, DTensor) else plain_params).append(p)
    if dtensor_params:
        torch.nn.utils.clip_grad_norm_(dtensor_params, max_norm)
    if plain_params:
        torch.nn.utils.clip_grad_norm_(plain_params, max_norm)


# --------------------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------------------
def trainable_state_dict(net):
    """On rank 0 under FSDP, gather full (unsharded) tensors for the trainable params.

    Sharded decoder-layer params are DTensors -> .full_tensor() materializes them on
    every rank; we keep only rank-0's copy. Non-sharded params (heads, time_embedder)
    are plain tensors. We save only requires_grad params (motion heads + trained
    gen/LoRA weights) to keep checkpoints small and resumable against the base ckpt.
    """
    from torch.distributed.tensor import DTensor

    sd = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        t = p.detach()
        if isinstance(t, DTensor):
            t = t.full_tensor()  # collective; all ranks participate
        if _is_rank0():
            sd[name] = t.to("cpu")
    return sd


def save_checkpoint(net, opt, step, out_dir, fsdp, save_optimizer=False):
    sd = trainable_state_dict(net)  # collective under FSDP (must run on all ranks)
    if not _is_rank0():
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_step{step:06d}.pt")
    payload = {
        "step": step,
        "model": sd,
        # Adam state for a 7B full-FT is ~56 GB fp32; off by default to keep
        # checkpoints small. Only meaningful for the non-FSDP single-GPU resume
        # path (under FSDP the states are sharded DTensors and not gathered here).
        "optimizer": opt.state_dict() if (save_optimizer and not fsdp) else None,
        "meta": {"motion_dim": MOTION_DIM, "fsdp": fsdp},
    }
    torch.save(payload, path)
    # publish "latest.pt" -> the step ckpt. Prefer a hardlink (instant, no extra
    # I/O); fall back to a copy+atomic-rename if hardlinks aren't supported.
    latest = os.path.join(out_dir, "latest.pt")
    latest_tmp = os.path.join(out_dir, "latest.pt.tmp")
    try:
        if os.path.exists(latest_tmp):
            os.remove(latest_tmp)
        os.link(path, latest_tmp)            # hardlink, same inode, ~free
        os.replace(latest_tmp, latest)       # atomic publish
    except OSError:
        import shutil
        shutil.copyfile(path, latest_tmp)
        os.replace(latest_tmp, latest)
    return path


# --------------------------------------------------------------------------------------
# main training loop
# --------------------------------------------------------------------------------------
def _load_viz_prompts(path):
    if not path or not os.path.exists(path):
        return VIZ_PROMPTS
    out = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            p, n = line.split("|", 1)
            out.append((p.strip(), n.strip()))
        else:
            out.append((line, f"p{i}"))
    return out or VIZ_PROMPTS


KIMODO_PY = "/home/jungbin_cho/miniforge3/envs/kimodo/bin/python"
RENDER_KIMODO = "/home/jungbin_cho/cosmos_motion_ft/render_kimodo.py"


def run_viz(net, tokenize, step, out_dir, args, log):
    """Rank-0 in-training visualization. Sample each eval prompt (cosmos env), decode
    to world joints with the verified motion_decode (bit-exact vs kimodo), save
    <name>_joints.npy, then render mp4s with kimodo's render_soma AS-IS via a kimodo-env
    subprocess (follow camera + floor/grid; matches kimodo training viz). Fail-safe:
    never crashes the training loop. net is set eval then back to train."""
    import sample_motion as sm  # lazy: avoids circular import at module load
    import subprocess
    prompts = _load_viz_prompts(args.viz_prompts_file)
    vdir = os.path.join(out_dir, "viz", f"step_{step:06d}")
    os.makedirs(vdir, exist_ok=True)
    skel = md.load_skeleton(args.skeleton)
    mean, std = md.load_stats(args.stats_dir)
    net.eval()
    try:
        with torch.no_grad():
            for prompt, name in prompts:
                try:
                    x0 = sm.sample(net, tokenize, prompt, args.viz_frames,
                                   args.viz_steps, args.viz_cfg, dtype=torch.bfloat16, seed=0)
                    x0_np = x0.cpu().numpy().astype(np.float32)
                    np.save(os.path.join(vdir, name + ".npy"), x0_np)
                    feat = torch.from_numpy(x0_np)
                    joints = md.decode_features_to_joints(
                        feat, skel, is_normalized=True, stats=(mean, std)
                    ).cpu().numpy()
                    np.save(os.path.join(vdir, name + "_joints.npy"), joints)
                except Exception as e:
                    log(f"  [viz] prompt {name!r} sample failed: {type(e).__name__}: {str(e)[:120]}")
        # render all saved *_joints.npy with kimodo's render_soma (as-is), in the kimodo env
        try:
            subprocess.run([KIMODO_PY, RENDER_KIMODO, "--dir", vdir, "--fps", "20"],
                           check=False, timeout=900,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"  [viz] kimodo render subprocess failed: {type(e).__name__}: {str(e)[:120]}")
        log(f"  [viz] step {step}: wrote joints+mp4 for {len(prompts)} prompts -> {vdir}")
    except Exception as e:
        log(f"  [viz] step {step} skipped: {type(e).__name__}: {str(e)[:120]}")
    finally:
        net.train()
        torch.cuda.empty_cache()


def run(args):
    torch.manual_seed(0 + int(os.environ.get("RANK", "0")))
    dtype = torch.bfloat16

    distributed = args.fsdp or args.ddp
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())

    os.makedirs(args.out, exist_ok=True) if _is_rank0() else None
    logfile = os.path.join(args.out, "train.log")
    log_fh = open(logfile, "a") if _is_rank0() else None

    # TensorBoard (rank 0 only) -> <out>/tb ; view with: tensorboard --logdir <out>/tb
    writer = None
    if _is_rank0():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.out, "tb"))
        except Exception as e:
            print(f"[tb] SummaryWriter unavailable ({e}); continuing without TB")

    def log(*a):
        if rank == 0:
            msg = " ".join(str(x) for x in a)
            print(msg, flush=True)
            if log_fh:
                log_fh.write(msg + "\n")
                log_fh.flush()

    # ---- data ----
    data_dir = args.data

    def _has_export(d):
        if d is None:
            return False
        if os.path.exists(os.path.join(d, "index.json")):
            return True  # single-dir export
        return len(glob(os.path.join(d, "shard_*", "index.json"))) > 0  # multi-shard

    used_synthetic = False
    if not _has_export(data_dir):
        used_synthetic = True
        data_dir = args.synthetic_dir
        if _is_rank0():
            make_synthetic_dataset(data_dir, n=64)
        if distributed:
            import torch.distributed as dist
            dist.barrier()  # ensure rank0 wrote it before others read
        log(f"[data] !!! real export NOT found at {args.data!r}; "
            f"USING SYNTHETIC data at {data_dir} (64 random samples) !!!")
    ds = TextMotionDataset(data_dir, max_frames=args.max_frames)
    log(f"[data] dir={data_dir} samples={len(ds)} synthetic={used_synthetic} "
        f"max_frames={args.max_frames}")

    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    def infinite(dl):
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b in dl:
                yield b
            epoch += 1

    data_iter = infinite(loader)

    # ---- text processor ----
    proc = build_text_processor()
    log(f"[proc] loaded Cosmos3-Nano processor: {type(proc).__name__}")

    def tokenize(texts):
        ids = []
        for t in texts:
            tid = proc.tokenize_text(t)  # list[int], applies the model's chat template
            if len(tid) == 0:
                tid = [SPECIAL_TOKENS["eos_token_id"]]
            ids.append(tid)
        return ids

    # null/unconditional tokens for CFG text-dropout (the SAME empty prompt the sampler
    # uses as the unconditional in sample_motion.py -> train/inference stay consistent).
    null_ids = tokenize([""])[0] if args.text_dropout > 0 else None
    _drop = [0, 0]  # [num dropped, num total] for logging

    # ---- model ----
    log(f"=== text->motion FT | config={'lora' if args.lora else 'full_generator'} | "
        f"tiny={args.tiny} fsdp={args.fsdp} world={world} | "
        f"snapshot={os.path.basename(NANO_SNAPSHOT) if NANO_SNAPSHOT else 'NONE'} ===")
    t0 = time.time()
    net, base_config = build_network(args.tiny, dtype)
    vocab_size = base_config.text_config.vocab_size

    if args.lora:
        net = inject_lora_pre_fsdp(
            net, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_target_modules=LORA_TARGETS,
        )

    net = materialize(net, dtype)
    if args.lora:
        init_lora_weights_post_materialization(net)
    log(f"[build] meta->cuda done in {time.time()-t0:.1f}s")

    if not args.tiny:
        load_gen_weights(net, verbose=(rank == 0))
    else:
        log("[load] tiny mode: random weights (no checkpoint load)")

    if args.lora:
        setup_lora(net)
    else:
        setup_full_generator(net)

    train, total = count_params(net)
    log(f"[params] trainable={train:,} total={total:,} ({100*train/total:.3f}%)")

    if args.fsdp:
        from torch.distributed.fsdp import fully_shard
        # Shard only the 36 decoder layers (where the ~7B _moe_gen weights live).
        # Sharding the top-level net would turn embed_tokens/heads into DTensors and
        # the hand-built scatter forward would mix Tensor/DTensor (PoC simplification #4).
        for layer in net.language_model.model.layers:
            fully_shard(layer)
        log(f"[fsdp] sharded {len(net.language_model.model.layers)} decoder layers "
            f"across {world} GPUs")

    trainable = [p for p in net.parameters() if p.requires_grad]
    # Per-group LR: new (random) motion heads get a higher LR than the pretrained
    # backbone (generator / LoRA adapters). Each group stores base_lr; the cosine
    # schedule multiplies base_lr by lr_factor(step) every step.
    head_params = [p for n, p in net.named_parameters() if p.requires_grad and is_motion_head(n)]
    rest_params = [p for n, p in net.named_parameters() if p.requires_grad and not is_motion_head(n)]
    groups = []
    if rest_params:
        groups.append({"params": rest_params, "base_lr": args.lr})
    if head_params:
        groups.append({"params": head_params, "base_lr": args.lr * args.head_lr_mult})
    # fused AdamW can't mix DTensor (sharded layers) + plain Tensor (heads); use foreach
    opt = torch.optim.AdamW(groups, lr=args.lr, fused=not args.fsdp)
    log(f"[opt] groups: backbone lr={args.lr:.2e} ({len(rest_params)} tensors), "
        f"motion-heads lr={args.lr*args.head_lr_mult:.2e} ({len(head_params)} tensors); "
        f"schedule={args.lr_schedule} warmup={args.warmup_steps}")

    # save a self-documenting config.json for this run (rank 0)
    if _is_rank0():
        cfg = dict(vars(args))
        cfg.update({
            "backbone_lr": args.lr,
            "head_lr": args.lr * args.head_lr_mult,
            "mode": "lora" if args.lora else "full_generator",
            "loss_weights": (md.DEFAULT_LOSS_WEIGHTS if args.loss == "kimodo" else None),
            "world_size": world,
            "trainable_params": train,
            "total_params": total,
            "snapshot": os.path.basename(NANO_SNAPSHOT) if NANO_SNAPSHOT else None,
        })
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        log(f"[config] wrote {os.path.join(args.out, 'config.json')}")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        from torch.distributed.tensor import DTensor, distribute_tensor

        payload = torch.load(args.resume, map_location="cpu")
        own = dict(net.named_parameters())
        n_loaded = 0
        with torch.no_grad():
            for k, v in payload["model"].items():
                if k not in own:
                    continue
                tgt = own[k]
                if isinstance(tgt, DTensor):
                    # checkpoint stores the FULL tensor (gathered at save time);
                    # re-shard it onto this rank's mesh/placement.
                    full = v.to(tgt.device, tgt.dtype)
                    tgt.copy_(distribute_tensor(full, tgt.device_mesh, tgt.placements))
                else:
                    tgt.copy_(v.to(tgt.device, tgt.dtype))
                n_loaded += 1
        start_step = payload.get("step", 0)
        if not args.fsdp and payload.get("optimizer"):
            opt.load_state_dict(payload["optimizer"])
        log(f"[resume] from {args.resume} step={start_step} (loaded {n_loaded} param tensors; "
            f"optimizer state {'restored' if not args.fsdp and payload.get('optimizer') else 'fresh'})")

    # ---- loss config (kimodo bones_seed FK loss) ----
    skeleton = stats = weights = None
    if args.loss == "kimodo":
        skeleton = md.load_skeleton(args.skeleton)
        skeleton = {
            "parents": skeleton["parents"].cuda(),
            "offsets": skeleton["offsets"].float().cuda(),
            "root_idx": int(skeleton["root_idx"]),
        }
        mean, std = md.load_stats(args.stats_dir)
        stats = (mean.cuda(), std.cuda())
        weights = dict(md.DEFAULT_LOSS_WEIGHTS)  # bones_seed_full.yaml weights
        log(f"[loss] mode=kimodo weights={weights} stats={args.stats_dir}")
    else:
        log("[loss] mode=mse (plain rectified-flow velocity MSE)")

    net.train()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    last_ckpt = None
    for step in range(start_step, args.steps):
        # LR schedule: linear warmup then cosine decay; multiplies each group's base_lr.
        lrf = lr_factor(step, args.warmup_steps, args.steps, args.min_lr_ratio, args.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lrf
        cur_lr = opt.param_groups[0]["lr"]  # backbone group (for logging)

        texts, motions = next(data_iter)
        # clip every motion to max_frames defensively (dataset already crops)
        motions = [m[: args.max_frames] for m in motions]
        text_ids = tokenize(texts)

        # text-dropout for CFG: replace caption with the null prompt with prob p (per sample)
        if args.text_dropout > 0 and null_ids is not None:
            for bi in range(len(text_ids)):
                _drop[1] += 1
                if torch.rand(1).item() < args.text_dropout:
                    text_ids[bi] = list(null_ids)
                    _drop[0] += 1

        opt.zero_grad(set_to_none=True)
        ps, sample_t = build_pack_from_batch(text_ids, motions, "cuda")
        loss, comps = forward_loss(
            net, ps, sample_t, dtype,
            loss_mode=args.loss, skeleton=skeleton, stats=stats, weights=weights,
        )
        loss.backward()
        if args.ddp:
            # plain data-parallel: each rank has the full (replicated) model; sync the
            # trainable grads by averaging across ranks (we don't use the torch DDP
            # wrapper because the forward is custom, not net.forward()).
            import torch.distributed as dist
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        if args.grad_clip > 0:
            clip_grads(trainable, args.grad_clip)
        opt.step()

        lv = loss.item()
        losses.append(lv)
        if writer is not None:
            writer.add_scalar("loss/total", lv, step)
            writer.add_scalar("lr", cur_lr, step)
            for k, v in comps.items():
                if k.startswith("l_"):
                    writer.add_scalar(f"loss/{k}", float(v), step)
        if step % args.log_every == 0 or step == args.steps - 1:
            peak = torch.cuda.max_memory_allocated() / 1e9
            if writer is not None:
                writer.add_scalar("mem/peak_gb", peak, step)
            n_mot = int(ps.action.sequence_indexes.numel())
            extra = ""
            if args.loss == "kimodo":
                extra = " " + " ".join(
                    f"{k}={comps[k].item():.3f}"
                    for k in ("l_fk", "l_global_rot_data", "l_local_joints_positions")
                    if k in comps
                )
            if args.text_dropout > 0 and _drop[1] > 0:
                extra += f" textdrop={_drop[0]}/{_drop[1]}({100*_drop[0]/_drop[1]:.0f}%)"
            log(f"  step {step:05d}  loss={lv:.5f}  lr={cur_lr:.2e}  "
                f"motion_tok={n_mot}  peak_mem_per_gpu={peak:.1f}GB{extra}")

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            last_ckpt = save_checkpoint(net, opt, step + 1, args.out, args.fsdp,
                                        save_optimizer=args.save_optimizer)
            log(f"  [ckpt] saved step {step+1} -> {last_ckpt}")

        # in-training test-set visualization (rank 0 samples + renders; others wait)
        if args.viz_every > 0 and (step + 1) % args.viz_every == 0:
            if _is_rank0():
                run_viz(net, tokenize, step + 1, args.out, args, log)
            if distributed:
                import torch.distributed as dist
                dist.barrier()

    # final checkpoint
    last_ckpt = save_checkpoint(net, opt, args.steps, args.out, args.fsdp,
                                save_optimizer=args.save_optimizer)
    if args.viz_every > 0:
        if _is_rank0():
            run_viz(net, tokenize, args.steps, args.out, args, log)
        if distributed:
            import torch.distributed as dist
            dist.barrier()
    peak = torch.cuda.max_memory_allocated() / 1e9
    log(f"[done] steps={args.steps} loss_first={losses[0]:.5f} loss_last={losses[-1]:.5f} "
        f"peak_mem_per_gpu={peak:.2f}GB ckpt={last_ckpt} synthetic={used_synthetic}")

    if _is_rank0():
        res = {
            "config": "lora" if args.lora else "full_generator",
            "fsdp": args.fsdp,
            "world_size": world,
            "trainable": train,
            "total": total,
            "peak_mem_per_gpu_gb": peak,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "losses": losses,
            "synthetic_data": used_synthetic,
            "data_dir": data_dir,
            "ckpt": last_ckpt,
        }
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)

    if writer is not None:
        writer.close()
    if log_fh:
        log_fh.close()
    if distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="export dir with features.npy + index.json")
    ap.add_argument("--synthetic_dir", type=str,
                    default="/tmp/cosmos_motion_synthetic",
                    help="where to write synthetic data if --data is missing")
    ap.add_argument("--out", type=str, required=True, help="run dir for ckpts + logs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--lora", action="store_true", help="LoRA on _moe_gen q/k/v/o + heads")
    ap.add_argument("--fsdp", action="store_true", help="FSDP2 shard decoder layers (for full-generator)")
    ap.add_argument("--ddp", action="store_true",
                    help="plain data-parallel (replicate model, all-reduce trainable grads); "
                         "use for LoRA/projection where the model fits on one GPU")
    ap.add_argument("--tiny", action="store_true", help="2-layer random model (fast smoke)")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--resume", type=str, default="", help="path to a ckpt .pt")
    ap.add_argument("--save_optimizer", action="store_true",
                    help="also save Adam state (large; only useful for non-FSDP resume)")
    ap.add_argument("--loss", type=str, default="kimodo", choices=["kimodo", "mse"],
                    help="kimodo = per-block weighted smooth-L1 + FK (bones_seed); mse = plain velocity")
    ap.add_argument("--skeleton", type=str,
                    default="/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz",
                    help="SOMASkeleton30 FK constants for the kimodo FK loss")
    ap.add_argument("--stats_dir", type=str,
                    default="/weka/jungbin/seed/stats/soma_uniform_motions_20fps/",
                    help="normalization stats dir (must match the export's stats)")
    ap.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"],
                    help="cosine = linear warmup then cosine decay (recommended); constant = old behavior")
    ap.add_argument("--warmup_steps", type=int, default=500, help="linear warmup steps")
    ap.add_argument("--min_lr_ratio", type=float, default=0.0,
                    help="cosine decays to peak_lr * this (0 = decay to 0)")
    ap.add_argument("--head_lr_mult", type=float, default=2.0,
                    help="LR multiplier for the new (random) motion heads vs the pretrained backbone")
    ap.add_argument("--text_dropout", type=float, default=0.0,
                    help="prob of replacing a caption with the null/empty prompt during training (CFG); Cosmos uses 0.1")
    ap.add_argument("--viz_every", type=int, default=0,
                    help="every N steps, sample+render eval prompts to <out>/viz/step_N/ (0=off)")
    ap.add_argument("--viz_prompts_file", type=str, default="",
                    help="optional 'prompt|name' per-line file; default = built-in VIZ_PROMPTS")
    ap.add_argument("--viz_frames", type=int, default=120)
    ap.add_argument("--viz_steps", type=int, default=50, help="denoising steps for viz sampling")
    ap.add_argument("--viz_cfg", type=float, default=2.5)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
