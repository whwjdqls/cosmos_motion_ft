# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
# WARNING
1. never use login node for memory and compute heavy tasks
2. if you want to run them, make a tmux session and just ssh in to a3ultravis-a3ultranodeset-0 or 1,2,3 and use them even though they are being used
3. even if you need to debug with gpu, tmux and ssh in to one of them and check the memory. and if there is room you can just use them.
4. only when the debuggin is done run sbatch
## Start here

**[`AGENTS.md`](./AGENTS.md) is the canonical, up-to-date navigation map** — commands, key file locations, per-area docs, common training/inference tasks, and gotchas. Read it first; this file does **not** repeat its tables. [`docs/code_structure.md`](./docs/code_structure.md) is the per-subpackage tour. This file adds the cross-file *architecture* that those don't spell out — the things you can only learn by reading many files at once.

## Environment (the non-obvious parts)

- **uv-managed, Python 3.13, torch 2.10** (`+cu128` or `+cu130`). The `*-train` dependency groups are the supported install path: `uv sync --all-extras --group=cu130-train` (CUDA 13, recommended) or `--group=cu128-train` (CUDA 12.8). `pip`/`conda` alone will **not** resolve the pinned `flash-attn`/`transformer-engine`/`natten` wheels — they live on a custom NVIDIA index that only `uv` is configured for.
- **Always `export LD_LIBRARY_PATH=` after activating** the venv, or `import torch` fails with `cannot import name '_functionalization'`. (Also in AGENTS.md — repeated here because nearly every fresh shell hits it.)
- **Attention backend is GPU-arch-specific** and auto-selected by `cosmos_framework/model/attention/`: Hopper → `flash_attn_3_nv`, Ampere/Ada → `flash_attn` (v2), Blackwell → `natten`. The `*-train` groups install all that apply; the framework degrades gracefully if one is missing.
- Run scripts from the repo root (so `import cosmos_framework` resolves) or set `PYTHONPATH=.`.

## What Cosmos3 actually is (the model)

Cosmos3 is an **omnimodal world model** (text, image, video, audio, action) with two runtime surfaces over **one shared transformer**:
- **Reasoner** — autoregressive, causal self-attention, next-token prediction (text out).
- **Generator** — diffusion with **rectified-flow / flow-matching** (not DDPM), full attention (vision/action/sound out).

Variants: **Nano** = 16B (Qwen3-VL-8B backbone, 36 layers, hidden 4096) — full-parameter SFT; **Super** = 64B — LoRA. **There is no "edge" variant.**

### Mixture-of-Transformers = dual weight pathways, one attention stack

This is the single most important thing to understand, and it is split across `cosmos_framework/model/vfm/mot/unified_mot.py` (`MoTDecoderLayer`, `PackedAttentionMoT`) and `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py` (`Cosmos3VFMNetwork`):

- **Every decoder layer carries two complete weight sets.** The *understanding/reasoner* pathway uses plain names (`q_proj`, `mlp`, `input_layernorm`, …); the *generation/diffusion* pathway uses the **same names suffixed `_moe_gen`** (`q_proj_moe_gen`, `mlp_moe_gen`, …). There is also a final dual norm (`norm` vs `norm_moe_gen`).
- **A token's role selects its pathway**, not a router. Text/condition tokens = **"und" / causal**; generation-modality tokens = **"gen" / full**. So "freeze the reasoner, train only the generator" ≡ *train params whose name contains `_moe_gen` (+ the modality I/O heads + `time_embedder`), freeze everything else (including `embed_tokens`).*
- **Attention is self-attention, not cross-attention.** Default `joint_attn_implementation="two_way"`: und tokens attend causally among themselves; gen tokens attend *fully over all tokens (und+gen)* — that joint full-attention **is** the conditioning mechanism. `three_way` exists for NATTEN sparsity / temporal-causal video and assumes all full tokens are one modality (`mot/attention.py`).

### Modalities are tokens in one packed sequence (no per-modality towers)

Each modality has only a thin **encoder→hidden** and **hidden→decoder** head plus a learned modality embedding; everything is scattered into a single 1-D packed sequence and mixed by the shared attention. See `Cosmos3VFMNetwork._encode_{text,vision,action,sound}` / `_decode_*` and `cosmos_framework/data/vfm/sequence_packing.py` (`PackedSequence`, `ModalityData`):

| Modality | encoder / decoder | extras |
| --- | --- | --- |
| text | `embed_tokens` | — |
| vision (video latent) | `vae2llm` / `llm2vae` (Linear) | 3D-mRoPE position ids |
| action | `action2llm` / `llm2action` (**DomainAwareLinear**, per-embodiment) | `action_modality_embed`, `raw_action_dim` masking |
| sound | `sound2llm` / `llm2sound` (Linear) | `sound_modality_embed` |

Conditioning (I2V/V2V) is expressed per-modality by `condition_mask` — clean frames stay clean; noised frames get an additive timestep embedding (`TimestepEmbedder`, computed in fp32) and are the loss targets. **There is no AdaLN/modulation** — timestep/modality/position are additive token biases. `action_gen` and `sound_gen` both **`assert vision_gen=True`** (`cosmos3_vfm_network.py:96`) — action/sound generation is *coupled* to vision generation; a vision-free generation modality would need a new path that omits that assert.

### Where the pieces meet

`cosmos_framework/model/vfm/omni_mot_model.py` (`OmniMoTModel`) is the top-level model: `build_net` instantiates `Cosmos3VFMNetwork`, `get_data_and_condition` extracts modalities from the data batch, `training_step` packs the sequence + adds rectified-flow noise + runs the net, and `_compute_losses` sums per-modality flow-matching losses (note `action_loss_weight=10.0` so the small action loss isn't swamped by video). The bare trainer (`cosmos_framework/trainer/__init__.py`) is model-agnostic: it calls `model.training_step(...)` for the loss, scales/accumulates, and steps.

## How a training run is configured (three layers)

`launch shell → cosmos_framework.scripts.train --sft-toml=<recipe>.toml → config → trainer`:

1. **Recipe TOML** (`examples/toml/sft_config/*.toml`) — run-level scalars only (`[job]`, `[model]`, `[optimizer]`, `[trainer]`, `[checkpoint]`, `[dataloader_train]`), with `${oc.env:VAR}` interpolation.
2. **Pydantic schema** (`cosmos_framework/configs/toml_config/sft_config.py`, `SFTExperimentConfig`, `extra="forbid"`) — validates the TOML; typos raise. `[job].task` (`vfm`|`vlm`) picks the base config.
3. **Hydra LazyConfig experiment SKU** (Python under `cosmos_framework/configs/base/experiment/`, selected by `[job].experiment`) — holds the full dataloader/dataset/optimizer/callbacks wiring. Trailing `key.path=value` overrides win last.

Finetuning scope is **selective full-parameter** via `optimizer.keys_to_select` — a substring allowlist over param names; anything not matched gets `requires_grad=False` (`cosmos_framework/utils/vfm/optimizer.py`). `lr_multipliers` similarly substring-matches. **LoRA** is opt-in via `lora_enabled` (default targets `q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen`; injected pre-FSDP in `omni_mot_model.add_lora` / `utils/vfm/lora.py`) and is the Super-tier path. Newly-added/resized heads go in `checkpoint.keys_to_skip_loading` so they init fresh against the base checkpoint.

## Data pipeline

Custom datasets use **`CosmosDataLoader`** (`cosmos_framework/data/vfm/dataflow/`), composed of four swappable roles in fixed order: **Distributor → Processor → Batcher → Collator** (see `docs/custom_dataset.md`). Use a `MapDistributor` for shuffle + **resumable** checkpoint/resume (iterable sources are not resumable). VFM batches keep media as per-sample lists via `VFMListCollator`. The action SFT path (`cosmos_framework/data/vfm/action/`) is the template for any new low-dim per-frame temporal modality.

## Training vs inference are two implementations of the same network

- **Native training stack**: `cosmos_framework/` (the classes above).
- **Diffusers inference shim**: `packages/diffusers-cosmos3/` (`Cosmos3OmniTransformer`, `Cosmos3OmniDiffusersPipeline`) — a re-implementation with a flat config, used by the published HF checkpoints (`nvidia/Cosmos3-Nano` is in **diffusers layout**).
- The **authoritative diffusers↔native weight key map** lives in `cosmos_framework/inference/model.py` (e.g. `add_q_proj → q_proj_moe_gen`, `proj_in → vae2llm`, `time_embedder.linear_1 → time_embedder.mlp.0`, plus a `language_model.model.` prefix). Use it whenever loading published checkpoints into the native network. `packages/transformers-cosmos3/` loads **only** the reasoner/understanding tower (drops all `_moe_gen`/action/sound weights).
- Inference entry: `cosmos_framework/scripts/inference.py`; `model_mode` selects the modality (`text2image`, `text2video`, `image2video`, `video2video`, `forward_dynamics`, `inverse_dynamics`, `policy`); the checkpoint registry is `_CHECKPOINTS` in `cosmos_framework/inference/args.py`.

## Local project: Cosmos-3 Nano → omnimodal human-motion world model

External project dir `/home/jungbin_cho/cosmos_motion_ft/` (see its `README.md` for the full
scope/plan; `cosmos_motion_poc/` holds the original PoC + `RESULTS.md`). It adds **3D human motion
as a new body-native continuous modality** to Cosmos-3 Nano.

**Scope (load-bearing):** Text/image/video/audio are **conditions, not targets** — we reuse Cosmos's
frozen understanding prior; we do **not** train a captioner/VLM. **Motion is both input and output.**
Valid directions: *observation/instruction → motion* (Text/Image/Video/Audio[+combos] → Motion) and
*motion + visual context → world* (Motion[+Text]+Initial Image → Future Video[+Audio]).
**Explicitly excluded:** `Video→Text`, `Motion→Text`, `Video+Motion→Text`, and context-free `Motion→Video` (ill-posed).

**Three methods** (shared: motion modality + freeze-reasoner/adapt-generator): (1) projection-only
(sanity), (2) motion heads + LoRA on `*_moe_gen` (current run), (3) **main** = frozen Cosmos + a
Motion-Transformer bridge cross-attending the frozen reasoner/generator (bidirectional, no text output).

**Implementation facts:** motion is vision-decoupled (no `vision_gen` assert path); reasoner frozen =
freeze non-`_moe_gen` params + `embed_tokens`; LoRA/full uses the `_moe_gen` pathway; weights loaded via
the diffusers↔native key map in `inference/model.py`. Production-faithful first-class-modality edits
(`motion_gen` flag, `_encode_motion`/`_decode_motion`, `sequence_packing.py` entry, loss/recipe,
`parallelize_vfm_network`) are listed in the PoC `RESULTS.md`.

## Conventions

- Cite code as `file:line`; when unsure, point to the nearest doc rather than guessing (per AGENTS.md).
- **Keep inference and training separate**: no training-time imports under `cosmos_framework/inference/`; keep heavy inference-only deps (vLLM, Ray, Gradio) behind optional extras.
- Add new code by concern, not convenience — see the "Where to Add New Code" table in `docs/code_structure.md`.
