# Additional Phase3 LoRA

Implemented on sep-1 on 2026-11-05; no training job was started or restarted.
The additional adapter and EMA are opt-in. Existing task/noise/loss settings
and bridge-only defaults are retained.

## Model and trainable scope

Each generation-attention Q/K/V/O projection in all 28 native layers computes
the existing frozen projection (base plus Phase1 EMA LoRA), then adds an
independent `alpha / rank * B(A(x))` Phase3 delta. There are 112 new adapters.
Phase1 rank16/alpha32, trained video/action I/O, modality embeddings, reasoner,
and Phase2 motion weights remain frozen. A uses Kaiming initialization; B
starts at zero, preserving the loaded specialist function exactly initially.

The new FP32 adapters belong to the registered `model.phase3_lora` ModuleDict;
forward hooks apply them to native projections. The native model retains its
original state-dict keys and frozen-parameter guards. DDP, optimizer, clipping,
checkpointing and optional EMA include both bridges and new adapters. Hook
execution is covered under non-reentrant activation-checkpoint recomputation.

Generation Q/K/V/O also process native camera tokens in camera variants. This
is a generation-stream adapter, not a video-token-only adapter. It receives
the existing mixed-task objective's gradients, including motion losses through
the generator; this implementation does not introduce loss-specific routing.

## Training flags

Supported by the joint, image-joint and pretrained-motion camera **trainers**.
The image-joint Foundary launcher also forwards the options to its preflight,
fresh/reload probes and production process. Camera launchers retain their strict
baseline source-identity gates; do not bypass those gates with an old contract.
The historical scratch-motion recipe remains separate and rejects LoRA/EMA.

| Flag | Default | Meaning |
|---|---:|---|
| `--phase3-lora-rank` | 0 | Disabled; use a positive rank to enable |
| `--phase3-lora-alpha` | rank | New delta scale is alpha/rank |
| `--phase3-lora-lr` | 1e-5 | Peak LR of its separate AdamW group |
| `--phase3-lora-weight-decay` | 0 | Weight decay for the new factors |
| `--bridge-lr` | existing recipe | Optional override of bridge peak LR |
| `--bridge-weight-decay` | existing recipe | Optional override; currently 0 |
| `--ema-decay` | 0 | Disabled; e.g. 0.999 enables joint trainable-weight EMA |

Both LR groups follow the same existing 1k warmup and 200k cosine schedule,
retaining their peak LR ratio. AdamW betas remain 0.9/0.99, eps1e-8, global
batch32 and combined gradient clipping1. EMA updates FP32 shadows after every
optimizer update and stores its decay and update count for exact continuation.
EMA tracks bridges and the additional LoRA together, never the frozen Phase1.

Example options for a *new* experiment (starting values, not validated optima):

```bash
--phase3-lora-rank 8 --phase3-lora-alpha 8 \
--phase3-lora-lr 1e-5 --bridge-lr 5e-5 \
--phase3-lora-weight-decay 0 --bridge-weight-decay 0
```

Add `--ema-decay 0.999` only when wanted. Supply exactly the same options to
preflight, production and resume. Use a new local output directory and its own
acceptance receipt. Changing the trainable architecture, EMA setting or LR
changes the recipe fingerprint; an old bridge-only run is not an exact resume
of this new experiment. Existing jobs/checkpoints should use their original
source snapshots when resuming against their original source fingerprints.

## Checkpoints and inference

Feature-enabled checkpoints use schema2; legacy bridge-only checkpoints retain
schema1. The original `bridges` field is preserved. Enabled adapters add a
separate `phase3_lora` state dictionary; geometry, scaling, optimizer settings
and target paths are recorded in the recipe/provenance. Enabled EMA adds a
matched bridge/adapter state plus decay/update count. Optimizer/RNG/global
draw cursor and scheduler metadata are saved as before.

Updated joint/image-joint/camera sampling entrypoints infer adapter geometry
from the checkpoint, install it, and strictly restore the complete trainable
state. Select `--weights raw` (default) or `--weights ema`. Missing adapters,
incompatible configuration, incomplete EMA pairs or absent requested EMA fail
instead of falling back silently. Sampling records the weight selection.
These loaders also accept legacy bridge-only checkpoints.

Do not evaluate new checkpoints with older scripts that only call
`model.bridges.load_state_dict(...)`: they cannot restore the additional delta.
Custom evaluators must call `adapters.configure_checkpoint(payload)` before
model construction, then `adapters.load_trainable_state(model, payload,
weights="raw" or "ema")` after checking frozen specialist identities. Pure
Phase1 reference inference needs its own model without the Phase3 hooks;
`bridges_enabled=False` disables bridges only, leaving the new adapter active.

## Verification

CPU tests cover zero-delta exact parity, unchanged native keys/frozen tensors,
FP32 adapter masters under BF16 autocast, A/B gradient onset, exact raw/EMA
save-resume-continue, LR-ratio retention, rejection of incomplete checkpoints,
legacy bridge-only loading, historical scratch-state recovery, and real
two-process Gloo DDP with checkpoint recomputation and an unused task branch.

The full native GPU/data preflight is still required before production.
No GPU sampling or training was performed for this implementation.

On this node, all10 new adapter tests passed, including the two-process DDP
test. A separate CPU check against NVIDIA's actual `LoraInjectedLinear` also
passed. Existing conditioning/camera/scratch tests passed16/17; the remaining
legacy parity test cannot read Git object `f9043d3`, which is absent from this
checkout. It was not changed or counted as a pass. Trainer CLI overlays and
sampling/launcher argument parsing were checked without allocating a GPU.
