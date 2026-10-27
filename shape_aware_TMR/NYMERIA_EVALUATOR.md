# Nymeria evaluator: C45 pipeline

This is a separate Nymeria evaluator, not a replacement for the BONES C45 bundle.
The shared model, losses, InfoNCE implementation and official initialization are
unchanged. The new code adapts data, validation, provenance, and cluster paths.

## Recipe

- Official TMR-SOMA-RP-v1 initialization, revision
  e427752ae3446dedba49e928c93ddc9f0e413401; fresh shape modules.
- Frozen official text encoder; motion/text posterior means during training.
- Batch 128, six loader workers, AdamW LR 5e-5, weight decay 0.01,
  1k warmup, 10k updates, existing cosine floor 1e-6.
- Reconstruction 0.01, motion KL 3e-6, text KL 0, contrastive 1.
- InfoNCE temperature 0.1, memory bank 8192, frozen LLM2Vec duplicate
  threshold 0.95; feature noise 0.02.
- Native source 20 fps -> raw SOMA30 joints -> resample 30 fps -> canonical
  TMR186 -> official 30-fps motion statistics. No generator283 z-scoring.
- Same C45 architecture: latent256, encoder6, decoder4, heads4, FF1024,
  dropout0.1. This is not a Cosmos motion-expert training run.
- Save/evaluate every1k. Validation selects best.pt by plain average-tie R@3;
  periodic checkpoints and last.pt remain available for recovery.

## Nymeria data contract

prepare_nymeria.py preserves the source recording-level test split. It uses an
existing val/validation split, or deterministically reserves 10% of the original
training recordings using seed0. The test recordings never enter training or
validation. Manifests, captions, recording lists, rejection reasons and source
provenance are saved, with prepared.json published last.

The default motion source is uniego_rep_camhead_v1. Native caption windows must
be usable, finite, in bounds, 10-200 frames long, and have a finite floor offset.
Known floor-quality drops are excluded. Additional exclusions are counted,
not silently replaced by unrelated samples. Captions use the existing camera
wearer expansion and Kimodo sanitization. There is no CFG caption dropout.

Each eligible raw window and neutral skeleton are cached as NPY files. Grounding
uses only the recorded GT floor offset plus the recorded calibration delta.
No grounding based on a generated motion's minimum foot height is introduced.
Training boundary jitter stays INSIDE the annotated window, unlike BONES jitter
which can extend outside a segment. BONES three-source balancing does not apply.
These are explicit dataset adaptations, not an identical window sampler.

Validation/test windows are deterministic and unaugmented. Padding masks mark
True=valid. Variable source lengths are retained; the Nymeria inference adapter
crops each item before 20-to-30-fps resampling, so padded tails are never scored.

## Launch on an owned compute allocation

Use one GPU and eight CPUs. All model/data work, including preparation, must run
on a compute node. The wrappers reject login hosts and cs-gpu-01. New Yonsei
allocations should also exclude node31 and node33. The scripts never cancel
an allocation and never replace an existing trainer.

Set COSMOS_SITE to yonsei or grasp. Optional environment overrides:

- OFFICIAL_TMR_ROOT: official snapshot with last_weights and stats/motion.
- SHAPE_TMR_BUNDLE: existing C45 bundle supplying pinned Kimodo and text cache.
- SHAPE_TMR_PYTHON / KIMODO_ROOT: explicit compatible runtime/source overrides.
- NYMERIA_TMR_ROOT: parent for caches and output.
- NYMERIA_TMR_DATA_DIR, NYMERIA_TMR_TEXT_CACHE, NYMERIA_TMR_OUT.
- NYMERIA_TMR_MANIFEST, NYMERIA_TMR_SPLIT_FILE, NYMERIA_TMR_MOTION_ROOT,
  NYMERIA_TMR_FLOOR, NYMERIA_TMR_REPRESENTATION: explicit input overrides.
- NYMERIA_TMR_STATS: explicit evaluator stats override for a separately named
  experiment; never point this at a generator's 283-D mean/std.
- NYMERIA_TMR_BASE_TEXT_CACHE: reusable compatible LLM2Vec cache.
- NYMERIA_TMR_TEXT_HUB: HF model cache containing LLM2Vec and its adapters.
  Yonsei defaults to /scratch2/whwjdqls99/cosmos/cache/phase2_c45_hf/hub,
  not the separate Cosmos Nano cache. The wrapper aligns HF_HUB_CACHE,
  HUGGINGFACE_HUB_CACHE and HUGGINGFACE_CACHE_DIR for tokenizer and model
  loading. Grasp retains its configured HF cache unless explicitly overridden.
- NYMERIA_TMR_EVAL_CKPT, NYMERIA_TMR_EVAL_SPLIT, NYMERIA_TMR_EVAL_OUT.

Yonsei defaults to:
 /scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/shape_tmr/nymeria_c45_camhead_v1_20260918

Grasp defaults to its normal WEKA run root. Physical storage is not shared.
Model downloads are not implicit: provide the official snapshot and existing
LLM2Vec weights before launch. The text-cache builder checks compatibility with
the BONES bundle's cached embeddings.

Run from the repository root:

~~~bash
export COSMOS_SITE=yonsei
bash shape_aware_TMR/train_nymeria_c45.sh all
~~~

Actions prepare, text, train, eval can also be run separately. A completed
prepared dataset is immutable and is reused only for matching preparation
arguments. A changed representation/split needs a new data directory.
The train action automatically resumes its own last.pt if present.

## Resume and evaluation provenance

st_train.py keeps dataset=bones as its default. Pretrained weights are now
fresh-initialization-only for both datasets: checkpoint restoration takes
precedence. A missing resume path is an error, not a silent fresh start.

Nymeria checkpoints bind manifest, text-cache and evaluator-statistics hashes
and the training recipe. Mismatched resumes fail. They also retain optimizer,
contrastive memory bank, RNG state, and validation-best bookkeeping.
The in-flight loader cursor and worker RNG states are not checkpointed, so
resumption is not a bitwise replay of an uninterrupted data stream.

NymeriaTMREmbedder in nymeria_eval.py exposes the normal C45 embed_text and
embed_motion interface, enforces Nymeria checkpoint/statistics compatibility,
and handles variable-length resampling correctly. It accepts raw joints or
UNNORMALIZED UniEgo283. A generator must use its OWN training mean/std before
calling this adapter. Existing BONES evaluation defaults are unchanged.

## Reporting and interpretation

Validation JSON/history and held-out test JSON always label the inputs as GT.
They include per-case ranks, case counts, unique captions, the existing C45
duplicate-aware protocol, strict plain ranks and average-tie plain ranks.
Collapse and duplicate concentration diagnostics are retained. No invalid
embedding rows are silently removed to improve a score.

GT-self FID is zero by definition. This retrieval training workflow does not
compute generated-motion FID, skating or penetration; physical metrics are
explicitly marked unavailable rather than fabricated. Existing generator
physical scoring remains independent of which retrieval evaluator is used.

Use validation for model selection or new experiments. A desired 80-90% R@3
is a target, not an acceptance criterion permitting test leakage or changes
to candidate pools. Report unsuccessful experiments as well.
Do not compare FID across different evaluator embedding spaces. Training this
evaluator does not demonstrate a fix for floating or foot contact.
