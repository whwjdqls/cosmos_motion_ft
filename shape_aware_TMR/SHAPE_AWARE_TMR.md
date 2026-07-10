# Shape-aware TMR — an evaluator for shape-aware text→motion generation

**Goal.** A TMR (text–motion retrieval, TMR/ACTOR-style dual-encoder VAE) whose motion **encoder and
decoder are conditioned on the actor's skeleton** (centered `neutral_joints (30,3)` — the *identical*
conditioning the shape-aware generation model in `../motion_expert/` uses), trained on **BONES-SEED
proportional** with **all 3 text sources** (natural/overview + single + multi timeline), consuming
**raw motion joints** exactly like TMR in `/home/jungbin_cho/TAP` (via kimodo `TMRMotionRep`).
Purpose: R@k / MedR / FID metrics (via `kimodo.metrics.tmr.compute_tmr_retrieval_metrics`) for
evaluating shape-aware motion generation.

**Why shape-aware:** the same action on a tall vs short body yields different joint trajectories; a
shape-blind evaluator conflates body size with motion semantics. Here the **text encoder is
shape-free**, so InfoNCE pushes shape OUT of the shared latent (the retrieval embedding scores
*semantics*); the **decoder gets shape back** so reconstruction doesn't force skeleton geometry into z.

## Architecture (dims = the TAP run config)

- **Motion encoder** `st_model.ShapeAwareMotionEncoder` — copy of kimodo `ACTORStyleEncoder`
  (kimodo/model/tmr.py:58-129) + one addition: sequence `[mu_tok, logvar_tok, SHAPE_TOK, proj(motion)]`.
  latent 256, 6 layers, 4 heads, ff 1024, dropout 0.1, gelu, VAE (mu/logvar tokens).
- **Text encoder** — kimodo `ACTORStyleEncoder(llm_shape=(1,4096), vae=True)` **unmodified**, trained
  from the frozen llm2vec cache (`/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt`,
  227,286 captions covering all 3 sources). No shape.
- **Motion decoder** `st_model.ShapeAwareDecoder` — copy of TAP `ACTORStyleDecoder`
  (tmr_g1/model/tmr_model.py:25-70) with memory `[z, shape_tok]` (2 tokens). 4 layers.
- **ShapeEncoder** — the generation model's MLP (Linear(90→256), GELU, LN, Linear). Separate instances
  for encoder and decoder. The shape token is **never dropped**.
- Total ~15.0M params.

## Motion representation — raw joints, TAP-exact

On-disk uniego `features (T,283)` (proportional tree,
`/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps/`) → `decode_uniego.decode_joints`
(bit-exact vs kimodo, parity 0.0e+00) → **raw `posed_joints (T,30,3)`** → kimodo
`TMRMotionRep(SOMASkeleton30, fps=20)` = **186-d**/frame: root_pos 3 + heading 2 + local joints 29×3 +
velocities 30×3 + foot contacts 4, **canonicalized (frame-0 heading+planar zero) + z-scored** inside
the rep. Canonicalization happens ONCE, in the rep — same convention for stats, training, eval, and
(later) generated motions. Stats built on the proportional train split by `build_stats.py`
(kimodo split layout `{global_root,local_root,body}/{mean,std}.npy` + flat) →
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/stats_v0`.

## Data

`st_dataset.ShapeTMRDataset(SOMABonesSeedDataset)` — inherits kimodo's 3-source in-memory index,
split filtering, ≤10 s windowing (random ≤2 s offset). Overrides:
- `sources=(...)` — build/serve only listed pools (**test splits have ZERO multi entries**; the parent
  raises on any empty pool; its final log line also hardcodes all 3 names — guarded).
- `_build_natural_pool` — frame counts from uniego `features.shape[0]`.
- `_resolve_segment` — **train-time ±0.3 s temporal jitter on BOTH window boundaries**
  (TAP `aug_time_jitter_sec`; `--aug-time-jitter-sec`, default 0.3).
- `__getitem__` — raw uniego window → NaN guard (~0.4% tainted) → decode → rep(canonicalize+normalize)
  → (T,186); centered `neutral_joints`.
- `collate_st` — pads to batch max; **`mask` True = VALID** (TAP convention — OPPOSITE of
  bs_dataset's `motion_pad_mask`).

## Training (`st_train.py`, TAP-exact recipe)

AdamW lr 3e-4 wd 0.01 · warmup 2k → cosine · **200k steps** · **batch 256** · clip 1.0 ·
loss = `0.1·recon(masked L2) + 1e-4·(KL_m+KL_t) + 1.0·InfoNCE` (temp 0.1, **memory bank 8192**,
projected-text-cos dup masking 0.9 — load-bearing: 3-source batches repeat captions) ·
reparam at train / mu at eval · ckpt + inline eval every 5k · TB + config.json.
Inline eval (`st_inline_eval.SplitEvaluator`): fixed deterministic pools from **test_content**
(natural + single, 500 each), R@k/MedR/FID, with TAP's **posterior-collapse guard**
(emb std < 1e-4 → COLLAPSED, not fake 100%).

## Eval & the deliverable API

- `st_eval.ShapeTMREmbedder`: `embed_motion(raw joints [B,T,30,3] OR unnormalized uniego [B,T,283],
  neutral_joints, lengths) → [B,256] unit vecs`; `embed_text(captions) → [B,256]`. This is what a
  generation eval calls (unnormalize the generator's 283-d output with the proportional Mean/Std first).
- `st_eval.main`: per-source retrieval on test_content (natural, single) + multi on a train slice
  (labeled TRAIN — no test multi exists).
- `st_shape_ablation.py`: permuted-shape sanity — encoder latent should be ~shape-INVARIANT
  (high cos; the fair-evaluator property), decoder recon error should clearly INCREASE with a wrong
  shape (shape path alive).

## Run

```bash
cd /home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR
bash st_run.sh build_stats.py --out .../shape_tmr/stats_v0 --n-motions 5000     # CPU, once
sbatch -p a2 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --wrap "bash st_run.sh st_train.py --stats-path .../shape_tmr/stats_v0 --out-dir .../shape_tmr/v0"
bash st_run.sh st_eval.py --ckpt .../v0/last.pt --stats-path .../stats_v0 --out .../v0/eval.json
bash st_run.sh st_shape_ablation.py --ckpt .../v0/last.pt --stats-path .../stats_v0
```

## v2 — SnapMoGen-evaluator architecture (`st_model_v2/st_train_v2/st_losses_v2/st_t5`)

Reference: github.com/snap-research/SnapMoGen `model/evaluator/*` + `config/evaluator.yaml`
(their eval model = full TMR recipe). Same dataset / same 186-d raw-joints rep / same shape
tokens (encoder in-context + decoder memory, never dropped). Differences vs v1:

| | v1 (TAP-style) | **v2 (SnapMoGen losses)** |
|---|---|---|
| Text | pooled llm2vec (cache) | **same** — pooled llm2vec caches (per user; no live T5). Text tower identical to v1 → v2-vs-v1 isolates the LOSS recipe |
| Loss balance | 0.1 recon / 1.0 NCE | **1.0 recon / 0.1 NCE** |
| Cross-modal recon | off | **on** (decode from z_m AND z_t, SmoothL1) |
| KL | unit (1e-4) | unit + **bidirectional cross-modal** (1e-5) |
| Latent align | — | SmoothL1(z_t, z_m) (1e-5) |
| NCE negatives | queue 8192, proj-space dup 0.9 | **batch-only, RAW-llm2vec sent-emb filtering 0.8** (temp 0.10) |
| Decoder layers | 4 | **6** |
| Optim | 3e-4 wd .01 cosine | **2e-4 wd 1e-5**, warmup 200 → ×0.1 @120k |

Trainable 17.1M. Eval = official testsuite + `benchmark_llm2vec.pt` (same path as v1).
(`st_t5.py` remains available — `TestsuiteEvaluator(text_cache=None)` + `text_embed_fn`
supports a token-level-T5 variant later; the kimodo env now has sentencepiece and T5 loads
via `use_safetensors=True`.)

## Status

- 2026-07-03: v1 implemented; CPU smokes passed (decode parity 0.0e+00; dataset 3-source + jitter +
  test-subset OK, ~22 ms/item; model 15.0M, grads reach both shape encoders; shape changes encoder out).
- v1 trained (run `v0/`, job 9081, 200k): benchmark content R@1 ≈ 47–52 / R@3 ≈ 72–78 / MedR 1–2
  (protocol w/ 0.99 dedup, 500-case pools). Two eval bugs found+fixed on the way: test-split captions
  barely covered by the train llm2vec cache (9.6%) → switched to the official testsuite
  (`Kimodo-Motion-Gen-Benchmark-20fps` + `benchmark_llm2vec.pt`, ~100% coverage, all 6 groups incl.
  timeline_multi); protocol metric fakes ~100% on untrained embeddings (init dup_frac ~0.4) → added
  plain_R@k/plain_MedR/dup_frac alongside.
- v2 (SnapMoGen recipe) launched: run `v2/`, job 9175. Rationale: v1's numbers below published SOMA-TMR
  (~86–89 R@3, uniform data); v2 brings token-level text + generation-grounded alignment losses.
