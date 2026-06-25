# MotionExpert POC — text → motion via a frozen Cosmos-3 reasoner

**Hypothesis.** Does the **frozen** Cosmos-3 Nano *reasoner* provide useful semantic conditioning for
human-motion generation, when a small trainable, motion-native transformer (**MotionExpert**) cross-attends
**one-way** to its hidden states `H_R`? Text→motion only — no video, no generator, no motion→reasoner feedback.

Target = **size-aware head/device UniEgoMotion** (`motion_uniego`, 283-D SOMA-30).

> **Status: v3 (current).** v1 (feature-MSE, velocity, on-the-fly reasoner) crumpled/spun. v2 fixed motion
> quality (x0-pred + decoded losses + cached H_R). v3 adds **floor grounding** + the **viz coordinate fix**.
> See "Version history" at the bottom.

## Information flow & attention contract
```
text → frozen reasoner → H_R [B,Ttext,4096]   (PRECOMPUTED & cached; frozen)   external semantics → cross-attn
skeleton neutral_joints [B,30,3] → ShapeEncoder → shape_tok [B,1,d]            actor size → in-context token
seq = [shape_tok, x_σ tokens] → MotionExpert(self-attn over seq, cross-attn KV=H_R) → x0_hat [B,T,283]
```
Required mask (R=reasoner, M=motion, G=generator):

```
            Key/Value
Query     R     M     G
R         ✓     ✗     ✗      (reasoner: causal, separate forward)
M         ✓     ✓     ✗      (M attends shape+motion (self) and H_R (cross))
G         ✗     ✗     ✗      (generator NOT instantiated)
```

**The contract holds by construction** because the reasoner and the MotionExpert are *separate forwards*:
- `M→M`: self-attention over `[shape_tok, motion_1..T]` (full/bidirectional denoiser attn).
- `M→R`: cross-attention with `KV = H_R`. (`H_R` is the reasoner's post-transformer hidden states, computed in a
  totally separate pass — now offline/cached — so `R→M` is *impossible*.)
- `M→G` / `G→*`: **there is no generator.** No G tokens, no reserved G position, no G↔M edge → nothing to mask.

### On "should there be a generator position?" (FAQ)
In the **full Cosmos MoT** (later-stage: motion conditioning the *video generator*), R/M/G live in **one packed
sequence** sharing a position/RoPE coordinate system; there you keep a G *position* (so RoPE indices stay
consistent) and **mask** the M↔G attention rather than deleting G. In **this standalone POC** there is no shared
packed sequence — the MotionExpert is its own transformer (motion frame-positions + a shape token), and `H_R` only
enters as cross-attention KV (the reasoner's own RoPE is already baked into `H_R`). So no G position is reserved and
nothing can leak. The spec's *"G is fully masked **or** not instantiated"* → we took **"not instantiated."**

### Design choices
- **Conditioning split:** external frozen semantics via **cross-attention** to `H_R`; owned actor-shape via an
  **in-context** `shape_tok` prepended to the self-attention sequence (not an additive bias).
- **No separate H-projection:** the cross-attn K/V linears use `kdim=vdim=4096`, consuming `H_R` directly.
- **Time conditioning:** DiT-style **AdaLN-zero** (per-block FiLM from the flow-time embedding). Required — a
  one-time additive time token is too weak for σ-dependent denoising (v1 lesson).
- **Prediction target: x0** (clean motion), not velocity — velocity errors compound through the residual
  canonical frame into spinning (v1 lesson). Rectified-flow noising is kept; sampler converts x0_hat→ε per step.

## Representation (283-D `motion_uniego`, SOMA-30, fps 20)
`[0:270]` per-joint local SE(3) (30×[6D rot ++ 3D trans]; **trans = joint position in the canonical frame**);
`[270:279]` canon_delta (head-canonical residual frame; frame-0 reset to identity per window); `[279:283]` foot
contacts. Decode to joints (no FK): `kimodo/motion_rep/uniego.py:uniego_world_joints_from_features(feat, n_joints=30)`
(ported, differentiable, bit-exact in `decode_uniego_torch.py`).

**Coordinate convention: Y-up, +Z-forward** (same as HumanML3D uniego; verified empirically — head sits at Y≈1.5 m,
body spans Y). matplotlib mplot3d is Z-up, so `viz.py` remaps world `(x,y,z)→plot(x,z,y)`, floor at y=0, negates X
(mirrors `kimodo/scripts/render_hml3d.py`). *Plotting `(x,y,z)` directly lays the skeleton on its side.*

## Data
- Motion: `/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep/{Sxx}/{seq}.npz` (`features`, `neutral_joints`).
- Text: reuses the camera-work caption manifest `.../video/manifest_video.jsonl` (per-window narration), sliced
  `features[start:end]` per window → **126,866 train / 13,374 val** (text, motion, skeleton) pairs (v3).
- Split: `.../train_test_split.json` (sequence-level, 642/71) — not re-invented.

### Floor grounding (v3) — IMPORTANT
The stored uniego features are **raw SLAM-world height** (`grounded=False`): feet float at a random per-window
vertical offset (measured: foot min-Y mean −2.3 m, std 3.6 m, range −12.8..1.5 m). We ground per atomic-action
window using `ground_offset_y` from `metadata/metadata_atomic_action_floor.jsonl` (also carried in the manifest;
see `nymeria_kimodo_pipeline/floor/SLICE_FLOOR_README.md`):
- grounding convention is `kimodo root_y -= ground_offset_y`. The uniego canonical frame is **pure-yaw** (no
  vertical component), so each joint's decoded height == its local-pose Y-translation → grounding = subtract
  `ground_offset_y` from **every joint's Y channel** `feat[..., j*9+7]` (`uniego_layout.ground_features`).
- Verified: feet −2.3 m → **−0.10 m (std 0.22)** — body now sits on the room floor (residual = sit/crouch).
- **Ambiguous floors** (GT multi-level/stairs `ambiguous`, or `est_ambiguous`) are **dropped** (1,088 windows,
  ~0.8%); the rest were already excluded by the manifest `usable` flag.
- Stats are computed on **grounded** windows (`compute_stats.py`), so max channel std dropped 6.5 → ~1.0.

### What goes into the model (the four data questions)
1. **Shape**: ✓ per-actor `neutral_joints (30,3)` → `ShapeEncoder` → in-context `shape_tok`.
2. **Floor**: ✓ grounded per window (v3; was raw height in v1/v2).
3. **Ambiguous floors**: ✓ dropped (v3).
4. **Canonicalization**: ✓ `canonicalize_frame0` per window (horizontal position + yaw; vertical = grounding).

## Files (all under `motion_expert/`)
| file | env | purpose |
|---|---|---|
| `uniego_layout.py` | both | 283-D layout consts + `canonicalize_frame0` + `ground_features` |
| `decode_uniego_torch.py` | both | differentiable 283-D→joints decode (bit-matches kimodo) for decoded losses + viz |
| `build_pairs.py` | cosmos | manifest ↔ uniego_rep windows + split + `ground_offset_y` + drop ambiguous → `pairs_{train,val}.jsonl` |
| `compute_stats.py` | cosmos | 283-D mean/std over **grounded** train windows → `stats/uniego283_{mean,std}.npy` |
| `reasoner.py` | cosmos | `FrozenReasoner`: text → `H_R [B,T,4096]` (reasoner_forward, frozen, no_grad) + sanity check |
| `precompute_hr.py` | cosmos | precompute H_R for all unique captions → sharded fp16 cache (~47 GB) |
| `hr_cache.py` | cosmos | `HRCache`: caption → cached `H_R` (memmap, batched + pad mask) |
| `motion_expert.py` | cosmos | `MotionExpert` (in-context shape, cross-attn to H_R, AdaLN-zero, x0-out) + `ShapeEncoder` |
| `flow.py` | cosmos | rectified-flow noising + **x0-prediction** DDIM-style sampler with CFG |
| `uniego_dataset.py` | cosmos | (text, motion[T,283] grounded+canon0+norm, neutral_joints, mask); 10% text-drop |
| `train.py` | cosmos | training (cached H_R; loss = feat-MSE(x0) + centroid-relative pose + joint-velocity smoothness) + `--smoke` |
| `sample.py` | cosmos | text → motion (x0 sampler, CFG) → unnormalized `.npy`; `--ablation cond/null/both` |
| `viz.py` | kimodo | decode → joints → stick-figure mp4 (Y-up) + cond-vs-null **ABLATION** side-by-side |
| `eval_watch.sh` | — | wait for milestone ckpts → sample + render the ablation |
| `run.sh` | — | launcher (cds into cosmos-framework — `build_network` uses a relative config path) |

## Loss (v3)
`loss = w_feat·MSE(x0_hat, x0) + w_joint·pose + w_smooth·smooth`, on valid (non-pad) frames, where
`pose` = centroid-relative decoded-joint L2 (targets *crumpling*) and `smooth` = decoded joint-velocity L2
(targets *jitter/spinning*). Defaults `w_feat=1, w_joint=10, w_smooth=50`.
**Why not absolute joint-position loss?** It is drift-dominated (the cumulative canonical frame), huge & unstable
(MSE 12–33, gradients ~1e5) → use bounded centroid-relative pose + velocity instead.

## Run (all on `ssh a3ultravis-a3ultranodeset-1`, via `run.sh`)
```bash
# Phase 0 (CPU): pairs (needs floor metadata) then stats (needs pairs)
bash run.sh build_pairs.py ; bash run.sh compute_stats.py
# H_R cache (one-time, ~11 min on 6 GPUs): for k in 0..5: CUDA_VISIBLE_DEVICES=k bash run.sh precompute_hr.py --shard k --nshards 6
# verify wiring
CUDA_VISIBLE_DEVICES=0 bash run.sh reasoner.py            # H_R sanity (distinct prompts → distinct H_R)
CUDA_VISIBLE_DEVICES=0 bash run.sh train.py --smoke       # finite loss, only expert grad, base frozen
# train
CUDA_VISIBLE_DEVICES=0 bash run.sh train.py --steps 100000 --batch_size 128 --w_joint 10 --w_smooth 50 --run_name <name>
# sample + render (the POC test)
CUDA_VISIBLE_DEVICES=0 bash run.sh sample.py --ckpt <run>/ckpt_step050000.pt --out <run>/samples --ablation both
/home/jungbin_cho/miniforge3/envs/kimodo/bin/python viz.py --dir <run>/samples
# (or auto-eval at milestones)  bash eval_watch.sh <run_dir> <gpu> 5000,20000,50000,100000
```
Runs/artifacts under `/weka/jungbin/cosmos_motion_ft_runs/<run_name>/`. H_R cache: `.../hr_cache/`.

## The hypothesis test (ablation)
`sample.py --ablation both` generates each prompt twice: **text-conditioned** (`H_R`) vs **null** (empty-prompt
`H_R`). `viz.py` stacks them side-by-side. If the reasoner provides useful semantics, the conditioned motion is
text-faithful and beats null (qualitatively + lower val MSE; measure cond-vs-null pose RMSE).

## Conventions / gotchas
- Train/sample in `cosmos` (torch 2.10); decode/render in `kimodo` (torch 2.4).
- `run.sh` is mandatory: `train_motion_ft.build_network` reads a **relative** `QWEN_JSON`, so cwd must be the
  cosmos-framework repo root. Also `unset LD_LIBRARY_PATH`.
- Python stdout is **block-buffered** to a file — monitor live progress via the TensorBoard events, not the `.log`.
- H_R cache is **text-only** → unaffected by motion-side changes (grounding, losses), so it's reused across v2/v3.

## Version history / lessons
- **v1** (velocity, feature-MSE only, on-the-fly reasoner): output crumpled & spun. Diagnosed: GT decodes
  perfectly (round-trip 3.6e-7) so it's the *model*; ~30× excess jitter (root-step 0.37 vs GT 0.015); feature-MSE
  doesn't penalize temporal coherence; on-the-fly reasoner = **94% of step time**.
- **v2**: **cache H_R** (47 GB fp16; mem 32→3.6 GB), **x0-prediction**, **decoded losses** (centroid-relative
  pose + joint-velocity) + **AdaLN-zero**. → jitter fixed: root-step 0.015 (=GT); text effect present (cond-vs-null
  pose RMSE 4–6 cm). Note: overfit-to-0 is **not** a valid test for flow-matching (memorization floor = E[ε²]=1).
- **v3** (current): **floor grounding** (feet → ~0) + drop ambiguous floors + **viz Y-up fix**.

## Out of scope (later)
camera-frame motion, video conditioning, generator interaction (the masked-but-positioned G above),
motion→reasoner feedback, foot-skating loss, larger model.
