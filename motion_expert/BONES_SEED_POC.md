# BONES-SEED shape-aware text→motion POC (flow matching, in-context llm2vec)

**Goal.** Prove that **shape-aware text→motion on the UniEgoMotion (283-D SOMA-30) representation works
with flow matching** — *without any Cosmos generator or reasoner*. A small motion-native transformer
(the existing `MotionExpert`) denoises 283-D uniego motion, conditioned **in-context** on (a) a cached
**llm2vec** text embedding and (b) a per-actor **skeleton** (`neutral_joints`) token.

This is a sibling of the original `motion_expert/` POC (nymeria, frozen-Cosmos-reasoner cross-attention).
**Same flow-matching motion model**, three deliberate differences:

| | original motion_expert (nymeria) | this POC (bones-seed) |
|---|---|---|
| Text conditioning | Cosmos reasoner `H_R [B,Ttext,4096]` via **cross-attention** | cached **llm2vec** pooled `[B,1,4096]`, **in-context** (prepend one token) |
| Data | NymeriaPlus uniego (`pairs_*.jsonl`, SLAM height → grounded) | **BONES-SEED** proportional uniego (already floor-grounded) |
| Env / deps | `cosmos` env + `cosmos-framework` + H_R cache | **`kimodo` env only** (torch 2.4); no Cosmos at all |

Everything else — rectified-flow x0-prediction, AdaLN-zero DiT blocks, the `ShapeEncoder` in-context
shape token, decoded-joint losses, x0 DDIM sampler, CFG — is reused.

---

## 0. What runs where (this A100 box)

- **All steps run in the `kimodo` conda env** (`/home/jungbin_cho/miniconda3/envs/kimodo/bin/python`,
  torch 2.4). Verified present: torch, numpy, imageio, matplotlib, tensorboard. No `cosmos` env, no
  `cosmos-framework`, no `train_motion_ft` import, no H_R precompute.
- **GPUs via Slurm** (`srun`/`sbatch`) — this is a head node (`nvidia-smi` fails here by design).
- Decode/render uses `decode_uniego_torch.py` (pure-torch, bit-exact) — no kimodo import needed, but
  kimodo is available if wanted.

## 1. Inputs (all verified on disk, 2026-06-25)

| What | Path | Notes |
|---|---|---|
| Motion (283-D uniego, **proportional/shape-aware**) | `/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps/<date>/<name>.npz` | keys: `features (T,283)`, `foot_contacts (T,4)`, `neutral_joints (30,3)`. 142,220 npz (incl. `_M` mirrors). |
| Norm stats (**proportional**, 283-D) | `…/soma_proportional_uniegomotion_20fps/{Mean,Std}_uniego.npy` | `(283,)` float32. **Use these — NOT** `motion_expert/stats/uniego283_*.npy` (those are nymeria-grounded). |
| Text embeddings (**llm2vec, cached**) | `/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt` | `{captions:[str]×227286, features:[227286,4096] f32, meta}`. **Pooled, one vector/caption.** `""` (null) at row 0. model = `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`. |
| Text↔clip↔window metadata | `/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv` (natural), `…/seed_metadata_v002_temporal_labels.jsonl` (single), `/home/jungbin_cho/seed/multi_timeline.jsonl` (multi) | the 3 BONES-SEED text sources (1:1:1), keyed by `filename` + `[start,end]` sec. |
| Train/test split | `/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/{train_split_paths,test_content_split_paths,test_repetition_split_paths}.txt` (+ `_small`/`_medium`) | one `<date>/<name>` per line (incl. `_M`). Maps 1:1 to the uniego npz path. |

### 283-D layout (`uniego_layout.py`, J=30, head_idx=6)
`[0:270]` per-joint local SE(3) = 30×[6D rot ++ 3D trans (=joint pos in canon frame)] · `[270:279]`
canon_delta (residual head-yaw frame; frame 0 = absolute `cM[0]`) · `[279:283]` 4 foot contacts.
Decode→joints (no FK): `decode_uniego_torch.decode_joints(feat_unnorm) → [B,T,30,3]`.

## 2. The text↔motion pairing — reuse kimodo's in-memory index (NO offline pairs file)

How kimodo trains bones-seed motion models (`kimodo/scripts/train.py:1970 build_soma_dataset`): it
instantiates **`SOMABonesSeedDataset`** directly. That class builds the (text, segment) **index in memory
at `__init__`** from the three metadata sources (natural/single/multi), restricted to `train_split_path`,
with an optional **`cache_index` JSON** for fast startup, and serves windowed segments lazily in
`__getitem__` (1:1:1 source mixture, the `_resolve_segment` duration rule = random offset + cap to
`max_clip_sec`). **There is no offline "pairs" file** — I was overcomplicating it.

We reuse exactly that. The only mismatch: `SOMABonesSeedDataset` emits **369-D KimodoMotionRep** (reads raw
`local_rot_mats` + runs FK), but our data is **precomputed 283-D uniego**. So we **subclass it** and
override only the motion I/O — inheriting all index/mixture/split/window logic:

- **`data_root` = the uniego tree** (`…/soma_proportional_uniegomotion_20fps`). Then `_build_path_index`
  resolves each `entry.motion_path` straight to the uniego npz. *(Verified: no raw motion tree exists on
  this box, so this is also required.)*
- Override **`_build_natural_pool`** to read a clip's frame count from `features.shape[0]` (the parent
  reads `local_rot_mats`). single/multi pools read only metadata times — inherited unchanged.
- Override **`__getitem__`** to load `features[sf:ef]` + `neutral_joints` from the npz (below). The text
  string `entry.text` is unchanged → llm2vec cache lookup just works.

Caption coverage: every `entry.text` must be in the llm2vec cache (`CachedTextEncoder` raises `KeyError`
on a miss). The cache was built over the same bones-seed captions, so add a one-time **coverage check** (a
small script that iterates the dataset's pools and asserts all `entry.text ∈ cache`).

## 2b. Why subclass (not reimplement, not the shape-aware subclass)

- **Not reimplement** the metadata parsing — the user pointed at `train.py`: reuse the maintained kimodo
  path. Subclassing inherits `_build_single/multi_timeline_pool`, `_build_index`, `cache_index`,
  `_resolve_segment`, `__len__`, split filtering, mirror handling.
- **Not `SOMABonesSeedDatasetShapeAware`** — that subclass reads `neutral_joints` from a separate actor
  pack; our uniego npz already carries `neutral_joints` per file, so we read it directly. Subclass the
  **base** class and pull neutrals from the npz.
- `__init__` still constructs a (CPU) `KimodoMotionRep` with `normalize=False` (no stats needed); we never
  call its FK/normalize/heading paths (all overridden away).

## 3. Dataset (`bs_dataset.py`) — `BonesSeedUniegoDataset(SOMABonesSeedDataset)`

Subclass; the parent picks the source/pool and `entry` (with `entry.motion_path` = uniego npz,
`entry.text`, `entry.seg_*_sec`). Overridden `__getitem__(i)`:
1. `sf, ef = self._resolve_segment(entry)` (inherited windowing: random offset + cap to `max_clip_sec`);
   clamp `ef` to the clip length.
2. `feats = np.load(entry.motion_path)["features"][sf:ef]` → `[n,283]`; drop if `n < min_frames`.
3. **`canonicalize_frame0(feats)`** — reset window frame-0 `canon_delta` to identity (`uniego_layout`).
4. **normalize** with the **proportional** Mean/Std (loaded once in `__init__`).
5. `neutral_joints = np.load(...)["neutral_joints"]`, **centered** (`- mean(0)`; scale = the size cue).
6. caption (text-drop → `""` with prob 0.10, for CFG). **Shape token never dropped** (no shape CFG).
7. return `{motion[n,283], length n, text, neutral_joints[30,3]}`. Pad to batch-max in a small `collate`
   (build `motion_pad_mask`; stack neutrals). Variable length up to `max_frames` — the parent's windowing,
   not a fixed-T crop.

**Dropped vs the nymeria `uniego_dataset.py`:** no `ground_features` (bones-seed uniego already
floor-grounded), no `ground_offset_y`, no ambiguous-floor filter. **No heading aug / `first_heading`** —
uniego frame-0 canonicalization already removes global yaw, so the 369-D heading machinery is unneeded.
**Imports kimodo** (`from kimodo.data.soma_text_motion import SOMABonesSeedDataset`) — fine, we run in the
`kimodo` env; set `PYTHONPATH=/home/jungbin_cho/kimodo_open`.

## 4. Model (`bs_model.py`) — `MotionExpert` with in-context text, no cross-attention  ✅ DECIDED

Identical `MotionExpert` backbone (d=512 / 8 layers / 8 heads / ffn 2048, AdaLN-zero time conditioning, x0
prediction, zero-init output head, `ShapeEncoder`, sinusoidal frame pos). Exactly two changes:

1. **Text = a prepended in-context token.** Sequence `[text_tok, shape_tok, motion_1..T]` (was
   `[shape_tok, motion…]`). `text_tok = text_proj(llm2vec_pooled) + text_type`, where
   `text_proj: Linear(4096→d)` and `text_type` is a learned embedding (mirrors how `shape_tok =
   ShapeEncoder(neutral_joints) + shape_type`). Output is read on the `T` motion positions only
   (`seq[:, 2:]`); the two conditioning tokens are dropped from the output.
2. **Remove cross-attention.** Each block becomes self-attn (AdaLN-zero) → FFN (AdaLN-zero) — a standard
   DiT/MDM block. Delete `cross_attn`/`n2` and the `H_R` args; there is no reasoner anymore.

**Dropping / CFG (decided):**
- **Text is dropped** (prob `text_drop_prob`) to the cache's **`""` row** — a real learned null-prompt
  embedding, not zeros. That same `""` embedding is the **unconditional** branch at sample time (CFG).
- **The shape token is NEVER dropped** — the actor skeleton is always provided; there is **no shape CFG**.
- `self_pad_mask` gains two always-valid leading positions (text, shape).

> Why in-context (not cross-attn): the llm2vec text is a **single pooled vector**, so it is naturally one
> token. Prepending it to the self-attention sequence (exactly like the existing shape token) is the
> faithful "in-context" conditioning the user specified — the MDM-style text-token recipe, plus the shape
> token. The backbone is otherwise byte-identical to `motion_expert.py:MotionExpert`.

## 5. Flow matching + loss — reuse `flow.py` and the `motion_expert` loss unchanged

- **Noising/target (rectified flow, x0-pred):** `x_σ = σ·ε + (1−σ)·x0`; model predicts **x0**.
  `flow.add_noise` / `flow.sample_x0` (DDIM-style σ:1→0, CFG on x0) used verbatim.
- **Loss** (`motion_expert/train.py:step_loss`): `w_feat·MSE(x0) + w_joint·centroid-relative decoded-joint
  L2 + w_smooth·decoded joint-velocity L2`, masked to valid frames. Decode via `decode_uniego_torch`
  (unnormalize → joints). Centroid-relative pose targets *crumpling*; velocity targets *jitter/spinning*.
  Start `w_feat=1, w_joint=1, w_smooth=5`; if jittery, raise toward the nymeria-tuned `1/10/50`.
- **Shape-fidelity (optional, the §14 recommendation):** add a bone-length term pulling decoded limb
  lengths toward the conditioned `neutral_joints` (`Σ|‖j_c−j_p‖ − target_len|`). The pose loss already
  supervises shape implicitly (GT carries the actor's bone lengths); add this only if shape tracking is
  weak.

## 6. Training recipe (from `bones_seed_skel_aware.yaml` + `motion_expert/train.py`)

`d=512, layers=8, heads=8, ffn=2048` · windows ≤**200** frames (`max_clip_sec=10`, padded to batch-max) ·
batch **128** · **AdamW lr 2e-4**, wd 0, betas (0.9,0.99), warmup 1000, cosine decay ·
`cfg_dropout=0.10` (shape token **never** dropped — no shape CFG) · grad-clip 1.0 **+ non-finite-grad skip**
· **fp32** (no autocast) · steps **200k** · ckpt/viz every 5k · no EMA. Single GPU (~41.6 M params). Runs/TB
under `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/<name>/` (`/weka` absent). *(As-run config is in each
run's `config.json`; the actual sweep is in §13.)*

## 7. The POC hypothesis tests (deliverables)

1. **Text faithfulness** — `bs_sample.py --ablation both`: each prompt sampled **cond** (its llm2vec) vs
   **null** (`""`), rendered **cond (left) | null (right)** side-by-side. Conditioned motion should be
   text-appropriate and differ from null.
2. **Shape awareness (the novel claim)** — `bs_sample.py --shape_swap`: the *same* caption with a **tall
   (left) vs short (right)** actor skeleton; decoded limb lengths should track the conditioned skeleton
   (it prints per-bone-length MAE vs the conditioned `neutral_joints`). This is what "shape-aware" must show.

Sanity first (CPU-OK): `bs_sample.py --sanity <npz>` decodes a **real** proportional clip → renders →
a coherent floor-grounded human, confirming decode+render before trusting generations.

### Visualization = kimodo's, GT|gen side-by-side
All viz (in-training `bs_train.do_viz` and `bs_sample.py`) renders **left|right** via
`bs_viz.render_pair`, which wraps **`kimodo.scripts.render_soma.render_sidebyside`** — the SAME renderer
`kimodo/scripts/train.py:viz_step` uses (blue GT vs red gen, fingertip-end joints dropped, follow camera).
In-training viz pairs **decoded GT (left) vs generated (right)** for held-out captions. The one difference
vs kimodo: our motion is the 283-D uniego rep and is **shape-aware**, so joints come from
`decode_uniego_torch` and carry each actor's bone lengths; the skeleton *topology* (`joint_parents`,
`skip_joints`) is the shared SOMASkeleton30 read from `skeleton_soma30.npz`.

## 8. Files (all under `motion_expert/`)

| file | status | purpose |
|---|---|---|
| `flow.py` | **reuse as-is** | rectified-flow noising + x0 DDIM sampler + CFG |
| `uniego_layout.py` | **reuse** (`canonicalize_frame0`, `FEAT_DIM=283`; `ground_features` unused) | 283-D layout + frame-0 canon |
| `decode_uniego_torch.py` | **reuse as-is** | differentiable 283-D→joints decode (loss + viz) |
| `skeleton_soma30.npz` (repo root) | reuse | `parents` + `joint_names` → render edges + fingertip `skip_joints` |
| `bs_text_cache.py` | **new** | load llm2vec `.pt`; `batch(captions)→[B,1,4096]`, `null()→""` row; KeyError on miss (mirror `kimodo … CachedTextEncoder`) |
| `bs_dataset.py` | **new** — `BonesSeedUniegoDataset(SOMABonesSeedDataset)` | inherit index/mixture/split/window; override `_build_natural_pool` (uniego frame count) + `__getitem__` (load uniego features+neutrals, canon0, uniego-norm, drop NaN windows) + small `collate` |
| `bs_check_cache.py` | **new** | one-time: iterate the dataset pools, assert every `entry.text ∈` llm2vec cache |
| `bs_model.py` | **new** (model of `motion_expert.py`) | `MotionExpertInContext`: `[text,shape,motion]` self-attn, no cross-attn, AdaLN-zero, x0 out |
| `bs_viz.py` | **new** | wraps `kimodo.scripts.render_soma.render_sidebyside` → `render_pair(gt, gen, …)` + `load_skeleton()` (parents + fingertip skip_joints). The viz is **identical to kimodo's**, shape-aware joints |
| `bs_train.py` | **new** (model of kimodo `train.py`) | flow x0 train loop; losses (feature-only when `w_joint=w_smooth=0` → decode skipped); GT\|gen viz via `bs_viz`; TB; ckpt; non-finite-grad skip; `--smoke` |
| `bs_sample.py` | **new** | text(+shape)→x0 sampler + CFG; `--ablation both` (cond\|null), `--shape_swap` (tall\|short + bone-len MAE), `--sanity`; side-by-side mp4s via `bs_viz` |
| `bs_run.sh` | **new** | `srun`/`sbatch` launcher (kimodo env, 1 GPU) |
| `BONES_SEED_POC.md` | this doc | design + data contract |

## 9. Build/run order (proposed)

```bash
PY=/home/jungbin_cho/miniconda3/envs/kimodo/bin/python
export PYTHONPATH=/home/jungbin_cho/kimodo_open   # for `import kimodo` in bs_dataset
# 0. one-time: dataset index builds in-memory at first run (cached to cache_index JSON);
#    verify every caption is in the llm2vec cache (CPU, login node OK)
$PY bs_check_cache.py --split .../train_split_paths.txt
# 1. decode+render sanity on a REAL clip (CPU)
$PY bs_sample.py --sanity <a uniego npz>
# 2. smoke (1 GPU via srun): finite loss, only model grad, mem fits
srun -p a2 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G ... $PY bs_train.py --smoke
# 3. train (1 GPU)
srun -p a2 --gres=gpu:1 --ntasks=1 ... $PY bs_train.py --steps 150000 --batch_size 128 --run_name bs_incontext_v1
# 4. ablations: text cond-vs-null, and tall-vs-short shape swap
$PY bs_sample.py --ckpt <run>/ckpt_step050000.pt --ablation both --shape_swap
```

## 10. Decisions — proposed defaults (confirm before/while coding)

1. **Output/run root** → `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/<name>/` (verified writable; `/weka` absent).
2. **File layout** → `bs_*` prefix in `motion_expert/` (reuses `flow.py`/`uniego_layout.py`/`decode_uniego_torch.py` by plain import).
3. **Slurm** → partition **`a2`** (the only/default partition; `gpu:16`/node), `srun -p a2 --gres=gpu:1
   --ntasks=1 --cpus-per-task=8 --mem=64G`. (`--ntasks=1` mandatory or srun fans out per-CPU and OOMs.)
4. **Test split** → `test_content_split_paths.txt` for val/eval; `_small` for quick iteration.

## 11. Build status (2026-06-25) — code written + CPU-validated

All 7 files written under `motion_expert/`; `flow.py` / `uniego_layout.py` / `decode_uniego_torch.py`
reused **verbatim** (the model forward signature `(x_sigma, sigma, text_emb, text_pad_mask,
neutral_joints, motion_pad_mask)` was chosen so `flow.sample_x0` works unchanged). CPU smoke (no GPU,
no full cache load) passed end-to-end:
- dataset builds the 1:1:1 index (natural/single/multi) and serves finite `[B,200,283]` batches +
  `neutral_joints` + captions;
- `MotionExpertInContext` = **41.6M params**; forward + `flow.sample_x0` (CFG) + `decode_joints` all finite;
- a real clip decodes to a coherent floor-grounded human (y ≈ 0.02–1.44 m);
- `bs_check_cache` on a 46,813-entry slice: **0 captions missing** from the llm2vec cache.

**Gotcha handled:** kimodo's `SOMABonesSeedDataset.__init__` raises if **any** of the 3 pools is empty —
true for *small* splits (often no `multi`). The full train split is fine; **viz** therefore loads its few
captions via a standalone `load_viz_items` (natural CSV ∩ split), not a second dataset. `bs_check_cache`
and the training dataset should be pointed at the **full** train split.

**GPU smoke PASSED** (job on `a2`): 5 steps finite, loss ~0.7–0.87, grad_norm ~1.8–3.9, **~10 GB** GPU
mem (batch 128, T≤200). Two fixes were needed and are in place:
- **NaN-tainted windows dropped** in `bs_dataset.__getitem__` (`np.isfinite` guard). The proportional
  bones-seed tree has some NaN files (kimodo's shape-aware loader drops ~679 via a `nan_audit`); ~0.4%
  of windows were NaN → poisoned the GT loss on ~40% of batch-128 steps. Now skipped deterministically.
- **Non-finite-grad skip** in both loops (backstop): a degenerate batch through the 200-frame cumulative
  SE(3) decode can NaN the grad; that step is skipped instead of poisoning the run. The smoke now mirrors
  the real recipe (warmup + clip 1.0).

## 12. Results — eval at ckpt 25k of `bs_incontext_v1` (2026-06-25)

`bs_sample.py` on `ckpt_step025000.pt`:
- **Shape-awareness WORKS (the core claim):** same caption, tall vs short skeleton → decoded bone-length
  **MAE 0.4–0.8 cm** vs the conditioned `neutral_joints`; stature scales **tall ~2.0–2.4 m vs short
  ~1.4–1.65 m**. Quantitative, every prompt.
- **Text faithfulness present but modest at 25k:** null output is constant across prompts (same `""`
  embedding), cond varies per prompt and differs from null. Likely sharpens with more training.
- Loss ~1.0 → 0.025 by 25k (feat-MSE dominated), 0 non-finite skips.

**Eval gotcha (important):** no live text encoder → `bs_sample.py` can ONLY condition on captions in the
cache. Eval prompts are pulled from `--prompts_split` (natural CSV ∩ cache); free-form strings KeyError.

## 13. Loss-weight sweep (running, sbatch on `a2`, 200k steps each)

| run | job | w_feat / w_joint / w_smooth |
|---|---|---|
| `bs_incontext_v1` | 8750 | 1 / 1 / 5 |
| `bs_incontext_featonly` | 8755 | 1 / 0 / 0 (decode skipped entirely) |
| `bs_incontext_w10_50` | 8756 | 1 / 10 / 50 |

Launch: pre-seed `<newrun>/bs_train_index.json` (copy v1's; same full train split, 383 MB) to skip the
natural-pool rebuild, then `sbatch -p a2 … --wrap "bash bs_run.sh bs_train.py --steps 200000 --batch_size
128 --w_joint N --w_smooth M --run_name <name>"`.

## 14. Changes (2026-06-26)

- **Viz is now kimodo's** GT|gen side-by-side: new `bs_viz.py` wraps `render_soma.render_sidebyside`;
  `bs_train.do_viz` pairs decoded GT vs generated for held-out captions; `bs_sample` renders cond|null and
  tall|short pairs. The old single-panel `render_joints` was removed. CPU-validated (parents + fingertip
  `skip_joints=[14,15,20,21]`, mp4s written). Running jobs keep the old single-panel viz (in-memory code);
  re-render side-by-side on their ckpts with `bs_sample.py` if wanted.
- Removed smoke leftovers (`_smoke_split.txt`, `_smoke_bs_index.json`). The pre-existing **nymeria**
  motion_expert files (`train.py`, `sample.py`, `viz.py`, `reasoner.py`, …) are a separate experiment and
  were left intact.
