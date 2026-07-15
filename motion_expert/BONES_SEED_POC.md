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
| Env / deps | `cosmos` env + `cosmos-framework` + H_R cache | **A100 `kimodo` env** (torch 2.4); no Cosmos model, but audited Cosmos Framework + diffusers are used by official UniPC |

Everything else — rectified-flow x0-prediction, AdaLN-zero DiT blocks, the `ShapeEncoder` in-context
shape token, decoded-joint losses, x0 DDIM sampler, CFG — is reused.

---

## 0. What runs where (this A100 box)

> **Machine boundary:** all `motion_expert/bs_*` training, sampling, and evaluation belongs on this
> A100 machine. Do not run or validate it on the H200 `a3ultra` machines. The H200 `kimodo`
> environment lacks the BONES POC's expected Cosmos/UniPC dependencies and its local Cosmos source
> uses the newer `model.vfm` namespace instead of the older `model.generator` namespace expected by
> the pulled BONES UniPC path. The A100 environment is the authoritative working environment for
> this POC. This does not constrain `motion_expert_joint_attention/` jobs on H200.

- **All steps run in the A100 `kimodo` conda env**
  (`/home/jungbin_cho/miniconda3/envs/kimodo/bin/python`, torch 2.4). Verified present: torch,
  numpy, imageio, matplotlib, tensorboard, and the audited Cosmos Framework/diffusers versions used
  only by official UniPC. There is no Cosmos reasoner/generator model, `train_motion_ft` import, or
  H_R precompute in this POC.
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

## 15. Native-schedule x0 Phase-2 POC (2026-07-11)

This is a controlled schedule ablation before moving the motion expert behind the Cosmos reasoner.
Architecture, proportional 283-D representation, shape token, cached LLM2Vec conditioning, CFG dropout,
optimizer, LR schedule, and `1/1/5` feature/joint/smooth losses stay identical to `bs_incontext_v1`.

The only training change is:

```text
raw sigma ~ sigmoid(N(0,1))
sigma = shift * raw_sigma / (1 + (shift - 1) * raw_sigma)
shift = 3
x_sigma = sigma * epsilon + (1 - sigma) * x0
target = x0
```

The sampler uses Cosmos's native inference ladder construction: start from `(N-1)/N = 0.999` for
`N=1000`, linearly select the requested number of base sigmas, apply the same rational shift, quantize
the model timestep to `int64(sigma * 1000)`, and append a final sigma of exactly zero. The model still
predicts x0; the update uses the induced flow velocity `(x_sigma - x0_hat) / sigma`. This is native-ladder
x0 DDIM/straight-path integration, not a local UniPC copy.

Files:

- `bs_native_flow.py`: dependency-free native training sampler and inference ladder.
- `bs_native_train.py`: controlled native defaults over the shared `bs_train.py` loop.
- `bs_native_sample.py`: forces native-ladder sampling through the shared evaluation UI.
- `test_bs_native_flow.py`: formula, ladder, timestep, and oracle-x0 CPU tests.
- `sbatch_bs_native_phase2.sh`: one-GPU `a2` smoke/production launcher.

Legacy checkpoints remain backward compatible: `bs_train.py` defaults to `--schedule legacy`, and
`bs_sample.py --sampler auto` dispatches from the checkpoint's recorded schedule. The full controlled
run is configured as `bs_native_x0_logitnormal_shift3_w1_1_5_200k`, batch 128, 200k steps, reusing the
baseline's existing `bs_train_index.json`.

Launch status:

- GPU smoke job `10386`: completed five finite optimizer steps, sigma mean `0.70-0.74`, peak 10.3 GB.
- Production job `10387`: submitted on `a2`; run directory
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_1_5_200k`.
- Slurm log: `/home/jungbin_cho/cosmos_motion_ft/slurm-bsnatp2-10387.out`.

## 16. In-memory C45 generation evaluation (2026-07-13)

`bs_tmr_eval.py` compares one or more BONES generation checkpoints using the shape-aware C45 TMR
and Kimodo benchmark physical metrics. It does not create `motion.npz`, MP4, or embedding files.
Only an aggregate result JSON is written.

The representation path is explicit:

1. Generate normalized 283-D proportional UniEgo at the model's native 20 FPS.
2. Unnormalize with the proportional BONES Mean/Std and decode to shape-aware SOMA-30 joint
   positions. The same actor `neutral_joints` conditions generation.
3. Resample decoded positions from 20 to 30 FPS, then run C45's canonicalizing/normalizing
   `TMRMotionRep` with official NVIDIA 30-FPS stats. Pass the same centered `neutral_joints` to
   C45's shape token.
4. Compute Kimodo protocol R-precision/FID plus no-dedup plain retrieval diagnostics entirely in
   memory.
5. At native 20 FPS, compute `FootSkateFromHeight`, `FootSkateFromContacts`,
   `FootContactConsistency`, and `FootSkateRatio`. Generated UniEgo contact channels are
   thresholded at 0.5. Also report bone-length MAE against the conditioned skeleton.

Both generators use each benchmark case's seed, so a native/legacy comparison starts from the
same per-case Gaussian noise. Generation uses the benchmark LLM2Vec cache because the training
cache omits held-out captions; both caches contain the same frozen LLM2Vec representation. The
default launcher evaluates the complete proportional
`content/overview` pool with 100 sampling steps, CFG 2.0, and C45 step 5000:

```bash
sbatch sbatch_bs_tmr_eval.sh
```

Use `BS_TMR_MAX_CASES=8` for a GPU smoke. Do not run this evaluator on the login node.

### Full C45 result

Slurm job `10570` completed the in-memory `content/overview` comparison on 2026-07-13. It found
917 test cases, used all 911 with finite proportional GT UniEgo windows, and excluded six
non-finite GT windows. Configuration: both step-200k checkpoints, C45 step 5000, 100 sampling
steps, CFG 2.0, identical per-case seeded noise, no post-processing, and no saved motions.

| metric | native schedule | legacy schedule | proportional GT |
|---|---:|---:|---:|
| protocol R@1 | 46.76 | 34.58 | 52.36 |
| protocol R@3 | 69.15 | 54.34 | 89.13 |
| protocol R@5 | 78.70 | 63.56 | 95.06 |
| plain R@3 | 58.84 | 42.81 | 80.90 |
| FID gen-GT | 0.05255 | 0.08028 | - |
| paired text-motion cosine | 0.7805 | 0.6758 | 0.9021 |
| foot skate from predicted contacts (cm/s, lower better) | 13.85 | 10.23 | 1.90 |
| foot skate from height (cm/s, lower better) | 28.43 | 27.20 | 19.96 |
| foot skate ratio (lower better) | 0.260 | 0.212 | 0.102 |
| foot contact consistency (higher better) | 0.818 | 0.879 | 1.000 |
| conditioned-skeleton bone MAE (cm, lower better) | 0.360 | 0.269 | 0.181 |

The native shifted schedule is substantially better semantically: +14.81 protocol R@3,
+16.03 plain R@3, and 34.5% lower gen-GT FID. It does not dominate physical quality. The legacy
model has lower skating, more consistent contact channels, and slightly better bone adherence.
This points to a follow-up loss/post-processing ablation rather than reverting the native schedule:
retain native semantic alignment while directly improving contact and foot-velocity behavior.

Aggregate result JSON:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_vs_legacy.json`.

### Native Heun RK2 follow-up

`bs_native_flow.sample_x0_heun` adds an explicit trapezoidal Heun solver over the same shifted
native sigma ladder. Each non-final interval predicts with the current x0-derived velocity,
evaluates velocity again at the Euler-predicted next state, and averages the two slopes. The final
interval uses Euler because its endpoint is sigma zero, where `(x - x0) / sigma` is undefined.

Use it through `bs_sample.py --sampler native --native_solver heun` or
`bs_tmr_eval.py --native-solver heun`. A 50-step sample uses `2*50-1=99` denoiser evaluations and
198 model forwards with two-branch CFG. This nearly matches Euler-100's 100 denoiser evaluations
and 200 CFG forwards, making the comparison compute controlled.

Five focused native-flow tests pass. Slurm smoke job `10606` passed the full generation, C45, and
foot-metric pipeline. Full job `10607` evaluated the same 911 cases, checkpoint, case seeds,
shift 3, and CFG 2.0:

| metric | Euler 100 | Heun 50 | Heun change |
|---|---:|---:|---:|
| denoiser evaluations | 100 | 99 | -1 |
| protocol R@1 | 46.76 | 46.54 | -0.22 |
| protocol R@3 | 69.15 | 69.59 | +0.44 |
| protocol R@5 | 78.70 | 78.38 | -0.32 |
| plain R@3 | 58.84 | 57.96 | -0.88 |
| FID gen-GT | 0.05255 | 0.05193 | -1.18% |
| paired text-motion cosine | 0.7805 | 0.7797 | -0.0008 |
| predicted-contact skate (cm/s) | 13.85 | 13.73 | -0.12 |
| maximum contact velocity (cm/s) | 82.81 | 81.60 | -1.21 |
| foot skate ratio | 0.260 | 0.258 | -0.002 |
| foot contact consistency | 0.818 | 0.820 | +0.002 |
| conditioned-skeleton bone MAE (cm) | 0.360 | 0.352 | -0.008 |

Heun-50 is effectively tied with Euler-100. It gives very small FID and physical-quality gains,
but protocol retrieval is mixed and plain R@3 is lower. The native model's skating gap is therefore
not primarily first-order integration error; training losses or post-processing remain the more
promising intervention.

Heun result JSON:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_heun50.json`.

## 17. Official native Cosmos-3 UniPC (2026-07-13)

The UniPC path uses NVIDIA's implementation rather than a local solver approximation.
`bs_native_flow.create_unipc_scheduler` imports
`cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc.FlowUniPCMultistepScheduler`
and makes the same constructor and `set_timesteps` calls as the official
`cosmos_framework.model.generator.diffusion.samplers.unipc.UniPCSampler` wrapper. The `kimodo`
environment contains `cosmos-framework==1.2.2` and `diffusers==0.39.0`. The installed scheduler
source has SHA-256
`03aef1959f273b704ca4954f69b2a34df0fdd412f6acc8ab91625eeed78cf4fe` and is byte-identical to the
file audited at Cosmos Framework commit `3d9c0878fd0dde76eac98161aed0493d85a036fd`.

No UniPC predictor/corrector equations are copied into this repository. The local code only adapts
the BONES model's guided clean-motion prediction to the flow velocity required by the official
scheduler:

```text
x0_cfg = x0_null + guidance * (x0_cond - x0_null)
velocity = (x_sigma - x0_cfg) / sigma
```

The scheduler's own `convert_model_output` computes `x_sigma - sigma * velocity`, recovering
exactly `x0_cfg`; its own `step` then performs every UniP/UniC update. The model receives the
official integer timestep divided by 1000 because `MotionExpertInContext` multiplies its normalized
input by 1000 before timestep embedding.

Exact official settings used here:

- 35 inference steps, the `UniPCSampler.forward` default
- shift 3 and 1000 train timesteps from the BONES native checkpoint
- `use_dynamic_shifting=False`
- untouched scheduler defaults: order 2, `bh2`, `flow_prediction`, `predict_x0=True`, corrector
  enabled, lower-order final, and final sigma zero
- CFG 2.0, requiring 70 model forwards for 35 denoiser evaluations

One subtle but material source-code behavior is retained. The official scheduler constructor
applies shift 3 to its 1000-step sigma range. `set_timesteps` then interpolates from that shifted
`sigma_max` and applies shift 3 again. Therefore official UniPC does not use the earlier POC's
single-shift Euler/Heun ladder. The UniPC comparison measures the complete real Cosmos sampler
behavior, not an isolated solver-order change.

Use the sampler for normal generation with:

```bash
bash bs_run.sh bs_sample.py --ckpt CHECKPOINT --out OUTPUT \
  --sampler native --native_solver unipc --steps 35
```

For the in-memory C45 benchmark, set `BS_TMR_NATIVE_SOLVER=unipc`,
`BS_TMR_STEPS=35`, and `BS_TMR_ONLY_NATIVE=1` when launching
`sbatch_bs_tmr_eval.sh`. Do not run either model path on the login node.

Smoke job `10608` passed eight cases through generation, C45, and physical metrics. Full Slurm job
`10609` completed all 911 finite content/overview cases:

| metric | Euler 100 | Heun 50 | official UniPC 35 |
|---|---:|---:|---:|
| denoiser evaluations | 100 | 99 | 35 |
| CFG model forwards | 200 | 198 | 70 |
| protocol R@1 | 46.76 | 46.54 | 46.65 |
| protocol R@3 | 69.15 | 69.59 | **69.81** |
| protocol R@5 | **78.70** | 78.38 | 78.59 |
| plain R@3 | 58.84 | 57.96 | **59.06** |
| FID gen-GT | 0.05255 | 0.05193 | **0.05143** |
| paired text-motion cosine | 0.7805 | 0.7797 | **0.7809** |
| predicted-contact skate (cm/s) | 13.85 | 13.73 | **13.54** |
| maximum contact velocity (cm/s) | 82.81 | 81.60 | **79.87** |
| foot skate ratio | 0.260 | 0.258 | **0.257** |
| foot contact consistency | 0.818 | 0.820 | **0.821** |
| conditioned-skeleton bone MAE (cm) | 0.360 | 0.352 | **0.352** |

UniPC-35 is the best overall sampler of these three. Against Euler-100 it gains 0.66 protocol R@3
and 0.22 plain R@3, lowers FID by 2.13%, modestly improves physical metrics, and uses 65% fewer
denoiser evaluations. The gains are small enough that this does not change the training diagnosis:
sampling was not the main semantic or foot-skating bottleneck, but official UniPC should be the
default candidate for efficient native-model inference.

Result JSON:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_unipc35.json`.

## 18. Foot-skating loss variants (2026-07-13)

Two controlled native-schedule variants were launched to address the UniPC-35 model's
`13.54 cm/s` predicted-contact skate. Both train from scratch for 200k steps with the baseline's
architecture, proportional 283-D UniEgo data, cached text, shape conditioning, batch 128, seed 0,
shifted-logitnormal shift 3 schedule, optimizer, LR schedule, and shared training index. Native
checkpoint visualizations now use official UniPC at 35 steps.

### Variant A: stronger general position and velocity reconstruction

This variant changes only the existing loss weights:

```text
L = 1 * L_feature + 10 * L_joint_position + 100 * L_joint_velocity
```

It has no contact-specific term. GPU smoke job `10610` passed five finite steps at 10.3 GB. Its
pre-clipping gradient norm was `24-58`, substantially above the baseline but finite; the existing
global gradient clip of 1 remains active. Early production warmup later reached roughly `140-360`;
the baseline also clips heavily during this phase, but this run is more strongly clip-limited as
expected from the requested 10x/100x scaling.

- Training job: `10622`
- Run directory:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_10_100_inline10k_200k`
- Per-checkpoint evaluations:
  `<run>/inline_eval/step_010000.json`, ..., `<run>/inline_eval/step_200000.json`

### Variant B: contact-aware physical objective

This variant retains baseline weights `1/1/5` and adds three terms:

```text
L = L_feature + L_joint_position + 5 * L_joint_velocity
    + 0.05 * L_contact_BCE
    + 1.0  * L_contact_horizontal_foot_velocity
    + 10.0 * L_contact_foot_height
```

The four contact channels map to SOMA-30 joints
`[LeftFoot, LeftToeBase, RightFoot, RightToeBase] = [24, 25, 28, 29]`.

- `L_contact_BCE` reconstructs raw contact values, uses logits
  `2 * (predicted_contact - 0.5)` so its decision boundary equals evaluation's 0.5 threshold, and
  uses per-channel `pos_weight=(1-p)/p` from training contact means. This balances positives and
  negatives despite contact occupancy of roughly 71%/81%.
- `L_contact_horizontal_foot_velocity` is mean squared horizontal speed in physical `m/s`, masked
  by GT contacts and valid frame pairs. GT masking prevents the model from evading the physical
  penalty by predicting no contact.
- `L_contact_foot_height` is physical Y-position reconstruction for contacting feet. UniEgo's
  canonical frame is yaw-only, so decoded Y equals the corresponding raw local-pose Y channel. The
  loss uses those exact raw channels rather than backpropagating through the cumulative decoder.

The initial decoded-height formulation was rejected after smoke ablations: at weight 10 it caused
pre-clipping gradient norms of `65-321` despite a small scalar loss. Contact-only job `10613` and
velocity-only job `10614` remained around `2-6`, while decoded-height-only job `10615` reproduced
the spike. Using the equivalent raw Y channels fixed it. Final combined smoke job `10616` passed
five finite steps at 10.3 GB with gradient norms `2.7-5.5`.

- Training job: `10623`
- Run directory:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_inline10k_200k`
- Per-checkpoint evaluations:
  `<run>/inline_eval/step_010000.json`, ..., `<run>/inline_eval/step_200000.json`

### In-process checkpoint evaluation

Separate dependent evaluation jobs `10619` and `10620` were canceled. The first production
processes `10617` and `10618` were also stopped before checkpointing so both variants would use the
same callback implementation from their first update.

`bs_train.py --inline_eval_every 10000` now evaluates the live training model immediately after
each 10k checkpoint, plus the final checkpoint. The C45 model, benchmark text cache, 911 usable
content/overview cases, GT embeddings, and GT physical metrics are initialized once in the same
GPU process and reused. Each callback:

1. saves `ckpt_stepXXXXXX.pt` and `latest.pt`;
2. switches the live generator to evaluation mode;
3. samples all 911 cases in memory with official UniPC-35 and CFG 2;
4. computes protocol/plain retrieval, FID, predicted-contact skate, height skate, maximum contact
   velocity, contact consistency, skate ratio, and shape bone MAE;
5. writes `<run>/inline_eval/step_XXXXXX.json` and updates `<run>/inline_eval/history.json`;
6. logs key metrics to TensorBoard and restores training mode and the pre-evaluation CPU/CUDA RNG
   states.

No generated motions or embeddings are saved. GPU integration smoke job `10621` trained one
update, saved the checkpoint, ran an eight-case live UniPC/C45 evaluation, wrote both JSON files,
and exited successfully. Production job `10622` also initialized all 911 references successfully
before beginning optimization. The launcher defaults to 10k checkpoint, visualization, and inline
evaluation intervals; `BS_NATIVE_INLINE_EVAL_EVERY` can override the interval.

### Early results and paired follow-up (2026-07-14)

Variant A improved monotonically through 60k: protocol/plain R@3 reached
`58.40/44.68`, FID reached `0.10762`, and predicted-contact skate reached
`12.92 cm/s`. Variant B produced much better physical metrics at 40k
(`6.47 cm/s` contact skate, `19.05 cm/s` height skate, `0.129` skate ratio), but
its protocol/plain R@3 was only `51.70/37.87`; at 50k, retrieval and FID
regressed to `48.85/35.68` and `0.18285`. Evaluation uses fixed per-case noise,
so that checkpoint regression is not caused by resampling.

Two paired 200k follow-ups keep Variant A's retrieval-favorable `1/10/100`
reconstruction weights and add half-strength physical supervision. They differ
only in whether contact BCE is enabled, isolating whether the classifier term
helps contact consistency or contributes to the semantic tradeoff:

```text
Variant C: L = L_feature + 10 * L_joint_position + 100 * L_joint_velocity
             + 0.5 * L_contact_horizontal_foot_velocity
             + 5.0 * L_contact_foot_height

Variant D: Variant C + 0.025 * L_contact_BCE
           (contact logit scale 2)
```

- Variant C job: `10644` (`bsnatfoot`)
- Variant C run:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_10_100_foot_v0p5_h5_inline10k_200k`
- Variant D job: `10645` (`bsnatsoftc`)
- Variant D run:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_10_100_softcontact_c0p025_v0p5_h5_s2_inline10k_200k`

Both train from scratch with the same index, seed, native schedule, optimizer,
UniPC-35 visualization, and 10k in-process 911-case C45 evaluation contract as
Variants A/B. At submission, Variant C was running and Variant D was queued for
resources.

### Causal shape-awareness evaluation (2026-07-14)

The original inline reports already measure generated bone-length MAE against
the conditioned proportional skeleton. Current values are `0.352 cm` for the
native 200k baseline, `0.387 cm` for Variant A at 160k, `0.412/0.398 cm` for
Variant B at 120k/140k, `0.519 cm` for Variant C at 80k, and `0.604 cm` for
Variant D at 50k; the GT decoding floor is `0.181 cm`. This is good adherence,
but MAE alone cannot rule out always generating a population-average skeleton.

`bs_tmr_eval.py` therefore adds two stronger aggregate checks:

1. The normal pass centers each bone length across the 911 test cases and reports
   generated-target correlation, response slope, variance ratio, and centered
   MAE. A shape-collapsed model has slope/variance near zero; ideal tracking is
   near one.
2. A paired counterfactual pass keeps every caption, requested duration, and
   initial noise tensor fixed, but replaces the conditioning skeleton with the
   held-out natural skeleton having the most different bone-length vector. It
   reports requested-versus-generated delta cosine/slope/magnitude, target MAE,
   target advantage over the original skeleton, and retrieval/FID retention.

The intervention remains inside the natural proportional test distribution;
it does not use artificial global scaling. No generated motions or embeddings
are saved. Future inline callbacks enable it by default through
`--inline_eval_shape_counterfactual farthest`; `none` disables the extra pass.
Shape metrics are added to `history.json` and TensorBoard. Jobs already running
when this code was added retain their imported old callback and are not
restarted.

- Initial smoke job `10650` failed before Python because Slurm's `/bin/sh` rejected
  `set -o pipefail`; the permanently blocked dependent job `10651` was canceled.
- Corrected GPU unit/integration smoke with explicit Bash: job `10679`, completed successfully
  (`5/5` metric tests plus an eight-case paired UniPC/C45 integration pass)
- Final-200k 911-case backfill, dependent on the smoke and jobs `10644/10645`: job `10680`
- Backfill output:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_shape_counterfactual_final200_ablation_runs.json`

Variant C job `10644` completed at 200k. Its final protocol/plain R@3 is
`66.52/55.32`, FID `0.05309`, contact skate `5.18 cm/s`, height skate
`19.53 cm/s`, consistency `0.960`, skate ratio `0.113`, and bone MAE
`0.409 cm`. Its peak retrieval checkpoint remains 170k at protocol/plain R@3
`68.17/55.98`.

## 19. Full-contact all-benchmark evaluation and 500k continuation (2026-07-15)

The full-contact Variant B step-200k checkpoint is being evaluated on every
applicable Kimodo text-to-motion suite:

```text
content/{overview,timeline_single,timeline_multi}
repetition/{overview,timeline_single,timeline_multi}
```

This is 9,162 discovered benchmark cases before per-suite data audits. Constraint-conditioned
categories are explicitly not applicable: this BONES MotionExpert accepts text and actor skeleton
shape, but no trajectory, keyframe, or end-effector constraint input. The evaluation samples
in-memory with official Cosmos UniPC-35, CFG 2, C45 step 5k, proportional skeleton conditioning,
foot/contact metrics, population shape tracking, and the farthest-natural same-text/same-noise
shape counterfactual. It writes six detailed JSON files and one case-weighted suite summary without
saving generated motions or embeddings.

- Step-200k full evaluation job: `10746`
- Step-200k report directory:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/full_contact_200k_all_text2motion_unipc35_shape_cf`
- Six-suite smoke job `10743` completed all 48 requested cases and built its aggregate report.

The original step-200k checkpoint contains model weights and arguments only. It has no AdamW
moments, RNG state, or data-loader position, so an exact optimizer resume is impossible. The 500k
run is therefore a documented model-weight warm start: it strictly verifies the architecture,
data representation, batch size, schedule, prediction target, and all loss settings, then starts a
fresh AdamW optimizer with a conservative `5e-5` restart LR, 1k local-step warmup, cosine decay over
300k new updates, and seed `200000`. Checkpoint labels remain global (`210000` through `500000`).

- Source checkpoint:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_inline10k_200k/ckpt_step200000.pt`
- Continuation job: `10747`
- Continuation run:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_continue200to500k_lr5e-5_seed200000`
- Final step-500k all-benchmark evaluation: job `10748`, dependency `afterok:10747`
- Final report directory:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/full_contact_500k_all_text2motion_unipc35_shape_cf`

Training retains the exact `1/1/5 + contact/foot-velocity/foot-height = 0.05/1/10` objective and
official shifted-logitnormal shift-3 x0 recipe. Every 10k updates it saves a checkpoint and runs the
911-case content/overview C45 evaluation with UniPC-35 and shape counterfactuals. GPU continuation
smoke job `10742` loaded the real checkpoint and completed five finite optimizer steps at 10.3 GB;
LR unit job `10744` passed both restart-schedule tests.
