# Cosmos3-Nano text→motion finetune on BONES-SEED — design & data notes

Goal: finetune the **Cosmos3-Nano GENERATOR** (reasoner frozen) to generate human motion
from a text caption, using the kimodo BONES-SEED dataset. 8× H200 on Slurm partition
`a3ultra`, env `cosmos`.

## CRITICAL difference vs kimodo: raw TEXT, not embeddings

kimodo precomputes **LLM2Vec embeddings (4096-d)** (`scripts/precompute_text_embeddings.py`)
and feeds embeddings to its denoiser. **Cosmos3 must NOT do this.** Cosmos3 ingests **raw
text strings**, tokenizes them with its OWN processor (`nvidia/Cosmos3-Nano`, via
`cosmos_framework.data.vfm.processors.build_processor_lazy`), and runs the tokens through
the **reasoner / understanding ("und"/causal) pathway**. The reasoner is FROZEN; its job is
to encode the caption into hidden states that the generation pathway cross-reads via the
two-way self-attention.

So our data export keeps the **raw `text` string** (the `"text"` field that
`SOMABonesSeedDataset.__getitem__` already returns). We do NOT run LLM2Vec at all.

## BONES-SEED motion representation (D = 369), from kimodo

`KimodoMotionRep`, 30-joint `SOMASkeleton30`, fps=20, T≤200 (10 s). Per-frame 369-D layout
(`kimodo/motion_rep/reps/kimodo_motionrep.py`):

| block | dims | slice | meaning |
| --- | --- | --- | --- |
| smooth_root_pos | 3 | [0:3] | low-passed root xyz |
| global_root_heading | 2 | [3:5] | (cos, sin) of frame-0-relative heading |
| local_joints_positions | 90 | [5:95] | 30 joints × xyz, root-relative |
| global_rot_data | 180 | [95:275] | 30 joints × 6D continuous global rotation (rot6d) |
| velocities | 90 | [275:365] | 30 joints × xyz velocity |
| foot_contacts | 4 | [365:369] | [L_heel, L_toe, R_heel, R_toe] binary |

Normalization stats: `/weka/jungbin/seed/stats/soma_uniform_motions_20fps/{global_root,body}`,
assembled as `cat([global_root(5), body(364)])` → (369,); formula `(x-mean)/sqrt(std²+1e-5)`.
We export **normalized** features (~unit scale per channel — ideal as flow-matching `x0`).
(`local_root` (4-d) is NOT part of the 369; ignore it.)

Decode back to joints for viz: `KimodoMotionRep(...).inverse(feats, is_normalized=True)["posed_joints"]`.

## Text sources (3, 1:1:1 mixture) — all keyed by filename + [start,end] sec window

- **natural**: `/weka/jungbin/seed/metadata/seed_metadata_v004.csv`, cols `content_natural_desc_1..4`, whole-clip window.
- **single**: `/weka/jungbin/seed/metadata/seed_metadata_v002_temporal_labels.jsonl`, per-event `description` + `[start,end]`.
- **multi**: `/weka/jungbin/seed/multi_timeline.jsonl`, `merged_description` + `[start,end]`.

Raw motion NPZs: `/weka/jungbin/seed/soma_uniform_motions_20fps/<date>/<filename>.npz`
(keys incl. `local_rot_mats (T,77,3,3)`, `root_positions (T,3)`; the dataset recomputes the
369-D feature on the fly). `*_M.npz` = mirrored (included).

## Two conda envs (cannot share a process)

- `kimodo` (Py 3.10, torch 2.4): owns `SOMABonesSeedDataset`. Used ONLY to **export** pairs.
- `cosmos` (Py 3.13, torch 2.10): owns Cosmos3-Nano. Used to **train**.

## Pipeline

### Step 1 — Export (kimodo env): `export_bones_seed_text_motion.py`
Instantiate `SOMABonesSeedDataset` with the `configs/training/bones_seed_full.yaml` paths,
`normalize=True`, iterate, and write a packed dataset the cosmos env can mmap:

- `features.npy` — float32 memmap, shape `[total_frames, 369]` (normalized).
- `index.json` — `{ "offsets": int64[N+1], "texts": [str]*N, "lengths": [int]*N,
  "filenames": [str]*N, "sources": [str]*N,
  "meta": {"fps":20, "dim":369, "normalized":true, "layout":"smooth_root3|heading2|jpos90|rot6d180|vel90|footc4"} }`
  (sample i = `features[offsets[i]:offsets[i+1]]`).

Output dir: `/weka/jungbin/seed/cosmos_text_motion/` (NFS/weka, visible to all nodes).
First do a SUBSET (`--max-samples 4000`) for a smoke test, then the full export.

### Step 2 — Train (cosmos env): `train_motion_ft.py` (extends `cosmos_motion_poc/motion_ft_poc.py`)
- Dataset: mmap `features.npy` + `index.json` → `(text:str, motion:[T,369] f32)`.
- Text: Cosmos3-Nano processor tokenizes `text` → text_token_ids → reasoner (und/causal, FROZEN).
- Motion: new 369-d modality (`motion2llm`/`llm2motion`/`motion_modality_embed`), gen pathway.
  Reuse the PoC's faithful forward (real MoT transformer, real `_moe_gen` weights loaded via
  the diffusers→native key map in `inference/model.py`, two-way attention, rectified-flow loss
  `v = noise − x0`).
- Default config: **full generator** (train all `_moe_gen` + motion heads, reasoner frozen)
  with **FSDP2 across 8 GPUs** (PoC measured 38.9 GB/GPU). LoRA selectable as a fallback.
- Batching B>1 with padding; AdamW; cosine/constant LR; DCP checkpoints to
  `/weka/jungbin/cosmos_motion_ft_runs/<name>/`; periodic loss logging.

### Step 3 — Launch: `sbatch_train_8gpu.sh`
1 node, `--gres=gpu:8`, partition `a3ultra`, `torchrun --nproc_per_node=8`.
Activate `cosmos`, `export LD_LIBRARY_PATH=`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Status / caveats
- This produces a TRAINING run (loss decreasing, checkpoints). A text→motion **sampling/inference**
  path (denoise loop + decode 369-d → joints via kimodo `inverse`) is a follow-on.
- The motion modality still "borrows" the packer's action layout for token-split / 3D-mrope /
  attention mask (see `cosmos_motion_poc/RESULTS.md`); making motion a first-class modality in
  `cosmos-framework` is the production-faithful follow-on.
</content>
