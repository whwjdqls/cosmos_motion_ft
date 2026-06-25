# Cosmos-3 Nano → Omnimodal Human-Motion World Model

## Project statement

> We adapt **Cosmos-3 Nano** into an omnimodal **human-motion** world model by introducing
> **3D human motion as a body-native continuous action modality**. The model generates human
> motion from language, images, videos, and audio, and uses human motion as a **control signal**
> for context-conditioned video/audio generation.

> Cosmos-3 Nano를 사람 모션 중심의 omnimodal world model로 확장한다. Text/image/video/audio를
> condition으로 받아 3D human motion을 생성하고, 반대로 human motion을 control signal로 사용해
> image context 기반 video/audio를 생성한다.

**Three load-bearing principles:**
1. **Text is a *condition*, not a target.** We do **not** train a captioner / VLM reasoner.
   We *reuse* Cosmos-3's frozen text/image/video/audio understanding prior as conditioning.
2. **Motion is both input and output** — the new continuous modality we add.
3. **Video/audio are generative outputs only when there is sufficient visual/context condition**
   (an initial image / scene). Pure `motion → video` is ill-posed and is avoided.

## Task space (what we train)

Direction is **observation/instruction → motion**, and **motion + context → world (video/audio)**.

### Motion as output (primary)
| Task | Dataset |
| --- | --- |
| Text → Motion | BONES-SEED |
| Image → Motion | NymeriaPlus (first/key frame) |
| Video → Motion | NymeriaPlus (egocentric) |
| Text + Image → Motion | NymeriaPlus |
| Text + Video → Motion | NymeriaPlus |
| Video + Audio → Motion | NymeriaPlus |
| Text + Video + Audio → Motion | NymeriaPlus |

### Motion as control signal → world generation (secondary, later phase)
| Task | Notes |
| --- | --- |
| Motion + Initial Image → Future Video | image = appearance/scene condition |
| Motion + Text + Initial Image → Future Video | best video-gen setting |
| Motion + Initial Image → Audio + Video | joint AV event generation |
| Motion + Text + Initial Image → Audio + Video | — |

### Explicitly OUT of scope (removed)
- `Video → Text` ❌  (captioning / VLM task — not ours)
- `Motion → Text` ❌  (motion captioning — not ours)
- `Video + Motion → Text` ❌
- `Motion → Video` (no image/context) — ⚠️ ill-posed; only use the **Motion + Image/Text → Video** forms above.

## Method (3 variants; #3 is the main method)

The motion modality and "freeze the reasoner, adapt the generator" pattern are shared. The three
variants differ in how much of Cosmos is adapted (see `DESIGN.md`, `RESULTS.md` in the PoC dir,
and `train_motion_ft.py`):

1. **Projection-only** — motion encoder/decoder projections into/out of Cosmos's action token
   space; everything else frozen. Sanity check.
2. **Motion enc/dec + LoRA** — train the motion heads + LoRA on the Cosmos generator
   (`*_moe_gen`). Performance variant. *(Current implemented run; see Status.)*
3. **Frozen Cosmos + Motion-Transformer bridge** — **main method.** A dedicated Motion
   Transformer cross-attends to a frozen Cosmos reasoner/generator:
   ```
   Text/Image/Video/Audio tokens
           ↓
   Frozen Cosmos reasoner/generator  ⇕ cross-attention  Motion Transformer → Human-Motion decoder
   ```
   Reverse direction (motion → world): Human motion → Motion Transformer ⇕ cross-attention →
   frozen Cosmos generator → Video / Audio+Video. **No text output in any direction.**

## Training plan (phased; motion-output first, motion→world last)

- **Phase 1 — Motion modality attachment.** Data: BONES-SEED + NymeriaPlus (motion only).
  Tasks: motion autoencoding / continuation / inpainting. Goal: stable 3D-motion encoder/decoder.
- **Phase 2 — Text→motion prior.** Data: BONES-SEED. Tasks: Text→Motion, Text+partial→full,
  Text+prefix→future. Goal: align Cosmos text prior with the motion latent.
- **Phase 3 — Visual/audio→motion grounding.** Data: NymeriaPlus. Tasks: Image/Video/Text+Image/
  Text+Video/Video+Audio/Text+Video+Audio → Motion. Goal: egocentric observation → motion.
- **Phase 4 — Motion-controlled world generation.** Data: NymeriaPlus. Tasks: Motion(+Text)+Initial
  Image → Future Video (+Audio). Goal: motion as a control signal for world gen (the reason to use Cosmos).
- **Phase 5 — Joint omnimodal training.** Mixture (start low on motion→world; it's harder/ill-posed):
  ```
  Text → Motion                 25%    Text + Video → Motion         15%
  Image → Motion                10%    Video + Audio → Motion        10%
  Video → Motion                20%    Motion + Image → Video         5%
  Text + Image → Motion         10%    Motion + Text + Image → Video  3%
                                       Motion + Image → Audio+Video   2%
  ```

## Motion representation
369-d / frame, 30-joint SOMASkeleton30, fps 20 (kimodo BONES-SEED rep):
`[0:3] smooth_root_pos · [3:5] heading(cos,sin) · [5:95] joint_pos(30×3) · [95:275] rot6d(30×6 global) · [275:365] vel(30×3) · [365:369] foot_contacts(4)`.
Normalized via `/weka/jungbin/seed/stats/soma_uniform_motions_20fps/`. Decode → joints via
`motion_decode.py` (verified bit-exact vs kimodo). See `DESIGN.md`.

## Status (current)
- ✅ Motion as a vision-decoupled generation modality on Cosmos-3 Nano (reasoner frozen).
- ✅ **Phase 2 (Text → Motion) running**: full BONES-SEED export (1,076,474 pairs,
  `/weka/jungbin/seed/cosmos_text_motion_full`), **Method 2** (motion heads + LoRA r16 on
  `*_moe_gen`), DDP×8, kimodo loss (per-block smooth-L1 + FK consistency), batch 64.
  TensorBoard at `<run>/tb` (`tensorboard --logdir /weka/jungbin/cosmos_motion_ft_runs`).
- ✅ Sampling/decode/render (`sample_motion.py`).
- ⏳ Known gaps to address: heading augmentation currently **OFF** (kimodo uses on-the-fly Y-rotation);
  fps fixed at 20 (no fps conditioning). Phases 1, 3–5 and Method 3 are future work.

## Key files
- `train_motion_ft.py` — text/…→motion trainer (`--lora` / `--ddp` / `--fsdp`, `--loss kimodo`, TB).
- `motion_decode.py` — pure-torch kimodo FK/decode + kimodo-weighted loss (bit-exact port).
- `sample_motion.py` — flow-matching sampler → decode → render.
- `export_bones_seed_full.py` / `run_shards_node.sh` — sharded (text, motion) export.
- `DESIGN.md` — data contract + pipeline; `SAMPLING_NOTES.md` — sampling ODE/CFG.
</content>
