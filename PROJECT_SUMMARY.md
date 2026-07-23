# Egocentric Omnimodal World Model on Cosmos-3 Nano (NymeriaPlus)

> Project summary (Notion source). Three-phase curriculum: (1) adapt the generator to the egocentric
> domain + metric camera space, (2) pretrain a motion-native expert against the frozen reasoner,
> (3) bridge the two generative streams with new bidirectional layers for co-generation.

## Preliminaries (why the phasing looks like this)

**Cosmos-3 Nano** is a 16B omnimodal world model (Qwen3-VL-8B backbone, 36 layers, hidden 4096) with a
**dual-pathway Mixture-of-Transformers**: every layer carries two weight sets — the *reasoner/understanding*
pathway (plain names) and the *generation* pathway (`_moe_gen` suffix). There is **no cross-attention
module**: conditioning is one joint self-attention ("two-way": text/condition tokens attend causally among
themselves, generation tokens attend fully over everything). Generation is trained with **rectified flow**
(x_σ = σε + (1−σ)x₀, predict v = ε − x₀); conditioning frames are kept clean via σ=0 condition masks.
Actions are a native modality: per-embodiment `DomainAwareLinear` I/O heads; **camera = `camera_pose`
domain (id 2), 9D = 3D Δpos + 6D Δrot**, relative pseudo-action ΔT_t = T_{t-1}⁻¹T_t, deliberately
**un-normalized** (the representation *is* the normalization). Video = Wan2.2 VAE latents (4×16×16, T=4N+1).

**Data**: NymeriaPlus — 19 subjects / 713 sequences of ego RGB (640², 20 fps), frame-aligned head-camera
SE(3), narration captions → **141,589 usable captioned windows**. Two corrections we established:
- **Camera frame**: stored Aria *device* pose ≠ Cosmos's OpenCV optical convention. Using the VRS
  device→RGB extrinsic (~39° optical tilt) + Rz(−90°) for the upright video raised no-alignment
  direction-cosine vs the pretrained model from **0.51 → 0.70**.
- **Floor**: per-window `ground_offset_y` grounding (+ per-seq floor calibration; ambiguous multi-level
  windows dropped).

**Zero-shot diagnosis (motivates Phase 1's "metric space alignment")**: pretrained Nano inverse dynamics
gets the *pattern* of camera motion right (trans-corr 0.6–0.8) but over-predicts translation
**7–17× (median 8.5×)**. k-sweep smoking gun: its per-action magnitude ≈ our GT displacement over
**~9 frames** — a learned action *temporal step* / units mismatch, **not** fps (sweep 10–160 refutes),
**not** coordinates (‖Δt‖ is frame-invariant), **not** a de-norm bug (output is raw). Forward dynamics
responds correctly only when we scale translations ×8 into its native range. External judge:
**VGGT-Omega nATE 0.032 vs Cosmos 0.057** on the same clips (VGGT = camera pseudo-GT for evals).

---

# Phase 1 — Finetune ONLY the Generator

**Goal**: domain adaptation (egocentric indoor, 20 fps) + **metric-space alignment** (teach the pretrained
`camera_pose` head our per-frame metric deltas — fixes the 8.5× / 9-frame-step mismatch above).

**Tasks (mixture 40 / 25 / 20 / 15, one stream per packed batch):**
- Forward dynamics — text (10% drop) + image + camera → video *(frame-0 + all actions clean)*
- Inverse dynamics — video → camera *(all frames clean, actions denoised; no instruction text —
  deliberate deviation from native, which keeps captions for all modes)*
- Policy — text (10% drop) + image → video + camera *(frame-0 clean, both denoised)*
- **image2video (15%)** — pure i2v retained as an anti-forgetting stream

**Setup (native-faithful):**
- **Native Cosmos trainer stack** (Hydra experiment + TOML + torchrun), with exactly one override:
  `get_data_and_condition` consumes **precomputed Wan-VAE latents** (127,956 windows, T=97 @ 256², fp16,
  byte-identical preprocessing incl. reflection-pad latent crop) → no VAE in the training step. All native
  noising/packing/losses/sampling contracts untouched.
- **Trainable**: LoRA (r=16, α=32) on `q/k/v/o_proj_moe_gen` **only** (generation pathway) + the
  **pretrained** camera action heads (`action2llm/llm2action/action_modality_embed` — kept, *not*
  re-initialized, since `camera_pose` is a pretrained domain) ≈ 32M params. Reasoner + embeddings frozen.
- **Current native recipe**: FusedAdamW LoRA lr `5e-5`, camera-interface LR
  `2e-4` through a 4x parameter-group multiplier, wd 0.05, LambdaLinear (factor
  0.4 for the first 500 steps then linear decay to 0 at 100k), clip 1.0, bf16,
  pure-replicate ×8, full activation checkpointing, and `raw_action_dim=9`
  masking (pad to 64). The original native baseline uses action loss ×10. The
  2026-07-21 video-quality A-E suite deliberately uses action loss ×2 and is an
  ablation, not the universal Phase-1 default.
- Engineering note: audit found a **finite-stream dataloader livelock** (native joint loader assumes
  infinite streams; training would silently freeze at ~1.1k/100k iters) — fixed with a cycling wrapper +
  `set_epoch`, verified by forced-exhaustion smoke + inference from the smoke checkpoint.
- Visual-quality note: historical MP4s and recent outputs were not initially
  resolution/provenance matched. The durable investigation, fp16 cache audit,
  shift-10 result, ranked hypotheses, and required controlled comparisons are
  in `native_phase_training/PHASE1_VISUAL_QUALITY_AUDIT.md`.

## Results
Forward dynamics — `t4_S01_20230607_s1_barbara_fd.mp4`
Policy — `t4_S01_20230607_s1_barbara_policy.mp4`
Inverse dynamics — camera trajectory visualizations
`invdyn_metric_montage.png` (+ images)

*Suggested metrics to report alongside the visuals*: per-frame ‖Δt‖ ratio vs GT (zero-shot baseline: 8.5×;
target ≈1×), no-align direction cosine (baseline 0.70), Umeyama ATE vs GT, and vs VGGT-Omega (0.032) as
the feed-forward reference.

---

# Phase 2 — Finetune ONLY the Motion Expert

**Goal**: motion-expert pretraining — text–motion alignment against the **frozen** reasoner, with zero risk
to the generator.

**Motion representation — `motion_uniego`** (283-D SOMA-30 UniEgoMotion, 20 fps): `[0:270]` 30 joints ×
(6D rot + 3D trans, translation = joint position — decode needs no FK); `[270:279]` head-canonical residual
frame (frame-0 canonicalized to identity per window); `[279:283]` foot contacts. **Size-aware**: per-actor
`neutral_joints (30,3)` → ShapeEncoder → **in-context shape token** (bone lengths conditioned, not baked
in). Floor-grounded per window (feet → y≈0; verified −2.3 m → −0.10 m).

**Architecture** (small motion-native transformer, ~77M): self-attention over `[shape_tok, motion tokens]`;
**one-way cross-attention to reasoner hidden states H_R** — critically, H_R = *post-transformer*
`reasoner_forward` output ([B,T,4096], semantic; `_encode_text` is just the embedding lookup and is NOT
sufficient); cross-attn K/V projections do the 4096→d adaptation directly (no separate resampler);
**AdaLN-zero** flow-time conditioning; **x₀-prediction** (velocity prediction compounded errors through the
residual canonical frame → spinning); losses = feature MSE + decoded centroid-relative pose +
joint-velocity smoothness (via a bit-exact differentiable decoder) — this fixed a 30× temporal-jitter
failure (root-step 0.37 → 0.015 = GT level).

**Tasks:**
- text → motion — text (10% drop) → motion *(CFG null = cached empty-prompt H_R)*
- image → motion — text (10% drop) + image → motion
  - (리즈너에 이미지 넣는게 제일 옵티멀 하지 않을 수도 있음) — two candidate routes to compare:
    **(a)** frozen ViT → reasoner multimodal prefill so image semantics land inside H_R (the framework's
    reasoner-only path supports image `pixel_values`; video only as frames-as-images); **(b)** bypass the
    reasoner: inject the image as VAE-latent tokens through a second cross-attn KV stream into the expert.
    (a) is architecturally uniform; (b) preserves pixel-level detail the reasoner may compress away.

**Validated**: the frozen reasoner *does* provide useful conditioning — cond-vs-null (H_R ablation) steers
motion semantically (e.g., "waves right hand" shows the largest pose divergence from the unconditional).
Infra: **H_R cache** (47 GB fp16, 117,702 captions) removes the 16B reasoner from the loop → ~30×
per-sample speedup (75 ms → 2.5 ms).

---

# Phase 3 — Modality Bridge

**Requirement**: co-generation of **motion + video + camera from text** requires the two *generative*
streams to condition each other during joint denoising — generator ⇄ motion expert, both directions.
One-directional adapters (cross-attn layers, AdaLN-style modulation) only give X→Y; bidirectional coupling
with symmetric information flow is exactly what **shared self-attention** provides. So the bridge = new
self-attention layers over `[generator tokens; motion-expert tokens]`.

**Resulting topology**: reasoner ⇄ generator (pretrained, Phase 1-adapted), reasoner ⇄ motion expert
(Phase 2-trained), + **new bridge: generator ⇄ motion expert**.

**Why not one shared 3-way attention (reasoner ⇄ generator ⇄ motion expert)?**
1. The reasoner⇄generator joint attention is **pre-trained, mid-trained, and finetuned as a calibrated
   system**. Injecting untrained motion K/V into those layers shifts the attention distributions of *every
   pretrained head* — the softmax now normalizes over new, initially-meaningless keys — risking degradation
   of exactly the capabilities Phases 1–2 secured.
2. This is a **falsifiable design choice**: ablation = 3-way joint attention vs. bridged attention;
   hypothesis = 3-way lowers video/camera metrics (and possibly motion metrics) relative to the bridge.
3. Mitigation built into the bridge design: **zero-gated insertion** (AdaLN-zero-style gates / zero-init
   output projections) → the bridge is an *identity at init*, so Phase-1 and Phase-2 behavior is exactly
   preserved at step 0 and coupling is learned gradually. (This mirrors how Cosmos itself adds modalities —
   thin I/O heads + shared attention — but confines the new K/V to *new* layers instead of the pretrained
   ones.)

**Alignment details to handle**: motion runs at 20 fps while video latents run at 5 latent-fps (4× temporal
compression) → align motion frame *i* ↔ latent frame *i//4* on Cosmos's unified 3D-MRoPE temporal axis with
a modality offset (the same mechanism Cosmos uses for action↔video supertokens).

**Unique evaluation this unlocks — cross-modal consistency**: the head trajectory is *derivable from
generated motion* (head joint), *commanded by generated camera actions*, and *observable in the generated
ego-video* (VGGT-Omega as judge). Agreement between the three is a consistency metric no single-modality
baseline has.

---

## Recurring design principles (thread through all phases)
1. **Freeze what is pretrained; add small trainable modules** (LoRA / heads / expert / zero-init bridge).
2. **Keep native contracts** — Cosmos-exact action representation (un-normalized 9D deltas), native trainer
   stack, native sequence plans.
3. **Precompute what is frozen** — Wan-VAE latents (128k windows), reasoner H_R (47 GB): both caches are
   exact because their producers are frozen.
4. **Every phase has a falsifiable ablation** — Phase 2: cond vs null-H_R; Phase 3: bridge vs 3-way joint
   attention.

## Known deviations from native (documented, deliberate)
| item | choice | native |
|---|---|---|
| latent cache dtype | fp16 | fp32 encode at train time |
| LoRA trainable set | excludes `time_embedder` | full-FT recipes include it |
| camera action step | per-frame (k=1) @ 20 fps, metric | learned coarser step (~9 frames) — the thing Phase 1 fixes |
