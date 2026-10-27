# BONES-only in-context LLM2Vec POC on Yonsei

Fresh reproduction of the full-contact Variant B recipe in `BONES_SEED_POC.md`.
The original checkpoint was not found in the local restored run roots.

## Launch

Run `COSMOS_SITE=yonsei bash motion_expert/train_bs_full_contact_site.sh`
inside an owned, idle single-4090 allocation. Initial allocation: 2275411,
node36, 8 CPUs. The launcher keeps the allocation alive on completion and
uses an allocation lock. It does not interfere with the Nymeria evaluator.
`COSMOS_SITE=grasp` selects Grasp data/output roots; `KIMODO_ROOT`,
`BS_RUN_ROOT`, `BS_POC_RUN_NAME`, and `BS_POC_TEXT_CACHE` can override defaults.

Yonsei output:
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_poc/bs_native_full_contact_llm2vec_w1_1_5_c0p05_v1_h10_yonsei_20260919`

## Unchanged learning recipe

- In-context pooled LLM2Vec text token plus centered actor shape token.
- Width 512, 8 self-attention blocks, 8 heads, FFN 2048; no Cosmos reasoner.
- BONES original proportional UniEgo283, 20 fps, frame-0 canonicalization.
- BONES proportional mean/std, with hashes recorded by the existing loader.
- Equal natural/single/multi source mixture and the original window sampler.
- Up to 200 frames, variable lengths with padding masks; batch 128, seed 0.
- Native shifted x0 flow, shift 3, 1000 timesteps, CFG text dropout 0.1.
- AdamW 2e-4, betas 0.9/0.99, no weight decay, 1000 warmup, cosine to zero.
- Feature/centroid-relative joint/joint-velocity losses 1/1/5.
- Contact BCE 0.05, horizontal foot velocity 1, foot height 10, logit scale 2.
- Gradient clip 1, FP32 training, 200k updates, checkpoints every 10k.

## Preparation and explicit differences

The training LLM2Vec cache was not found locally. Preparation builds the
original three-pool index and encodes its exact raw captions, plus held-out
visualization captions, on the assigned GPU. It uses the existing C45-compatible
LLM2Vec encoder and its built-in comparison against the benchmark cache.
No caption subject rewriting or Nymeria data is introduced. The pipeline
automatically starts training only after the complete cache is available.
Read `pipeline_stage.txt` and `pipeline_job2275411.log` for actual progress;
launching preparation does not mean optimizer updates have begun.

This is a fresh recipe reproduction, not a bitwise historical replay: text
embeddings are rebuilt, hardware/software differ, and checkpoint labels count
completed updates rather than historical preincrement labels. Historical global
DataLoader RNG behavior is explicitly selected. Inline C45 evaluation is disabled;
evaluate checkpoints offline with both generated and GT metrics. Original POC
checkpoints remain model-only, without optimizer/RNG state, so a restart needs an
explicit warm-start decision rather than claiming exact resume.

## Phase2-aligned visualization

Every 10k updates, render four fixed BONES held-out natural-caption cases from
the content test split, up to 200 frames, seed 0, UniPC35, CFG 2, shift 3.
The old small split was not restored, so these are not claimed to be the same
historical four examples or the Edge fixed cohort.

The default tracking-camera renderer is shared with current Phase2:
GT blue on the left, generated red on the right, enlarged/thinner skeleton,
first-frame head axes from decoded feature rotations, above-floor bones drawn
above the floor, shaded only below-floor fragments, and no head-angle numbers.
No floor snapping or recanonicalization of generated outputs is performed.
Legacy explicitly fixed-camera clients retain their former renderer.

Outputs: `visualizations/step_000010000/` and subsequent 10k steps, containing
MP4s, raw generated/GT feature and joint NPZs, and a caption/seed/source manifest.
The model, loss, and shared Phase2 renderer implementations are unchanged.
