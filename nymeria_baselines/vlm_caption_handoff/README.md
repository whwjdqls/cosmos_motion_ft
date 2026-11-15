# Evaluated random20 captions and independent text guidance

The selected full-set generated-caption recipe is **Q40 / video CFG 1.5 /
text CFG 0.5** with the supplied-shape directional W1/no-camera EMA200k model.
Random20 means **1,420 windows: 20 per recording across 71 recordings**.
The caption model is Qwen3.8-27B, revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, with 32 full-frame 512px inputs.
Canonical captions were generated on A100 with batch 4; fresh PRO-B1
recaptions and Q77 are different experiments.

`Q40_prompt.txt` is the exact captioning instruction, including the three
unrelated training style examples. The per-window **generated captions** are
different strings and are delivered in the data bundle, not committed here.

## Export the actual evaluated captions

The export checks the exact frozen cohort and prompt hashes, model revision,
all 1,420 original caption JSON hashes against the accepted winning prediction
receipts, and GT-free VLM input flags. It preserves decoded strings byte-for-byte
in UTF-8, including any historical formatting. It does not rerun the VLM or
choose captions/guidance per case.

```bash
python -m nymeria_baselines.vlm_caption_handoff export \
  --recipe /scratch2/minsu1206/cosmos/operations/vlm_caption_final_audit_20261115/BEST_GLOBAL_RECIPE_REPRODUCIBLE_AND_SOURCE_BOUND.json \
  --output-dir /PATH/TO/NEW/random20_best_q40_v1p5_t0p5_20261115
python -m nymeria_baselines.vlm_caption_handoff verify \
  --bundle-dir /PATH/TO/NEW/random20_best_q40_v1p5_t0p5_20261115
```

An existing output/archive is never overwritten. `--cases`, `--captions-root`
and `--receipts` support relocated source copies while enforcing the same
accepted hashes. The export requires access to the prompt/result paths named
in the recipe; downloading/verifying the finished bundle does not require any
model environment. The exporter/verifier uses only Python's standard library.

The deterministic ZIP contains `captions_by_id.json`, ordered caption JSON/CSV,
window bounds, native and VLM seeds, exact prompt, portable recipe, measured
generated/GT/control metrics, provenance and checksums. It excludes target GT
captions, motion, images, checkpoints, raw model reasoning and credentials.

## Native inference adapter

`modality_cfg.py` and `motion_text.py` are byte-identical copies of the adapters
used by the completed native full1420 evaluation. `SOURCE_PROVENANCE.json`
records their original hashes. They alter inference routing, not training or
checkpoint weights. Shape remains supplied in every branch.

For frozen model predictions `p(video,text)`, first apply video CFG separately
with and without text, then interpolate those results using text CFG:

```text
g_text = p(no_video, text) + video_CFG * (p(video, text) - p(no_video, text))
g_null = p(no_video, no_text) + video_CFG * (p(video, no_text) - p(no_video, no_text))
motion = g_null + text_CFG * (g_text - g_null)
```

Use `MotionTextScope(model.motion)` and `ModalityGuided(model, scope, 1.5, 0.5)`
inside `scope.use(generated_captions)`. The sampler's ordinary `captions` remain
empty: text is injected only into the motion expert. Call `assert_counts` for
the native 35 calls, then `close()` on both wrappers to restore hooks. This is
an integration adapter for the pinned model/source snapshot, not a standalone
sampler or evidence that another environment has passed native GPU parity.

The baseline sampler, shifts, 97-frame/20fps input, 256px motion-evaluation
video, normalization, decoding, seeds and metrics stay unchanged.

## Upload from the Drive-owning account

The prepared Yonsei handoff is documented in
[`migration/RANDOM20_BEST_GENERATED_CAPTIONS_20261115.md`](../../migration/RANDOM20_BEST_GENERATED_CAPTIONS_20261115.md).
`upload_drive.sh` verifies its local checksums and 1,420-case coverage before
copying with the operator's existing rclone login, then downloads remote content
for comparison and records a receipt. It does not change Drive sharing.
An archive prepared locally is not an upload-completion claim.

## Scope of the result

Native full1420 MPJPE: selected generated text **14.627242 cm**, video-only
**16.125472**, fixed GT text CFG1 **13.073182**, matching GT text CFG0.5
**13.505623**. These are benchmark-development results with repeated GT/RGB/
caption/metric inspection, not held-out generalization or a universal optimum.
FID still favors video-only; penetration is worse than generated text CFG1.
All ten metrics and actual GT are retained in `recipe.json` and the
[experiment report](https://app.notion.com/p/415d2a44f44b80b2b687dd273fa22f36).
