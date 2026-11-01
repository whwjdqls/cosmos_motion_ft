# Phase2 left/right audit: A2 Kimodo369, step25000

Exact sample: `ti2m_001135_1340b22562f6`, sourceNymeria, seed21310939.
Run: `/scratch2/minsu1206/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/A2_kimodo369_selfcross_gbs256_a100_minsu_20261031`.
Evidence: `/scratch2/whwjdqls99/cosmos/operations/phase2_leftright_audit_20261101`.

## Finding

This example does not show a simple systematic left/right swap. Both GT and
prediction initially rotate left and subsequently right. Prediction starts with
an incorrect body/head relationship and substantially under-rotates later.
Head yaw is atan2(head local+Z.x, head local+Z.z), unwrapped in Y-up world;
positive yaw is a left turn for the canonical +Z-facing SOMA convention.
Body yaw here uses the summed left-minus-right hip/shoulder axes, not a fitted
camera trajectory. Head orientation and body orientation need not coincide.

| Measurement | GT | Generated |
|---|---:|---:|
| Initial body yaw | -69.94 deg | -2.74 deg |
| Initial head yaw | 0.00 deg | 1.80 deg |
| First-half head left peak | 29.96 deg | 34.45 deg |
| Final head yaw | -141.44 deg | -10.69 deg |
| Head turn from that peak to end | -171.40 deg | -45.14 deg |
| Initial-to-first body left turn | +24.40 deg | +47.37 deg |
| Final body yaw | -135.10 deg | +12.18 deg |

The GT walks largely toward initial camera-right while looking initially into
the living area; the model starts approximately facing its initial head and
walks forward/left. A canonical head direction does not imply canonical body
facing. This is an example-level prediction error, not proof of a dataset-wide
label reversal or a causal diagnosis of model training.

## Alignment checks

- Unique source annotation: S02/20230725_s1_julie_taylor_act4_iemk0l,
  frames2330:2430, of which TI2M uses2330:2427 (97 frames).
- Source caption uses standalone C; policy replaces only C with The camera
  wearer. No left/right text replacement or anatomical joint swapping found.
- The first RGB source frame after the same resize matches the fixed TI2M
  conditioning tensor exactly (maximum pixel difference0).
- Saved GT normalized features equal the frozen fixed sample exactly. Both GT
  and generated mean/std reconstruction errors are0. Standard deviations positive.
- Kimodo369 direct joint decode matches saved joints within1.8e-7m. The source
  camera/head-aligned UniEgo window matches saved GT positions to0.0001423m
  after the expected constant vertical grounding shift; no horizontal reflection
  or fitted horizontal rotation is needed. Source-to-saved drift includes the
  float32 normalization/rotation accumulation path.
- Renderer complete world-to-plot map is(-X,Z,Y), determinant+1. All joints and
  head vectors use the same map. Camera tracks root translation, not body yaw;
  screen movement alone is not an anatomical-turn label.

## Presentation bug and fix

`render_motion.py` truncated captions at100 characters, hiding the later
"Then, she turns right..." clause in this exact video. The model receives the
full caption. Both single and comparison render paths now wrap the full caption
and reserve title space. Existing predictions/NPZs/MP4s are unchanged.

Rerendered this sample as `sample_25k_full_caption.mp4` and made
`topdown_heading_comparison.mp4` with shared fixed axes, body-facing arrows and
head-direction arrows. `trajectory_headings.png` shows measured paths/yaw.
Regression check confirms the full later-turn clause survives the display helper;
actual rerendered first frame was inspected. Production source fingerprints may
notice this renderer-only change on a future resume; do not bypass their guards.

Compute: short diagnostic allocations2655586/2655588 on node24. The first
source-lookup import lacked COSMOS_SITE and was corrected; its earlier computed
numbers were independently rerun successfully. Redundant CPU-only pending job
2655576 was cancelled. No pre-existing allocation or training run was changed.

Published and remotely verified: https://app.notion.com/p/binne/409d2a44f44b81ff9d2cd1efaedd666e
Notion tables read back; one plot, two diagnostic videos and evidence ZIP
remotely SHA256 verified (NOTION_COMPLETE.json,4assets).
