# Camera video-weight-1 evaluation: joint 40k and no-joint 30k

Completed and published on 2026-11-04T22:51:19.461126+00:00. [Quantitative results on Notion](https://app.notion.com/p/40dd2a44f44b80459383d6df25184a73).

Both checkpoints were the latest saved snapshots when selected. These are **40k and 30k**, not 50k. The unequal budgets prevent a causal conclusion about joint training. Four training runs continued throughout. Evaluation temporarily shared our training GPUs 1 and 5 and has now released all GPU memory.

| Model | Condition | MPJPE cm | FID | R@1 % |
| --- | --- | ---: | ---: | ---: |
| Camera joint, video weight1 (40k): video | video | 23.2132 | 0.290691 | 20.83 |
| Camera joint, video weight1 (40k): video_text | video_text | 17.8636 | 0.127015 | 44.79 |
| Camera no joint, video weight1 (30k): video | video | 27.3021 | 0.338681 | 22.92 |
| Camera no joint, video weight1 (30k): video_text | video_text | 19.1894 | 0.148580 | 43.23 |

Same original 71 held-out 97-frame recording windows, same seeds and neutral shapes; video-only and video+text, guidance 1, UniPC 35, shift 3, 20 fps. All 284 predictions completed with zero skipped cases. No GT motion or camera conditioning. Text is an evaluation-only intervention through both frozen reasoners. Camera tokens are generated from noise; camera accuracy is not measured.

Matched Nymeria TMR evaluator SHA256 `91f9d8adec6bcaa448571dade670da0f94916e448b875a0853eda29ef44a1e21`, original text rows, original normalization, train-only Body23→SOMA30 completion, 20→30 fps conversion. Fresh GT retrieval/completion parity passed separately for both models. Retrieval uses 32 candidates over three grouping seeds (64 cases per grouping); FID and physical metrics use all 71.

At matched step 40k, joint video weight 1 improves MPJPE versus the already evaluated video weight 10 checkpoint by 6.8580 cm for video-only (paired 95% CI −9.7415 to −4.1652) and 4.6977 cm for video+text (−7.8540 to −2.2808). This is one training seed; paired intervals do not cover training-replica uncertainty.

## Checkpoints and evidence

- joint_w1: step 40,000; `/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_wv1_latest_matched71_20261104T2146Z/checkpoints/joint_w1/step_000040000.pt`; SHA256 `f84b33a32883136fad6663eddae6ee77d4b211cdd90910d590b5380836d5c930`.
- nojoint_w1: step 30,000; `/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_wv1_latest_matched71_20261104T2146Z/checkpoints/nojoint_w1/step_000030000.pt`; SHA256 `dd17654d02eff13821dfc7fb282fe6f1728efae8be6a94366b4689fa5ad4f894`.

Node-local sep-1 operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_wv1_latest_matched71_20261104T2146Z`.
Evidence ZIP SHA256: `2ed6c93ddf44b23e888b0211416d71dc212cd466e6cecc225614592362dc99ae` (973,402 bytes).

Root `COMPLETE.json` is `published_verified`: all saved numeric table rows were read back from Notion, and the uploaded evidence ZIP was downloaded and matched by byte count and SHA256. Per-window physical scores, embeddings, retrieval groups, paired bootstrap contrasts, configurations, exact code and acceptance/provenance receipts are included. The operation retains generated motion and camera NPZs locally. No shared-storage copies were performed.
