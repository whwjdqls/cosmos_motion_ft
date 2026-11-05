# Standard Phase1 EMA200k camera→video on exact fixed71 (sep-2)

Completed and published on 2026-11-05. All71 exact windows scored, zero skipped;
32 new native clips plus39 reused predictions. The Notion table and all seven
uploaded files passed readback and downloaded-SHA256 verification. Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_phase1_video_io200k_camera2video_matched71_20261105T081600Z`.

Destination: https://app.notion.com/p/40fd2a44f44b80489a0bdecc5d9410bb

- Standard global_lora_video_io Phase1 EMA200k, prefix1, camera action heads,
  trainable video I/O, frozen VAE. Native UniPC30/shift10, camera CFG2, text1,
  empty captions, seed0, 97 frames/20fps/256.
- Exact current Phase3 matched71 recording/start list, not the old Phase1
  one-window-per-recording list (only39 physical windows overlap).
- Reused39 existing generated MP4s; generated32 missing windows. All71 were
  freshly scored against the fixed-bundle lossless RGB, frames1–96, with PSNR,
  SSIM, LPIPS-Alex, DreamSim, I3D400-FVD and VideoMAE1408 CD-FVD.
- Prediction MP4 decoding is retained consistently for all71; cached initial
  RGB came from an encoded GT clip, unlike the32 new lossless initial RGBs.
  This and CFG/solver controls distinguish this baseline from recent lossless
  pre-MP4 Phase3/Naeive scoring. Cases record all provenance/input differences.
- Scripts/preparation verified, inputs local, checkpoint and camera-CFG hashes
  checked. User explicitly approved sharing own GPU3 with trainer PID3620615;
  fresh ownership/headroom admission passed. Generation and scoring ran
  sequentially on GPU3. The evaluator exited0 and released its GPU memory.
- METRICS_COMPLETE.json establishes local completion; COMPLETE.json and
  publication.json record remote table and attachment download SHA checks.
  The evidence ZIP contains configuration, exact case map, per-case scores,
  FVD features and evaluation scripts. Weights, videos and other large artifacts
  remain outside Git.

## Completed and published

All71 exact windows scored, zero skipped;32 new native clips plus39 reused
predictions. The evaluator exited0 and released GPU3. Native camera-CFG
audits passed for every new sample. Coverage, six qualitative clips and ZIP
integrity passed; the Notion table and all seven uploaded files were read back
and downloaded-SHA256 verified.

| Metric | Standard Phase1 EMA200k |
|---|---:|
| PSNR dB |21.4995436174489|
| SSIM |0.6641601211994782|
| LPIPS-Alex |0.2162697606634222|
| DreamSim |0.13621237004796666|
| I3D400-FVD |286.67786033205397|
| VideoMAE-v2 CD-FVD |320.36747503640254|

Evidence ZIP SHA256:
`2ad2947f94f7282d41196d5d4ffe6abe7516430e10905ea082854d2f32d1f0a6`.

The codec/cached-initial-image and sampler controls documented above remain
part of the interpretation; no architecture-only winner is inferred.

A first-case GT feature diagnostic is saved locally in
`GT_feature_reference_diagnostics.json`: current native feature batch1 versus
recent Phase3 batch2 gives bit-identical VideoMAE GT features, while I3D has
relativeL2 0.00024538. Both current GT and generated features use the same
native batch1 policy; no cached GT features from a different policy are mixed.
