# Public Drive handoff: directional temporal-local W1 no-camera EMA100k

Completed 2026-11-09T00:03:37.446548+00:00. User explicitly requests publishing this selected100k
model with anyone-with-link download access for a collaborator's motion demo.

- Model: `sep1_dir5050_w1_EMA0`, W&B p3-d2d167e7b78773091498.
- Directional temporal-local MMDiT, videoW1, no camera prediction, no captions,
  V2M50/M2V50/joint0, EMA0.999 from step0 (100000 updates), no extra Phase3 LoRA.
- Folder: https://drive.google.com/drive/folders/1d16BDPRcmyJJHP5d5SofLFuLYzOVvvo-
- EMA inference checkpoint: https://drive.google.com/file/d/1Inau9dZN5Lfl9cdlAlCv6C1PbIxEqoM-/view
- Original full recovery checkpoint: https://drive.google.com/file/d/15JfopTKtxCGSbYgiEo1jUS_rRsQXwtqV/view

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| EMA inference | 939800605 | `101d3e4c0ede32f0603bd66480683f39747c4f937bdb19a460d6d9803fac14b9` |
| Original full100k | 3759187999 | `65a04432e1b8ad91e9aa6b6130dfc856777e68aae6a5185f5d7b19028b1072ec` |

The inference export contains the exact98 EMA bridge tensors (939800605 bytes),
with both top-level and EMA bridge slots referencing EMA tensors for sampling
compatibility. Optimizer/RNG omitted; do not resume training from that file.
The original full100k checkpoint is uploaded byte-for-byte unchanged and retains
raw/EMA/AdamW/scheduler/two-rank RNG; select `--weights ema` for inference.

Both checkpoints and README passed anonymous HTTP206 range downloads (first1024
bytes matched local content) using a fresh requests session with no credentials,
netrc authentication or cookies. Every uploaded file passed remote size/MD5
comparison. Folder permission was read back as type=anyone, role=reader,
allowFileDiscovery=false. Existing parent/dependency sharing was not changed.

Source checkpoint SHA verified; all98 exported EMA tensors matched exactly on
CPU roundtrip; all seven actual234938382-parameter bridge modules loaded strictly
on CPU. No new GPU sampling test or collaborator-environment acceptance claimed.

Public folder includes README, demo_motion.py, exact385-file source snapshot,
model/provenance contracts, normalization vectors and checksums. Sampling helper
uses caption-free V2M, native35-step UniPC and optional whole-bridge-off guidance.
Output is normalized97x283 motion features requiring the matching motion decoder.

Frozen Phase1 video-I/O EMA200k (metadata b2b3374b...) and selected Phase2 200k,
native Edge base/tokenizer/framework, matching motion assets and Wan VAE/video
preprocessing remain separate requirements. Their access permissions are unchanged;
the Phase3 checkpoint is not a standalone complete frozen-specialist bundle.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/upload_phase3_directional_w1_nocamera_100k_ema_20261108T234953Z`.
COMPLETE, UPLOAD_VERIFIED, PUBLIC_ACCESS_VERIFIED, PACKAGE_VERIFIED and manifests
record transfer checks, exact files, public permissions and anonymous access.
No live training, original protected checkpoints or shared storage was modified.
