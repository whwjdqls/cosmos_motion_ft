# One-frame LoRA + video projections, final200k evaluation

Requested Notion destination:
https://app.notion.com/p/408d2a44f44b80e882cfcc8d8f4b7869

Operation on sep-2:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_phase1_video_io200k_20261101T035914Z`.
Pointer: `operations/eval_phase1_video_io200k_current_ops.txt`.

The final `global_lora_video_io` checkpoint is
`edge_phase1_T97_20fps_2gpu_bs16_lora_video_io_exclude5_fd_invdyn_no_text_p1_200k_sep2_20261028/checkpoints/iter_000200000`
under the node-local `cosmos_storage/runs/cosmos3_camera/camera_world_edge` root.
Its model metadata SHA256 is
`b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`.
The DCP includes FP32 EMA weights and biases for both `vae2llm` and `llm2vae`;
all metadata storage extents were checked before inference.

## Matched benchmark

- The same71 fixed held-out T97 windows as the destination page, one per
  recording. This is not the full annotation-window test cohort.
- One initial RGB frame,20fps,256×256, native EMA UniPC30/shift10,seed0.
- Empty captions. Forward camera CFG2 / native text CFG1; inverse native
  guidance1. The camera guidance source matches the published baseline hash.
- Official native inference resolves the saved adaptation contract. Diffusion
  cache, text-KV reuse, compile and CUDA graphs are disabled.
- GPU2 runs forward inference; GPU3 runs inverse inference and then scoring.
  Other jobs are preserved. All data, caches and outputs are node-local.

The published quantitative ZIP was downloaded from the exact destination page.
All71 baseline case files match its configuration SHA256
`006fa35e0eaea6446fdd766f682be3034bc774f463d1d8f2b85604614fc69bc6`.
Every local GT video and camera action JSON matches the published hash.
First-frame PNGs are reconstructed from those clips. The six qualitative
reference PNGs have identical RGB pixels; different PNG metadata does not imply
different conditioning. The historical71-case PNG byte hashes were not bundled,
so independent exact-file parity with that old run is not claimed.

## Metrics and publication

Forward PSNR/SSIM/LPIPS-Alex score only generated frames1–96, retaining per-frame,
horizon and per-case results. Additional DreamSim, I3D400 FVD and FP32 VideoMAE-v2
CD-FVD are measured for the new checkpoint; old-baseline values for those metrics
are unavailable. I3D FVD and CD-FVD are labeled separately. GT self-reference is
measured, with the existing finite100dB PSNR convention and explicit zero
distribution-distance reference.

Inverse metrics retain rotation error, translation direction cosine, scale and
path-length ratios, scale-matched translation error and Sim3-aligned ATE. GT is
recomputed and must match all71 published GT cases within1e-8, including the
nonzero regularized rotation residual. No raw metric trajectory-error claim.
Paired changes against the published baselines use5000 recording bootstrap
samples, seed20261029. They do not capture training-replica or generation-seed
uncertainty.

Six fixed qualitative indices0,7,14,28,42,63 show GT / standard two-task200k /
two-task200k with video projections. Their reference predictions are the prior
audited Foundary samples from the exact two-task200k checkpoint, with matching
RGB pixels, seed, sampler and guidance. They are illustrative; published scalar
baseline values retain their historical source. Two inverse plots show GT and
the new aligned prediction. Frame counts and green/red conditioning borders are
audited on decoded publication media.

`METRICS_COMPLETE.json`, `MEDIA_COMPLETE.json`, `publication.json`, remote SHA256
receipts and final `COMPLETE.json` distinguish completed stages. Live logs and
markers establish actual progress; this implementation note is not a completion
claim. `report.md` and the quantitative ZIP retain exact settings and evidence.

## Completed measurements (2026-11-01 UTC)

Both native inference managers exited 0. All 71 forward and inverse cases were
scored with no skips; GT parity and all 71 camera-CFG branch audits passed.
Forward PSNR/SSIM/LPIPS-Alex are 19.847175 dB / 0.620123 / 0.275601, versus
19.750820 / 0.618447 / 0.278854 for standard two-task 200k. The paired PSNR and
LPIPS intervals favor the new checkpoint; the SSIM interval crosses zero.
Inverse rotation is 0.304028 degrees, direction cosine 0.793551, scale-matched
translation 4.933739 mm and Sim3-aligned ATE 2.917261 cm. Their paired changes
favor the new checkpoint; scale/path-length ratio changes remain inconclusive.

Supplementary DreamSim is 0.173211, I3D FVD 312.848708 and VideoMAE-v2 CD-FVD
287.678816. Historical baselines for these extra metrics are unavailable.
The six videos and two inverse figures have already passed remote SHA256
verification. Final archive publication is recorded separately in the operation
receipts and is complete only when `COMPLETE.json` exists.

Publication completed at 2026-11-01 04:22:14 UTC on the requested Notion page.
All nine new files (six MP4s, two PNGs, one quantitative ZIP) were downloaded
from Notion and matched their local SHA256 and byte sizes. All 19 pre-existing
file references remain present. `publication.json=complete_verified` and
`COMPLETE.json` record the final verified result. The archive SHA256 is
`0e7742fa9a559d155bd43f65a88a4bef3aa09f6158bc046a27413894a646914f`.
