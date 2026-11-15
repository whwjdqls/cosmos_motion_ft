# Nymeria random20 v1 evaluation handoff

User chooses20 random physical windows per each of71 test recordings, without
replacement:1420 total, sampled from11677 accepted physical windows. The parent
eligibility set and exact indices are pinned before model generation. No first
window is forced; the previous first/spread20 candidates are superseded.

Authoritative repository metadata: `artifacts/nymeria_random20_v1/`.
Manifest SHA256:`47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df`.
Shared: `/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/evaluation/nymeria_random20_v1/`.
Yonsei/private existing Drive handoff: https://drive.google.com/file/d/150K2s5UhFJtB3M75uawXoV1z6wFIGPI-/view.
Archive338928bytes; remote MD5/SHA256 and read-only account access verified.

Run `python verify_selection.py` after copying. Use UUID and20FPS start as the
physical key, and preserve the caption and sampling seed from the manifest.
Start/end indices refer to the resampled20FPS timeline, not video packets.
The README documents selection, indices, preprocessing, metrics and batch layout.

Future offline defaults, explicitly requested now: whole-bridge reference,
M2V CFG1.1; V2M and video+text→motion CFG1.5. Values are prespecified using the
prior directional W1 EMA150k fixed71 sweep. Image-metric and temporal-error
tradeoffs are recorded; no single universally best CFG or model is claimed.
Source settings file: `artifacts/nymeria_random20_v1/evaluation_defaults.json`.
Existing training caption/loss/periodic-monitor contracts are unchanged.

Current sep-1 evaluation: **directional W1 EMA200k, no camera prediction**.
Checkpoint SHA256:`c180e9d595b4971158177574c96bb22816ae93e400a6869e39d767a01ea77bf5`.
Operation:`/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_dirw1_200k_random20_20261110`.
Uses initially empty GPUs4–7, four355-window shards,3tasks/4260 predictions.
The two camera training ranks on2/3 remain active and untouched.

Inputs are prepared and scored in64-case chunks. Native frozen-source sampler,
35UniPC steps, video/motion shifts10/3,97frames/20FPS/256px, supplied shape,
M2V32/V2M64 sampling, decode4. All generator captions empty; only explicit
video+text inference sends the supplied text to the frozen motion expert.

Scoring retains per-window PSNR/SSIM/LPIPS/DreamSim and native I3D features,
physical/UPE/LPE/sJPE measures and matched91f9d8 TMR embeddings. Final FID/FVD
and R32 retrieval are recomputed over the complete subset. R32 scores1408cases
per grouping and records12 dropped IDs; physical/FID metrics use all1420.
Recording-block intervals preserve all20 windows when resampling71 recordings.
Native sampler parity is checked separately on each worker; all four passed.

Status: `RUNNING.json`, `shards/*/status.json`, per-worker logs. `ERROR.json`
records failures. `COMPLETE.json` is written only after all355-case shards and
full global aggregation/output audit pass. Final artifacts: `summary.json`,
`metrics.csv`, `report.md`, `quantitative_evidence.zip`; per-case metric arrays
and embeddings remain available. No final metric is claimed from setup alone.
