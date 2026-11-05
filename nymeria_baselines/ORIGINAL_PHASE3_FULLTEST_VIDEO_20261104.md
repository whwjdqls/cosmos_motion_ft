# Original Phase3 full-test video reference

User requested November4 on Foundary sep-2, using idle GPUs and large batches.
Report: https://app.notion.com/p/40ed2a44f44b802eb554ea55a82b4611 .

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_original_p3_m2v_fulltest_20261104T204800Z`.
The adjacent `original_phase3_fulltest_video_current_ops.txt` points here.

Checkpoint: original September two-task Phase3, exactly200k, SHA256
`3dc642508076c2d4199014a8bae82bbec330fc45d8de23cf02eff75089c3bc1d`.
Original Phase1 EMA200k and selected Phase2 200k are preserved. The newer
video-projection Phase1 specialist is not substituted.

Inputs reuse the already verified full-test Nymeria aligned-T97 cohort:
11911 eligible annotation entries,11677 distinct recording/start windows,
all71 held-out recordings,27 explicitly recorded invalid-motion rejections.
This is not the fixed71-window reconstruction benchmark. Duplicate physical
windows with different annotation entries are retained in the primary result.
All input bundle hashes and the checkpoint checksum were reverified locally.

Native UniPC35, video shift10, guidance1;97frames/20fps/256square. Full
GT283-D motion and neutral shape plus exactly one observed image; no semantic
caption. Unknown future video latents are replaced with per-case Gaussian
noise. BF16 sampling disables TF32. Batches8/16/32/64 were benchmarked;
batch64 gives approximately1clip/s per GPU and passed native single-case
sampler parity and packed-batch separation checks. BF16 batch shapes can
change floating-point rounding; bit-exact batch-versus-single inference is
not claimed.

GPU2 runs shard0. GPU4 first completes the additionally requested exact100k
directional/bidirectional V2M comparison, then runs shard1. Each generation
worker subsequently scores its shard. No training source, weights or workers
are changed. Dataset, checkpoint, evaluator and output I/O stays on local NVMe.
Metric followers also score concurrently while generation continues; the
manager's final scoring call waits on the same shard lock and skips completed
outputs. Actual mixed-workload generation speed may be lower than the isolated
batch benchmark. The original native metric kernels are retained for parity.

Metrics: PSNR, SSIM, LPIPS-Alex, DreamSim, I3D-FVD and VideoMAE-v2 CD-FVD.
Only future RGB frames1–96 are scored, using lossless native Wan decode before
MP4 compression. Native resize/pad, scikit-image SSIM and official metric
weights are retained. LPIPS/DreamSim use frame batches128; native FP32 FVD
uses video batches2. The metric runtime is the existing `naeive_eval` env.
It retains the previous evaluator's convolution-TF32 policy (on) with FP32
matmul TF32 off; sampling TF32 remains off. An initial strict GT-feature
parity check caught this policy difference and passed after restoring the
previous evaluator policy. Original GT features may then be reused for the
identical case manifest. DreamSim large-frame-batch parity also passed.
An optional SDPA experiment failed the stricter native-feature tolerance
and was not adopted; production uses the original evaluator kernels.

GT identity references are PSNR100dB cap,SSIM1,LPIPS/DreamSim0 and FVD0.
FVD must be computed over the complete cohort, not averaged across batches.
The prior Naeive Adaptation full-test used30 sampling steps, so this35-step
reference is not a compute-matched architecture-only comparison.

The managers are `gpu2.sh`, `gpu4.sh` and `finalize.sh` in the operation.
Generation and metric outputs resume by case. Trust live progress and exact
coverage checks rather than this launch-time note. `METRICS_COMPLETE.json`
means local metrics/report/archive completed; it does not by itself establish
Notion publication. Require a separately verified publication receipt.

The additional V2M operation is
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_phase3_5050_100k_v2m_20261104T204800Z`.
Both100k checkpoints were hardlinked before retention;100k was the newest
common saved step. It repeats the same71 cases, seeds, shapes, native sampler,
physical/TMR evaluator and video/video+text intervention as the50k report.
Its results append separately to
https://app.notion.com/p/40dd2a44f44b80459383d6df25184a73 .

The100k comparison completed21:11UTC with284 predictions and all physical/TMR
metrics. Its table and899121-byte evidence archive passed remote readback and
downloaded SHA256 verification (`6cac856765ca51bb53b7148cee951b6d3cc5af7fb9597ce2166e5064ca7f8766`).
Video/video+text MPJPE is26.9017/19.5434cm for bidirectional and
32.1353/18.1352cm for directional. This does not mark the full-test M2V task
complete; consult that operation's separate completion/publication markers.
