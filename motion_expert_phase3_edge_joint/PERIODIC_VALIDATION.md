# Grouped BF16 training and periodic fixed71 monitoring

The final user-selected run uses `motion_expert_phase3_edge_mmdit.launch`:
**directional conditioning,50% V2M /50% M2V, no joint objective**, on GPUs0/4.
It retains clean-motion E(0), video weight10 and the optimization settings below.
The image-joint launcher described below remains a separate supported recipe.

The new `motion_expert_phase3_edge_image_joint.launch_mmdit` entry point selects
`--bridge-parameterization mmdit --bridge-attention-backend grouped
--bridge-precision bf16_bridge`. This is option 2 from the precision trial:
BF16 Q/K/V/O and attention matrix multiplication, FP32 normalization/softmax,
FP32 master parameters, gradients and Adam state. The extra Phase3 LoRA and EMA
remain off by default. Frozen Phase1 LoRA/video I/O and the Phase2 expert stay
frozen. Shape remains excluded from bridge attention.

The image-joint recipe stays 40% V2M / 40% M2V / 20% joint, bidirectional,
clean motion E(0), video weight10, global batch32, bridge LR2e-4,
1000-step warmup then cosine through200k, AdamW(.9,.99), WD0, gradient clip1.
Existing processes and their source snapshots are unaffected.

## Periodic evaluation

Supply `--validation-config /mnt/local/.../validation.json`. The new launcher
requires it; older entry points keep validation opt-in. Defaults:

- Every10000 optimizer steps, after saving the recoverable checkpoint.
- The same fixed71 test windows, one per recording, seeds0–70. These are test
  monitoring results; there is no test-based checkpoint selection.
- V2M: full clean video and supplied neutral shape, no text/head/camera input.
- M2V: clean motion/shape and one first-image latent, no text/camera input.
- Raw current bridge/optional-LoRA weights, UniPC35, CFG1, shift3 for motion
  and10 for video. No motion-null or image-guidance intervention.
- Sampling batch32 **per GPU**, sharded across the training ranks. CUDA OOM
  halves the local batch and retries using the same case seeds. Last batches
  have their natural size. Effective batches are recorded per rank.
- VAE decode batch4 with OOM backoff; LPIPS/DreamSim frame batches32 and TMR
  embedding batches32. Evaluation uses the training GPUs synchronously.

V2M reports Body23 MPJPE/PA-MPJPE, root ADE/FDE, relative root ADE, acceleration
error, skating, penetration and airborne fraction; matched September19 Nymeria
C45 TMR FID, retrieval R@1/2/3, matching distance and diversity. Its body completion
uses the existing train-only adapter, not predicted extra joints.

M2V reports PSNR, SSIM, LPIPS-Alex, DreamSim ensemble and global FVD. It scores
lossless decoded uint8 RGB frames1–96, excluding the conditioning frame. FVD
uses the reviewed EgoControl I3D400 logits, six contiguous16-frame segments per
window,426 descriptors overall. It must not be compared with a71-descriptor FVD.

`scripts/stream_phase3_wandb.py` sends both event types to the same training run:
`train/*` uses `train/step`; `val/v2m/*` and `val/m2v/*` use `val/step`. The
internal W&B event index remains append-only. Final validation is drained before
the logger exits. No checkpoint, dataset or generated media is uploaded.

Python/NumPy/Torch/current-device CUDA RNG, Torch Hub directory and module train/eval flags are
restored even on validation failure. Gradients/optimizer/data cursors are not
advanced. All ranks report failures together. Validation errors stop training
after checkpoint save instead of silently dropping metrics. Restarting at a
milestone retries pending evaluation; completed events are not duplicated.

Outputs are node-local under `RUN/validation/step_*/`: pinned identity,
per-rank effective batches, both prediction sets, per-case metrics, GT retrieval
reference, summary and `COMPLETE.json`. The checkpoint contract pins execution
settings, evaluation protocol, case manifest, evaluator sources and assets.

Case seeds and noise draws are independent of batch partition. Native BF16
kernels can produce slightly different outputs at different batch sizes; this
is not a claim of bitwise equivalence to historical batch-one sampling.

## Local setup

Install `migration/environment/phase3-validation.txt` in each node's existing
project environment. Use `python -m scripts.prepare_phase3_validation --help`
to pin an existing local fixed71 manifest, TMR weights/stats/text cache,
train-only body adapter, Wan VAE, reviewed I3D source/weights/preprocessing,
and DreamSim cache. AlexNet/LPIPS asset paths are discovered and pinned too.
It checks all71 bundle checksums and does no copying.

Launch with the same exact video-I/O Phase1 EMA200k and Phase2 200k paths:

```bash
python -m motion_expert_phase3_edge_image_joint.launch_mmdit \
  --gpus GPU_A,GPU_B --phase1 /mnt/local/.../iter_000200000 \
  --phase2 /mnt/local/.../step_000200000.pt \
  --cache /mnt/local/.../phase3_edge_camhead_T97_v1 \
  --out /mnt/local/.../NEW_RUN \
  --validation-config /mnt/local/.../validation.json
```

For the requested directional two-task run replace the command module above
with `motion_expert_phase3_edge_mmdit.launch`; its spawned DataLoader workers
also restore the50/50 policy, so no worker can revert to joint sampling.

Select two coordinated, empty GPUs. The launcher checks occupancy, performs a
fresh DDP preflight and disposable save/reload probes, then creates a fresh
production run and W&B sidecar. A previous architecture's acceptance/checkpoint
does not authorize this run. Training microbatch stays16 per GPU; evaluation's
batch32 is an independent setting.

On sep-1 the prepared assets/settings are recorded under
`/mnt/local/jinhyung.kim/cosmos/operations/phase3_mmdit_bf16_validation_20261105/`.
Native GPU and CPU test evidence lives there; no production run has been started
by this implementation. The other user's jobs on5/6 must not be interrupted.
