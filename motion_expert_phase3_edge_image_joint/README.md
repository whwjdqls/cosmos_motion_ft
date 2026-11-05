# Image-only joint video/motion denoising

Current Phase3 policy forbids captions in every task. See
[sep-2 replacement record](SEP2_NO_TEXT_REPLACEMENT_20261105.md) for the fresh
caption-free directional/bidirectional handover. Read operation receipts for
live status; the experiment definition alone does not establish GPU acceptance.

## Objective

The task probabilities are 40% `video2motion`, 40% `motimg2video`, and 20% `joint`.
All three tasks receive empty captions, with caption dropout0.
**Joint has no semantic text:** the only measured conditions are the initial
RGB image and supplied neutral body shape. Required native structural prompt
tokens remain. No clean target motion or future video enters the joint inputs.
The loader, noising function, model entry point and samplers all remove
captions for every task. Motion retains its usual camera/head-aligned 283-D representation.

Using our convention, sigma0 is clean and sigma1 is noise. For each joint row:

```text
sigma_M ~ shifted-logitnormal(shift=3)
sigma_V ~ Waver(shift=3)                 # independent draw, not a shared time
M_t = (1 - sigma_M) M_0 + sigma_M epsilon_M
V_t = (1 - sigma_V) V_0 + sigma_V epsilon_V
V_t[:, frame0] = V_0[:, frame0]          # always clean, excluded from video loss
L_joint = L_motion_native(x0_hat, M_0) + 10 L_video_native(v_hat, epsilon_V - V_0)
L = .4 E[L_V2M] + .4 E[L_M2V] + .2 E[L_joint]
```

Native motion losses remain feature1/joint1/temporal5/contact0.05/foot-velocity1/
foot-height10. Native video reduction retains the full denominator, including
the masked first frame. Neither joint loss is halved. Expected target exposure
is60% motion and60% video; this is an extra training objective, not an exposure-
matched50/50 control.

The seven bridges use the selected directional or bidirectional conditioning
attention across time-aligned intervals. Shape stays in the motion expert and
is excluded from bridge Q/K/V. This does not make
all frames globally cross-attend. Clean M2V motion receives E_M(0); noisy motion
receives E_M(sigma_M). Video and motion keep separate timestep embeddings.

## Relationship to JUICE and the old joint checkpoint

Reference: [JUICE loss at commit1bc0e5f](https://github.com/Ced3-han/JUICE-Code/blob/1bc0e5f34a2b47d9a4cbb5bc76821091b5994984/juice/train/loss.py),
[paper](https://arxiv.org/abs/2610.28801).
JUICE independently noises its two modalities and supervises both on joint rows.
It uses clean-state predictions with velocity-space losses, a0.05 denominator
floor, and active-token normalization. Its time direction is the reverse of ours:
`t_JUICE = 1 - sigma_ours`. Here the user explicitly chose to retain the native
specialist losses/distributions and40/40/20 stochastic task sampling. This adapts
the joint-denoising idea; it does not reproduce JUICE's loss, mixture, or full
architecture. No JUICE dependencies, datasets or weights are needed.

The previous joint200k recipe already used independent times, but its joint
task received image+text and used the old conditioning/time policy. This new
joint task receives an image without a caption and uses the same trained
video-input/output-projection Phase1 as the new50/50 comparisons.

## Frozen specialists and training

Use exactly the selected Phase1 video-projection EMA200k checkpoint, not the
original two-task no-video-projection model or the sep-1 stopped125764 ablation:

```text
edge_phase1_T97_20fps_2gpu_bs16_lora_video_io_exclude5_fd_invdyn_no_text_p1_200k_sep2_20261028/checkpoints/iter_000200000
model metadata SHA256: b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8
```

The model requires this metadata identity and all four trained video projection
tensors. Phase2 is the same selected200k specialist:

```text
edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921/checkpoints/step_000200000.pt
checkpoint SHA256: 6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a
```

Only seven fresh zero-gated bridges train. Both specialists remain frozen.
T97/20fps/256; global batch32=2x16; accumulation1; AdamW2e-4; warmup1000;
fresh200k cosine schedule to zero. The validated cache is
`phase3_edge_camhead_T97_v1` (112503 accepted training windows). Keep its index
and normalization unchanged. Checkpoint every10k; retain latest+100k. All
datasets, caches, environments, logs and checkpoints must be on node-local NVMe.

## Run on sep-1

Pull `master` into that node's existing checkout without overwriting local
edits. Keep its own environment and `.site.local.sh`; they are node-specific.
The exact selected Phase1, Phase2 and validated cache must already be staged on
sep-1. **Git does not transfer these artifacts, and their presence on sep-1 has
not been verified by this handoff.** Do not silently choose a different checkpoint.

After coordinating two empty GPUs and setting these four absolute local paths:

```bash
git pull --ff-only origin master
export PHASE1_VIDEOIO_200K=/mnt/local/your/local/selected/phase1/checkpoints/iter_000200000
export PHASE2_200K=/mnt/local/your/local/selected/phase2/checkpoints/step_000200000.pt
export PHASE3_CACHE=/mnt/local/your/local/caches/phase3_edge_camhead_T97_v1
export PHASE3_NEW_RUN=/mnt/local/your/local/runs/p3_imagejoint402020_bidir_clean_t0
export PHASE3_GPUS=2,3  # replace with the two GPUs coordinated on sep-1

bash motion_expert_phase3_edge_image_joint/run_foundary.sh \
  --gpus "$PHASE3_GPUS" --phase1 "$PHASE1_VIDEOIO_200K" \
  --phase2 "$PHASE2_200K" --cache "$PHASE3_CACHE" --out "$PHASE3_NEW_RUN" \
  --check-only

# Start only after the checks pass; run in your normal persistent terminal.
bash motion_expert_phase3_edge_image_joint/run_foundary.sh \
  --gpus "$PHASE3_GPUS" --phase1 "$PHASE1_VIDEOIO_200K" \
  --phase2 "$PHASE2_200K" --cache "$PHASE3_CACHE" --out "$PHASE3_NEW_RUN"
```

If the checkout lacks `.site.local.sh`, set `COSMOS_SITE_CONFIG` to that node's
existing config. It must select the node's Cosmos Python, Edge framework and
cache/storage roots. No installation or large transfer is performed by the
launcher. It refuses occupied GPUs immediately; it does not queue or share them.
Its default W&B destination is `jungbinc-upenn/cosmos-motion-ft` and can be
changed with `--wandb-entity/--wandb-project`. Existing `wandb login` is required.

Original cache latent paths are translated only at read time and retain their
size/mtime checks. Defaults map `/lustre/whwjdqls99/cosmos` and
`/scratch2/whwjdqls99/cosmos` under `$COSMOS_STORAGE_ROOT/restored/`. If sep-1 uses
different staging, set explicit mappings before the checks, for example:

```bash
export PHASE3_IMAGE_JOINT_PATH_MAP='{"/scratch2/whwjdqls99/cosmos":"/mnt/local/your/cosmos/restored/scratch2/whwjdqls99/cosmos","/lustre/whwjdqls99/cosmos":"/mnt/local/your/cosmos/restored/lustre/whwjdqls99/cosmos"}'
```

The launcher performs a real two-GPU native-specialist/bridge acceptance gate,
caption-invariance and independent-time checks, both-target gradient checks,
and actual fresh4→reload6 optimizer/RNG updates before starting production fresh.
Disposable weights never initialize production. Logs, snapshots, plans, contracts
and W&B logger state live under `$PHASE3_NEW_RUN/launch/`.
There are no in-training qualitative previews; the scalar logger is the existing
Phase3 sidecar. Use offline inference for qualitative evaluation.

For resume, use the recorded source snapshot/environment and this run's own
acceptance/checkpoint with `-m motion_expert_phase3_edge_image_joint.train`;
do not rerun the fresh launcher on an existing output. Fingerprints remain strict.
This package applies two explicit runtime-only diagnostic/probe adaptations to
the unchanged trainer: the zero-bridge reference receives sanitized captions,
and `--probe-updates` bounds disposable probes without shortening the schedule.
The contract hashes this package, the original trainer, shared bridge and cached
execution policy. Running sep-2 baseline files were left byte-for-byte unchanged.

## Checks and inference

In the configured Cosmos environment/PYTHONPATH:

```bash
python -m unittest motion_expert_phase3_edge_image_joint.test_objective \
  motion_expert_phase3_edge_joint.test_conditioning_attention -v
python -m motion_expert_phase3_edge_image_joint.train --help
```

The CPU checks cover caption removal at every entry, original noising/RNG parity,
independent modality times, first-image preservation, spawned-worker path/task
policy, checkpoint policy rejection, and actual native coupled UniPC clocks with
no target leakage. The existing bidirectional/cache/legacy checks are included.
CPU success does not certify GPU acceptance or completed training.

Offline inference uses `python -m motion_expert_phase3_edge_image_joint.sample`
with the inherited checkpoint/cases/phase1/phase2/out flags and `--mode joint`.
It rejects old image+text joint checkpoints and uses empty text, separate
motion-shift3/video-shift10 UniPC35 clocks, paired predictions before either state
advances, and first-image re-pinning. Guidance must be1 for every task (no caption CFG).
