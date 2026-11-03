# Camera-prediction variant of the current image-joint run

This is a fresh controlled variant of `motion_expert_phase3_edge_image_joint`.
It retains the same frozen Phase1 LoRA+video-projection EMA200k specialist,
selected Phase2 200k, fresh seven bridges, 112503-window cache, seed and global
draw order, caption policy, native losses, and 40/40/20 task mixture.

The user explicitly selected **noise-corrupted ground-truth camera inputs** and
**camera tokens participating in native generator attention**. Every task now
includes96 noisy native action tokens, raw9 relative SE(3) padded to64, domain2.
No camera token is marked clean. The seven bridges still connect only the same
1600 video /98 motion tokens; no direct camera-motion bridge edges are added.
Native camera input/output projections and all other specialist parameters stay
frozen. Only the original117469198 bridge parameters train.

| Task | Motion/video behavior | Additional camera clock |
|---|---|---|
| video2motion40% | full clean video, noisy motion | motion sigma |
| motimg2video40% | clean motion, initial image, noisy video | video sigma |
| joint20% | initial image/shape, independently noisy motion/video, no caption | video sigma |

Camera noise uses a private generator keyed by the global draw. The existing
motion/video random draws and targets are unchanged. Only motion and video
appear in the target dictionary and loss function. The final camera velocity
readout is detached and no camera loss is computed. Camera hidden states remain
in native attention, so motion/video losses may affect camera predictions
indirectly through the trained bridges. Noisy GT camera also supplies residual
camera information; this experiment does not isolate a purely unused readout.

Recipe: T97/20fps/256; global32=2x16, accumulation1, or oneGPU x16 with
accumulation2; AdamW2e-4, betas0.9/0.99,
weight decay0, clip1; warmup1000, cosine to zero at fresh200000 updates;
checkpoint10k and retain latest+100k; unchanged native physical motion losses
and video velocity weight10. All artifacts and repeated reads use local NVMe.

## Launch

The exact frozen checkpoints/cache must already be local. Set the same
`PHASE3_IMAGE_JOINT_PATH_MAP` as the existing run, and use one or two coordinated
empty GPUs. The launcher verifies the saved baseline contract's recipe, source hashes,
specialist identities and training cohort before launching. It does not stop,
queue behind or share an occupied GPU. Preserve the current baseline.

```bash
bash motion_expert_phase3_edge_image_joint_camera/run_foundary.sh \
  --gpus "$GPUS" --phase1 "$PHASE1_VIDEOIO_200K" --phase2 "$PHASE2_200K" \
  --cache "$PHASE3_CACHE" --camera-cache "$NATIVE_CAMERA_CACHE" \
  --reference-contract "$BASELINE_RUN/contract.json" --out "$NEW_LOCAL_RUN" \
  --check-only

# After resource and local-input checks pass, omit --check-only.
```

Required production gates at the actual GPU count and microbatch: native specialist parity, finite
motion/video gradients, frozen-specialist checks, camera predictions for all
three tasks, zero camera loss/readout gradient, joint-caption invariance,
and full fresh4→reload6 optimizer/RNG probes. Production starts fresh; disposable
probe weights never initialize it. W&B uses the existing tracked scalar sidecar.
Single GPU preserves the original two 16-sample loss groups and global draw
order, executing those groups sequentially before the same optimizer update.
It does not promise bitwise equality of random noise or floating-point sums
across GPU counts. Execution topology is recorded in the saved run contract.
Earlier microbatch4 diagnostic acceptance cannot authorize production.

CPU checks:

```bash
python -m unittest motion_expert_phase3_edge_image_joint_camera.test_camera_prediction -v
```

Offline `python -m motion_expert_phase3_edge_image_joint_camera.sample` accepts
the usual checkpoint/cases/mode/phase1/phase2/out flags and native guidance1.
It returns motion, video and camera arrays. Camera generation starts from noise;
no GT camera is supplied at inference. Motion/video draw order, separate clocks
and initial-image pinning are retained; camera shares the active target clock.
Camera outputs are diagnostics, not evidence of learned camera accuracy.

Consult the operation receipts for GPU acceptance and launch status. This file
does not claim that a new production run has started.
