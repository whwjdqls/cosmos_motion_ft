# sep-2 temporal-local cross-only camera Phase3

User requested a fresh cross-attention-only bridge variant, keeping temporal
locality, camera prediction without camera loss, video weight1 and EMA.
The agreed reference is the directional camera/no-joint run: V2M50/M2V50/J0,
no captions, EMA0.999 from0, global32=2x16/no accumulation, target200k.

Package: `motion_expert_phase3_edge_camera_cross_only`.
The seven private MMDiT norm/Q/K/V/output projections and gates are unchanged
(234,938,382 trainable parameters). Both same-modality bridge quadrants are
blocked; cross edges retain the exact Wan temporal grouping and directional
clean-query rules. No-key queries produce exact zero residuals. Shape bypass
and frozen native specialist attention remain unchanged. Camera states still
enter native generator attention, with detached readout and zero camera loss.

Selected empty GPUs: **2,4** on `lg-gwangmo-4xh200-sep-2`. Existing directional
camera trainers on3/5 and1/7 are unchanged. No shared-storage transfers.
Fresh production started at2026-11-08 23:30UTC after actual2rank/micro16
preflight and full-state4→6 checkpoint reload. Disposable probe weights are
not used by production. Six independent CPU attention/policy tests passed.

W&B:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-0b54e17752808987cb55

Operation and startup verification:

```text
/mnt/local/jinhyung.kim/cosmos_storage/operations/phase3_camera_cross_only_20261108
```

`COMPLETE.json` is authoritative for production finite-step, raw/EMA/optimizer/
RNG reload, frozen-source, exact-reference-recipe and remote W&B verification.
`verification_status.json` records pending progress; `VERIFICATION_FAILED.json`
requires investigation. Always verify current PID/start identities.

Run directory:

```text
/mnt/local/jinhyung.kim/cosmos_storage/runs/motion_expert_phase3_edge/edge_p3_mmdit_crossonly_directional_nojoint5050_camera_nocamloss_wv1_ema0999_temporal_local_bf16_T97_gbs32_sep2_20261108
```

The source manifest includes the cross-only package plus every underlying
adapter. Resume only this package with its own frozen source/contract. Generic
joint-attention loading is not sufficient: inference must call this package's
`policy.configure_checkpoint` and `sample.sample`, which restore the topology.

Validation: camera-aware native UniPC35/CFG1, EMA, exact fixed71 every10k,
batch32/rank with adaptive fallback, RGB decode4. Camera starts from private
noise, not GT. Camera outputs are currently not saved or assigned metrics.

Exact100k is retained natively and separately protected by an independent-copy
CPU watcher under:

```text
/mnt/local/jinhyung.kim/cosmos_storage/preserved_milestones/phase3_camera_cross_only_100k_20261108
```

Its registry contains only `cross_only_local`; the two other camera runs keep
their existing watcher. `VERIFIED.json` will appear only after the actual100k
checkpoint exists and raw/EMA/full-resume state and copied hashes pass.
