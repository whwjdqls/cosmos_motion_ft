# Exact Phase1 video-I/O EMA200k handoff for Yonsei / minsu1206

This is the unchanged checkpoint required by the shared camera trainer:
`edge_phase1_T97_20fps_2gpu_bs16_lora_video_io_exclude5_fd_invdyn_no_text_p1_200k_sep2_20261028`. Its model metadata SHA256 is
`b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`. Both trained video projections and all
original model/optimizer/scheduler/trainer DCP shards, config and native
adaptation contract are included. This is not the standard LoRA-only two-task
model and not either local projection-only ablation.

Drive folder: https://drive.google.com/drive/folders/1d49-FqpeYu9TW8sJGkCnfzLLBjvxgf5E
Archive: https://drive.google.com/file/d/1MuZs66AhDLyAWvSLgfdFfMz3KeH3sdqo/view
Bytes: 27141017600
Archive SHA256: `94f237c198d1777d8ce114f959fc608dda444b883b2c53a8121e366ff9117e50`

## Download and restore

Use your own authorized Drive remote; do not copy anyone else's credentials.
Choose a writable checkpoint location with at least55GB free for download plus
extraction. The destination below is an example; no Yonsei filesystem or
minsu1206 permissions were inspected from Foundary.

```bash
REMOTE=YOUR_DRIVE_REMOTE
DEST=/scratch2/minsu1206/cosmos/checkpoints/phase1_video_io_ema200k
mkdir -p "$DEST"
rclone copy "$REMOTE:" "$DEST" --drive-root-folder-id 1d49-FqpeYu9TW8sJGkCnfzLLBjvxgf5E --checksum --transfers 1 --checkers 2
cd "$DEST"
sha256sum -c SHA256SUMS
tar --no-same-owner -xf phase1_video_io_ema200k_b2b3374b.tar
sha256sum -c PAYLOAD_SHA256SUMS
PHASE1="$DEST/edge_phase1_T97_20fps_2gpu_bs16_lora_video_io_exclude5_fd_invdyn_no_text_p1_200k_sep2_20261028/checkpoints/iter_000200000"
sha256sum "$PHASE1/model/.metadata"
```

The final command must print `b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`.
Pass this `PHASE1` path to the camera trainer's `--phase1` argument. Keep
`config.yaml` and `native_phase1_contract.json` at the extracted run root,
because the native loader resolves its adaptation contract there. No training
or constructor source fingerprints should be bypassed to use different weights.

For Phase3 this is a frozen specialist, not a new trainable initialization.
Other base Cosmos/Phase2/stat/cache dependencies remain separate and are not
included in this single checkpoint handoff. No destination GPU acceptance is
claimed; the identical payload is already used by the current sep-1 trainers.

## Verified publication

Uploaded and verified on2026-11-04 at03:26UTC. All14 original source files
passed SHA256 verification while archiving; DCP extents and all four EMA video
projection tensors were checked. Drive archive size/MD5 match the source;
restoration/checksum/READY files were downloaded and SHA256-verified. The
provided archive SHA256 should also be checked after download on Yonsei.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/upload_phase1_video_io_ema200k_20261104`.
The user confirmed that minsu1206 will use existing authorized Drive access;
no new public sharing permission was added.
