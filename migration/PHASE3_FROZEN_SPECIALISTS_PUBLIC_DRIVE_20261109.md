# Public Phase3 handoff and exact frozen Phase1/Phase2 dependencies

Completed 2026-11-09T17:19:42.228482+00:00. User explicitly authorized anyone-with-link reader access for the corresponding checkpoints and handoff folders.

| Component | Public folder | Checkpoint |
|---|---|---|
| Phase3 directional W1, no camera, EMA100k | [Phase3 folder](https://drive.google.com/drive/folders/1d16BDPRcmyJJHP5d5SofLFuLYzOVvvo-) | [EMA inference export, 939800605 bytes](https://drive.google.com/file/d/1Inau9dZN5Lfl9cdlAlCv6C1PbIxEqoM-/view) |
| Phase1 video-I/O EMA200k | [Phase1 folder](https://drive.google.com/drive/folders/1d49-FqpeYu9TW8sJGkCnfzLLBjvxgf5E) | [Exact tar archive, 27141017600 bytes](https://drive.google.com/file/d/1MuZs66AhDLyAWvSLgfdFfMz3KeH3sdqo/view) |
| Phase2 selected shape-conditioned camera-head 200k | [Phase2 folder](https://drive.google.com/drive/folders/11Vs0D3euL_qk1P0Ioq4R4OfUYG8CVc0l) | [Exact checkpoint, 1973517263 bytes](https://drive.google.com/file/d/1ZiOqTXAUaNd1YUqU6KZFuAtyeddxpjxr/view) |

Phase1 is the required video-I/O specialist. Its model/.metadata SHA256 is `b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`. Archive SHA256: `94f237c198d1777d8ce114f959fc608dda444b883b2c53a8121e366ff9117e50`. Extract it and retain the run-root config.yaml and native_phase1_contract.json layout expected by the loader.

Phase2's original full-file SHA256 is `6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a`. It is byte-for-byte unchanged and matches the Phase3 model card. Its public folder also contains the exact motion mean/std, contract, identity receipt, restoration instructions and SHA256SUMS.

The publication script verified anyone/reader/unlisted ACLs for all three folders and the four checkpoint files (including Phase3's optional full resume checkpoint). Fresh unauthenticated sessions downloaded matching first1024-byte prefixes from all four checkpoint files and the dependency guide. Anonymous folder requests returned200 and contained the correct folder names. Phase2's entire local SHA was verified before upload, and all uploaded files passed remote size/MD5 comparisons. The existing Phase1 archive's remote size/MD5 matches its prior verified publication receipt.

PUBLIC_CHECKPOINT_DEPENDENCIES.md was added to the Phase2 and Phase3 folders. Original Phase3 payload files/checksums were retained. Older statements that dependencies require existing authorized access describe the earlier publication and are superseded by this public-sharing request.

Native Edge base weights/tokenizer/framework, the Kimodo decoding stack and matching Wan VAE/video preprocessing remain separate runtime requirements. No new GPU or collaborator-environment acceptance is claimed. Training, original checkpoints, shared storage and broader backup-parent permissions were unchanged.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/publish_phase3_frozen_dependencies_20261109T171247Z`. COMPLETE.json, UPLOAD_VERIFIED.json, PUBLIC_PERMISSIONS_VERIFIED.json and ANONYMOUS_ACCESS_VERIFIED.json record completion and exact identities. Credentials are excluded.
