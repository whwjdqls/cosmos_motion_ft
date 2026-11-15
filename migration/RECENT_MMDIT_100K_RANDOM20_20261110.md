# Recent MMDiT EMA100k random20 evaluation

User explicitly requests only the four recent MMDiT checkpoints on sep-1. Older FiLM and pre-MMDiT image-joint prototypes are excluded.

Git was committed/merged (origin already up to date) and published before the100k queue: `b2a678d3d3b05b9de1b7faecc34166b6ae16c423`. The connected GitHub API published the exact reviewed tree because CLI HTTPS credentials were unavailable; no force push. Local master matches origin. Original local commits are retained on `foundry/sep1-pre-api-publication-20261110`.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_mmdit100k_random20_20261110`. ManagerPID2216862. Existing camera trainer411844/411845 onGPUs2/3 remains untouched.

| Model ID | GPU | Step/weights | Video weight | Direction | Camera prediction | EMA start |
|---|---:|---|---:|---|---|---:|
| sep1_dir5050_w1_EMA0 | 4 |100000/EMA| 1 | directional | No | 0 |
| sep1_dir5050_w10_lateEMA | 5 |100000/EMA| 10 | directional | No | 50043 |
| sep1_bidir5050_w10_lateEMA | 6 |100000/EMA| 10 | bidirectional | No | 50143 |
| sep1_bidir5050_w1_camera_nocamloss_EMA0 | 7 |100000/EMA| 1 | bidirectional | Yes; no camera loss | 0 |

All four use V2M50/M2V50/joint0, no training captions, groupedBF16 MMDiT, temporal-local video/motion bridges, per-modality norms/QKV/O, frozen matched Phase1/2 specialists, no additional Phase3 LoRA.

Cohort: the frozen1420 random windows/71recordings/20each; manifest SHA256 `47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df`. Source `migration/artifacts/nymeria_random20_v1/`. Do not rerandomize on another node.

Three conditions per model: M2V whole-bridge CFG1.1; V2M CFG1.5; video+text→motion CFG1.5 (only frozen motion expert gets text, same text in both branches). Native UniPC35, shifts10/3,97frames/20FPS/256px, supplied neutral shape. Camera model uses its native camera-aware sampler from independent private Gaussian, no test GT camera; camera solver uses conditional readout without separate camera CFG.

Existing fully verified200k input bundles are reused, with no data copy or VAE re-encode. Predictions are new. Native sampler parity and exact EMA/source checks occur per shard; all metric arrays/embeddings are saved for future rescoring. Source files are frozen per checkpoint; evaluator driver is frozen from published b2a678d.

Each model runs four355-case virtual shards sequentially on its assigned GPU. Sampling batches32 M2V /64 V2M, decode4,64-case chunks. Total17040 predictions/12settings. Scoring includes PSNR/SSIM/LPIPS/DreamSim/native8520-segment I3D FVD, Body23 positional/physical errors, UPE/LPE/sJPE, matched91f9d8 TMR FID/retrieval. Full-cohort distribution metrics are recomputed from features, never averaged from shards.

The original directional W1 EMA200k random20 operation has completed all4260 outputs and global audit. It remains a separate200k reference, not a100k result.

All four100k milestone files are independent preserved copies, SHA-reverified before launch. Source paths and identities:

- `sep1_dir5050_w1_EMA0`: `/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108/sep1_dir5050_w1_EMA0/step_000100000.pt`; SHA256 `65a04432e1b8ad91e9aa6b6130dfc856777e68aae6a5185f5d7b19028b1072ec`.
- `sep1_dir5050_w10_lateEMA`: `/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108/sep1_dir5050_w10_lateEMA/step_000100000.pt`; SHA256 `788e5ee9a2374b38006f160a733915a3259ec92085c2b61dd578691ce9561d92`.
- `sep1_bidir5050_w10_lateEMA`: `/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108/sep1_bidir5050_w10_lateEMA/step_000100000.pt`; SHA256 `d42c2c2f6c6905ee2d1df1efdb168f55aa4b73e2ae62ae5e2dd6341f4b7ecdb0`.
- `sep1_bidir5050_w1_camera_nocamloss_EMA0`: `/mnt/local/jinhyung.kim/cosmos/checkpoints/phase3_milestones_100k_20261108/sep1_bidir5050_w1_camera_nocamloss_EMA0/step_000100000.pt`; SHA256 `f595ff4a7fa7a37ec78d629e467fa6009da9a80361173d891d0eb4dd831579cc`.

Status: root RUNNING/MANAGER_ERROR/ERROR.json; per-model shards/*/status.json and ERROR.json. Per-model COMPLETE requires all1420 cases/global metrics/audit. Root COMPLETE requires all4 models. Details/results go to the existing Final Variations page, preserving sep-2 entries.

Cross-node code: `nymeria_baselines/random20_eval/`. Supply node-local paths/snapshot in plan.json and use each architecture’s native sampler/policy. Cross-only requires explicit camera_cross_only policy_module and sample_module. The shared indices/defaults work on sep-1, sep-2 and Yonsei; checkpoint paths above are sep-1 NVMe, not shared NAS.
