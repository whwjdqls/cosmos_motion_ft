# Naeive adaptation, video weight one — sep-2

Fresh ablation authorized2026-11-10. Same native Edge action-stream LoRA16/alpha32,
283-D motion/padded288, neutral prefix and97-frame/20FPS/256px data as the completed
Naeive x0 model. No motion expert/Phase3 bridge. Equal25% T2M/TI2M/V2M/M2V;
text captions only for T2M/TI2M (dropout0.1), one initial image for TI2M/M2V.

The only motion-contract change is video_loss_scale10→1. Motion loss stays1,
feature/joint/displacement1/1/5, contact/foot velocity/foot height0.05/1/10.
Native shared WAVER shift3, clean x0 motion target, native video velocity.
Starts from official Edge base, fresh action heads/LoRA/optimizer/native power
EMA (rate0.1), never the completed200k model. Global32=2x16/accumulation1,
GPUs6/7 checked empty. Same LR/scheduler/optimizer and full local train cohort.

Resolved config/contract comparison passed (RECIPE_PARITY.json). The original
reconstruction and physical-loss source bytes were retained from the accepted
Oct28 snapshot; their SHA values match the original contract exactly. Current
two-rank preview handling retained. No source file or data from another run was
modified.424 source/config/launcher files frozen and SHA-verified. Nine focused
CPU tests including actual native video loss/gradient at scales1 and10 passed.

Native EMA previews for all four tasks are generated at launch/every5k, with W&B
media and scalar losses in jungbinc-upenn/cosmos3_camera. Checkpoint every5k plus
same15-minute rolling recovery policy as the original. Full200k horizon.

Operation: `/mnt/local/jinhyung.kim/cosmos_storage/operations/naeive_adaptation_x0_wv1_20261110`.
Run: `/mnt/local/jinhyung.kim/cosmos_storage/runs/cosmos3_camera/motion_world_edge/naeive_adaptation_x0_wv1_edge_T97_gbs32_lora_4tasks_200k_sep2_20261110`.
Launch: `bash /mnt/local/jinhyung.kim/cosmos_storage/operations/naeive_adaptation_x0_wv1_20261110/train.sh`.
The frozen source snapshot is used for production. Do not resume this W1 run
using the old W10 run's identity. Resume only this run's own explicit checkpoint.
Check STARTED.json identity, status/VERIFICATION_FAILED/COMPLETE, training.log,
task_audit{,_rank1}.json and W&B before claiming startup completion.
No shared storage transfer. Existing trainers and evaluation untouched.

## Startup verified

COMPLETE.json confirms finite production through step99, four-task gradient audits
on both GPUs and actual M2V total loss equal to1x native video loss.424 frozen
sources verified. All four initial EMA previews were confirmed uploaded with
nonzero remote file sizes, and remote validation metrics are finite. W&B:
https://wandb.ai/jungbinc-upenn/cosmos3_camera/runs/d6z7q2gi
Display name: sep2_naive4task_w1_x0_EMA.
Production agent2852517, ranks2852772/2852773 on GPUs6/7; verify identities and
live step before later actions. No stopped legacy run was resumed.
