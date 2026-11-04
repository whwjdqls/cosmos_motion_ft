# Handover audit — 2026-11-02

Historical handover inventory checked on sep-2 at 2026-11-02 23:58 UTC. Follow-up on 2026-11-03 01:25 UTC traced the live trainers to their actual source checkout and located the camera-prediction-omitted Phase3 code. The previous claim that its source could not be located is corrected by that runtime verification.

| Workstream | Verified handover state | Outstanding work / limits |
| --- | --- | --- |
| Phase3 50/50 directional vs bidirectional | Current local logs: bidirectional37,610 on GPUs0/3; directional38,700 on1/7. Recent100 loss/gradient records finite, roughly1.98s/update. Both add learned E_M(0) to clean M2V motion. Matched initial bridges, original validated cache, global32 and frozen specialists. | Continue existing training toward200k; no launch/config change made during this audit. |
| Prior control and video-projection40/40/20 trainers | Stop receipts record exit0 and complete preserved recovery state at178,363 and17,672 respectively. | They are stopped; older status reports that show them running are historical. |
| Image-only joint40/40/20 on sep-1 | Live GitHub master is d267a56225eac342733d7f76287e86cd6237167c, tree fb856f3388dbc5d1fe805f8eeee0e512954443f3. Handoff records12 CPU checks. All12 baseline-source hashes still match the handoff. Joint has initial image+supplied shape, empty semantic caption, independent modality times and summed native losses. | New variant was not launched or queued by this handoff. Exact artifact staging and two-GPU acceptance on sep-1 remain unverified. |
| Camera prediction: existing Phase1 video→camera evaluation | Completed and published for final one-frame LoRA+video-projection EMA200k:71 inverse and71 camera-guided forward windows, zero skipped cases; quantitative evidence and media verified. | This completed71-case Phase1 benchmark is distinct from the active Phase3 runtime described in the next row. No full-test camera sweep is claimed. |
| Camera-prediction ablation — active Phase3 runtime | Located through live worker PIDs3620614/3620873, their REPO_ROOT/PYTHONPATH/environment and adapter.py. Source is motion_expert_phase3_edge_joint in the isolated phase3_videoio_joint4020_20261101 checkout. The packed sequence has no action tokens; forward decodes video/motion only; the loss supervises video/motion only. | Both running50/50 arms omit camera prediction. Their recorded controlled difference is conditioning_attention (directional versus bidirectional). Phase1 camera heads remain loaded and frozen; they are unused in this Phase3 forward. No source path is needed from the user for these active sep-2 runs. |
| Naeive Adaptation whole test | Final COMPLETE.json at23:50 UTC:11,911 cases/task,47,644 saved predictions; reconstruction/physics, video metrics, motion FID/diversity and all supported text TMR retrieval/matching metrics complete and published. Exact caption cache staged and verified; final archive remotely SHA256-verified. | Earlier text-cache-blocked snapshot at22:29 UTC is superseded. Camera-action input/output tasks remain unsupported by this motion-head checkpoint; multimodality is unavailable with one sample/case. |
| M2V qualitative hand cohorts | Both cohorts complete and published: initial hands visible and hands absent initially/entering later. Each has8 recordings×2seeds×5models,16 comparison videos and33 verified uploaded artifacts. | No new inference pending in these completed handovers. |
| Control150k V2M evaluation and shared backup | All284 predictions/physical/semantic scores complete;16 rows added to existing tables. Full150k milestone is in shared storage with READY/manifest/checksums. | Evaluated150k snapshot is distinct from the preserved178,363 stopped recovery state. |
| Baseline/FiLM checkpoint transfers | Original two-task Phase3 and equal-thirds joint Phase3 final200k, hidden-state FiLM200k and trajectory FiLM180k are staged locally; prior checksums and completed comparison evidence exist. Shared FiLM handoff metadata inspected. | Shared trajectory package is180k; its old handoff says training continued. A final200k trajectory package is not established by those records. Frozen specialist identities must remain matched. |
| Phase2 timestep embedding diagnostic | Selected200k checkpoint: relative L2 E_M(0) vs E_M(0.01)=0.947585959%, cosine0.999994627504; native encoder hook, strict heads load and FP64 checks passed. | Diagnostic made no training changes. Original clean-motion omission was later superseded only in the new 50/50 and image-joint recipes that add E_M(0). |
| sep-1 no-shape Phase2 and EgoControl evaluation | Notion handovers record successful EgoControl200k→fresh no-shape Phase2 handoff and a launched full11,677-window EgoControl evaluation. | Last published progress snapshots are Nov1; they do not establish live sep-1 status on Nov2. Direct access remains unavailable. |

Sources:

- [50/50 conditioning comparison](https://app.notion.com/p/40cd2a44f44b815cae2edb72a7fdc539)
- [Image-only joint sep-1 handoff](https://app.notion.com/p/40cd2a44f44b81278a2cee606a0935ab)
- [Phase1 forward/inverse evaluation](https://app.notion.com/p/408d2a44f44b80e882cfcc8d8f4b7869)
- [Final Naeive Adaptation evaluation](https://app.notion.com/p/40ad2a44f44b8017b16aeb87901f236d)
- [Both M2V hand cohorts](https://app.notion.com/p/40bd2a44f44b80c5a349ca6df079d9be)
- [Control/FiLM evaluation and checkpoint handoffs](https://app.notion.com/p/40ad2a44f44b808c8a6eedd4ab2d13c2)
- [Phase2 timestep diagnostic](https://app.notion.com/p/3fbd2a44f44b811a9d14e00c7b8bddc5)
- [sep-1 no-shape Phase2 handover](https://app.notion.com/p/40bd2a44f44b8187850ac41dd36ef75c)
- [sep-1 EgoControl evaluation handover](https://app.notion.com/p/40bd2a44f44b8185b567d54f0e5be3b8)

Local evidence: `/mnt/local/jinhyung.kim/cosmos_storage/operations/handover_audit_20261102/evidence.json`. No training process, checkpoint, package, or Git branch was changed during this review.
