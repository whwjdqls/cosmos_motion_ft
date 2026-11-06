# Phase3 validation metric visibility

User reported no visible validation metrics for `p3-eed7a56cdcf72f3f4569`.
Remote API inspection confirmed the completed10k event is present in both
summary and actual history: internal event1001, `val/step=10000`, including
all V2M/M2V metrics. The training sidecar continues streaming. This was not
missing uploaded validation data; the existing UI layout was not audited.
There is only one completed validation round so far.

Created and readback-verified a dedicated saved workspace:

https://wandb.ai/scene_motion/cosmos-motion-ft?nw=yowy3nwtcw4

- Only the selected Phase3 run is included, using its immutable run ID.
- V2M/M2V charts explicitly use `val/step`, with point markers and no
  smoothing; no intermediate validation measurements are fabricated.
- 13 metric plots,11 latest-value cards, and the matching measured motion GT
  and RGB identity references are visible in four expanded sections.
- Charts read the existing live history and include later10k milestones.
  The dashboard link was also appended to the original run's notes.
- Existing project views, trainer, checkpoint, optimizer, losses, schedule,
  validation frequency and metric history were not modified.

Operation on Scratch2:
`cosmos/operations/wandb_validation_audit_20261106` contains `REMOTE.json`,
the workspace creation script, `WORKSPACE_CREATED.json`, serialized readback,
and `WORKSPACE_VERIFIED.json`. W&B's official workspace SDK was installed
in a separate operation-local UI venv, leaving the training environment and
its active W&B SDK unchanged. All API/setup commands ran on owned compute.

The contemporaneous Phase2 no-shape run has training/visualization logging
but no quantitative validation hook in its inherited launcher. The user
clarified this request concerns Phase3, so no Phase2 evaluation was started.
