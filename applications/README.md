# Applications of the trained models

Inference experiments live here, with scene assets, simulator environments and
generated results in site-specific bulk storage. Training packages and saved
evaluations remain the source of model and sampler behavior.

The first application is [Habitat-GS video rollout](habitat_gs/README.md):
render a scene image → generate video from text + image → infer camera motion
from the generated video → integrate the final camera pose → render again.

Each application must record checkpoint identity, its resolved training/inference
contract, the exact evaluated input profiles, sampler and seeds, camera/frame
conventions, source revisions and input/output provenance. Reuse the existing
inference entrypoints and normalization helpers. New inference settings require
a separately named experiment, rather than silently changing the baseline.

Human-motion rollout is now available in
[the replay contract](habitat_gs/HUMAN_REPLAY.md): generated video + caption
goes through the evaluated Phase3 baseline, then its decoded Head and measured
train-only head/camera transform drive the next simulator observation. It saves
portable GS assets, raw predictions, world joint/camera poses, full RGB frames,
scene replays, normalization and instructions for a model-free demo website.
The camera-only application remains a separate inference path.

[Motion reconstruction figures](motion_reconstruction/README.md) convert saved
UniEgo283 or global body poses into standard SOMA meshes with joint-derived
surface error colors, synchronized comparison videos and publication PNG/PDF/SVG
artifacts. It reuses saved predictions and never runs or changes training.
