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

Human-motion rollout is a subsequent application. It will additionally require
the trained motion representation/statistics, neutral shape, floor coordinate
mapping and the measured train-only head-to-upright-RGB transform. The first
camera-only application does not introduce a skeleton or synthetic head offset.
