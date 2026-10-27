"""Start official EgoAllo training with the Nymeria-SOMA body-only adapter."""

import runpy
from pathlib import Path

from egoallo import network


OPS = Path("/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015")
CACHE = Path("/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_camhead_v1_20261015")
api = runpy.run_path(str(OPS / "upstream/egoallo/1_train_motion_prior.py"))
config = api["EgoAlloTrainConfig"](
    experiment_name="nymeria_soma_body_t97_shape16_floor_b256_pinned_20261015",
    dataset_hdf5_path=CACHE,
    dataset_files_path=CACHE,
    nymeria_cache=CACHE,
    model=network.EgoDenoiserConfig(
        include_hands=False,
        body_joint_count=22,
        contact_joint_count=4,
        include_hand_positions_cond=False,
    ),
    batch_size=256,
    num_workers=4,
    subseq_len=97,
    train_splits=("train",),
)
api["run_training"](config)
