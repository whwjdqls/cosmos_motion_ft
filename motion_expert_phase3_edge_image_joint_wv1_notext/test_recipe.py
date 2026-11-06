import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_image_joint_wv1.policy import configure as captioned
from .policy import configure, check_checkpoint


@pytest.fixture(autouse=True)
def restore_recipe():
    saved = copy.deepcopy(RECIPE)
    yield
    RECIPE.clear()
    RECIPE.update(saved)


def test_only_caption_policy_and_experiment_identity_change():
    before = copy.deepcopy(captioned())
    after = copy.deepcopy(configure())
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    assert changed == {"experiment", "caption_policy", "cfg_dropout"}
    assert after["caption_policy"] == {mode: "empty" for mode in after["tasks"]}
    assert after["video_weight"] == 1 and after["cfg_dropout"] == 0
    check_checkpoint({"scheduler": {"recipe": after}})
    with pytest.raises(ValueError, match="experiment"):
        check_checkpoint({"scheduler": {"recipe": before}})


def test_real_noise_and_model_boundary_remove_poisoned_m2v_captions():
    from motion_expert_phase3_edge_image_joint.objectives import noise_batch
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3, OriginalModel
    configure()
    modes = ["video2motion", "motimg2video", "joint"]
    raw = dict(motion=torch.zeros(3, 7, 3), video=torch.zeros(3, 2, 3, 2, 2),
               neutral=torch.zeros(3, 30, 3), modes=modes, captions=["private caption"] * 3)
    assert noise_batch(raw, "cpu")[0]["captions"] == [""] * 3
    assert raw["captions"] == ["private caption"] * 3
    model = EdgePhase3.__new__(EdgePhase3)
    torch.nn.Module.__init__(model)
    with patch.object(OriginalModel, "forward", autospec=True,
                      side_effect=lambda self, **kwargs: kwargs):
        inputs = {key: torch.zeros(1) for key in ("motion", "neutral", "video", "motion_sigma", "video_sigma")}
        assert model(captions=raw["captions"], modes=modes, **inputs)["captions"] == [""] * 3


def checked_collate(rows):
    from motion_expert_phase3_edge_joint.data import collate
    assert RECIPE["video_weight"] == 1
    assert RECIPE["cfg_dropout"] == 0
    assert RECIPE["tasks"] == {"video2motion": .4, "motimg2video": .4, "joint": .2}
    assert RECIPE["conditioning_attention"] == "bidirectional"
    return collate(rows)


def test_spawned_worker_keeps_weight_one_and_never_returns_target_captions(tmp_path):
    from motion_expert_phase3_edge_joint.data import identity
    from .data import Phase3Dataset
    latent = tmp_path / "latent.npz"
    np.savez(latent, latents=np.zeros((48, 25, 16, 16), np.float16))
    np.save(tmp_path / "sample.motion.npy", np.zeros((1, 97, 283), np.float32))
    np.save(tmp_path / "sample.neutral.npy", np.zeros((30, 3), np.float32))
    row = dict(latent=str(latent), latent_identity=identity(latent), array="sample", row=0,
               caption="Private target caption", uuid="test", start=0)
    (tmp_path / "index.json").write_text(json.dumps(dict(rows=[row])))
    configure()
    with patch.dict("os.environ", {"PHASE3_IMAGE_JOINT_PATH_MAP": "{}",
                                  "PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION": "directional"}):
        batches = list(torch.utils.data.DataLoader(Phase3Dataset(tmp_path), batch_size=8,
            sampler=[(i, 0) for i in range(32)], num_workers=1,
            multiprocessing_context="spawn", collate_fn=checked_collate))
    assert {mode for batch in batches for mode in batch["modes"]} == {"video2motion", "motimg2video", "joint"}
    assert all(caption == "" for batch in batches for caption in batch["captions"])
