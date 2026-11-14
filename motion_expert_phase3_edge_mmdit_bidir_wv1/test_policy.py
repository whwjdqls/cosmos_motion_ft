import copy
import json
import os
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from motion_expert_phase3_edge_joint import settings
from motion_expert_phase3_edge_joint.data import collate, identity
from .data import Phase3Dataset
from .policy import configure, TASKS


def test_sampling_never_selects_joint_and_both_tasks_remain_active():
    previous = copy.deepcopy(settings.RECIPE)
    try:
        configure()
        draws = [settings.sample_task(random.Random(i)) for i in range(10000)]
        assert set(draws) == {"video2motion", "motimg2video"}
        assert abs(draws.count("video2motion") / len(draws) - .5) < .02
        assert settings.RECIPE["conditioning_attention"] == "bidirectional"
        assert settings.RECIPE["clean_motion_timestep"] == "zero"
        assert settings.RECIPE["video_weight"] == 1
        with patch.dict(os.environ, {"PHASE3_MMDIT_CONDITIONING_ATTENTION": "bidirectional"}):
            configure()
            assert settings.RECIPE["conditioning_attention"] == "bidirectional"
            assert settings.RECIPE["tasks"] == TASKS and "joint" not in settings.RECIPE["tasks"]
    finally:
        settings.RECIPE.clear(); settings.RECIPE.update(previous)


def test_spawned_worker_retains_two_task_policy_and_local_data_translation():
    previous = copy.deepcopy(settings.RECIPE)
    try:
        with tempfile.TemporaryDirectory(prefix="mmdit_worker_") as directory:
            root = Path(directory)
            video = root / "latent.npz"
            np.savez(video, latents=np.zeros((48, 25, 16, 16), np.float16))
            np.save(root / "test.motion.npy", np.zeros((1, 97, 283), np.float32))
            np.save(root / "test.neutral.npy", np.zeros((30, 3), np.float32))
            row = dict(latent="/old_mmdit_test/latent.npz", latent_identity=identity(video),
                       array="test", row=0, caption="The wearer walks.", uuid="test", start=0)
            (root / "index.json").write_text(json.dumps(dict(rows=[row])))
            settings.RECIPE["tasks"] = {"joint": 1.}
            with patch.dict(os.environ, {"PHASE3_IMAGE_JOINT_PATH_MAP": json.dumps({"/old_mmdit_test": directory})}):
                loader = torch.utils.data.DataLoader(Phase3Dataset(root), batch_size=8, num_workers=1,
                    sampler=[(i, 0) for i in range(32)], multiprocessing_context="spawn", collate_fn=collate)
                batches = list(loader)
            modes = sum([b["modes"] for b in batches], [])
            assert set(modes) == set(TASKS) and "joint" not in modes
            for batch in batches:
                assert all(caption == "" for caption in batch["captions"])
    finally:
        settings.RECIPE.clear(); settings.RECIPE.update(previous)
