"""Regressions for caption leakage, independent clocks and known-image inputs."""
import copy
import json
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import numpy as np

from motion_expert_phase3_edge_joint import settings
from motion_expert_phase3_edge_joint.objectives import noise_batch as original_noise_batch
from .data import Phase3Dataset, OriginalDataset
from .model import EdgePhase3, OriginalModel
from .objectives import noise_batch
from .policy import check_checkpoint, configure
from .sample import sample_joint


class ImageJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.recipe = copy.deepcopy(settings.RECIPE)
        configure()

    def tearDown(self):
        settings.RECIPE.clear()
        settings.RECIPE.update(self.recipe)

    def test_dataset_removes_all_captions_and_preserves_metadata(self):
        dataset = Phase3Dataset.__new__(Phase3Dataset)
        for mode in ("joint", "video2motion", "motimg2video"):
            original = {"mode": mode, "caption": "private target caption", "draw": 123, "motion": torch.ones(2)}
            with mock.patch.object(OriginalDataset, "__getitem__", return_value=original):
                actual = dataset[0]
            self.assertEqual(actual["caption"], "")
            self.assertEqual(original["caption"], "private target caption")
            self.assertIs(actual["motion"], original["motion"])
            self.assertEqual(actual["draw"], 123)

    def test_model_sanitizes_before_both_specialist_paths(self):
        model = EdgePhase3.__new__(EdgePhase3)
        torch.nn.Module.__init__(model)
        inputs = {key: torch.zeros(1) for key in ("motion", "neutral", "video", "motion_sigma", "video_sigma")}
        with mock.patch.object(OriginalModel, "forward", autospec=True,
                               side_effect=lambda self, **kwargs: kwargs):
            actual = model(captions=["joint secret", "inverse secret", "forward caption"],
                           modes=["joint", "video2motion", "motimg2video"], **inputs)
        self.assertEqual(actual["captions"], ["", "", ""])
        self.assertIs(actual["motion"], inputs["motion"])

    def test_all_tasks_have_no_caption_conditioning_or_dropout(self):
        recipe = configure()
        self.assertEqual(recipe["caption_policy"],
                         {mode: "empty" for mode in recipe["tasks"]})
        self.assertEqual(recipe["cfg_dropout"], 0.0)
        raw = self.batch(["video2motion", "motimg2video", "joint"])
        poisoned, _ = noise_batch(raw, "cpu")
        self.assertEqual(poisoned["captions"], ["", "", ""])
        self.assertEqual(raw["captions"], ["secret"] * 3)

    def test_conditional_sampler_removes_caption_before_model(self):
        from .sample import sample
        for mode in ("video2motion", "motimg2video"):
            with mock.patch("motion_expert_phase3_edge_image_joint.sample.conditional_sample",
                            return_value="sampled") as call:
                self.assertEqual(sample(object(), mode=mode, caption="private target caption"), "sampled")
                self.assertEqual(call.call_args.kwargs["caption"], "")
            with self.assertRaisesRegex(ValueError, "no caption guidance"):
                sample(object(), mode=mode, caption="secret", guidance=2)

    @staticmethod
    def batch(modes):
        count = len(modes)
        return {"motion": torch.randn(count, 7, 3), "video": torch.randn(count, 2, 3, 2, 2),
                "neutral": torch.randn(count, 30, 3), "modes": modes, "captions": ["secret"] * count}

    def test_joint_times_are_independent_and_image_is_always_clean(self):
        torch.manual_seed(14)
        raw = self.batch(["joint"] * 1024)
        inputs, target = noise_batch(raw, "cpu")
        self.assertEqual(inputs["captions"], [""] * 1024)
        a, b = inputs["motion_sigma"], inputs["video_sigma"]
        self.assertTrue(bool(((a > 0) & (a < 1) & (b > 0) & (b < 1)).all()))
        self.assertLess(abs(torch.corrcoef(torch.stack((a, b)))[0, 1].item()), .12)
        self.assertTrue(torch.equal(inputs["video"][:, :, 0], raw["video"][:, :, 0]))
        self.assertFalse(torch.equal(inputs["motion"], raw["motion"]))
        self.assertFalse(torch.equal(inputs["video"][:, :, 1:], raw["video"][:, :, 1:]))
        self.assertTrue(torch.equal(target["motion"], raw["motion"]))

    def test_native_noising_rng_and_conditional_inputs_unchanged(self):
        raw = self.batch(["video2motion", "motimg2video", "joint"])
        torch.manual_seed(17)
        before, before_target = original_noise_batch(raw, "cpu")
        rng = torch.random.get_rng_state()
        torch.manual_seed(17)
        after, after_target = noise_batch(raw, "cpu")
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key in ("motion", "video", "neutral", "motion_sigma", "video_sigma"):
            self.assertTrue(torch.equal(before[key], after[key]), key)
        for key in before_target:
            self.assertTrue(torch.equal(before_target[key], after_target[key]), key)
        self.assertEqual(after["motion_sigma"][1].item(), 0)
        self.assertEqual(after["video_sigma"][0].item(), 0)
        self.assertTrue(torch.equal(after["motion"][1], raw["motion"][1]))
        self.assertTrue(torch.equal(after["video"][0], raw["video"][0]))
        self.assertEqual(after["captions"], ["", "", ""])

    def test_spawned_worker_restores_joint_policy_and_local_path_translation(self):
        from motion_expert_phase3_edge_joint.data import collate, identity
        with tempfile.TemporaryDirectory(prefix="image_joint_worker_") as directory:
            root = Path(directory)
            video = root / "latent.npz"
            np.savez(video, latents=np.zeros((48, 25, 16, 16), dtype=np.float16))
            np.save(root / "test.motion.npy", np.zeros((1, 97, 283), dtype=np.float32))
            np.save(root / "test.neutral.npy", np.zeros((30, 3), dtype=np.float32))
            row = {"latent": "/old_imagejoint_test/latent.npz", "latent_identity": identity(video),
                   "array": "test", "row": 0, "caption": "must disappear", "uuid": "test", "start": 0}
            (root / "index.json").write_text(json.dumps({"rows": [row]}))
            # Parent globals cannot stand in for initializing a spawned worker.
            settings.RECIPE["tasks"] = {"video2motion": 1.0}
            with mock.patch.dict(os.environ, {"PHASE3_IMAGE_JOINT_PATH_MAP": json.dumps({"/old_imagejoint_test": directory})}):
                dataset = Phase3Dataset(root)
                draw = next(i for i in range(100) if random.Random(1000003 * (i + 1)).random() >= .8)
                loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,
                    sampler=[(draw, 0)], multiprocessing_context="spawn", collate_fn=collate)
                actual = next(iter(loader))
            self.assertEqual(actual["modes"], ["joint"])
            self.assertEqual(actual["captions"], [""])
            self.assertEqual(tuple(actual["motion"].shape), (1, 97, 283))

    def test_coupled_sampling_uses_distinct_clocks_without_target_leakage(self):
        image = torch.randn(1, 48, 25, 16, 16)

        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, **kwargs):
                self.calls.append(kwargs)
                return {"motion": torch.zeros_like(kwargs["motion"]), "video": torch.zeros_like(kwargs["video"])}

        outputs = []
        for value in (10., -20.):
            model = Recorder()
            gt_motion = torch.full((1, 97, 283), value)
            gt_video = image.clone()
            gt_video[:, :, 1:] = value
            outputs.append(sample_joint(model, caption="secret", motion=gt_motion, video=gt_video,
                                        neutral=torch.zeros(1, 30, 3), seed=9, steps=5))
            self.assertTrue(any(not torch.equal(c["motion_sigma"], c["video_sigma"]) for c in model.calls))
            for call in model.calls:
                self.assertEqual(call["captions"], [""])
                self.assertEqual(call["modes"], ["joint"])
                self.assertTrue(torch.equal(call["video"][:, :, 0], image[:, :, 0]))
            self.assertFalse(torch.equal(model.calls[0]["motion"], gt_motion))
        for key in ("motion", "video"):
            self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), key)
        self.assertTrue(torch.equal(outputs[0]["video"][:, :, 0], image[:, :, 0]))
        with self.assertRaisesRegex(ValueError, "no caption guidance"):
            sample_joint(Recorder(), guidance=2)

    def test_checkpoint_cannot_be_misidentified_as_old_text_joint(self):
        recipe = copy.deepcopy(configure())
        check_checkpoint({"scheduler": {"recipe": recipe}})
        for field, wrong in (("joint_conditioning", "first_image_and_text"),
                             ("conditioning_attention", "directional"),
                             ("clean_motion_timestep", "omit"),
                             ("caption_policy", {"video2motion": "empty", "motimg2video": "caption_dropout_0.1", "joint": "empty"})):
            bad = copy.deepcopy(recipe)
            bad[field] = wrong
            with self.assertRaisesRegex(ValueError, field):
                check_checkpoint({"scheduler": {"recipe": bad}})
        with self.assertRaises(ValueError):
            check_checkpoint({})


if __name__ == "__main__":
    unittest.main()
