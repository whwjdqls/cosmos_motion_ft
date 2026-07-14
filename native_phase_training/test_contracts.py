"""Focused CPU tests for native Phase 1 train/inference contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader

from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.vfm.augmentors.duration_fps_text_timestamps import DEFAULT_TEMPLATE as DURATION_TEMPLATE
from cosmos_framework.data.vfm.augmentors.resolution_text_info import DEFAULT_VIDEO_TEMPLATE as RESOLUTION_TEMPLATE
from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.inference.action import _format_prompt as format_official_action_prompt
from cosmos_framework.inference.inference import _format_prompt_with_template

from native_phase_training import checkpoint_eval_callback
from native_phase_training.checkpoint_eval_callback import (
    NativeCheckpointEvalSubmitter,
    build_eval_submission_command,
)
from native_phase_training.latent_nymeria_dataset import (
    LatentAwareIterativeJointDataLoader,
    _format_prompt_for_mode,
)
from native_phase_training.prep_test_eval import (
    ACTION_CHUNK_SIZE,
    FPS,
    IMAGE_SIZE,
    NUM_FRAMES,
    RESOLUTION,
    SHIFT,
    build_inference_records,
)


def _prompt_sample(caption: str = "Walk forward") -> dict:
    return {
        "ai_caption": caption,
        "video": torch.zeros(3, NUM_FRAMES, 1, 1),
        "action": torch.zeros(ACTION_CHUNK_SIZE, 64),
        "conditioning_fps": torch.tensor(FPS),
        "image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
        "viewpoint": "ego_view",
    }


class PromptContractTest(unittest.TestCase):
    def test_action_prompt_matches_official_inference(self) -> None:
        sample = _prompt_sample()
        actual = _format_prompt_for_mode(copy.deepcopy(sample), "forward_dynamics", ActionPromptJsonFormatter())
        expected = format_official_action_prompt(
            prompt=sample["ai_caption"],
            view_point=sample["viewpoint"],
            video=sample["video"],
            action=sample["action"],
            fps=sample["conditioning_fps"],
            image_size=sample["image_size"],
        )
        self.assertEqual(actual["ai_caption"], json.loads(expected))
        self.assertEqual(json.dumps(actual["ai_caption"]), expected)

    def test_inverse_prompt_stays_exactly_empty(self) -> None:
        sample = _prompt_sample(caption="")
        actual = _format_prompt_for_mode(sample, "inverse_dynamics", ActionPromptJsonFormatter())
        self.assertEqual(actual["ai_caption"], "")

    def test_image2video_prompt_matches_official_inference(self) -> None:
        sample = _prompt_sample()
        sample.pop("action")
        actual = _format_prompt_for_mode(sample, "image2video", ActionPromptJsonFormatter())
        expected = _format_prompt_with_template(
            "Walk forward",
            fps=FPS,
            num_frames=NUM_FRAMES,
            duration_template=DURATION_TEMPLATE,
            resolution_template=RESOLUTION_TEMPLATE,
            h=IMAGE_SIZE,
            w=IMAGE_SIZE,
        )
        self.assertEqual(actual["ai_caption"], expected)
        self.assertNotIn("first-person", actual["ai_caption"])


class PackingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = LatentAwareIterativeJointDataLoader.__new__(LatentAwareIterativeJointDataLoader)
        self.loader.patch_spatial = 2
        self.loader.tokenizer_spatial_compression_factor = 16
        self.loader.tokenizer_temporal_compression_factor = 4
        self.loader.sound_latent_fps = 0
        self.loader.audio_sample_rate = 48000

    def test_cached_latents_are_counted(self) -> None:
        sample = {
            "text_token_ids": [torch.arange(11)],
            "video": [torch.empty(3, NUM_FRAMES, 1, 1)],
            "video_latents": torch.empty(1, 48, 25, 16, 16),
            "action": [torch.empty(ACTION_CHUNK_SIZE, 64)],
        }
        expected = 11 + 1 + (25 * 8 * 8 + 2) + ACTION_CHUNK_SIZE
        self.assertEqual(self.loader._compute_num_tokens_per_sample(sample), expected)
        self.assertLess(26 * expected, 45056)
        self.assertGreaterEqual(27 * expected, 45056)

    def test_nonlatent_batch_falls_back_to_parent_counter(self) -> None:
        sample = {
            "text_token_ids": [torch.arange(3)],
            "video": [torch.empty(3, NUM_FRAMES, 256, 256)],
        }
        self.assertEqual(self.loader._compute_num_tokens_per_sample(sample), 3 + 1 + 1600 + 2)

    def test_fixed_sample_cap_yields_exactly_four(self) -> None:
        sample = {
            "text_token_ids": torch.arange(11),
            "video": torch.empty(3, NUM_FRAMES, 1, 1),
            "video_latents": torch.empty(48, 25, 16, 16),
            "action": torch.empty(ACTION_CHUNK_SIZE, 64),
        }
        child = DataLoader(
            dataset=[sample] * 8,
            batch_size=1,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
        loader = LatentAwareIterativeJointDataLoader(
            dataloaders={"test": {"dataloader": child, "ratio": 1}},
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            patch_spatial=2,
            max_sequence_length=None,
            max_samples_per_batch=4,
            prewarm=False,
        )

        batch = next(iter(loader))
        self.assertEqual(batch["_num_samples"], 4)
        self.assertEqual(len(batch["video_latents"]), 4)


class EvaluationContractTest(unittest.TestCase):
    def test_records_pin_native_phase1_contract(self) -> None:
        records = build_inference_records(
            name="sample",
            first_frame=Path("/tmp/first.png"),
            gt_clip=Path("/tmp/clip.mp4"),
            action_path=Path("/tmp/action.json"),
            caption="Walk forward",
            seed=7,
        )
        for mode, record in records.items():
            self.assertEqual(record["model_mode"], mode)
            self.assertEqual(record["num_frames"], NUM_FRAMES)
            self.assertEqual(record["fps"], FPS)
            self.assertEqual(record["resolution"], RESOLUTION)
            self.assertEqual(record["shift"], SHIFT)
        for mode in ("inverse_dynamics", "forward_dynamics", "policy"):
            self.assertEqual(records[mode]["action_chunk_size"], ACTION_CHUNK_SIZE)
            self.assertEqual(records[mode]["image_size"], IMAGE_SIZE)
            self.assertEqual(records[mode]["num_steps"], 30)
            self.assertEqual(records[mode]["guidance"], 1.0)
        for record in records.values():
            self.assertEqual(record["aspect_ratio"], "1,1")
        self.assertEqual(records["image2video"]["num_steps"], 35)
        self.assertEqual(records["image2video"]["guidance"], 6.0)
        self.assertEqual(records["inverse_dynamics"]["prompt"], "")
        self.assertEqual(len({record["name"] for record in records.values()}), 4)
        for mode, record in records.items():
            self.assertTrue(record["name"].endswith(f"_{mode}"))

    def test_checkpoint_eval_submission_is_isolated_and_explicit(self) -> None:
        command = build_eval_submission_command(
            sbatch_script="/repo/sbatch_eval.sh",
            checkpoint_path=Path("/run/checkpoints/iter_000005000"),
            eval_input_dir=Path("/run/eval_inputs"),
            eval_output_dir=Path("/run/checkpoint_evals/iter_000005000"),
        )
        self.assertEqual(command[:2], ["sbatch", "--parsable"])
        self.assertIn("CHECKPOINT_PATH=/run/checkpoints/iter_000005000", command[2])
        self.assertIn("EVAL_INPUT_DIR=/run/eval_inputs", command[2])
        self.assertIn("EVAL_OUTPUT_DIR=/run/checkpoint_evals/iter_000005000", command[2])
        self.assertEqual(command[-1], "/repo/sbatch_eval.sh")

        disabled = NativeCheckpointEvalSubmitter(
            enabled=False,
            sbatch_script="/repo/sbatch_eval.sh",
            eval_input_dir="/run/eval_inputs",
        )
        disabled.on_save_checkpoint_success(iteration=5000)

    def test_checkpoint_eval_callback_submits_once_after_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint = run_dir / "checkpoints" / "iter_000005000"
            checkpoint.mkdir(parents=True)
            eval_inputs = run_dir / "eval_inputs"
            eval_inputs.mkdir()
            for name in ("fd_input.jsonl", "invdyn_input.jsonl", "policy_input.jsonl", "i2v_input.jsonl"):
                (eval_inputs / name).write_text("{}\n")

            callback = NativeCheckpointEvalSubmitter(
                enabled=True,
                sbatch_script="/repo/sbatch_eval.sh",
                eval_input_dir=str(eval_inputs),
            )
            callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(run_dir)))
            completed = SimpleNamespace(stdout="12345\n")
            with (
                mock.patch.object(checkpoint_eval_callback.distributed, "is_rank0", return_value=True),
                mock.patch.object(checkpoint_eval_callback.subprocess, "run", return_value=completed) as submit,
            ):
                callback.on_save_checkpoint_success(iteration=5000)
                callback.on_save_checkpoint_success(iteration=5000)

            submit.assert_called_once()
            marker = run_dir / "checkpoint_evals" / "submitted" / "iter_000005000.job"
            self.assertEqual(marker.read_text(), "12345\n")

    def test_checkpoint_eval_callback_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "checkpoints" / "iter_000005000").mkdir(parents=True)
            eval_inputs = run_dir / "eval_inputs"
            eval_inputs.mkdir()
            for name in ("fd_input.jsonl", "invdyn_input.jsonl", "policy_input.jsonl", "i2v_input.jsonl"):
                (eval_inputs / name).write_text("{}\n")
            (eval_inputs / "policy_input.jsonl").write_text("")

            callback = NativeCheckpointEvalSubmitter(
                enabled=True,
                sbatch_script="/repo/sbatch_eval.sh",
                eval_input_dir=str(eval_inputs),
            )
            callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(run_dir)))
            with (
                mock.patch.object(checkpoint_eval_callback.distributed, "is_rank0", return_value=True),
                mock.patch.object(checkpoint_eval_callback.subprocess, "run") as submit,
            ):
                callback.on_save_checkpoint_success(iteration=5000)

            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
