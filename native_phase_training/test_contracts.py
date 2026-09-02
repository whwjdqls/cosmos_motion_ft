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
from torch import nn
from torch.utils.data import DataLoader

from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.data.vfm.augmentors.duration_fps_text_timestamps import DEFAULT_TEMPLATE as DURATION_TEMPLATE
from cosmos_framework.data.vfm.augmentors.resolution_text_info import DEFAULT_VIDEO_TEMPLATE as RESOLUTION_TEMPLATE
from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.action import _format_prompt as format_official_action_prompt
from cosmos_framework.inference.inference import _format_prompt_with_template

from native_phase_training import checkpoint_eval_callback
from native_phase_training.checkpoint_eval_callback import (
    NativeCheckpointEvalSubmitter,
    build_eval_submission_command,
)
from native_phase_training.evaluate_prefix_suite import _rgb_prefix_length, _source_name
from native_phase_training.latent_nymeria_dataset import (
    LatentAwareIterativeJointDataLoader,
    _format_prompt_for_mode,
    build_cached_index,
    latent_path,
    load_quality_filter_exclusions,
    replace_standalone_c_with_camera_wearer,
    replace_standalone_c_with_person,
    rgb_prefix_to_latent_frames,
    validate_training_cache_contract,
    validate_prefix_sampling,
)
from native_phase_training.latent_cache_contract import (
    CACHE_CONTRACT_KIND,
    CACHE_CONTRACT_VERSION,
    LatentCacheContract,
    ensure_latent_cache_contract,
)
from native_phase_training.prepare_phase1_eval_tier import convert_record
from native_phase_training.validate_eval_inputs import validate_record
from native_phase_training.camera_token_lora import (
    CameraTokenLoraLinear,
    build_camera_token_mask,
    camera_token_mask_context,
)
from native_phase_training.prep_test_eval import (
    ACTION_CHUNK_SIZE,
    FPS,
    IMAGE_SIZE,
    NUM_FRAMES,
    RESOLUTION,
    SHIFT,
    build_inference_records,
    build_prefix_inference_records,
)
from native_phase_training.prefix_inference import install_action_prefix_support
from native_phase_training.run_contract import (
    CONTRACT_FILENAME,
    contract_from_config,
    load_contract_for_checkpoint,
    persist_run_contract,
    resolve_eval_contract,
    write_eval_resolution,
)
from native_phase_training.run_latent_train import _scheduler_compatible_cpu_affinity
from native_phase_training.sanitize_prefix_inference_inputs import (
    runtime_mode_matches,
    sanitize_record,
)
from native_phase_training.visualize_checkpoint import (
    _annotated_video_filter,
    _video_frame_provenance,
)
from cosmos_framework.data.vfm.sequence_packing import PackedSequence, _pack_vision_tokens
from cosmos_framework.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss


def _prompt_sample(caption: str = "Walk forward") -> dict:
    return {
        "ai_caption": caption,
        "video": torch.zeros(3, NUM_FRAMES, 1, 1),
        "action": torch.zeros(ACTION_CHUNK_SIZE, 64),
        "conditioning_fps": torch.tensor(FPS),
        "image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
        "viewpoint": "ego_view",
    }


def _native_run_config(
    run_dir: Path,
    *,
    adaptation_mode: str,
    active_modes: tuple[str, ...],
    prefix_lengths: tuple[int, ...] = (1, 9, 17, 33, 49),
    model_resolution: str = "256",
    training_shift: float = 3.0,
    model_family: str = "nano",
    vision_loss_scale: float = 1.0,
    image_loss_scale: float | None = 1.0,
) -> SimpleNamespace:
    lora_enabled = adaptation_mode != "action_only"
    targets = (
        "k_proj_moe_gen,v_proj_moe_gen"
        if adaptation_mode == "camera_kv_lora"
        else "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    )
    streams = {
        mode: {
            "dataloader": {
                "dataset": {
                    "prefix_lengths": list(prefix_lengths),
                    "num_frames": NUM_FRAMES,
                    "fps": float(FPS),
                }
            }
        }
        for mode in active_modes
    }
    return SimpleNamespace(
        job=SimpleNamespace(path_local=str(run_dir)),
        model=SimpleNamespace(
            model_family=model_family,
            adaptation_mode=adaptation_mode,
            config=SimpleNamespace(
                lora_enabled=lora_enabled,
                lora_target_modules=targets,
                resolution=model_resolution,
                diffusion_expert_config={"base_fps": 24.0},
                rectified_flow_training_config={
                    "shift": {model_resolution: training_shift},
                    "loss_scale": vision_loss_scale,
                    "image_loss_scale": image_loss_scale,
                    "action_loss_weight": 10.0,
                },
            ),
        ),
        dataloader_train=SimpleNamespace(dataloaders=streams),
    )


class NativeRunContractTest(unittest.TestCase):
    def test_contract_captures_edge_fps_and_released_loss_scales(self) -> None:
        contract = contract_from_config(
            _native_run_config(
                Path("/run"),
                adaptation_mode="global_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy", "image2video"),
                model_family="edge",
                vision_loss_scale=10.0,
                image_loss_scale=None,
            )
        )
        self.assertEqual(contract.model_family, "edge")
        self.assertEqual(contract.conditioning_fps, 20.0)
        self.assertEqual(contract.base_fps, 24.0)
        self.assertEqual(contract.training_shift, 3.0)
        self.assertEqual(contract.inference_shift, 10.0)
        self.assertEqual(contract.vision_loss_scale, 10.0)
        self.assertIsNone(contract.image_loss_scale)
        self.assertEqual(contract.action_loss_weight, 10.0)
        self.assertEqual(contract.dropped_modes, ())

    def test_contract_captures_camera_kv_architecture_and_dropped_i2v(self) -> None:
        config = _native_run_config(
            Path("/run"),
            adaptation_mode="camera_kv_lora",
            active_modes=("forward_dynamics", "inverse_dynamics", "policy"),
        )
        contract = contract_from_config(config)
        self.assertEqual(contract.adaptation_mode, "camera_kv_lora")
        self.assertEqual(contract.dropped_modes, ("image2video",))
        self.assertEqual(contract.lora_target_modules, ("k_proj_moe_gen", "v_proj_moe_gen"))
        self.assertEqual(contract.training_prefix_lengths, (1, 9, 17, 33, 49))
        self.assertEqual(contract.model_resolution, "256")
        self.assertEqual(contract.training_shift, 3.0)
        self.assertEqual(contract.inference_shift, 3.0)

    def test_contract_captures_released_720_shift(self) -> None:
        contract = contract_from_config(
            _native_run_config(
                Path("/run"),
                adaptation_mode="global_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy", "image2video"),
                model_resolution="720",
                training_shift=10.0,
            )
        )
        self.assertEqual(contract.model_resolution, "720")
        self.assertEqual(contract.num_frames, NUM_FRAMES)
        self.assertEqual(contract.training_shift, 10.0)
        self.assertEqual(contract.inference_shift, 10.0)

    def test_persisted_contract_is_immutable_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            global_config = _native_run_config(
                run_dir,
                adaptation_mode="global_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy", "image2video"),
            )
            path = persist_run_contract(global_config)
            self.assertEqual(path, run_dir / CONTRACT_FILENAME)
            self.assertEqual(persist_run_contract(global_config), path)

            incompatible = _native_run_config(
                run_dir,
                adaptation_mode="camera_kv_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy"),
            )
            with self.assertRaisesRegex(RuntimeError, "contract mismatch on resume"):
                persist_run_contract(incompatible)

    def test_eval_resolves_saved_contract_and_rejects_conflicting_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            config = _native_run_config(
                run_dir,
                adaptation_mode="camera_kv_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy"),
            )
            persist_run_contract(config)
            checkpoint = run_dir / "checkpoints" / "iter_000005000"
            checkpoint.mkdir(parents=True)

            contract, source, resolved_checkpoint, resolved_run = resolve_eval_contract(
                checkpoint,
                environ={
                    "NATIVEP1_ADAPTATION_MODE": "camera_kv_lora",
                    "NYMERIA_DROP_MODES": "image2video",
                },
            )
            self.assertEqual(source, CONTRACT_FILENAME)
            self.assertEqual(contract.adaptation_mode, "camera_kv_lora")
            self.assertEqual(resolved_checkpoint, checkpoint.resolve())
            self.assertEqual(resolved_run, run_dir.resolve())

            with self.assertRaisesRegex(ValueError, "conflicts with the checkpoint contract"):
                resolve_eval_contract(
                    checkpoint,
                    environ={
                        "NATIVEP1_ADAPTATION_MODE": "global_lora",
                        "NYMERIA_DROP_MODES": "image2video",
                    },
                )
            with self.assertRaisesRegex(ValueError, "NYMERIA_DROP_MODES conflicts"):
                resolve_eval_contract(
                    checkpoint,
                    environ={
                        "NATIVEP1_ADAPTATION_MODE": "camera_kv_lora",
                        "NYMERIA_DROP_MODES": "",
                    },
                )

    def test_legacy_edge_contract_upgrades_to_official_inference_shift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            config = _native_run_config(
                run_dir,
                adaptation_mode="global_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy", "image2video"),
                model_family="edge",
                vision_loss_scale=10.0,
                image_loss_scale=None,
            )
            path = persist_run_contract(config)
            saved = json.loads(path.read_text())
            saved["schema_version"] = 3
            saved.pop("inference_shift")
            path.write_text(json.dumps(saved))
            checkpoint = run_dir / "checkpoints" / "iter_000000003"
            checkpoint.mkdir(parents=True)

            contract, _, _, _ = resolve_eval_contract(checkpoint, environ={})
            self.assertEqual(contract.training_shift, 3.0)
            self.assertEqual(contract.inference_shift, 10.0)

            with self.assertRaisesRegex(ValueError, "NATIVEP1_INFERENCE_SHIFT conflicts"):
                resolve_eval_contract(
                    checkpoint,
                    environ={"NATIVEP1_INFERENCE_SHIFT": "3"},
                )

    def test_legacy_config_recovery_does_not_default_to_global_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint = run_dir / "checkpoints" / "iter_000005000"
            checkpoint.mkdir(parents=True)
            config = _native_run_config(
                run_dir,
                adaptation_mode="action_only",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy"),
            )
            config_dict = {
                "model": {
                    "adaptation_mode": config.model.adaptation_mode,
                    "config": {
                        "lora_enabled": config.model.config.lora_enabled,
                        "lora_target_modules": config.model.config.lora_target_modules,
                    },
                },
                "dataloader_train": {"dataloaders": config.dataloader_train.dataloaders},
            }
            (run_dir / "config.yaml").write_text(json.dumps(config_dict))

            contract, source, _, _ = load_contract_for_checkpoint(checkpoint)
            self.assertEqual(source, "config.yaml (legacy recovery)")
            self.assertEqual(contract.adaptation_mode, "action_only")
            self.assertFalse(contract.lora_enabled)
            self.assertEqual(contract.dropped_modes, ("image2video",))

    def test_legacy_fixed_prefix_global_lora_config_is_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint = run_dir / "checkpoints" / "iter_000035000"
            checkpoint.mkdir(parents=True)
            streams = {
                mode: {"dataloader": {"dataset": {}}}
                for mode in ("forward_dynamics", "inverse_dynamics", "policy")
            }
            config_dict = {
                "model": {
                    "config": {
                        "lora_enabled": True,
                        "lora_target_modules": (
                            "q_proj_moe_gen,k_proj_moe_gen,"
                            "v_proj_moe_gen,o_proj_moe_gen"
                        ),
                    },
                },
                "dataloader_train": {"dataloaders": streams},
            }
            (run_dir / "config.yaml").write_text(json.dumps(config_dict))

            contract, source, _, _ = load_contract_for_checkpoint(checkpoint)
            self.assertEqual(source, "config.yaml (legacy recovery)")
            self.assertEqual(contract.adaptation_mode, "global_lora")
            self.assertEqual(contract.training_prefix_lengths, (1,))
            self.assertEqual(contract.dropped_modes, ("image2video",))

    def test_eval_resolution_writes_validated_shell_environment_and_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "run"
            config = _native_run_config(
                run_dir,
                adaptation_mode="camera_kv_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy"),
                prefix_lengths=(1,),
            )
            persist_run_contract(config)
            checkpoint = run_dir / "checkpoints" / "iter_000005000"
            checkpoint.mkdir(parents=True)
            record_path, env_path = write_eval_resolution(
                checkpoint_path=checkpoint,
                output_dir=root / "eval",
                environ={},
            )
            record = json.loads(record_path.read_text())
            self.assertEqual(record["adaptation_mode"], "camera_kv_lora")
            self.assertEqual(record["training_prefix_lengths"], [1])
            self.assertIn("NATIVEP1_ADAPTATION_MODE=camera_kv_lora", env_path.read_text())
            self.assertIn("NYMERIA_DROP_MODES=image2video", env_path.read_text())
            self.assertIn("NYMERIA_RESOLUTION=256", env_path.read_text())
            self.assertIn("NATIVEP1_EFFECTIVE_SHIFT=3.0", env_path.read_text())

    def test_edge_eval_uses_shift_ten_without_changing_training_shift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "edge-run"
            config = _native_run_config(
                run_dir,
                adaptation_mode="global_lora",
                active_modes=("forward_dynamics", "inverse_dynamics", "policy", "image2video"),
                model_family="edge",
                vision_loss_scale=10.0,
                image_loss_scale=None,
            )
            persist_run_contract(config)
            checkpoint = run_dir / "checkpoints" / "iter_000005000"
            checkpoint.mkdir(parents=True)
            record_path, env_path = write_eval_resolution(
                checkpoint_path=checkpoint,
                output_dir=root / "eval",
                environ={},
            )

            record = json.loads(record_path.read_text())
            environment = env_path.read_text()
            self.assertEqual(record["training_shift"], 3.0)
            self.assertEqual(record["inference_shift"], 10.0)
            self.assertIn("NATIVEP1_SHIFT_OVERRIDE=3.0", environment)
            self.assertIn("NATIVEP1_TRAINING_SHIFT=3.0", environment)
            self.assertIn("NATIVEP1_INFERENCE_SHIFT=10.0", environment)
            self.assertIn("NATIVEP1_EFFECTIVE_SHIFT=10.0", environment)

    def test_eval_shell_resolves_contract_before_importing_inference_config(self) -> None:
        script = Path(__file__).with_name("sbatch_checkpoint_eval.sh").read_text()
        self.assertLess(script.index("run_contract.py"), script.index("cosmos_framework.scripts.inference"))
        self.assertIn("resolved_run_contract.json", script)

        full71_wrapper = Path(__file__).with_name("sbatch_checkpoint_eval_full71.sh").read_text()
        self.assertIn("run_full71_all_checkpoints.sh", full71_wrapper)
        full71_driver = Path(__file__).with_name("run_full71_all_checkpoints.sh").read_text()
        self.assertLess(
            full71_driver.index("run_contract.py"),
            full71_driver.index("cosmos_framework.scripts.inference"),
        )
        self.assertIn("resolved_run_contract.json", full71_driver)


class PromptContractTest(unittest.TestCase):
    def test_standalone_subject_replacement_does_not_change_words(self) -> None:
        caption = "C carries a cup. ABC, c, coffee, C2, _C, and (C) stay distinct. C turns."
        expected = (
            "The camera wearer carries a cup. ABC, c, coffee, C2, _C, and "
            "(the camera wearer) stay distinct. The camera wearer turns."
        )
        self.assertEqual(replace_standalone_c_with_camera_wearer(caption), expected)

    def test_historical_person_subject_replacement_remains_available(self) -> None:
        self.assertEqual(
            replace_standalone_c_with_person("C walks, then C turns."),
            "The person walks, then the person turns.",
        )

    def test_gpu_cpu_affinity_stays_inside_scheduler_allocation(self) -> None:
        self.assertEqual(
            _scheduler_compatible_cpu_affinity({64, 65, 96}, {32, 33, 64, 65}),
            [64, 65],
        )
        self.assertEqual(
            _scheduler_compatible_cpu_affinity({128, 129}, {32, 33}),
            [32, 33],
        )

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

    def test_720_tier_four_clip_pack_stays_below_native_budget(self) -> None:
        sample = {
            "text_token_ids": [torch.arange(64)],
            "video": [torch.empty(3, NUM_FRAMES, 1, 1)],
            "video_latents": torch.empty(1, 48, 25, 40, 40),
            "action": [torch.empty(ACTION_CHUNK_SIZE, 64)],
        }
        per_clip = self.loader._compute_num_tokens_per_sample(sample)
        self.assertEqual(per_clip, 64 + 1 + (25 * 20 * 20 + 2) + ACTION_CHUNK_SIZE)
        self.assertLess(4 * per_clip, 45056)
        self.assertGreaterEqual(5 * per_clip, 45056)

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


class PrefixContractTest(unittest.TestCase):
    def test_visualization_marks_gt_prefix_and_generated_suffix(self) -> None:
        provenance = _video_frame_provenance(9, NUM_FRAMES)
        self.assertEqual(provenance["gt_reference_frames"], [0, 96])
        self.assertEqual(provenance["generated_panel"]["gt_condition_frames"], [0, 8])
        self.assertEqual(provenance["generated_panel"]["generated_frames"], [9, 96])

        generated_filter = _annotated_video_filter(
            input_index=1,
            output_label="pred",
            width=256,
            height=256,
            header_height=28,
            font_size=13,
            label="FD P9",
            prefix_length=9,
        )
        self.assertIn("color=lime", generated_filter)
        self.assertIn("enable='lt(n,9)'", generated_filter)
        self.assertIn("GT CONDITION", generated_filter)
        self.assertIn("color=red", generated_filter)
        self.assertIn("enable='gte(n,9)'", generated_filter)
        self.assertIn("GENERATED", generated_filter)

        reference_filter = _annotated_video_filter(
            input_index=0,
            output_label="gt",
            width=256,
            height=256,
            header_height=28,
            font_size=13,
            label="GT REFERENCE",
            prefix_length=None,
        )
        self.assertIn("color=lime", reference_filter)
        self.assertNotIn("color=red", reference_filter)
        self.assertNotIn("GENERATED", reference_filter)

    def test_exact_rgb_to_wan_latent_boundaries(self) -> None:
        self.assertEqual(
            [rgb_prefix_to_latent_frames(value, NUM_FRAMES) for value in (1, 9, 17, 33, 49)],
            [1, 3, 5, 9, 13],
        )
        with self.assertRaisesRegex(ValueError, r"1 \+ 4N"):
            rgb_prefix_to_latent_frames(8, NUM_FRAMES)

    def test_prefix_weights_are_validated(self) -> None:
        lengths, weights = validate_prefix_sampling([1, 9, 17], [1.0, 2.0, 3.0], NUM_FRAMES)
        self.assertEqual(lengths, (1, 9, 17))
        self.assertEqual(weights, (1.0, 2.0, 3.0))
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_prefix_sampling([1, 9], [1.0], NUM_FRAMES)

    def test_packer_marks_clean_prefix_and_suffix_only_loss(self) -> None:
        latent_t = 25
        latent_prefix = 9
        packed = PackedSequence()
        _pack_vision_tokens(
            packed_seq=packed,
            input_vision_tokens=torch.zeros(1, 2, latent_t, 4, 4),
            condition_frame_indexes_vision=list(range(latent_prefix)),
            input_timestep=500.0,
            curr_rope_id=0,
            latent_patch_size=2,
            temporal_compression_factor=4,
        )
        condition_mask = packed.vision.condition_mask[0].flatten()
        self.assertTrue(torch.equal(condition_mask[:latent_prefix], torch.ones(latent_prefix)))
        self.assertTrue(torch.equal(condition_mask[latent_prefix:], torch.zeros(latent_t - latent_prefix)))
        self.assertEqual(packed.vision.noisy_frame_indexes[0].tolist(), list(range(latent_prefix, latent_t)))
        tokens_per_frame = 4
        self.assertEqual(
            packed.vision.mse_loss_indexes,
            list(range(latent_prefix * tokens_per_frame, latent_t * tokens_per_frame)),
        )

    def test_prefix_does_not_shift_camera_alignment(self) -> None:
        plan = build_sequence_plan_from_mode(
            mode="forward_dynamics",
            video_length=NUM_FRAMES,
            action_length=ACTION_CHUNK_SIZE,
            video_temporal_downsample=4,
        )
        plan.condition_frame_indexes_vision = list(range(rgb_prefix_to_latent_frames(33, NUM_FRAMES)))
        self.assertEqual(plan.action_start_frame_offset, 1)
        self.assertEqual(plan.condition_frame_indexes_action, list(range(ACTION_CHUNK_SIZE)))
        self.assertEqual(plan.condition_frame_indexes_vision, list(range(9)))

    def test_active_normalization_gives_each_suffix_equal_sample_weight(self) -> None:
        class UnitWeightFlow:
            @staticmethod
            def train_time_weight(timestep, _kwargs):
                return torch.ones_like(timestep, dtype=torch.float32)

        pred = [torch.ones(1, 5, 1, 1), torch.ones(1, 5, 1, 1)]
        target = [torch.zeros_like(value) for value in pred]
        masks = [
            torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32).reshape(5, 1, 1),
            torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32).reshape(5, 1, 1),
        ]
        normalized, per_instance = compute_flow_matching_loss(
            pred=pred,
            target=target,
            condition_mask=masks,
            timesteps=torch.ones(2, 1),
            has_valid_tokens=True,
            rectified_flow=UnitWeightFlow(),
            tensor_kwargs_fp32={"dtype": torch.float32, "device": "cpu"},
            normalize_by_active=True,
        )
        self.assertTrue(torch.equal(per_instance, torch.ones(2)))
        self.assertEqual(float(normalized), 1.0)

        diluted, diluted_per_instance = compute_flow_matching_loss(
            pred=pred,
            target=target,
            condition_mask=masks,
            timesteps=torch.ones(2, 1),
            has_valid_tokens=True,
            rectified_flow=UnitWeightFlow(),
            tensor_kwargs_fp32={"dtype": torch.float32, "device": "cpu"},
            normalize_by_active=False,
        )
        self.assertLess(float(diluted_per_instance[1]), float(diluted_per_instance[0]))
        self.assertLess(float(diluted), 1.0)

    def test_official_inference_record_strips_only_local_metadata(self) -> None:
        record = {
            "model_mode": "forward_dynamics",
            "name": "sample_p009_forward_dynamics",
            "source_name": "sample",
            "rgb_prefix_length": 9,
            "latent_prefix_length": 3,
            "condition_frame_indexes_vision": [0, 1, 2],
            "prompt": "test",
        }
        sanitized = sanitize_record(record, "forward_dynamics")
        self.assertEqual(
            sanitized,
            {
                "model_mode": "forward_dynamics",
                "name": "sample_p009_forward_dynamics",
                "condition_frame_indexes_vision": [0, 1, 2],
                "prompt": "test",
            },
        )
        with self.assertRaisesRegex(ValueError, "condition indexes"):
            sanitize_record({**record, "condition_frame_indexes_vision": [0, 1]}, "forward_dynamics")
        with self.assertRaisesRegex(ValueError, "inconsistent with source_name"):
            sanitize_record({**record, "name": "different_p009_forward_dynamics"}, "forward_dynamics")
        with self.assertRaisesRegex(ValueError, "does not match latent prefix"):
            sanitize_record({**record, "rgb_prefix_length": 8}, "forward_dynamics")

    def test_legacy_fixed_prefix_record_is_supported(self) -> None:
        record = {
            "model_mode": "forward_dynamics",
            "name": "sample_forward_dynamics",
            "prompt": "test",
        }
        self.assertEqual(sanitize_record(record, "forward_dynamics"), record)
        self.assertEqual(_source_name(record), "sample")
        self.assertEqual(_rgb_prefix_length(record), 1)

    def test_edge_policy_is_sanitized_to_official_wam_mode(self) -> None:
        record = {
            "model_mode": "policy",
            "name": "sample_policy",
            "vision_path": "/input.png",
        }
        sanitized = sanitize_record(record, "policy", model_family="edge")
        self.assertEqual(sanitized["model_mode"], "wam")
        self.assertTrue(
            runtime_mode_matches(actual_mode="wam", canonical_mode="policy")
        )
        self.assertFalse(
            runtime_mode_matches(actual_mode="wam", canonical_mode="forward_dynamics")
        )

    def test_inference_caption_replaces_only_standalone_subject_marker(self) -> None:
        record = {
            "model_mode": "forward_dynamics",
            "name": "sample_forward_dynamics",
            "prompt": "C carries C2 near ABC, c, _C, and (C).",
        }
        sanitized = sanitize_record(
            record,
            "forward_dynamics",
            model_family="edge",
            replace_standalone_c=True,
            standalone_c_subject="camera_wearer",
        )
        self.assertEqual(
            sanitized["prompt"],
            "The camera wearer carries C2 near ABC, c, _C, and (the camera wearer).",
        )


class CameraTokenLoraContractTest(unittest.TestCase):
    @staticmethod
    def _linear() -> CameraTokenLoraLinear:
        torch.manual_seed(3)
        layer = CameraTokenLoraLinear(nn.Linear(4, 4, bias=False), rank=2, alpha=2)
        layer.lora_A = nn.Linear(4, 2, bias=False)
        layer.lora_B = nn.Linear(2, 4, bias=False)
        nn.init.normal_(layer.lora_A.weight)
        nn.init.zeros_(layer.lora_B.weight)
        return layer

    def test_zero_initialization_matches_base_and_mask_is_exact(self) -> None:
        layer = self._linear()
        x = torch.randn(5, 4)
        mask = torch.tensor([False, False, False, True, True])
        expected = nn.functional.linear(x, layer.weight, layer.bias)
        with camera_token_mask_context(mask):
            actual = layer(x)
        self.assertTrue(torch.equal(actual, expected))

        nn.init.normal_(layer.lora_B.weight)
        with camera_token_mask_context(mask):
            adapted = layer(x)
        self.assertTrue(torch.equal(adapted[~mask], expected[~mask]))
        self.assertFalse(torch.equal(adapted[mask], expected[mask]))

    def test_video_attention_loss_reaches_lora_but_not_frozen_base(self) -> None:
        key = self._linear()
        value = self._linear()
        nn.init.normal_(key.lora_B.weight)
        nn.init.normal_(value.lora_B.weight)
        key.weight.requires_grad_(False)
        value.weight.requires_grad_(False)
        x = torch.randn(4, 4)
        camera_mask = torch.tensor([False, False, True, True])
        visual_query = torch.randn(2, 4)
        with camera_token_mask_context(camera_mask):
            keys = key(x)
            values = value(x)
        video_hidden = torch.softmax(visual_query @ keys.T, dim=-1) @ values
        video_hidden.square().mean().backward()
        for layer in (key, value):
            self.assertIsNone(layer.weight.grad)
            self.assertGreater(float(layer.lora_A.weight.grad.norm()), 0.0)
            self.assertGreater(float(layer.lora_B.weight.grad.norm()), 0.0)

    def test_joint_action_indexes_map_to_generation_rows(self) -> None:
        source = SimpleNamespace(
            action=SimpleNamespace(sequence_indexes=torch.tensor([4, 5], dtype=torch.long))
        )
        pack = {
            "_causal_indices": torch.tensor([0, 1], dtype=torch.int32),
            "_full_indices": torch.tensor([2, 3, 4, 5], dtype=torch.int32),
            "full_only_seq": torch.zeros(4, 8),
            "is_sharded": False,
        }
        self.assertEqual(build_camera_token_mask(source, pack).tolist(), [False, False, True, True])
        source.action = None
        self.assertEqual(build_camera_token_mask(source, pack).tolist(), [False] * 4)

    def test_state_dict_keeps_native_lora_keys(self) -> None:
        original = self._linear()
        restored = self._linear()
        restored.load_state_dict(original.state_dict())
        self.assertEqual(set(original.state_dict()), {"weight", "lora_A.weight", "lora_B.weight"})


class LatentCacheContractTest(unittest.TestCase):
    @staticmethod
    def _contract() -> LatentCacheContract:
        return LatentCacheContract(
            schema_version=CACHE_CONTRACT_VERSION,
            kind=CACHE_CONTRACT_KIND,
            source_manifest="/data/manifest.jsonl",
            split_file="/data/split.json",
            split="train",
            source_window_count=120,
            expected_file_count=116,
            num_frames=NUM_FRAMES,
            fps=float(FPS),
            spatial_transform_resolution="480",
            model_resolution_tier="720",
            expected_image_hw=(640, 640),
            expected_latent_shape=(48, 25, 40, 40),
            expected_camera_shape=(ACTION_CHUNK_SIZE, 9),
            latent_dtype="float16",
            vae_path="/models/Wan2.2_VAE.pth",
            num_shards=24,
            limit_per_shard=None,
        )

    def test_cache_contract_is_atomic_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            contract = self._contract()
            path = ensure_latent_cache_contract(root, contract)
            self.assertEqual(path.name, "latent_cache_contract.json")
            self.assertEqual(ensure_latent_cache_contract(root, contract), path)

            incompatible = LatentCacheContract(
                **{
                    **contract.to_dict(),
                    "expected_latent_shape": (48, 25, 16, 16),
                }
            )
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                ensure_latent_cache_contract(root, incompatible)

    def test_training_rejects_wrong_model_tier_or_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            ensure_latent_cache_contract(root, self._contract())
            validated = validate_training_cache_contract(
                latent_root=str(root),
                num_frames=NUM_FRAMES,
                fps=float(FPS),
                model_resolution_tier="720",
                expected_latent_hw=40,
                expected_image_hw=640,
                require_contract=True,
            )
            self.assertIsNotNone(validated)
            with self.assertRaisesRegex(ValueError, "model tier mismatch"):
                validate_training_cache_contract(
                    latent_root=str(root),
                    num_frames=NUM_FRAMES,
                    fps=float(FPS),
                    model_resolution_tier="256",
                    expected_latent_hw=40,
                    expected_image_hw=640,
                    require_contract=True,
                )
            with self.assertRaisesRegex(ValueError, "spatial shape mismatch"):
                validate_training_cache_contract(
                    latent_root=str(root),
                    num_frames=NUM_FRAMES,
                    fps=float(FPS),
                    model_resolution_tier="720",
                    expected_latent_hw=16,
                    expected_image_hw=640,
                    require_contract=True,
                )


class HighTierEvaluationInputContractTest(unittest.TestCase):
    @staticmethod
    def _base(mode: str) -> dict:
        record = {
            "model_mode": mode,
            "fps": 20,
            "shift": 3.0,
            "num_frames": NUM_FRAMES,
            "resolution": "256",
            "aspect_ratio": "1,1",
        }
        if mode != "image2video":
            record.update({"image_size": 256, "action_chunk_size": ACTION_CHUNK_SIZE})
        return record

    def test_720_action_record_uses_image_size_and_explicit_t97(self) -> None:
        record = convert_record(
            self._base("forward_dynamics"),
            resolution_tier="720",
            shift=10.0,
        )
        self.assertEqual(record["num_frames"], NUM_FRAMES)
        self.assertEqual(record["image_size"], 480)
        self.assertNotIn("resolution", record)
        self.assertNotIn("aspect_ratio", record)
        validate_record(
            record,
            context="forward",
            expected_shift=10.0,
            expected_resolution="720",
            expected_num_frames=NUM_FRAMES,
        )

    def test_720_i2v_record_uses_640_square_output_bucket(self) -> None:
        record = convert_record(
            self._base("image2video"),
            resolution_tier="720",
            shift=10.0,
        )
        self.assertEqual(record["num_frames"], NUM_FRAMES)
        self.assertEqual(record["resolution"], "480")
        self.assertEqual(record["aspect_ratio"], "1,1")
        self.assertEqual(VIDEO_RES_SIZE_INFO["480"]["1,1"], (640, 640))
        validate_record(
            record,
            context="i2v",
            expected_shift=10.0,
            expected_resolution="720",
            expected_num_frames=NUM_FRAMES,
        )

    def test_missing_num_frames_is_rejected(self) -> None:
        record = convert_record(
            self._base("image2video"),
            resolution_tier="720",
            shift=10.0,
        )
        record.pop("num_frames")
        with self.assertRaisesRegex(ValueError, "num_frames is required"):
            validate_record(
                record,
                context="i2v",
                expected_shift=10.0,
                expected_resolution="720",
                expected_num_frames=NUM_FRAMES,
            )


class QualityFilterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        build_cached_index.cache_clear()
        load_quality_filter_exclusions.cache_clear()

    def tearDown(self) -> None:
        build_cached_index.cache_clear()
        load_quality_filter_exclusions.cache_clear()

    @staticmethod
    def _artifact(*, num_frames: int = NUM_FRAMES, duplicate: bool = False) -> dict:
        exclusion = {
            "split": "train",
            "uuid": "S01/sequence",
            "start": 0,
            "end": NUM_FRAMES,
            "dataset_row_multiplicity": 2,
            "reasons": ["camera_translation_jump"],
            "metrics": {"max_camera_translation_step_m": 1.0},
        }
        exclusions = [exclusion, copy.deepcopy(exclusion)] if duplicate else [exclusion]
        return {
            "kind": "nymeria_camera_motion_quality_filter",
            "version": 1,
            "num_frames": num_frames,
            "summary_by_split": {
                "train": {"excluded_unique_physical_windows": 2 if duplicate else 1},
                "test": {"excluded_unique_physical_windows": 0},
            },
            "excluded_windows": exclusions,
        }

    def test_one_physical_exclusion_removes_duplicate_caption_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = root / "manifest.jsonl"
            split_file = root / "split.json"
            latent_root = root / "latents"
            latent = Path(latent_path("S01/sequence", 0, str(latent_root)))
            latent.parent.mkdir(parents=True)
            latent.touch()
            record = {
                "uuid": "S01/sequence",
                "nb_frames": NUM_FRAMES,
                "camera_path": "/camera.npz",
                "vision_path": "/video.mp4",
                "t2w_windows": [
                    {"usable": True, "caption": "first", "start_frame": 0, "end_frame": NUM_FRAMES},
                    {"usable": True, "caption": "second", "start_frame": 0, "end_frame": NUM_FRAMES},
                ],
            }
            manifest.write_text(json.dumps(record) + "\n")
            split_file.write_text(json.dumps({"train": ["S01/sequence"], "test": []}))
            filter_path = root / "filter.json"
            filter_path.write_text(json.dumps(self._artifact()))

            unfiltered = build_cached_index(
                str(manifest), str(split_file), "train", NUM_FRAMES, str(latent_root)
            )
            filtered = build_cached_index(
                str(manifest),
                str(split_file),
                "train",
                NUM_FRAMES,
                str(latent_root),
                str(filter_path),
            )
            self.assertEqual(len(unfiltered), 2)
            self.assertEqual(filtered, [])

    def test_filter_rejects_wrong_temporal_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            filter_path = Path(temporary_dir) / "filter.json"
            filter_path.write_text(json.dumps(self._artifact(num_frames=33)))
            with self.assertRaisesRegex(ValueError, "T mismatch"):
                load_quality_filter_exclusions(str(filter_path), NUM_FRAMES)

    def test_filter_rejects_duplicate_physical_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            filter_path = Path(temporary_dir) / "filter.json"
            filter_path.write_text(json.dumps(self._artifact(duplicate=True)))
            with self.assertRaisesRegex(ValueError, "duplicate physical window"):
                load_quality_filter_exclusions(str(filter_path), NUM_FRAMES)


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

    def test_prefix_records_cover_all_visual_tasks(self) -> None:
        prefixes = {value: Path(f"/tmp/prefix_{value}.mp4") for value in (1, 9, 17, 33, 49)}
        records = build_prefix_inference_records(
            name="sample",
            gt_clip=Path("/tmp/clip.mp4"),
            prefix_paths=prefixes,
            action_path=Path("/tmp/action.json"),
            caption="Walk forward",
            seed=7,
        )
        self.assertEqual(len(records["inverse_dynamics"]), 1)
        for mode in ("forward_dynamics", "policy", "image2video"):
            self.assertEqual([record["rgb_prefix_length"] for record in records[mode]], list(prefixes))
            self.assertEqual(
                [record["latent_prefix_length"] for record in records[mode]],
                [1, 3, 5, 9, 13],
            )

    def test_official_action_inference_receives_exact_prefix_plan(self) -> None:
        import cosmos_framework.inference.inference as inference_module

        original = inference_module.get_sample_data
        installed_before = getattr(inference_module, "_native_phase_prefix_patch_installed", False)
        if installed_before:
            delattr(inference_module, "_native_phase_prefix_patch_installed")
        plan = SimpleNamespace(condition_frame_indexes_vision=[0])

        def fake_get_sample_data(_sample_args, _model, *, device="cuda"):
            del device
            return {"sequence_plan": [plan]}

        try:
            inference_module.get_sample_data = fake_get_sample_data
            install_action_prefix_support()
            sample_args = SimpleNamespace(
                model_mode=SimpleNamespace(value="forward_dynamics"),
                condition_frame_indexes_vision=[0, 1, 2],
            )
            inference_module.get_sample_data(sample_args, object(), device="cpu")
            self.assertEqual(plan.condition_frame_indexes_vision, [0, 1, 2])
            sample_args.condition_frame_indexes_vision = [0, 2]
            with self.assertRaisesRegex(ValueError, "contiguous causal prefix"):
                inference_module.get_sample_data(sample_args, object(), device="cpu")
        finally:
            inference_module.get_sample_data = original
            if installed_before:
                inference_module._native_phase_prefix_patch_installed = True
            elif hasattr(inference_module, "_native_phase_prefix_patch_installed"):
                delattr(inference_module, "_native_phase_prefix_patch_installed")

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

    def test_checkpoint_eval_callback_obeys_ten_thousand_step_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            for iteration in (5000, 10000):
                (run_dir / "checkpoints" / f"iter_{iteration:09d}").mkdir(parents=True)
            eval_inputs = run_dir / "eval_inputs"
            eval_inputs.mkdir()
            for name in ("fd_input.jsonl", "invdyn_input.jsonl", "policy_input.jsonl", "i2v_input.jsonl"):
                (eval_inputs / name).write_text("{}\n")

            callback = NativeCheckpointEvalSubmitter(
                enabled=True,
                sbatch_script="/repo/sbatch_eval.sh",
                eval_input_dir=str(eval_inputs),
                every_n_iterations=10000,
            )
            callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(run_dir)))
            completed = SimpleNamespace(stdout="12345\n")
            with (
                mock.patch.object(checkpoint_eval_callback.distributed, "is_rank0", return_value=True),
                mock.patch.object(checkpoint_eval_callback.subprocess, "run", return_value=completed) as submit,
            ):
                callback.on_save_checkpoint_success(iteration=5000)
                submit.assert_not_called()
                callback.on_save_checkpoint_success(iteration=10000)
                submit.assert_called_once()

    def test_full71_callback_requires_only_inverse_and_forward_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "checkpoints" / "iter_000010000").mkdir(parents=True)
            eval_inputs = run_dir / "eval_inputs"
            eval_inputs.mkdir()
            for name in ("fd_input.jsonl", "invdyn_input.jsonl"):
                (eval_inputs / name).write_text("{}\n")

            callback = NativeCheckpointEvalSubmitter(
                enabled=True,
                sbatch_script="/repo/sbatch_eval_full71.sh",
                eval_input_dir=str(eval_inputs),
                output_subdir="eval_full71_inverse_forward",
                every_n_iterations=10000,
                required_input_files=("fd_input.jsonl", "invdyn_input.jsonl"),
            )
            callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(run_dir)))
            completed = SimpleNamespace(stdout="54321\n")
            with (
                mock.patch.object(checkpoint_eval_callback.distributed, "is_rank0", return_value=True),
                mock.patch.object(checkpoint_eval_callback.subprocess, "run", return_value=completed) as submit,
            ):
                callback.on_save_checkpoint_success(iteration=10000)

            submit.assert_called_once()
            marker = (
                run_dir
                / "eval_full71_inverse_forward"
                / "submitted"
                / "iter_000010000.job"
            )
            self.assertEqual(marker.read_text(), "54321\n")


if __name__ == "__main__":
    unittest.main()
