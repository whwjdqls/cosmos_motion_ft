"""Persist and resolve architecture-critical native Phase-1 run settings."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


CONTRACT_FILENAME = "native_phase1_contract.json"
RESOLVED_CONTRACT_FILENAME = "resolved_run_contract.json"
RESOLVED_ENV_FILENAME = "resolved_run_contract.env"
SCHEMA_VERSION = 4
COSMOS_FRAMEWORK_EDGE_INFERENCE_SHIFT = 10.0

ALL_TASKS = frozenset({"forward_dynamics", "inverse_dynamics", "policy", "image2video"})
ADAPTATION_MODES = frozenset({"global_lora", "action_only", "camera_kv_lora"})


def _value(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        if key not in container:
            raise KeyError(key)
        return container[key]
    return getattr(container, key)


def _parse_targets(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        targets = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        targets = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise TypeError(f"lora_target_modules must be a string or list, got {type(value).__name__}")
    if not targets:
        raise ValueError("lora_target_modules must not be empty")
    return targets


def _parse_drop_modes(value: str) -> tuple[str, ...]:
    modes = tuple(sorted({part.strip() for part in value.split(",") if part.strip()}))
    unknown = set(modes) - ALL_TASKS
    if unknown:
        raise ValueError(f"unknown dropped modes: {sorted(unknown)}")
    return modes


def _infer_adaptation_mode(*, lora_enabled: bool, lora_target_modules: tuple[str, ...]) -> str:
    """Recover the adaptation mode for runs created before it was persisted."""

    if not lora_enabled:
        return "action_only"
    targets = set(lora_target_modules)
    if targets == {
        "q_proj_moe_gen",
        "k_proj_moe_gen",
        "v_proj_moe_gen",
        "o_proj_moe_gen",
    }:
        return "global_lora"
    if targets == {"k_proj_moe_gen", "v_proj_moe_gen"}:
        return "camera_kv_lora"
    raise ValueError(
        "cannot infer adaptation mode from legacy LoRA configuration: "
        f"enabled={lora_enabled} targets={lora_target_modules}"
    )


@dataclass(frozen=True)
class NativePhase1RunContract:
    schema_version: int
    adaptation_mode: str
    active_modes: tuple[str, ...]
    dropped_modes: tuple[str, ...]
    lora_enabled: bool
    lora_target_modules: tuple[str, ...]
    training_prefix_lengths: tuple[int, ...]
    model_resolution: str
    num_frames: int
    training_shift: float
    inference_shift: float
    model_family: str
    conditioning_fps: float
    base_fps: float
    vision_loss_scale: float
    image_loss_scale: float | None
    action_loss_weight: float

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported native Phase-1 contract schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.adaptation_mode not in ADAPTATION_MODES:
            raise ValueError(f"unsupported adaptation mode: {self.adaptation_mode!r}")
        active = set(self.active_modes)
        dropped = set(self.dropped_modes)
        if not active or active | dropped != ALL_TASKS or active & dropped:
            raise ValueError(
                f"active/dropped modes do not partition {sorted(ALL_TASKS)}: "
                f"active={sorted(active)} dropped={sorted(dropped)}"
            )
        if self.adaptation_mode == "action_only" and self.lora_enabled:
            raise ValueError("action_only contract cannot enable LoRA")
        if self.adaptation_mode != "action_only" and not self.lora_enabled:
            raise ValueError(f"{self.adaptation_mode} contract must enable LoRA")
        expected_targets = {
            "global_lora": {
                "q_proj_moe_gen",
                "k_proj_moe_gen",
                "v_proj_moe_gen",
                "o_proj_moe_gen",
            },
            "camera_kv_lora": {"k_proj_moe_gen", "v_proj_moe_gen"},
        }
        if self.adaptation_mode in expected_targets and set(self.lora_target_modules) != expected_targets[
            self.adaptation_mode
        ]:
            raise ValueError(
                f"{self.adaptation_mode} has unexpected LoRA targets: {self.lora_target_modules}"
            )
        if self.adaptation_mode in {"action_only", "camera_kv_lora"} and "image2video" in active:
            raise ValueError(f"{self.adaptation_mode} contract must exclude image2video from training")
        if not self.training_prefix_lengths or any(value <= 0 for value in self.training_prefix_lengths):
            raise ValueError(f"invalid training prefix lengths: {self.training_prefix_lengths}")
        if not self.model_resolution:
            raise ValueError("model_resolution must not be empty")
        if self.num_frames <= 0 or self.num_frames % 4 != 1:
            raise ValueError(f"num_frames must be positive and 4N+1, got {self.num_frames}")
        if self.training_shift <= 0.0:
            raise ValueError(f"training_shift must be positive, got {self.training_shift}")
        if self.inference_shift <= 0.0:
            raise ValueError(f"inference_shift must be positive, got {self.inference_shift}")
        if self.model_family not in {"nano", "edge"}:
            raise ValueError(f"unsupported model family: {self.model_family!r}")
        if self.conditioning_fps <= 0.0 or self.base_fps <= 0.0:
            raise ValueError(
                f"FPS values must be positive: conditioning={self.conditioning_fps} base={self.base_fps}"
            )
        if self.vision_loss_scale < 0.0 or self.action_loss_weight < 0.0:
            raise ValueError("vision/action loss scales must be non-negative")
        if self.image_loss_scale is not None and self.image_loss_scale < 0.0:
            raise ValueError("image_loss_scale must be None or non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativePhase1RunContract":
        legacy_expected = {
            "schema_version",
            "adaptation_mode",
            "active_modes",
            "dropped_modes",
            "lora_enabled",
            "lora_target_modules",
            "training_prefix_lengths",
        }
        schema2_expected = legacy_expected | {"model_resolution", "num_frames", "training_shift"}
        schema3_expected = schema2_expected | {
            "model_family",
            "conditioning_fps",
            "base_fps",
            "vision_loss_scale",
            "image_loss_scale",
            "action_loss_weight",
        }
        schema4_expected = schema3_expected | {"inference_shift"}
        schema_version = int(value.get("schema_version", -1))
        if schema_version == 1:
            missing = legacy_expected - set(value)
            extra = set(value) - legacy_expected
            if missing or extra:
                raise ValueError(
                    f"invalid legacy run contract fields: missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )
            upgraded = dict(value)
            upgraded.update(
                {
                    "schema_version": 2,
                    "model_resolution": "256",
                    "num_frames": 97,
                    "training_shift": 3.0,
                }
            )
            value = upgraded
            schema_version = 2

        if schema_version == 2:
            missing = schema2_expected - set(value)
            extra = set(value) - schema2_expected
            if missing or extra:
                raise ValueError(
                    f"invalid schema-2 run contract fields: missing={sorted(missing)} extra={sorted(extra)}"
                )
            upgraded = dict(value)
            upgraded.update(
                {
                    "schema_version": 3,
                    "model_family": "nano",
                    "conditioning_fps": 20.0,
                    "base_fps": 24.0,
                    "vision_loss_scale": 1.0,
                    "image_loss_scale": 1.0,
                    "action_loss_weight": 10.0,
                }
            )
            value = upgraded
            schema_version = 3

        if schema_version == 3:
            missing = schema3_expected - set(value)
            extra = set(value) - schema3_expected
            if missing or extra:
                raise ValueError(
                    f"invalid schema-3 run contract fields: missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )
            upgraded = dict(value)
            model_family = str(value["model_family"]).strip().lower()
            upgraded.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "inference_shift": (
                        COSMOS_FRAMEWORK_EDGE_INFERENCE_SHIFT
                        if model_family == "edge"
                        else float(value["training_shift"])
                    ),
                }
            )
            value = upgraded

        missing = schema4_expected - set(value)
        extra = set(value) - schema4_expected
        if missing or extra:
            raise ValueError(f"invalid run contract fields: missing={sorted(missing)} extra={sorted(extra)}")
        return cls(
            schema_version=int(value["schema_version"]),
            adaptation_mode=str(value["adaptation_mode"]),
            active_modes=tuple(str(mode) for mode in value["active_modes"]),
            dropped_modes=tuple(str(mode) for mode in value["dropped_modes"]),
            lora_enabled=bool(value["lora_enabled"]),
            lora_target_modules=tuple(str(module) for module in value["lora_target_modules"]),
            training_prefix_lengths=tuple(int(length) for length in value["training_prefix_lengths"]),
            model_resolution=str(value["model_resolution"]),
            num_frames=int(value["num_frames"]),
            training_shift=float(value["training_shift"]),
            inference_shift=float(value["inference_shift"]),
            model_family=str(value["model_family"]),
            conditioning_fps=float(value["conditioning_fps"]),
            base_fps=float(value["base_fps"]),
            vision_loss_scale=float(value["vision_loss_scale"]),
            image_loss_scale=(
                None if value["image_loss_scale"] is None else float(value["image_loss_scale"])
            ),
            action_loss_weight=float(value["action_loss_weight"]),
        )


def contract_from_config(config: Any) -> NativePhase1RunContract:
    model = _value(config, "model")
    model_config = _value(model, "config")
    dataloader_train = _value(config, "dataloader_train")
    dataloaders = _value(dataloader_train, "dataloaders")
    if not isinstance(dataloaders, Mapping):
        raise TypeError("dataloader_train.dataloaders must be a mapping")

    active_modes = tuple(sorted(str(mode) for mode in dataloaders))
    unknown = set(active_modes) - ALL_TASKS
    if unknown:
        raise ValueError(f"unknown active training modes: {sorted(unknown)}")
    dropped_modes = tuple(sorted(ALL_TASKS - set(active_modes)))

    prefix_sets: set[tuple[int, ...]] = set()
    num_frame_sets: set[int] = set()
    fps_sets: set[float] = set()
    for mode, stream in dataloaders.items():
        try:
            loader = _value(stream, "dataloader")
            dataset = _value(loader, "dataset")
            try:
                configured_prefixes = _value(dataset, "prefix_lengths")
            except (AttributeError, KeyError):
                # Fixed-prefix Phase-1 runs created before the variable-prefix
                # ablations always conditioned visual tasks on one RGB frame.
                configured_prefixes = (1,)
            prefixes = tuple(int(length) for length in configured_prefixes)
            try:
                configured_num_frames = int(_value(dataset, "num_frames"))
            except (AttributeError, KeyError):
                configured_num_frames = 97
            try:
                configured_fps = float(_value(dataset, "fps"))
            except (AttributeError, KeyError):
                configured_fps = 20.0
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(f"cannot resolve prefix lengths for training mode {mode!r}") from error
        prefix_sets.add(prefixes)
        num_frame_sets.add(configured_num_frames)
        fps_sets.add(configured_fps)
    if len(prefix_sets) != 1:
        raise ValueError(f"training streams disagree on prefix lengths: {sorted(prefix_sets)}")
    if len(num_frame_sets) != 1:
        raise ValueError(f"training streams disagree on num_frames: {sorted(num_frame_sets)}")
    if len(fps_sets) != 1:
        raise ValueError(f"training streams disagree on conditioning FPS: {sorted(fps_sets)}")

    lora_enabled = bool(_value(model_config, "lora_enabled"))
    lora_target_modules = _parse_targets(_value(model_config, "lora_target_modules"))
    try:
        adaptation_mode = str(_value(model, "adaptation_mode")).strip().lower()
    except (AttributeError, KeyError):
        adaptation_mode = _infer_adaptation_mode(
            lora_enabled=lora_enabled,
            lora_target_modules=lora_target_modules,
        )
    try:
        model_resolution = str(_value(model_config, "resolution"))
    except (AttributeError, KeyError):
        model_resolution = "256"
    try:
        flow_config = _value(model_config, "rectified_flow_training_config")
        shift_config = _value(flow_config, "shift")
        if isinstance(shift_config, Mapping):
            training_shift = float(shift_config[model_resolution])
        else:
            training_shift = float(shift_config)
    except (AttributeError, KeyError, TypeError):
        if model_resolution != "256":
            raise ValueError(
                f"legacy config does not record a shift for resolution {model_resolution!r}"
            )
        training_shift = 3.0

    try:
        model_family = str(_value(model, "model_family")).strip().lower()
    except (AttributeError, KeyError):
        try:
            vlm_config = _value(model_config, "vlm_config")
            model_name = str(_value(vlm_config, "model_name")).lower()
            model_family = "edge" if "edge" in model_name else "nano"
        except (AttributeError, KeyError):
            model_family = "nano"
    try:
        diffusion_config = _value(model_config, "diffusion_expert_config")
        base_fps = float(_value(diffusion_config, "base_fps"))
    except (AttributeError, KeyError):
        base_fps = 24.0
    flow_config = _value(model_config, "rectified_flow_training_config")
    try:
        vision_loss_scale = float(_value(flow_config, "loss_scale"))
    except (AttributeError, KeyError):
        vision_loss_scale = 10.0 if model_family == "edge" else 1.0
    try:
        image_loss_scale_raw = _value(flow_config, "image_loss_scale")
    except (AttributeError, KeyError):
        image_loss_scale_raw = vision_loss_scale
    image_loss_scale = None if image_loss_scale_raw is None else float(image_loss_scale_raw)
    try:
        action_loss_weight = float(_value(flow_config, "action_loss_weight"))
    except (AttributeError, KeyError):
        action_loss_weight = 10.0

    return NativePhase1RunContract(
        schema_version=SCHEMA_VERSION,
        adaptation_mode=adaptation_mode,
        active_modes=active_modes,
        dropped_modes=dropped_modes,
        lora_enabled=lora_enabled,
        lora_target_modules=lora_target_modules,
        training_prefix_lengths=next(iter(prefix_sets)),
        model_resolution=model_resolution,
        num_frames=next(iter(num_frame_sets)),
        training_shift=training_shift,
        inference_shift=(
            COSMOS_FRAMEWORK_EDGE_INFERENCE_SHIFT if model_family == "edge" else training_shift
        ),
        model_family=model_family,
        conditioning_fps=next(iter(fps_sets)),
        base_fps=base_fps,
        vision_loss_scale=vision_loss_scale,
        image_loss_scale=image_loss_scale,
        action_loss_weight=action_loss_weight,
    )


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(contents)
    os.replace(temporary, path)


def load_run_contract(path: Path) -> NativePhase1RunContract:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read native Phase-1 run contract {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"native Phase-1 run contract must contain a JSON object: {path}")
    return NativePhase1RunContract.from_dict(raw)


def persist_run_contract(config: Any) -> Path:
    """Atomically create the run contract, rejecting incompatible resume settings."""

    run_dir = Path(str(_value(_value(config, "job"), "path_local")))
    path = run_dir / CONTRACT_FILENAME
    contract = contract_from_config(config)
    if path.exists():
        existing = load_run_contract(path)
        if existing != contract:
            raise RuntimeError(
                f"native Phase-1 run contract mismatch on resume: {path}\n"
                f"saved={existing.to_dict()}\ncurrent={contract.to_dict()}"
            )
        return path
    _atomic_write(path, json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _checkpoint_and_run_dir(checkpoint_path: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint_path.expanduser().resolve()
    if checkpoint.name == "model":
        checkpoint = checkpoint.parent
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    if checkpoint.parent.name != "checkpoints" or not checkpoint.name.startswith("iter_"):
        raise ValueError(
            "checkpoint path must be <run>/checkpoints/iter_XXXXXXXXX or its model subdirectory: "
            f"{checkpoint}"
        )
    return checkpoint, checkpoint.parent.parent


def load_contract_for_checkpoint(checkpoint_path: Path) -> tuple[NativePhase1RunContract, str, Path, Path]:
    checkpoint, run_dir = _checkpoint_and_run_dir(checkpoint_path)
    contract_path = run_dir / CONTRACT_FILENAME
    if contract_path.is_file():
        return load_run_contract(contract_path), CONTRACT_FILENAME, checkpoint, run_dir

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"checkpoint has neither {CONTRACT_FILENAME} nor a legacy config.yaml: {checkpoint}"
        )
    try:
        config = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read legacy run config {config_path}: {error}") from error
    if not isinstance(config, Mapping):
        raise ValueError(f"legacy run config must contain a mapping: {config_path}")
    return contract_from_config(config), "config.yaml (legacy recovery)", checkpoint, run_dir


def resolve_eval_contract(
    checkpoint_path: Path, environ: Mapping[str, str] | None = None
) -> tuple[NativePhase1RunContract, str, Path, Path]:
    contract, source, checkpoint, run_dir = load_contract_for_checkpoint(checkpoint_path)
    environment = os.environ if environ is None else environ

    if "NATIVEP1_ADAPTATION_MODE" in environment:
        requested_mode = environment["NATIVEP1_ADAPTATION_MODE"].strip().lower()
        if requested_mode != contract.adaptation_mode:
            raise ValueError(
                "NATIVEP1_ADAPTATION_MODE conflicts with the checkpoint contract: "
                f"environment={requested_mode!r} checkpoint={contract.adaptation_mode!r}"
            )
    if "NATIVEP1_MODEL_FAMILY" in environment:
        requested_family = environment["NATIVEP1_MODEL_FAMILY"].strip().lower()
        if requested_family != contract.model_family:
            raise ValueError(
                "NATIVEP1_MODEL_FAMILY conflicts with the checkpoint contract: "
                f"environment={requested_family!r} checkpoint={contract.model_family!r}"
            )
    if "NYMERIA_DROP_MODES" in environment:
        requested_drops = _parse_drop_modes(environment["NYMERIA_DROP_MODES"])
        if requested_drops != contract.dropped_modes:
            raise ValueError(
                "NYMERIA_DROP_MODES conflicts with the checkpoint contract: "
                f"environment={requested_drops} checkpoint={contract.dropped_modes}"
            )
    if "NYMERIA_RESOLUTION" in environment:
        requested_resolution = environment["NYMERIA_RESOLUTION"].strip()
        if requested_resolution != contract.model_resolution:
            raise ValueError(
                "NYMERIA_RESOLUTION conflicts with the checkpoint contract: "
                f"environment={requested_resolution!r} checkpoint={contract.model_resolution!r}"
            )
    if "NYMERIA_NUM_FRAMES" in environment:
        requested_frames = int(environment["NYMERIA_NUM_FRAMES"])
        if requested_frames != contract.num_frames:
            raise ValueError(
                "NYMERIA_NUM_FRAMES conflicts with the checkpoint contract: "
                f"environment={requested_frames} checkpoint={contract.num_frames}"
            )
    if "NATIVEP1_SHIFT_OVERRIDE" in environment and environment[
        "NATIVEP1_SHIFT_OVERRIDE"
    ].strip():
        requested_shift = float(environment["NATIVEP1_SHIFT_OVERRIDE"])
        if abs(requested_shift - contract.training_shift) > 1e-9:
            raise ValueError(
                "NATIVEP1_SHIFT_OVERRIDE conflicts with the checkpoint contract: "
                f"environment={requested_shift} checkpoint={contract.training_shift}"
            )
    if "NATIVEP1_INFERENCE_SHIFT" in environment and environment[
        "NATIVEP1_INFERENCE_SHIFT"
    ].strip():
        requested_inference_shift = float(environment["NATIVEP1_INFERENCE_SHIFT"])
        if abs(requested_inference_shift - contract.inference_shift) > 1e-9:
            raise ValueError(
                "NATIVEP1_INFERENCE_SHIFT conflicts with the checkpoint contract: "
                f"environment={requested_inference_shift} "
                f"checkpoint={contract.inference_shift}"
            )
    return contract, source, checkpoint, run_dir


def write_eval_resolution(
    *, checkpoint_path: Path, output_dir: Path, environ: Mapping[str, str] | None = None
) -> tuple[Path, Path]:
    contract, source, checkpoint, run_dir = resolve_eval_contract(checkpoint_path, environ=environ)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / RESOLVED_CONTRACT_FILENAME
    env_path = output_dir / RESOLVED_ENV_FILENAME
    record = {
        **contract.to_dict(),
        "checkpoint_path": str(checkpoint),
        "run_dir": str(run_dir),
        "resolved_from": source,
    }
    _atomic_write(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    drop_modes = ",".join(contract.dropped_modes)
    env_contents = "\n".join(
        (
            f"export NATIVEP1_MODEL_FAMILY={shlex.quote(contract.model_family)}",
            f"export NATIVEP1_ADAPTATION_MODE={shlex.quote(contract.adaptation_mode)}",
            f"export NYMERIA_DROP_MODES={shlex.quote(drop_modes)}",
            f"export NYMERIA_RESOLUTION={shlex.quote(contract.model_resolution)}",
            f"export NYMERIA_NUM_FRAMES={contract.num_frames}",
            f"export NATIVEP1_SHIFT_OVERRIDE={contract.training_shift}",
            f"export NATIVEP1_TRAINING_SHIFT={contract.training_shift}",
            f"export NATIVEP1_INFERENCE_SHIFT={contract.inference_shift}",
            f"export NATIVEP1_EFFECTIVE_SHIFT={contract.inference_shift}",
            f"export NATIVEP1_CONDITIONING_FPS={contract.conditioning_fps}",
            f"export NATIVEP1_BASE_FPS={contract.base_fps}",
            "",
        )
    )
    _atomic_write(env_path, env_contents)
    return record_path, env_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record_path, env_path = write_eval_resolution(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
    )
    resolved = json.loads(record_path.read_text())
    print(
        "[native-contract] "
        f"family={resolved['model_family']} mode={resolved['adaptation_mode']} "
        f"dropped={resolved['dropped_modes']} fps={resolved['conditioning_fps']} "
        f"resolution={resolved['model_resolution']} "
        f"training_shift={resolved['training_shift']} "
        f"inference_shift={resolved['inference_shift']} "
        f"source={resolved['resolved_from']} record={record_path} env={env_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
