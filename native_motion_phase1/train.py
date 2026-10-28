"""Separate native entrypoint; camera run contracts remain untouched."""
import argparse
import json
import os
from pathlib import Path
import socket

import native_motion_phase1.experiment  # registers recipe
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.scripts.train import launch
from cosmos_framework.utils.serialization import to_yaml
from native_phase_training.run_latent_train import (
    _configure_tensorboard_log_dir, _disable_incompatible_sigma_loss_analysis,
    _install_scheduler_compatible_cpu_affinity,
)
from native_motion_phase1.data import TASKS, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.config, args.deterministic, args.attach_vscode_debugger = args.sft_toml, False, False
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    _configure_tensorboard_log_dir(config)
    _disable_incompatible_sigma_loss_analysis(config)
    root = Path(config.job.path_local)
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=1, architecture="native Edge action-stream + global QKVO LoRA16/alpha32",
                    representation="camera_head_recanonicalization_v1", motion_shape=[97, 283], padded_head_dim=288,
                    tasks={t: .25 for t in TASKS}, fps=20, image_hw=256, reference_frames=1,
                    neutral="centered [30,3] in first 90 channels of always-observed prefix token",
                    action_start_frame_offset=-1, domain_id=31,
                    text2motion_visual="fixed zero latent; no example RGB", captions="T2M/TI2M only, dropout0.1",
                    cache=os.environ["EDGE_PHASE3_CACHE"],
                    cache_index_sha256=sha256(Path(os.environ["EDGE_PHASE3_CACHE"]) / "index.json"),
                    mean_sha256=sha256(os.environ["MOTION_STATS_MEAN"]), std_sha256=sha256(os.environ["MOTION_STATS_STD"]),
                    global_batch=(config.dataloader_train.max_samples_per_batch
                                  * config.model.config.parallelism.data_parallel_replicate_degree
                                  * config.model.config.parallelism.data_parallel_shard_degree), accumulation=1,
                    horizon=200000, source_host=socket.gethostname())
    objective = os.environ.get("MOTIONP1_OBJECTIVE", "native_velocity")
    if objective != "native_velocity":
        from native_motion_phase1.objectives import OBJECTIVE, MOTION_WEIGHTS
        if objective != OBJECTIVE:
            raise ValueError(objective)
        contract.update(schema=2, motion_objective=OBJECTIVE, motion_prediction="direct clean x0",
                        motion_weights=MOTION_WEIGHTS, motion_loss_scale=1., video_loss_scale=10.,
                        training_noise="shared native WAVER shift3; independent_action_schedule=False",
                        sampling="native UniPC; x0 motion converted with actual integration sigma",
                        reconstruction_loss_sha256=sha256(Path(__file__).resolve().parents[1] / "motion_expert_t2m_edge/losses.py"),
                        physical_loss_sha256=sha256(Path(__file__).resolve().parents[1] / "motion_expert_joint_attention/motion_losses.py"))
    path = root / "motion_contract.json"
    if int(os.environ.get("RANK", "0")) == 0:
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("Existing run motion contract differs")
        path.write_text(json.dumps(contract, indent=2) + "\n")
    if args.dryrun:
        to_yaml(config, str(root / "config.yaml"))
        print(root / "config.yaml")
        return
    _install_scheduler_compatible_cpu_affinity()
    launch(config, args)


if __name__ == "__main__":
    main()
