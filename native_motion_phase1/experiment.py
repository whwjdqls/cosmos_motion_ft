"""Reuse native Phase-1 LoRA/loss/EMA machinery with motion in the action stream."""
import copy
import os
from hydra.core.config_store import ConfigStore
from torch.utils.data import DistributedSampler
from cosmos_framework.utils.lazy_config import LazyCall as L
from native_phase_training.experiment import world_camera_nymeria_latent
from native_phase_training.latent_nymeria_dataset import CyclingDataLoader
from native_motion_phase1.data import MotionDataset, TASKS

recipe = copy.deepcopy(world_camera_nymeria_latent)
recipe.job.group = "motion_world_edge"
recipe.job.name = "native_motion_phase1_edge"
recipe.model.config.max_action_dim = 288
_objective = os.environ.get("MOTIONP1_OBJECTIVE", "native_velocity")
if _objective == "x0_reconstruction_shared_waver_v1":
    from native_motion_phase1.model import MotionX0Model
    recipe.model._target_ = MotionX0Model
    recipe.model.motion_mean_path = os.environ["MOTION_STATS_MEAN"]
    recipe.model.motion_std_path = os.environ["MOTION_STATS_STD"]
    recipe.model.config.rectified_flow_training_config.action_loss_weight = 0.
elif _objective != "native_velocity":
    raise ValueError(f"Unknown motion objective: {_objective}")
recipe.dataloader_train.dataloaders = {
    task: dict(ratio=25, dataloader=L(CyclingDataLoader)(
        dataset=L(MotionDataset)(cache=os.environ["EDGE_PHASE3_CACHE"], task=task,
                                 tokenizer_config="${model.config.vlm_config.tokenizer}",
                                 mean_path=os.environ["MOTION_STATS_MEAN"],
                                 std_path=os.environ["MOTION_STATS_STD"]),
        batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
        prefetch_factor=2, in_order=False,
        sampler=L(DistributedSampler)(dataset="${..dataset}", shuffle=True, seed=42, drop_last=True)))
    for task in TASKS
}
if os.environ.get("MOTIONP1_PROBE_CYCLE") == "1":
    recipe.dataloader_train.seed = None
    for stream in recipe.dataloader_train.dataloaders.values():
        stream.ratio = 1
for key in ("fixed_validation", "checkpoint_eval", "checkpoint_eval_full71"):
    recipe.trainer.callbacks.pop(key, None)
from native_motion_phase1.validation import MotionValidation, TaskAudit
recipe.trainer.callbacks.motion_audit = L(TaskAudit)()
if os.environ.get("MOTIONP1_PREVIEWS", "1") == "1":
    recipe.trainer.callbacks.motion_validation = L(MotionValidation)()
ConfigStore.instance().store(group="experiment", package="_global_", name="native_motion_phase1_edge", node=recipe)
