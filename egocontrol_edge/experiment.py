"""Clone Phase1's recipe, changing only the declared EgoControl baseline axes."""
from __future__ import annotations

import copy
import os

from hydra.core.config_store import ConfigStore
from cosmos_framework.utils.lazy_config import LazyCall as L
from native_phase_training.experiment import world_camera_nymeria_latent
from egocontrol_edge.model import EgoControlOmniModel
from egocontrol_edge.data import get_dataset

experiment = copy.deepcopy(world_camera_nymeria_latent)
experiment.model._target_ = EgoControlOmniModel
experiment.model.conditioning_layout = os.environ.get('EGOCONTROL_CONDITIONING_LAYOUT', 'phase3_seven')
experiment.model.config.action_gen = False
experiment.job.project = 'cosmos3_camera'
experiment.job.group = 'egocontrol_edge'
experiment.optimizer.keys_to_select = ['lora_A', 'lora_B', 'egocontrol']
experiment.optimizer.lr_multipliers = {'egocontrol': 4.0}
for stream in experiment.dataloader_train.dataloaders.values():
    ds = stream.dataloader.dataset
    ds._target_ = get_dataset
    ds.cache = os.environ['EGOCONTROL_CACHE']
    ds.motion_mean = os.environ['EGOCONTROL_MOTION_MEAN']
    ds.motion_std = os.environ['EGOCONTROL_MOTION_STD']
    stream.dataloader.sampler.dataset = copy.deepcopy(ds)
    stream.ratio = 1
from egocontrol_edge.callback import AcceptanceAudit, ComparisonContract
experiment.trainer.callbacks.egocontrol_contract = L(ComparisonContract)()
from egocontrol_edge.validation import FixedValidation
experiment.trainer.callbacks.egocontrol_validation = L(FixedValidation)(
    configuration=os.environ['EGOCONTROL_VALIDATION_CONFIG'],
    indices=[int(x) for x in os.environ.get('EGOCONTROL_VALIDATION_INDICES', '0,23,46,70').split(',')],
    every=int(os.environ.get('EGOCONTROL_VALIDATION_EVERY', '5000')),
    first_step=int(os.environ.get('EGOCONTROL_VALIDATION_FIRST', '1')),
)
if os.environ.get('EGOCONTROL_ACCEPTANCE') == '1':
    experiment.trainer.callbacks.egocontrol_acceptance = L(AcceptanceAudit)()
ConfigStore.instance().store(group='experiment', package='_global_',
                             name='egocontrol_edge_m2v', node=experiment)
