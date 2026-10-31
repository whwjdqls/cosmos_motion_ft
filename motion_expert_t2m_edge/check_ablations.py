"""Compute-node CPU acceptance; no model weights are instantiated."""
import argparse
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

import config
from audit_motion_stats import _load_kept_windows
from feature_runtime import decode_joints
from feature_runtime import convert_source_row
import ablation_settings
from flow import add_noise_velocity_masked
from losses import t2m_reconstruction_loss
from kimodo_codec import encode_uniego
from uniego_layout import canonicalize_frame0, ground_features
from runtime_paths import resolve_legacy_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in os.uname().nodename:
        raise RuntimeError('Allocated compute node required')
    torch.set_num_threads(1)
    loader = unittest.TestLoader()
    test_dir = str(Path(__file__).with_name('tests'))
    tests = loader.discover(test_dir, pattern='test_ablations.py')
    tests.addTests(loader.discover(test_dir, pattern='test_motion_heads.py'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful(): raise RuntimeError('Ablation CPU unit tests failed')
    source_args=argparse.Namespace(manifest=config.NYMERIA_MANIFEST,split_file=config.NYMERIA_SPLIT_FILE,
                                   calibration=config.FLOOR_CALIBRATION_JSON,split='train')
    by_sequence,raw,dropped=_load_kept_windows(source_args)
    uuid=sorted(by_sequence)[0]
    start,end,offset=by_sequence[uuid][0]
    with np.load(Path(config.NYMERIA_UNIEGO_ROOT)/(uuid+'.npz'),allow_pickle=False) as data:
        ny=canonicalize_frame0(ground_features(data['features'][start:end].astype(np.float32),offset))
    # Inspect an actual source row, resolving imported Grasp paths through the site mapper.
    with open(config.BONES_PAIRS_TRAIN) as handle: row=json.loads(next(handle))
    path=resolve_legacy_path(row['uniego_path'])
    with np.load(path,allow_pickle=False) as data:
        bones=canonicalize_frame0(data['features'][row['start']:row['end']][:200].astype(np.float32))
    observations=[];initialization=[]
    mean=torch.from_numpy(np.load(config.MOTION_STATS_MEAN).astype(np.float32))
    std=torch.from_numpy(np.load(config.MOTION_STATS_STD).astype(np.float32))
    for name,values in [('nymeria',ny),('bones',bones)]:
        features=torch.from_numpy(values)
        converted=encode_uniego(features)
        error=float((decode_joints(converted[None])-decode_joints(features[None])).abs().max())
        if error>1e-5 or not torch.equal(converted[:,-4:],features[:,-4:]):
            raise RuntimeError(f'{name} representation/contact roundtrip failed')
        normalized=(features-mean)/std
        row=dict(motion=normalized,motion_pad_mask=torch.zeros(len(features),dtype=torch.bool),
                 neutral_joints=torch.zeros(30,3),reasoner_image=None,caption='unchanged')
        with mock.patch.object(ablation_settings,'CODEC','kimodo369_v1'),mock.patch(
                'feature_runtime.feature_statistics',return_value=(np.zeros(369,np.float32),np.ones(369,np.float32))):
            adapted=convert_source_row(row)
        adapter_error=float((decode_joints(adapted['motion'][None])-decode_joints(features[None])).abs().max())
        if adapter_error>1e-5 or adapted['neutral_joints'] is not row['neutral_joints']:
            raise RuntimeError('Source adapter changed physical motion or shape')
        observations.append(dict(source=name,frames=len(features),max_joint_error_m=error,
                                 adapter_joint_error_m=adapter_error,contacts_exact=True))
        for s in (.1,.5,1.):
            torch.manual_seed(1)
            x=normalized[None];pad=row['motion_pad_mask'][None]
            noisy,sigma,velocity,valid=add_noise_velocity_masked(x,pad,torch.tensor([s]))
            for objective,prediction in [('x0',torch.zeros_like(x)),('velocity',noisy)]:
                overrides=dict(feature_prediction=torch.zeros_like(x),feature_target=velocity) if objective=='velocity' else {}
                loss,terms=t2m_reconstruction_loss(prediction,x,valid,mean,std,w_joint=1,w_smooth=5,
                    w_contact=.05,w_foot_vel=1,w_foot_height=10,**overrides)
                initialization.append(dict(source=name,sigma=s,objective=objective,loss=float(loss),
                    triggers_initial_nano_guard=float(loss)>50,terms={k:float(v) for k,v in terms.items()}))
    report=dict(status='CPU_ACCEPTANCE_PASSED',gpu_acceptance='not_run',tests=result.testsRun,
        host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],real_windows=observations,
        zero_head_initialization=initialization,
        source_population=dict(raw=raw,dropped=sum(dropped.values()),windows=sum(map(len,by_sequence.values())),
            frames=sum(end-start for windows in by_sequence.values() for start,end,_ in windows)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__': main()
