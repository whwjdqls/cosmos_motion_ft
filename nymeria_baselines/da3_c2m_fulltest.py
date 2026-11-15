"""Inference-only DA3 camera replacement for the frozen full Nymeria cohort.

Preparation, conversion and inference require owned compute. Existing training,
checkpoints, measured-camera evaluations and native samplers remain unchanged.
"""
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import argparse
import copy
import fcntl
import json
import os
import random
import time

import numpy as np

from nymeria_baselines import fulltest_v2m as frozen
from nymeria_baselines.predicted_camera import (
    remove_initial_gauge, egoego_condition, uem_condition, egoallo_poses,
    egoego_to_common, common_from_world, yaw_rotation)

BASE = frozen.BASE / 'operations'
DEFAULT = BASE / 'da3_c2m_fulltest_20261116'
REFERENCE = BASE / 'fulltest_v2m_ablations_20261104'
EGOEGO = BASE / 'egoego_fulltest_20261107'
UEM = BASE / 'idle_gpu_followups_20261104/uem_final350'
EGOALLO = Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105')
LABELS = dict(egoego='egoego_da3_trainprior', uem='uem_da3_trainprior', egoallo='egoallo_da3_trainprior')
SOURCES = [Path(__file__), Path(__file__).with_name('predicted_camera.py')]


def fingerprint():
    return {p.name: frozen.sha(p) for p in SOURCES}


def pose_prior(cache):
    """Equal-record training-only yaw-free orientation and median height."""
    from scipy.spatial.transform import Rotation
    index = frozen.read(cache / 'index.json')
    groups = defaultdict(dict)
    test = {r['uuid'] for r in index['rows'] if r['split'] == 'test'}
    for r in index['rows']:
        if r['split'] == 'train': groups[r['array']][r['start']] = r
    heights, rotations = [], []
    for key, starts in sorted(groups.items()):
        rows = list(starts.values())
        assert not {r['uuid'] for r in rows} & test
        source = np.load(cache / 'records' / f'{key}.pose_yup.npy', mmap_mode='r')
        poses = np.asarray(source[sorted(starts)], dtype=np.float64)
        by_start = [starts[s] for s in sorted(starts)]
        floor = np.array([0. if r['ground_offset_y'] is None else r['ground_offset_y'] for r in by_start])
        heights.append(float(np.median(poses[:, 1, 3] - floor)))
        yawfree = np.stack([yaw_rotation(r).T @ r for r in poses[:, :3, :3]])
        rotations.append(Rotation.from_matrix(yawfree).mean().as_matrix())
    initial = Rotation.from_matrix(np.stack(rotations)).mean().as_matrix()
    # Remove any numerical residual heading from the rotation mean.
    initial = yaw_rotation(initial).T @ initial
    assert len(groups) == 578
    return dict(schema='train_only_initial_camera_gauge_v1', train_recordings=len(groups),
        unique_train_windows=sum(map(len, groups.values())), height_m=float(np.median(heights)),
        initial_rotation_cv_to_yup=initial.tolist(), cache_index_sha256=frozen.sha(cache / 'index.json'),
        policy='Per-record median first-window camera height above training floor, then median across578 training records. SO3 mean of yaw-removed initial camera rotations per record, then equal-record SO3 mean. No test pose/floor/height/shape or scale fitting.',
        limitation='DA3 export has no absolute gravity/floor/initial height. These are assumed from training, not estimated from the test video. Vertical position/tilt errors from this prior count in results.')


def prepare(a):
    import shutil
    frozen.guard()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'PREPARED.json').exists():
        assert frozen.read(out / 'PREPARED.json')['implementation'] == fingerprint()
        return
    raw = out / 'handoff/nymeria_predicted_cameras_20261115'
    cameras = frozen.read(raw / 'camera_manifest_full.json')['rows']
    lookup = {(r['uuid'], r['start']): r for r in cameras}
    cfg, rows = frozen.load_common(SimpleNamespace(out=REFERENCE))
    assert len(rows) == 11911 and len(lookup) == len(cameras) == 11677
    assert {(r['uuid'], r['start']) for r in rows} == set(lookup)
    prior = pose_prior(frozen.CACHE)
    frozen.save(out / 'training_camera_prior.json', prior)
    calibration = frozen.read(EGOEGO / 'calibration.json')
    assert calibration['train_recordings'] == 578
    frozen.save(out / 'calibration.json', calibration)
    # Independent measured-camera control: new adapters must match the actual
    # original training/evaluation conventions before seeing DA3 predictions.
    from nymeria_baselines.egoego_shape.data import MotionDataset
    from nymeria_baselines.evaluate_egoego_fulltest import common_frame
    ego_cfg = frozen.read(EGOEGO / 'configuration.json')
    ds = MotionDataset(ego_cfg['models'][0]['stats'], 'test')
    by_key = {(r['uuid'],r['start']):i for i,r in enumerate(ds.rows)}
    truth = np.load(BASE / 'motion_metrics_20261105/gt_body23.npy',mmap_mode='r')
    seen=set();maximum_condition=0.;maximum_common=0.
    for i,row in enumerate(rows):
        if row['uuid'] in seen: continue
        seen.add(row['uuid']);idx=by_key[(row['uuid'],row['start'])]
        target,native_camera,_,_=ds.raw(idx)
        camera=np.array(np.load(frozen.CACHE/'records'/(row['array']+'.pose_yup.npy'),mmap_mode='r')[row['start']:row['start']+97],copy=True)
        camera[:,1,3]-=0. if row['floor_offset'] is None else row['floor_offset']
        error=float(np.abs(egoego_condition(camera)-native_camera).max())
        assert error<3e-5,(row['id'],'Camera convention',error)
        maximum_condition=max(maximum_condition,error)
        normalized=(target-ds.mean)/ds.std
        native=common_frame(ds,idx,normalized,calibration,truth[i])
        adapted,_=egoego_to_common(normalized,ds.mean,ds.std,camera,calibration)
        error=float(np.abs(adapted-native['joints']).max())
        assert error<3e-5,(row['id'],'Common-frame convention',error)
        maximum_common=max(maximum_common,error)
    assert len(seen)==71
    frozen.save(out/'MEASURED_ADAPTER_PARITY.json',dict(recordings=71,maximum_camera9_difference=maximum_condition,
        maximum_common_joint_difference_m=maximum_common,GT_used='Read-only control/audit only; never in DA3 conversion or inference'))
    physical = {}
    for number, (key, r) in enumerate(sorted(lookup.items())):
        item = r['cameras']['da3_nested_1p1']
        path = raw / item['file']
        assert frozen.sha(path) == item['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert str(z['uuid']) == r['uuid'] and int(z['start']) == r['start'] and int(z['fps']) == 20
            pose = remove_initial_gauge(z['camera_to_world_cv'], prior['initial_rotation_cv_to_yup'], prior['height_m'])
            # Scale in the export is ALREADY metric; never multiply it again.
        dest = out / 'conditions' / (r['id'] + '.npz')
        frozen.npz(dest, camera_floor_yup=pose.astype(np.float32), egoego_camera9=egoego_condition(pose),
            uem_camera18=uem_condition(pose), egoallo_camera_zup98=egoallo_poses(pose),
            source_sha256=np.array(item['sha256']))
        physical[key] = dict(file=str(dest), sha256=frozen.sha(dest), raw_sha256=item['sha256'], raw_file=str(path))
        if number % 1000 == 0: frozen.progress(out, 'converting', completed=number, total=len(lookup))
    mapping = {r['id']: physical[(r['uuid'], r['start'])] for r in rows}
    frozen.save(out / 'case_conditions.json', mapping)
    measured = dict(egoego=EGOEGO, uem=UEM, egoallo=EGOALLO)
    names = dict(egoego='egoego_camera_shape_ema_final555800', uem='uem_final_trajectory', egoallo='egoallo_final200k_camera_shape')
    specs = []
    for family, root in measured.items():
        if family == 'egoallo':
            protocol = frozen.read(root / 'camera_to_motion_protocol.json')
            spec = dict(label=LABELS[family], family=family, checkpoint=protocol['checkpoint'],
                checkpoint_sha256=protocol['checkpoint_sha256'], step=200000)
        else:
            spec = copy.deepcopy(next(s for s in frozen.read(root / 'configuration.json')['models'] if s['label'] == names[family]))
            spec.update(label=LABELS[family])
        spec.update(measured_reference=str(root), measured_label=names[family],
            inputs='DA3 metric camera with train-only initial gravity/height prior; original supplied shape; no RGB/text/test-GT pose/floor/scale')
        specs.append(spec)
    cfg = copy.deepcopy(cfg)
    cfg.update(models=specs, predicted_camera=dict(method='da3_nested_1p1', manifest_sha256=frozen.sha(raw / 'camera_manifest_full.json'),
        mapping_sha256=frozen.sha(out / 'case_conditions.json'), prior=prior, calibration=calibration, implementation=fingerprint()),
        protocol='DA3 camera-to-motion, full11911 cases/11677 physical windows/71 recordings; original native samplers/seeds/weights/shape/evaluator. Training-only initial gravity/height prior; no oracle alignment.')
    frozen.save(out / 'configuration.json', cfg)
    shutil.copyfile(REFERENCE / 'cases.json', out / 'cases.json')
    for name in ('llm2vec_fulltest_extended.pt', 'llm2vec_fulltest_extended.provenance.json'):
        target = out / name
        if not target.exists(): target.symlink_to(REFERENCE / name)
    # Same frozen GT, targets, cohort, evaluator/statistics and completion adapter.
    target = out / 'scores/GT'; target.mkdir(parents=True, exist_ok=True)
    source = EGOEGO / 'scores/GT'
    for name in ('embeddings.npz', 'per_case.json', 'retrieval_groups.json'):
        dest = target / name
        if not dest.exists(): dest.symlink_to(source / name)
    gt = frozen.read(source / 'summary.json')
    assert gt['cases'] == len(rows) and gt['text_cache_sha256'] == frozen.sha(out / 'llm2vec_fulltest_extended.pt')
    gt.update(input_configuration_sha256=frozen.sha(out / 'configuration.json'),
        reused_reference_summary=str(source / 'summary.json'), reused_reference_sha256=frozen.sha(source / 'summary.json'))
    frozen.save(target / 'summary.json', gt)
    frozen.save(target / 'COMPLETE.json', dict(cases=len(rows), summary_sha256=frozen.sha(target / 'summary.json')))
    frozen.save(out / 'PREPARED.json', dict(cases=len(rows), physical_windows=len(physical), recordings=71,
        implementation=fingerprint(), configuration_sha256=frozen.sha(out / 'configuration.json')))


def context(a):
    frozen.guard()
    cfg, rows = frozen.load_common(a)
    assert cfg['predicted_camera']['implementation'] == fingerprint(), 'Adapter changed after preparation'
    assert frozen.sha(a.out / 'case_conditions.json') == cfg['predicted_camera']['mapping_sha256']
    spec = next(s for s in cfg['models'] if s['label'] == LABELS[a.model])
    return cfg, rows, spec, frozen.read(a.out / 'case_conditions.json')


def read_condition(mapping, row):
    item = mapping[row['id']]
    assert frozen.sha(item['file']) == item['sha256']
    with np.load(item['file']) as z: return {k: z[k].copy() for k in z.files}


def read_shape(row):
    assert frozen.sha(row['bundle']) == row['bundle_sha256']
    # Deliberately do not read the raw target-motion member in the sampler.
    with np.load(row['bundle']) as z: return z['neutral'].copy()


class EgoEgoRunner:
    def __init__(self):
        from nymeria_baselines.evaluate_egoego_fulltest import load_model
        self.model, cfg, _ = load_model(SimpleNamespace(out=EGOEGO))
        with np.load(cfg['models'][0]['stats']) as z:
            self.mean, self.std, self.cmean, self.cstd = [z[k].copy() for k in ('mean','std','camera_mean','camera_std')]

    def sample(self, rows, conditions, shapes, native=False):
        import torch
        from nymeria_baselines.evaluate_egoego_fulltest import sample_batch
        camera = torch.from_numpy(np.stack([(c['egoego_camera9']-self.cmean)/self.cstd for c in conditions])).cuda()
        shape = torch.from_numpy(np.stack(shapes)).cuda()
        if native: return self.model.sample(camera, shape, seed=rows[0]['seed'])
        return sample_batch(self.model, camera, shape, [r['seed'] for r in rows])

    def decode(self, x, condition, calibration):
        return egoego_to_common(x, self.mean, self.std, condition['camera_floor_yup'], calibration)


class UEMRunner:
    def __init__(self):
        from nymeria_baselines.fulltest_v2m_batched import Runner
        self.runner = Runner(SimpleNamespace(out=UEM, model='uem_final_trajectory'))
        self.runner.std = self.runner.std + np.float32(1e-6)
        with np.load(frozen.CACHE / 'train_stats.npz') as z:
            self.mean, self.std = z['camera_mean'].copy(), z['camera_std'].copy()

    def sample(self, rows, conditions, shapes, native=False):
        import torch
        count = len(rows)
        y = dict(valid_frames=torch.ones(count,97,dtype=torch.long,device='cuda'),
            neutral_joints=torch.from_numpy(np.stack(shapes)).cuda().float(),
            traj=torch.from_numpy(np.stack([(c['uem_camera18']-self.mean)/(self.std+1e-6) for c in conditions])).cuda().float(),
            traj_mask=torch.zeros(count,97,dtype=torch.long,device='cuda'))
        return self.runner.sample(rows, y, native=native)

    def decode(self, x, condition, calibration): return self.runner.decode(x)


class EgoAlloRunner:
    def __init__(self, spec):
        import torch
        from safetensors.torch import load_file
        from nymeria_baselines.shape_conditioned.egoallo import activate
        self.up = frozen.OVERLAY
        activate(self.up)
        from egoallo import network
        protocol = frozen.read(EGOALLO / 'camera_to_motion_protocol.json')
        self.model = network.EgoDenoiser(network.EgoDenoiserConfig(**protocol['training_recipe']['model_config'])).cuda().eval()
        self.model.load_state_dict(load_file(spec['checkpoint']), strict=True)
        with np.load(frozen.CACHE / 'body_shape16.npz') as z: self.basis = {k:z[k].copy() for k in z.files}

    def sample(self, rows, conditions, shapes, native=False):
        import torch
        from scipy.spatial.transform import Rotation
        from nymeria_baselines.shape_conditioned.sampling import sample_egoallo
        from nymeria_baselines.body_shape import BODY_JOINTS
        assert len(rows) == 1, 'Keep native EgoAllo per-case sampler unchanged'
        pose = conditions[0]['egoallo_camera_zup98']
        q = Rotation.from_matrix(pose[:, :3, :3]).as_quat()[:,[3,0,1,2]]
        transforms = torch.from_numpy(np.concatenate([q,pose[:,:3,3]],-1).astype(np.float32)).cuda()
        shape = shapes[0][list(BODY_JOINTS)].astype(np.float64)
        vec = (shape-shape[:1]).reshape(-1)
        beta = ((self.basis['components'] @ (vec-self.basis['mean']))/self.basis['scale']).astype(np.float32)
        betas = torch.from_numpy(beta[None,None]).cuda()
        seed = rows[0]['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode(): pred = sample_egoallo(self.model,self.up,transforms,betas)
        assert torch.equal(pred.betas,betas.expand(1,97,16))
        return pred

    def decode(self, pred, condition, calibration):
        import torch
        from nymeria_baselines.body_shape import PARENTS
        from nymeria_baselines.shape_conditioned.sampling import egoallo_fk_with_supplied_shape
        local = pred.body_rotmats[0].cpu().numpy()
        rotations = np.broadcast_to(np.eye(3),(97,23,3,3)).copy()
        for j in range(1,23): rotations[:,j] = rotations[:,PARENTS[j]] @ local[:,j-1]
        with torch.inference_mode(): fk = egoallo_fk_with_supplied_shape(pred.body_rotmats,pred.betas[:,:1],self.basis)[0].cpu().numpy()
        camera = condition['camera_floor_yup'].astype(np.float64)
        cr, cp = camera[:,:3,:3], camera[:,:3,3]
        root_r = cr @ np.asarray(calibration['head_rotation_in_camera']) @ rotations[:,6].swapaxes(-1,-2)
        head = cp + np.einsum('tij,j->ti',cr,calibration['head_offset_camera'])
        root_p = head-np.einsum('tij,tj->ti',root_r,fk[:,6])
        transform = common_from_world(camera,calibration)
        joints = (root_p[:,None]+np.einsum('tij,tkj->tki',root_r,fk)) @ transform[:3,:3].T+transform[:3,3]
        return joints.astype(np.float32),(transform[:3,:3] @ root_r[:,None] @ rotations).astype(np.float32)


def run(a):
    import torch
    from nymeria_baselines.body_shape import BODY_JOINTS
    cfg, rows, spec, mapping = context(a)
    assert frozen.sha(spec['checkpoint']) == spec['checkpoint_sha256']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = a.out / 'models' / LABELS[a.model]; root.mkdir(parents=True,exist_ok=True)
    lock = (root / f'rank{a.rank}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner = dict(egoego=EgoEgoRunner,uem=UEMRunner,egoallo=lambda:EgoAlloRunner(spec))[a.model]()
    calibration = cfg['predicted_camera']['calibration']
    # Real, full native sampler acceptance before production; no step shortcut.
    first = rows[:1]; conditions = [read_condition(mapping,first[0])]; shapes = [read_shape(first[0])]
    with torch.inference_mode():
        native = runner.sample(first,conditions,shapes,native=True)
        replay = runner.sample(first,conditions,shapes)
    nx = native.pack() if a.model == 'egoallo' else native
    rx = replay.pack() if a.model == 'egoallo' else replay
    assert torch.equal(nx,rx), ('Native singleton replay mismatch',float((nx-rx).abs().max()))
    joints, rotations = runner.decode(native if a.model=='egoallo' else native[0].cpu().numpy(),conditions[0],calibration)
    assert joints.shape==(97,23,3) and np.isfinite(joints).all() and np.isfinite(rotations).all()
    batch_check={}
    if a.model!='egoallo' and a.batch>1:
        check_rows=rows[:a.batch]
        with torch.inference_mode():
            batched=runner.sample(check_rows,[read_condition(mapping,r) for r in check_rows],[read_shape(r) for r in check_rows])
        assert torch.isfinite(batched).all()
        bj,_=runner.decode(batched[0].cpu().numpy(),conditions[0],calibration)
        distance=np.linalg.norm(bj-joints,axis=-1)*100
        batch_check=dict(mean_joint_difference_cm=float(distance.mean()),max_joint_difference_cm=float(distance.max()))
        assert distance.mean()<=.1 and distance.max()<=1.,('Material batch-dependent change',batch_check)
        del batched
    frozen.save(root / f'ACCEPTED_rank{a.rank}.json',dict(native_singleton_bitwise=True,full_native_steps=True,
        checkpoint_sha256=spec['checkpoint_sha256'],implementation=fingerprint(),device=torch.cuda.get_device_name(),batch=a.batch,
        production_batch_parity=batch_check))
    if a.command == 'preflight': return
    selected = rows[a.rank::a.world]
    if a.limit: selected = selected[:a.limit]
    count=0; start=time.monotonic()
    size = 1 if a.model=='egoallo' else a.batch
    for offset in range(0,len(selected),size):
        batch = selected[offset:offset+size]
        pending = []
        for r in batch:
            path=root/'predictions'/(r['id']+'.npz')
            if path.exists():
                with np.load(path) as z:
                    assert str(z['checkpoint_sha256'])==spec['checkpoint_sha256'] and str(z['condition_sha256'])==mapping[r['id']]['sha256']
            else: pending.append(r)
        if pending:
            conds=[read_condition(mapping,r) for r in pending]; shapes=[read_shape(r) for r in pending]
            with torch.inference_mode(): x=runner.sample(pending,conds,shapes)
            if a.model!='egoallo': x=x.float().cpu().numpy()
            for i,(r,c,shape) in enumerate(zip(pending,conds,shapes)):
                joints,rotations=runner.decode(x if a.model=='egoallo' else x[i],c,calibration)
                assert joints.shape==(97,23,3) and rotations.shape==(97,23,3,3)
                assert np.isfinite(joints).all() and np.isfinite(rotations).all()
                neutral=shape[list(BODY_JOINTS)]
                if a.model=='egoallo':
                    beta=x.betas[0,0].cpu().numpy()
                    neutral=(runner.basis['mean']+(beta*runner.basis['scale'])@runner.basis['components']).reshape(23,3)
                    neutral=neutral-neutral[:1]
                frozen.npz(root/'predictions'/(r['id']+'.npz'),joints=joints,rotations=rotations,
                    neutral23=neutral.astype(np.float32),case_id=np.array(r['id']),seed=r['seed'],
                    checkpoint_sha256=np.array(spec['checkpoint_sha256']),condition_sha256=np.array(mapping[r['id']]['sha256']))
        count+=len(batch)
        frozen.save(root/f'progress_rank{a.rank}.json',dict(completed=count,total=len(selected),seconds=time.monotonic()-start,
            rank=a.rank,world=a.world,batch=size,job=os.environ.get('SLURM_JOB_ID'),time=time.time()))
        print(json.dumps(dict(model=a.model,rank=a.rank,completed=count,total=len(selected),seconds=time.monotonic()-start)),flush=True)
    if a.limit: return
    frozen.save(root/f'COMPLETE_rank{a.rank}.json',dict(cases=len(selected),rank=a.rank,world=a.world,
        checkpoint_sha256=spec['checkpoint_sha256'],implementation=fingerprint()))


def finish(a):
    cfg,rows,spec,mapping=context(a)
    root=a.out/'models'/LABELS[a.model]
    total=0
    for rank in range(a.world):
        receipt=frozen.read(root/f'COMPLETE_rank{rank}.json')
        assert receipt['world']==a.world and receipt['implementation']==fingerprint()
        total+=receipt['cases']
    assert total==len(rows)
    for r in rows:
        with np.load(root/'predictions'/(r['id']+'.npz')) as z:
            assert str(z['case_id'])==r['id'] and int(z['seed'])==r['seed']
            assert str(z['condition_sha256'])==mapping[r['id']]['sha256']
            assert str(z['checkpoint_sha256'])==spec['checkpoint_sha256']
            assert z['joints'].shape==(97,23,3) and np.isfinite(z['joints']).all()
            assert z['rotations'].shape==(97,23,3,3) and np.isfinite(z['rotations']).all()
    frozen.save(root/'INFERENCE_COMPLETE.json',dict(cases=total,recordings=71,checkpoint_sha256=spec['checkpoint_sha256']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','preflight','sample','finish'])
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--model',choices=list(LABELS),default='egoego')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();assert 0<=a.rank<a.world
    {'prepare':prepare,'preflight':run,'sample':run,'finish':finish}[a.command](a)


if __name__=='__main__': main()
