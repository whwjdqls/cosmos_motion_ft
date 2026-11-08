"""Frozen full-test EgoEgo evaluation; never edits the trained source package.

Preserves the trained native 1000-step x0 DDPM and per-case RNG streams while
batching independent clips. Uses checkpoint EMA, as in training validation.
Only camera and supplied neutral skeleton enter the sampler. A train-only
camera/head calibration maps predictions into the common benchmark frame.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter

import numpy as np
import torch

from nymeria_baselines.fulltest_v2m import read, save, sha, npz, guard, load_common

REFERENCE = Path('/scratch2/whwjdqls99/cosmos/operations/fulltest_v2m_ablations_20261104')
TRAIN = Path('/scratch2/minsu1206/cosmos/operations/egoego_shape_20261104')
GT_CACHE = Path('/scratch2/whwjdqls99/cosmos/operations/motion_metrics_20261105')
LABEL = 'egoego_camera_shape_ema_final555800'


def link_or_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha(source) == sha(destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def prepare(a):
    guard()
    from nymeria_baselines.egoego_shape import upstream
    from nymeria_baselines.egoego_shape.data import MotionDataset, CACHE
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, rows = load_common(argparse.Namespace(out=REFERENCE))
    assert len(rows) == 11911 and len({r['uuid'] for r in rows}) == 71
    assert len({(r['uuid'], r['start']) for r in rows}) == 11677
    original = TRAIN/'training/epoch=349-step=555800.ckpt'
    assert read(TRAIN/'training/COMPLETE.json')['step'] == 555800
    snapshot = a.out/'snapshots/final555800.ckpt'
    link_or_copy(original, snapshot)
    stats = a.out/'snapshots/training_stats.npz'
    link_or_copy(TRAIN/'data/stats.npz', stats)
    checkpoint = torch.load(snapshot, map_location='cpu', mmap=True, weights_only=False)
    experiment = checkpoint['hyper_parameters']['experiment']
    assert checkpoint['global_step'] == 555800 and checkpoint['epoch'] == 349
    assert experiment['stats_sha256'] == sha(stats)
    assert experiment['cache_index_sha256'] == sha(CACHE/'index.json') == cfg['cache_index_sha256']
    assert experiment['upstream'] == upstream.identity()
    package = Path(upstream.__file__).parent
    for name, digest in experiment['source_sha256'].items():
        assert sha(package/name) == digest, ('Trained source changed', name)
    optimizer = checkpoint['optimizer_states'][0]
    assert optimizer['current_step'] == 555800 and optimizer['decay'] == .999
    assert len(optimizer['ema']) == 79
    ds = MotionDataset(stats, 'test')
    indices = sorted(range(len(ds)), key=lambda i:(ds.rows[i]['uuid'], ds.rows[i]['start'], ds.rows[i]['caption']))
    for number, (r, index) in enumerate(zip(rows, indices)):
        d = ds.rows[index]
        assert (r['uuid'], r['start'], r['array']) == (d['uuid'], d['start'], d['array'])
        expected = f'{number:06d}_' + hashlib.sha256(json.dumps([d['uuid'], d['start'], d['caption']]).encode()).hexdigest()[:12]
        assert r['id'] == expected
    calibration = Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105/calibration_rank0.json')
    assert read(calibration)['train_recordings'] == 578
    link_or_copy(calibration, a.out/'calibration.json')
    spec = dict(label=LABEL, family='egoego', step=555800, epoch=349,
        checkpoint=str(snapshot), checkpoint_sha256=sha(snapshot), stats=str(stats), stats_sha256=sha(stats),
        original_checkpoint=str(original), weights='EMA0.999 from optimizer_states[0].ema, original parameter order',
        inputs='Measured97-frame camera trajectory and supplied centered SOMA30 neutral skeleton; no RGB/text/GT motion',
        sampler=dict(steps=1000, objective='pred_x0', schedule='native cosine DDPM', clipping=False,
            precision='FP32, TF32 off', random_stream='Independent generator per original case seed; native draw order'),
        alignment='Training camera anchor -> world -> camera plus train-only head-calibration common yaw/origin. No test-motion placement or grounding.',
        training_experiment=experiment)
    evaluation = copy.deepcopy(cfg)
    evaluation.update(models=[spec], protocol='All11911 floor-clean Nymeria annotation windows /11677 physical windows /71 recordings. Camera-to-motion, supplied shape, EMA, native1000-step DDPM.',
        reference=str(REFERENCE), reference_configuration_sha256=sha(REFERENCE/'configuration.json'),
        evaluator_driver_sha256=sha(__file__), calibration_sha256=sha(a.out/'calibration.json'))
    if (a.out/'configuration.json').exists():
        assert read(a.out/'configuration.json') == evaluation, 'Frozen evaluation changed'
    else:
        save(a.out/'configuration.json', evaluation)
    link_or_copy(REFERENCE/'cases.json', a.out/'cases.json')
    save(a.out/'case_dataset_indices.json', indices)
    for name in ('llm2vec_fulltest_extended.pt', 'llm2vec_fulltest_extended.provenance.json'):
        link_or_copy(REFERENCE/name, a.out/name)
    # Reuse GT embeddings only after proving the entire cohort, evaluator,
    # normalization, adapter, text and GT targets are unchanged. The separate
    # provenance receipt retains the original summary's immutable checksum.
    assert read(GT_CACHE/'gt_cache.json')['cases_sha256'] == cfg['cases_sha256']
    assert sha(GT_CACHE/'gt_body23.npy') == read(GT_CACHE/'gt_cache.json')['sha256']
    source = REFERENCE/'scores/GT'
    target = a.out/'scores/GT'; target.mkdir(parents=True, exist_ok=True)
    gt = read(source/'summary.json')
    assert gt['input_configuration_sha256'] == sha(REFERENCE/'configuration.json')
    for name in ('embeddings.npz', 'per_case.json', 'retrieval_groups.json'):
        link_or_copy(source/name, target/name)
    gt.update(input_configuration_sha256=sha(a.out/'configuration.json'),
        reused_reference_summary=str(source/'summary.json'), reused_reference_sha256=sha(source/'summary.json'))
    extra = read(REFERENCE/'extended_metrics/GT/summary.json')
    assert extra['cases'] == len(rows) and extra['complete']
    gt['physical'].update(extra['physical'])
    gt['recording_balanced_physical'].update(extra['recording_balanced_physical'])
    save(target/'summary.json', gt)
    save(target/'COMPLETE.json', dict(cases=len(rows), summary_sha256=sha(target/'summary.json'),
        reuse='Identical frozen cases, targets, evaluator, shape completion and text; model-list-only configuration change'))
    save(a.out/'PREPARED.json', dict(passed=True, cases=len(rows), physical_windows=11677,
        recordings=71, checkpoint_sha256=spec['checkpoint_sha256'], input_configuration_sha256=sha(a.out/'configuration.json'),
        gt_reference_sha256=sha(source/'summary.json'), gt_joints_sha256=sha(GT_CACHE/'gt_body23.npy')))
    print('PREPARED', json.dumps(read(a.out/'PREPARED.json')), flush=True)


def load_model(a):
    from nymeria_baselines.egoego_shape.model import CameraShapeDiffusion
    cfg, rows = load_common(a)
    spec = cfg['models'][0]
    assert sha(__file__) == cfg['evaluator_driver_sha256']
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    model = CameraShapeDiffusion()
    model.load_state_dict({k.removeprefix('net.'):v for k,v in checkpoint['state_dict'].items()}, strict=True)
    pairs = list(model.named_parameters())
    opt = checkpoint['optimizer_states'][0]
    order = [index for group in opt['opt']['param_groups'] for index in group['params']]
    assert order == list(range(len(pairs))) and len(pairs) == len(opt['ema']) == 79
    with torch.no_grad():
        for (name, parameter), ema in zip(pairs, opt['ema']):
            assert parameter.shape == ema.shape and torch.isfinite(ema).all(), name
            parameter.copy_(ema)
            assert torch.equal(parameter, ema), name
    model.cuda().eval().requires_grad_(False)
    return model, cfg, rows


@torch.inference_mode()
def sample_batch(model, camera, shape, seeds, steps=None):
    """Same native arithmetic and per-case draws; no change to trained model."""
    generators = [torch.Generator(device=camera.device).manual_seed(int(s)) for s in seeds]
    def noise():
        return torch.cat([torch.randn((1,97,207), device=camera.device, generator=g) for g in generators])
    x = noise()
    condition = model.condition(camera, noise())
    timeline = list(reversed(range(model.num_timesteps)))
    if steps is not None:
        timeline = timeline[:steps]
    for index in timeline:
        t = torch.full((len(x),), index, device=x.device, dtype=torch.long)
        clean = model.denoise_fn(torch.cat((x, condition), dim=-1), t, shape=shape)
        mean, _, log_variance = model.q_posterior(clean, x, t)
        x = mean + (index != 0) * (0.5*log_variance).exp() * noise()
    assert torch.isfinite(x).all()
    return x


def dataset(a, cfg):
    from nymeria_baselines.egoego_shape.data import MotionDataset
    ds = MotionDataset(cfg['models'][0]['stats'], 'test')
    indices = read(a.out/'case_dataset_indices.json')
    return ds, indices


def inputs(ds, indices, selected):
    values = [ds[indices[i]] for i in selected]
    return {k:torch.from_numpy(np.stack([v[k] for v in values])).cuda()
            for k in ('camera', 'shape')}


def common_frame(ds, index, normalized, calibration, truth):
    """Never use GT in the prediction transform; GT only audits its own frame."""
    from motion_expert_joint_attention.camera_head_recanonicalization import yaw_rotation_y
    from nymeria_baselines.body_shape import BODY_JOINTS
    target, camera, neutral, (anchor, origin) = ds.raw(index)
    extrinsic_r = np.asarray(calibration['head_rotation_in_camera'])
    extrinsic_p = np.asarray(calibration['head_offset_camera'])
    cam_position = camera[0,:3].astype(np.float64)
    cam_rotation = camera[0,3:].reshape(2,3).astype(np.float64)
    cam_rotation = np.concatenate((cam_rotation, np.cross(cam_rotation[:1], cam_rotation[1:])), axis=0)
    # Both source camera and predicted/target bodies are already floor-aligned.
    cam_rotation = anchor @ cam_rotation
    cam_position = anchor @ cam_position + origin
    forward = (cam_rotation @ extrinsic_r)[:,2]
    yaw = yaw_rotation_y(np.arctan2(forward[0], forward[2]))
    predicted_origin = cam_position + cam_rotation @ extrinsic_p
    predicted_origin *= [1.,0.,1.]
    transform = yaw.T @ anchor
    translation = yaw.T @ (origin-predicted_origin)
    unnormalized = normalized * ds.std + ds.mean
    positions = unnormalized[:,:69].reshape(97,23,3).astype(np.float64)
    row6 = unnormalized[:,69:].reshape(97,23,6)
    first = row6[...,:3].astype(np.float64)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = row6[...,3:].astype(np.float64)
    second -= (first*second).sum(-1,keepdims=True)*first
    second /= np.maximum(np.linalg.norm(second,axis=-1,keepdims=True),1e-12)
    rotations = np.stack((first,second,np.cross(first,second)),axis=-2)
    joints = (np.einsum('ij,tkj->tki', transform, positions)+translation).astype(np.float32)
    rotations = (transform @ rotations).astype(np.float32)
    # Independent GT check uses its own head origin solely for this audit.
    gt_world = np.einsum('ij,tkj->tki',anchor,target[:,:69].reshape(97,23,3))+origin
    gt_origin = gt_world[0,6]*[1.,0.,1.]
    gt_common = np.einsum('ij,tkj->tki',yaw.T,gt_world-gt_origin)
    gt_error = float(np.max(np.abs(gt_common-truth)))
    assert gt_error < 2e-3, ('Common-frame GT audit failed',gt_error)
    assert joints.shape == (97,23,3) and rotations.shape == (97,23,3,3)
    assert np.isfinite(joints).all() and np.isfinite(rotations).all()
    return dict(generated=normalized, raw=unnormalized, joints=joints, rotations=rotations,
        neutral23=neutral[list(BODY_JOINTS)], supplied_neutral30=neutral,
        camera_normalized=(camera-ds.camera_mean)/ds.camera_std,
        common_from_training_rotation=transform, common_from_training_translation=translation,
        gt_frame_max_abs_m=gt_error)


def preflight(a):
    model, cfg, rows = load_model(a)
    ds, indices = dataset(a,cfg)
    calibration = read(a.out/'calibration.json')
    truth = np.load(GT_CACHE/'gt_body23.npy', mmap_mode='r')
    # Across recordings, verify camera/row conventions before sampling.
    chosen = sorted(set([0,len(rows)//2,len(rows)-1] + [next(i for i,r in enumerate(rows) if r['uuid']==uid) for uid in sorted({r['uuid'] for r in rows})]))
    max_gt = 0.
    for i in chosen:
        target,_,_,_ = ds.raw(indices[i])
        norm = (target-ds.mean)/ds.std
        mapped = common_frame(ds,indices[i],norm,calibration,truth[i])
        max_gt = max(max_gt,mapped['gt_frame_max_abs_m'])
    values = inputs(ds,indices,[0,1])
    seeds = [rows[i]['seed'] for i in (0,1)]
    torch.cuda.synchronize();clock=time.monotonic()
    native = model.sample(values['camera'][:1],values['shape'][:1],seed=seeds[0])
    own = sample_batch(model,values['camera'][:1],values['shape'][:1],seeds[:1])
    assert torch.equal(native,own), ('Native singleton parity',float((native-own).abs().max()))
    batched = sample_batch(model,values['camera'],values['shape'],seeds)
    difference = float((native-batched[:1]).abs().max())
    # Batched GEMM kernels may differ in FP32 roundoff, never in RNG draws.
    assert difference < 1e-3, ('Batched sampler differs materially',difference)
    repeated = sample_batch(model,values['camera'],values['shape'],seeds)
    assert torch.equal(repeated,batched), 'Same-batch seeded replay failed'
    for j,i in enumerate((0,1)):
        mapped=common_frame(ds,indices[i],batched[j].cpu().numpy(),calibration,truth[i])
        npz(a.out/'acceptance'/f'case_{i:06d}.npz',**mapped)
    benchmarks=[]
    for batch in (8,32,64,128,256):
        selected=list(range(batch));v=inputs(ds,indices,selected)
        torch.cuda.synchronize();clock=time.monotonic();torch.cuda.reset_peak_memory_stats()
        sample_batch(model,v['camera'],v['shape'],[rows[i]['seed'] for i in selected],steps=20)
        torch.cuda.synchronize();elapsed=time.monotonic()-clock
        entry=dict(batch=batch,steps=20,seconds=elapsed,
            estimated_full_seconds_per_case=elapsed*50/batch,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        benchmarks.append(entry);print('BENCHMARK',json.dumps(entry),flush=True)
    best=min(benchmarks,key=lambda x:x['estimated_full_seconds_per_case'])
    result=dict(passed=True,native_singleton_bitwise=True,batched_same_seed_bitwise=True,
        batch2_vs_singleton_max_abs_normalized=difference,gt_geometry_cases=len(chosen),gt_frame_max_abs_m=max_gt,
        benchmark=benchmarks,recommended_batch=best['batch'],weights=cfg['models'][0]['weights'],
        checkpoint_sha256=cfg['models'][0]['checkpoint_sha256'],source_sha256=sha(__file__))
    save(a.out/'PREFLIGHT_ACCEPTED.json',result);print('PREFLIGHT_ACCEPTED',json.dumps(result),flush=True)


def sample(a):
    accepted=read(a.out/'PREFLIGHT_ACCEPTED.json');assert accepted['passed'] and accepted['source_sha256']==sha(__file__)
    root=a.out/'models'/LABEL;root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'rank{a.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,cfg,rows=load_model(a);spec=cfg['models'][0];ds,indices=dataset(a,cfg)
    truth=np.load(GT_CACHE/'gt_body23.npy',mmap_mode='r');calibration=read(a.out/'calibration.json')
    selected=list(range(a.rank,len(rows),a.world));batch=a.batch or accepted['recommended_batch']
    durations=[];started=time.monotonic();completed=0
    for start in range(0,len(selected),batch):
        current=selected[start:start+batch];pending=[]
        for i in current:
            dest=root/'predictions'/(rows[i]['id']+'.npz')
            if dest.exists():
                with np.load(dest) as z:
                    assert str(z['case_id'])==rows[i]['id'] and int(z['seed'])==rows[i]['seed']
                    assert str(z['checkpoint_sha256'])==spec['checkpoint_sha256']
                    assert str(z['weights'])==spec['weights'] and np.isfinite(z['joints']).all()
            else:pending.append(i)
        if pending:
            v=inputs(ds,indices,pending);torch.cuda.synchronize();clock=time.monotonic()
            generated=sample_batch(model,v['camera'],v['shape'],[rows[i]['seed'] for i in pending]).cpu().numpy()
            torch.cuda.synchronize();elapsed=time.monotonic()-clock;durations.append(elapsed/len(pending))
            for offset,i in enumerate(pending):
                row=rows[i];mapped=common_frame(ds,indices[i],generated[offset],calibration,truth[i])
                npz(root/'predictions'/(row['id']+'.npz'),**mapped,case_id=row['id'],seed=row['seed'],
                    checkpoint_sha256=spec['checkpoint_sha256'],weights=spec['weights'],sampler_steps=1000)
        completed+=len(current)
        status=dict(stage='sampling',rank=a.rank,world=a.world,batch=batch,completed=completed,total=len(selected),
            seconds=time.monotonic()-started,seconds_per_case=float(np.mean(durations[-10:])) if durations else 0,
            latest_case=rows[current[-1]]['id'],time=time.time(),job=os.environ['SLURM_JOB_ID'])
        save(root/f'progress_rank{a.rank}.json',status);print(json.dumps(status),flush=True)
    save(root/f'COMPLETE_rank{a.rank}.json',dict(rank=a.rank,cases=completed,
        checkpoint_sha256=spec['checkpoint_sha256'],source_sha256=sha(__file__)))


def finish_inference(a):
    cfg,rows=load_common(a);root=a.out/'models'/LABEL;spec=cfg['models'][0]
    total=0
    for rank in range(a.world):
        done=read(root/f'COMPLETE_rank{rank}.json')
        assert done['checkpoint_sha256']==spec['checkpoint_sha256'];total+=done['cases']
    assert total==len(rows)
    for r in rows:
        with np.load(root/'predictions'/(r['id']+'.npz')) as z:
            assert str(z['case_id'])==r['id'] and int(z['seed'])==r['seed']
            assert str(z['checkpoint_sha256'])==spec['checkpoint_sha256']
            assert z['joints'].shape==(97,23,3) and np.isfinite(z['joints']).all()
    save(root/'INFERENCE_COMPLETE.json',dict(cases=total,recordings=71,checkpoint_sha256=spec['checkpoint_sha256']))


def main():
    guard();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','sample','finish'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=2);parser.add_argument('--batch',type=int,default=0)
    a=parser.parse_args();assert 0<=a.rank<a.world
    {'prepare':prepare,'preflight':preflight,'sample':sample,'finish':finish_inference}[a.command](a)


if __name__=='__main__':main()
