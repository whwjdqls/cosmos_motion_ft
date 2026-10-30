"""Reproduce the three-modality full test and add audited scene collisions."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

import numpy as np

from diagnose_image_conditioning import atomic_json, sha


def prepare(args):
    source = args.source_operation
    old = json.loads((source / 'configuration.json').read_text())
    assert sha(source / 'cases.json') == old['cases_sha256']
    assert sha(old['manifest']) == old['manifest_sha256']
    assert old['batch_cases'] == 4 and old['cases'] == 11892
    assert old['variants'] == ['image_only', 'text_only', 'image_text']
    assert old['sampling']['steps'] == 35 and old['sampling']['shift'] == 3 and old['sampling']['CFG'] == 2
    audited = json.loads((args.audited_gt_operation / 'COMPLETE.json').read_text())
    assert audited['state'] == 'complete_verified' and audited['windows'] == 71
    rows = json.loads((source / 'cases.json').read_text())
    checks = json.loads((args.audited_gt_operation / 'per_case.json').read_text())
    assert {r['uuid'] for r in rows} == {c['uuid'] for c in checks}
    assert all(c['coverage']['matching_graph_count'] == c['coverage']['raw_point_count'] for c in checks)
    cfg = {**old, 'schema': 'edge_phase2_modalities_scene_collision_v1', 'shards': 1,
        'reference_configuration': old, 'reference_operation': str(source),
        'audited_gt_operation': str(args.audited_gt_operation),
        'source_sha256': sha(__file__),
        'scene_engine_sha256': sha(Path(__file__).parents[1] / 'nymeria_baselines/phase2_collision_scene.py'),
        'collision_body': 'exclude_thigh_shin_foot',
        'collision_protocol': 'SDK legacy filter, 2cm voxels, >=3 connected voxels >2cm inside16capsules; hips retained',
        'scene_placement': 'fixed source-window yaw/origin/floor; GT frame0 canonicalized; predictions never recentered, grounded or fitted',
        'scene_support': 'full approved cloud; shared conservative region covers GT AND all three predicted trajectories',
        'saved_generations': True, 'batch_claiming': 'per-batch flock; workers may join or resume on replacement allocations'}
    if (args.out / 'configuration.json').exists():
        assert json.loads((args.out / 'configuration.json').read_text()) == cfg
        return
    (args.out / 'cases.json').write_bytes((source / 'cases.json').read_bytes())
    atomic_json(args.out / 'configuration.json', cfg)
    (args.out / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out / 'scene_engine_source.py').write_bytes((Path(__file__).parents[1] / 'nymeria_baselines/phase2_collision_scene.py').read_bytes())
    atomic_json(args.out / 'protocol_audit.json', {'cases': len(rows), 'sequences': 71,
        'case_manifest_identical': True, 'checkpoint_unchanged': True, 'same_sampling_and_noise_policy': True,
        'approved_graphs_cover_all_cases': True, 'gt_reported_for_this_full_cohort': True,
        'previous_71_window_GT_rate_not_reused': True})


def worker(args):
    import torch
    import evaluate_generation_modalities as generation
    generation.base.setup()
    from checkpoint import configure_attention_from_checkpoint, load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from shape_tmr_eval_common import stable_seed
    from decode_uniego_torch import decode_joints
    from uniego_layout import ground_features
    from nymeria_eval import NymeriaTMREmbedder
    from nymeria_data import NymeriaShapeTMRDataset
    from st_dataset import collate_st
    from nymeria_baselines.phase2_collision_scene import initialize, measure_case
    # The standalone scene evaluator exposes NumPy helpers through the joint
    # directory. Restore Edge's established import order before the shared
    # evaluator imports `sample`; it must resolve the Edge adapter.
    generation.base.setup()
    import eval_phase2_shape_tmr as ref
    import config
    quant = generation.quant
    cfg = json.loads((args.out / 'configuration.json').read_text())
    rows = json.loads((args.out / 'cases.json').read_text())
    assert sha(__file__) == cfg['source_sha256']
    assert sha(Path(__file__).parents[1] / 'nymeria_baselines/phase2_collision_scene.py') == cfg['scene_engine_sha256']
    assert sha(args.out / 'cases.json') == cfg['cases_sha256']
    assert sha(cfg['checkpoint']) == cfg['checkpoint_sha256']
    assert configure_attention_from_checkpoint(cfg['checkpoint']) == cfg['checkpoint_contract']
    for key, path in [('motion_mean', config.MOTION_STATS_MEAN), ('motion_std', config.MOTION_STATS_STD)]:
        assert sha(path) == cfg['checkpoint_contract'][key]
    print('[loading] original200k checkpoint and Nymeria evaluator', flush=True)
    source = cfg['reference_request']
    assert sha(source['evaluator_checkpoint']) == cfg['evaluator_sha256']
    embedder = NymeriaTMREmbedder(source['evaluator_checkpoint'], source['evaluator_stats'], source['text_cache'], 'cuda')
    assert embedder.contract['manifests']['test'] == source['evaluator_training_test_manifest_sha256']
    gt_dataset = NymeriaShapeTMRDataset(cfg['manifest'], embedder.rep, train=False, src_fps=20., rep_fps=embedder.fps)
    model = EdgePhase2MotionExpert(device='cuda', dtype=torch.bfloat16)
    payload = load_checkpoint(cfg['checkpoint'], model=model); assert payload['step'] == 200000
    del payload
    model.eval().requires_grad_(False)
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    skeleton = ref.SOMASkeleton30().cuda()
    variants = ('image_only', 'text_only', 'image_text')
    batch_size = cfg['batch_cases']
    work = args.out / 'shard_0'; work.mkdir(exist_ok=True)
    samples = args.out / 'samples'; samples.mkdir(exist_ok=True)
    (args.out / 'claims').mkdir(exist_ok=True)
    worker_id = os.environ['SLURM_JOB_ID'] + '_' + str(os.getpid())
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.cpu_workers,
        mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(args.out),))
    pending = []
    beginning = time.monotonic(); accepted = 0
    def finish(item):
        nonlocal accepted
        start, batch, values, tasks, lock = item
        collision = [future.result() for future in tasks]
        case_dir = args.out / 'collision_cases'; case_dir.mkdir(exist_ok=True)
        for row, result in zip(batch, collision):
            assert row['case_id'] == result['case_id']
            evidence = result.pop('evidence')
            quant.atomic_npz(case_dir / (row['case_id'] + '.npz'), **evidence)
            atomic_json(case_dir / (row['case_id'] + '.json'), result)
        quant.atomic_npz(work / f'batch_{start:07d}.npz', **values)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        accepted += len(batch)
        done = sum(min(batch_size, len(rows)-int(p.stem.split('_')[-1])) for p in work.glob('batch_*.npz'))
        elapsed = time.monotonic()-beginning
        status = {'completed': done, 'total': len(rows), 'worker': worker_id,
            'newly_measured': accepted, 'cases_per_second': accepted/elapsed,
            'eta_hours_single_worker': (len(rows)-done)*elapsed/max(1, accepted)/3600,
            'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'time_unix': time.time()}
        atomic_json(args.out / ('progress_' + worker_id + '.json'), status)
        print('[progress] ' + json.dumps(status), flush=True)
    print('[sampling] same4-case batches and three actual input modalities; scene scoring in CPU workers', flush=True)
    try:
        for start in range(0, len(rows), batch_size):
            if (work / f'batch_{start:07d}.npz').exists(): continue
            lock = (args.out / 'claims' / f'{start:07d}.lock').open('a')
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); continue
            if (work / f'batch_{start:07d}.npz').exists(): lock.close(); continue
            batch = rows[start:start+batch_size]; n = len(batch)
            old_shard = (start//batch_size) % cfg['reference_configuration']['shards']
            with np.load(Path(cfg['reference_operation']) / f'shard_{old_shard}' / f'batch_{start:07d}.npz') as z:
                old_hashes = z['image_sha256'].copy()
            def load(row, field, digest):
                path = Path(source['data_dir']) / row[field]
                assert sha(path) == row[digest]
                return np.load(path).astype(np.float32)
            gt = torch.from_numpy(np.stack([ground_features(load(r, 'features_file', 'features_sha256'), r['floor_offset']) for r in batch])).cuda()
            neutral = torch.from_numpy(np.stack([load(r, 'neutral_file', 'neutral_sha256') for r in batch])).cuda()
            neutral = neutral-neutral.mean(1, keepdim=True)
            image_tensors = []
            for i, row in enumerate(batch):
                path = Path(row['image_file'])
                assert sha(path) == old_hashes[i, 0] == old_hashes[i, 2]
                pixels = np.load(path)
                assert pixels.ndim == 3 and pixels.shape[0] == 3 and pixels.dtype == np.uint8
                image_tensors.append(torch.from_numpy(pixels))
            sample_path = samples / f'batch_{start:07d}.npz'
            if sample_path.exists():
                with np.load(sample_path) as z:
                    assert z['case_id'].tolist() == [r['case_id'] for r in batch]
                    raw = torch.from_numpy(z['raw_features'].reshape(3*n,97,283)).cuda()
                    audit = z['token_audit'].copy()
            else:
                captions = ['']*n + [r['text'] for r in batch]*2
                modes = ['textimg2motion']*n + ['text2motion']*n + ['textimg2motion']*n
                cond = model.prepare_conditions(captions, modes=modes, reasoner_images=image_tensors+[None]*n+image_tensors)
                null = model.prepare_conditions(['']*(3*n), modes=['text2motion']*(3*n), reasoner_images=[None]*(3*n))
                empty = model._edge.tokenize_text('')
                assert all(torch.equal(c['input_ids'].cpu(), empty.cpu()) for c in null)
                assert captions[:n] == ['']*n and captions[n:2*n] == captions[2*n:]
                assert all(torch.equal(cond[n+i]['input_ids'].cpu(), model._edge.tokenize_text(row['text']).cpu())
                           for i, row in enumerate(batch))
                audit = np.array([[len(c['input_ids']), int(c.get('visual_pos_mask',torch.zeros(1)).sum())] for c in cond]).reshape(3,n,2).transpose(1,0,2)
                assert (audit[:,0,1] > 0).all() and (audit[:,1,1] == 0).all() and (audit[:,2,1] > 0).all()
                noise = torch.stack([torch.randn((97,283), generator=torch.Generator().manual_seed(stable_seed('nymeria32:'+r['case_id']))) for r in batch]).cuda().repeat(3,1,1)
                shape = neutral.repeat(3,1,1); pad = torch.zeros((3*n,97), device='cuda', dtype=torch.bool)
                def closure(context):
                    return lambda x,sigma: model(reasoner_inputs=context, x_sigma=x, sigma=sigma, neutral_joints=shape, motion_pad_mask=pad)
                normalized = sample_x0_unipc(closure(cond), predict_null=closure(null), T=97, motion_dim=283,
                    steps=35, guidance=2, batch=3*n, device='cuda', initial_noise=noise, shift=3)
                assert torch.isfinite(normalized).all()
                raw = normalized.float()*std+mean
                quant.atomic_npz(sample_path, raw_features=raw.cpu().numpy().reshape(3,n,97,283),
                    case_id=np.array([r['case_id'] for r in batch]), token_audit=audit)
                if start == 0:
                    with np.load(Path(cfg['reference_operation'])/'qualitative_samples.npz') as z:
                        delta = raw.cpu().numpy().reshape(3,n,97,283)-z['raw_features']
                    parity = {'max_absolute_feature_difference':float(abs(delta).max()),
                              'mean_absolute_feature_difference':float(abs(delta).mean()), 'bitwise':bool((delta == 0).all())}
                    atomic_json(args.out/'first_batch_generation_parity.json', parity)
                    assert parity['max_absolute_feature_difference'] < .001 and parity['mean_absolute_feature_difference'] < .00001, parity
            posed = decode_joints(raw); lengths = torch.full((n,),97,device='cuda',dtype=torch.long)
            values = {'index':np.arange(start,start+n), 'case_id':np.array([r['case_id'] for r in batch]),
                      'token_audit':audit, 'image_sha256':old_hashes,
                      'text_embedding':embedder.embed_text([r['text'] for r in batch]).cpu().numpy()}
            for j, label in enumerate(('GT',)+variants):
                features = gt if label == 'GT' else raw[(j-1)*n:j*n]
                positions = decode_joints(gt) if label == 'GT' else posed[(j-1)*n:j*n]
                measured = quant.measure(ref, embedder, skeleton, positions, features[...,279:283]>.5, lengths, neutral)
                if label == 'GT':
                    packed = collate_st([gt_dataset[row['index']] for row in batch])
                    measured['embedding'] = embedder.embed_motion_features(packed['features'],packed['mask'],packed['neutral_joints']).cpu().numpy()
                    for i,row in enumerate(batch):
                        begin = row['index']//64*64
                        with np.load(Path(source['reference_dir'])/f'batch_{begin:07d}.npz') as z:
                            assert z['case_id'][row['index']-begin] == row['case_id']
                            for key in ['embedding','foot_skate_from_height','foot_depth_mean_cm']:
                                assert np.allclose(measured[key][i],z[key][row['index']-begin],atol=2e-5,rtol=2e-4), (row['case_id'], key)
                values.update({label+'__'+k:v for k,v in measured.items()})
            raw_numpy = raw.cpu().numpy().reshape(3,n,97,283)
            tasks = [pool.submit(measure_case, row, raw_numpy[:, i]) for i,row in enumerate(batch)]
            pending.append((start,batch,values,tasks,lock))
            if len(pending) >= 3: finish(pending.pop(0))
            if args.max_batches and accepted+sum(len(p[1]) for p in pending) >= args.max_batches*batch_size: break
        while pending: finish(pending.pop(0))
    finally:
        pool.shutdown(wait=True)
    print('[worker_finished] ' + worker_id, flush=True)


def summarize(args):
    import evaluate_generation_modalities as generation
    generation.base.setup()
    generation.summarize(args)
    from nymeria_baselines.phase2_collision_report import summarize_collisions
    summarize_collisions(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','worker','summarize'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-operation', type=Path)
    parser.add_argument('--audited-gt-operation', type=Path)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname(): raise RuntimeError('Compute allocation required')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage != 'prepare':
        import torch
        torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    globals()[args.stage](args)


if __name__ == '__main__': main()
