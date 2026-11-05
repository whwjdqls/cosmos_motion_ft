"""Backfill decoded-joint errors without inference or changing original audits.

Writes additive extended_metrics/ results beside each existing experiment.
Original summary/COMPLETE files remain valid, and the pending full-test model
is completed automatically when its original scoring marker appears.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from nymeria_baselines.motion_error_metrics import extra_metrics, protocol

BASE = Path('/scratch2/whwjdqls99/cosmos/operations')
DEFAULT_REFERENCE = BASE/'fulltest_v2m_ablations_20261104'
DEFAULT_FINAL = BASE/'idle_gpu_followups_20261104/uem_final350'
DEFAULT_OUT = BASE/'motion_metrics_20261105'
POSITION_KEYS = ('mpjpe_m', 'pa_mpjpe_m', 'root_err_m')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def guard():
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in os.uname().nodename
    assert os.uname().nodename not in ('cs-gpu-01', 'node37', 'node29')


def identity(out):
    return dict(protocol=protocol(), metric_source_sha256=sha(Path(__file__).with_name('motion_error_metrics.py')),
                backfill_source_sha256=sha(__file__), gt_cache_sha256=read(out/'gt_cache.json')['sha256'])


def prepare_cache(args):
    guard()
    from nymeria_baselines.visualize_official_baselines import decode
    from nymeria_baselines.body_shape import BODY_JOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'gt_cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg, rows = read(args.reference/'configuration.json'), read(args.reference/'cases.json')
        assert sha(args.reference/'cases.json') == cfg['cases_sha256']
        if (args.out/'gt_cache.json').exists():
            info = read(args.out/'gt_cache.json')
            assert info['cases_sha256'] == cfg['cases_sha256']
            assert sha(args.out/'gt_body23.npy') == info['sha256']
            return
        temp = args.out/'gt_body23.building.npy'
        target = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(rows),97,23,3))
        for i, row in enumerate(rows):
            assert sha(row['bundle']) == row['bundle_sha256']
            with np.load(row['bundle']) as data:
                joints, _ = decode(data['raw'].copy(), 30)
            target[i] = joints[:,BODY_JOINTS].astype(np.float32)
            if i%1000 == 0:
                print(json.dumps(dict(stage='gt_cache', completed=i, total=len(rows))), flush=True)
        target.flush()
        del target
        temp.replace(args.out/'gt_body23.npy')
        save(args.out/'gt_cache.json', dict(cases=len(rows), cases_sha256=cfg['cases_sha256'],
            shape=[len(rows),97,23,3], dtype='float32', sha256=sha(args.out/'gt_body23.npy'),
            policy='Exact original fulltest decoder and float32 body23 subset; no new normalization or motion changes.'))


def specifications(args):
    tasks = []
    for root in (args.reference, args.final):
        cfg = read(root/'configuration.json')
        excluded = set(read(root/'EXCLUDED_MODELS.json')['excluded_models']) if (root/'EXCLUDED_MODELS.json').exists() else set()
        for model in [dict(label='GT', step=0), *cfg['models']]:
            if model['label'] not in excluded:
                tasks.append((str(root), model['label'], str(args.out)))
    return tasks


def measure_model(task):
    guard()
    from motion_expert_joint_attention.eval_motion_recon import joint_metrics
    root, label, out = Path(task[0]), task[1], Path(task[2])
    cfg, rows = read(root/'configuration.json'), read(root/'cases.json')
    receipt = identity(out)
    assert read(out/'gt_cache.json')['cases_sha256'] == cfg['cases_sha256'] == sha(root/'cases.json')
    target = root/'extended_metrics'/label
    target.mkdir(parents=True, exist_ok=True)
    worker_lock = (target/'worker.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX)
    original = root/'scores'/label
    original_done = (original/'COMPLETE.json').exists()
    original_hash = sha(original/'summary.json') if original_done else None
    if (target/'COMPLETE.json').exists():
        completed = read(target/'COMPLETE.json')
        if completed['identity'] == receipt and completed['original_summary_sha256'] == original_hash:
            return str(target/'summary.json')
    reference = {}
    if original_done:
        assert read(original/'COMPLETE.json')['summary_sha256'] == original_hash
        reference = {r['id']:r['metrics'] for r in read(original/'per_case.json')}
        assert set(reference) == {r['id'] for r in rows}
    spec = next((s for s in cfg['models'] if s['label'] == label), dict(label='GT', step=0))
    gt = np.load(out/'gt_body23.npy', mmap_mode='r')
    records = []
    started = time.monotonic()
    maximum_difference = {k:0. for k in POSITION_KEYS}
    for start in range(0,len(rows),256):
        selected = [(i, r) for i,r in enumerate(rows[start:start+256],start)
                    if label == 'GT' or (root/'models'/label/'predictions'/(r['id']+'.npz')).exists()]
        dest = target/'batches'/f'{start:06d}.json'
        batch = None
        if dest.exists():
            saved = read(dest)
            if saved['identity'] == receipt and saved['ids'] == [r['id'] for _,r in selected]:
                batch = saved['records']
        if batch is None:
            batch = []
            for i,row in selected:
                truth = gt[i]
                if label == 'GT':
                    prediction = truth
                else:
                    with np.load(root/'models'/label/'predictions'/(row['id']+'.npz')) as z:
                        assert str(z['case_id']) == row['id'] and int(z['seed']) == row['seed']
                        assert str(z['checkpoint_sha256']) == spec['checkpoint_sha256']
                        prediction = z['joints'].copy()
                measured = joint_metrics(prediction, truth)
                metrics = {k:measured[k] for k in POSITION_KEYS}
                metrics.update(extra_metrics(prediction, truth, fps=20.))
                batch.append(dict(id=row['id'], uuid=row['uuid'], start=row['start'], seed=row['seed'],
                    joints_sha256=hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(), metrics=metrics))
            save(dest, dict(identity=receipt, ids=[r['id'] for _,r in selected], records=batch))
        for item in batch:
            if reference:
                for key in POSITION_KEYS:
                    diff = abs(item['metrics'][key]-reference[item['id']][key])
                    maximum_difference[key] = max(maximum_difference[key],diff)
                    assert diff <= 1e-6, (label,item['id'],key,diff,'Original position metric changed')
        records.extend(batch)
        if start%1024 == 0:
            save(target/'progress.json', dict(model=label, completed=len(records), total=len(rows), seconds=time.monotonic()-started))
    assert len({r['id'] for r in records}) == len(records)
    if not records:
        return None
    keys = list(records[0]['metrics'])
    physical = {k:float(np.mean([r['metrics'][k] for r in records])) for k in keys}
    grouped = defaultdict(list)
    for r in records:
        grouped[r['uuid']].append(r['metrics'])
    balanced = {k:float(np.mean([np.mean([r[k] for r in values]) for values in grouped.values()])) for k in keys}
    full = len(records) == len(rows) and original_done
    result = dict(model=label, step=spec['step'], cases=len(records), expected=len(rows), complete=full,
        physical=physical, recording_balanced_physical=balanced, identity=receipt,
        original_summary_sha256=original_hash, position_metric_parity_max_abs_m=maximum_difference,
        input_configuration_sha256=sha(root/'configuration.json'), checkpoint_sha256=spec.get('checkpoint_sha256'),
        protocol=protocol(), completed=time.time(), original_result=str(original/'summary.json'))
    save(target/'per_case.json', records)
    save(target/'summary.json', result)
    if full:
        save(target/'COMPLETE.json', dict(identity=receipt, cases=len(records), summary_sha256=sha(target/'summary.json'),
            original_summary_sha256=original_hash))
    print(json.dumps(dict(model=label, cases=len(records), complete=full, seconds=time.monotonic()-started)), flush=True)
    return str(target/'summary.json')


def comparison(args):
    rows = [['Experiment/model','Step','Cases','MPJPE cm','PA-MPJPE cm','Root ADE cm',
             'UPE cm','LPE cm','sJPE mean','Noise sJPE mean','Static sJPE mean','sJPE native joint-sum','Status']]
    table = []
    complete = True
    for root,label,_ in specifications(args):
        path = Path(root)/'extended_metrics'/label/'summary.json'
        scope = 'Final350 UEM' if Path(root) == args.final else 'Full-test V2M'
        if path.exists():
            result = read(path)
            v = result['physical']
            values = [f"{v[k]*scale:.6f}" for k,scale in [('mpjpe_m',100),('pa_mpjpe_m',100),
                ('root_err_m',100),('upe_m',100),('lpe_m',100),('sjpe',1),('sjpe_noise',1),
                ('sjpe_static',1),('sjpe_discord_v1_jointsum',1)]]
            row = [scope+'/'+label,str(result['step']),str(result['cases']),*values,'Complete' if result['complete'] else 'Partial']
            table.append({**result, 'experiment':str(root), 'result':str(path)})
            complete &= result['complete']
        else:
            row = [scope+'/'+label,'—','0',*['—']*9,'Pending']
            complete = False
        rows.append(row)
    with (args.out/'comparison.csv').open('w') as stream:
        csv.writer(stream).writerows(rows)
    save(args.out/'comparison.json', dict(rows=rows, models=table, complete=bool(complete), protocol=protocol()))
    if complete:
        save(args.out/'COMPLETE.json', dict(complete=True, model_rows=len(rows)-1, comparison_sha256=sha(args.out/'comparison.json')))
    return complete


def run(args):
    import multiprocessing
    prepare_cache(args)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(measure_model, task) for task in specifications(args)]
        for future in as_completed(futures):
            future.result()
    return comparison(args)


def publish(args):
    """Append/update one additive table; preserve original audited tables."""
    import runpy
    helper = runpy.run_path(str(BASE/'phase1_notext_fourway_55k_20261019/upload_notion.py'))
    api = helper['Notion'](helper['read_token']())
    rt, plain = helper['rich_text'], helper['plain_text']
    current = read(args.out/'comparison.json')
    rows = current['rows']
    page = '40dd2a44-f44b-8080-aad3-cda6b5fea7c8'
    note = ('All accepted11911 cases across71 recordings; GT included. Position errors are cm on the common23 body joints. '
        'UPE uses14 spine/head/arm joints; LPE uses pelvis+8 leg joints. PA-MPJPE preserves the original whole-clip similarity fit; '
        'Root ADE is absolute Hips trajectory error. sJPE compares wrist/toe jerk magnitudes at20fps, with no smoothing or alignment. '
        'sJPE mean averages time and four end effectors (0..1); native DisCoRD calculate_area sums four joints, hence its score is4x this mean. '
        'Both raw native v1 and v2 outputs, noise/static components and exact source revision are saved. MPJVE omitted at user request. '
        'Partial rows are not final full-test metrics. Phase1 video-only predictions have no joint metrics. Artifacts: '+str(args.out))
    statefile = args.out/'notion_publication.json'
    with (args.out/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if statefile.exists():
            state = read(statefile)
            existing = {b['id'] for b in api.request('GET','/blocks/'+state['table']+'/children?page_size=100')['results']}
            for uid,row in zip(state['rows'],rows,strict=True):
                if uid in existing:
                    api.request('PATCH','/blocks/'+uid,dict(table_row=dict(cells=[rt(x) for x in row])))
        else:
            children = [dict(object='block',type='heading_2',heading_2=dict(rich_text=rt('Motion position and DisCoRD sJPE — full-test metrics'))),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(note))),
                dict(object='block',type='table',table=dict(table_width=len(rows[0]),has_column_header=True,has_row_header=False,
                    children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(x) for x in row])) for row in rows]))]
            response = api.request('PATCH','/blocks/'+page+'/children',dict(children=children))
            table = response['results'][-1]['id']
            blocks = api.request('GET','/blocks/'+table+'/children?page_size=100')['results']
            state = dict(page=page, table=table, heading=response['results'][0]['id'], rows=[b['id'] for b in blocks])
            save(statefile,state)
        blocks = api.request('GET','/blocks/'+state['table']+'/children?page_size=100')['results']
        expected = dict(zip(state['rows'],rows,strict=True))
        for block in blocks:
            if block['id'] in expected:
                assert [plain(c) for c in block['table_row']['cells']] == expected[block['id']]
        state.update(verified=True, verified_time=time.time(), complete=current['complete'])
        save(statefile,state)
    print(json.dumps(dict(notion_verified=True, complete=current['complete'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run','publish','watch'])
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--final', type=Path, default=DEFAULT_FINAL)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.command == 'publish':
        publish(args)
        return
    if args.command == 'run':
        run(args)
        return
    # Metadata-only watcher runs on login; every numerical pass is dispatched
    # to an owned compute allocation. No inference is restarted or changed.
    import subprocess
    while True:
        pending = [(root,label) for root,label,_ in specifications(args)
                   if not (Path(root)/'extended_metrics'/label/'COMPLETE.json').exists()]
        ready = any((Path(root)/'scores'/label/'COMPLETE.json').exists() for root,label in pending)
        if ready:
            subprocess.run(['bash',str(args.out/'run_compute.sh')],check=True)
        if (args.out/'comparison.json').exists():
            publish(args)
        if not pending:
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
