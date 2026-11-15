"""Post-hoc matched-cohort rescore excluding the exact19 UEM extreme cases.

Use cached per-case physical metrics and paired TMR embeddings. No inference,
normalization, original prediction files or full-cohort results are changed.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import fcntl
import json
import runpy
import time
import urllib.parse

import numpy as np

from nymeria_baselines.fulltest_v2m import read, save, sha, npz, guard
from nymeria_baselines.da3_c2m_report import LABELS, FIELDS, PAGE

TITLE = 'DA3 camera-to-motion | matched cohort excluding19 UEM extreme cases | 2026-11-16'


def rescore(root, out):
    guard()
    from nymeria_baselines.evaluate_reconstruction_tmr import fid, retrieval_details
    out.mkdir(parents=True, exist_ok=True)
    original = read(root/'cases.json')
    assert len(original) == 11911
    ids = [r['id'] for r in original]
    exclusions = []
    # Use exactly the previously reported >10m absolute-coordinate predicate,
    # not the different threshold on mean MPJPE (which would remove only15).
    for row in original:
        path = root/'models/uem_da3_trainprior/predictions'/(row['id']+'.npz')
        with np.load(path) as z:
            maximum = float(np.abs(z['joints']).max())
        if maximum > 10.:
            exclusions.append(dict(id=row['id'],uuid=row['uuid'],start=row['start'],
                maximum_absolute_joint_m=maximum,prediction_sha256=sha(path)))
    audit = read(root/'UEM_SCORING_DTYPE_AUDIT.json')
    assert len(exclusions) == audit['large_position_cases_over10m'] == 19
    assert {r['id'] for r in exclusions} == {r['id'] for r in audit['largest_cases'] if r['maximum_absolute_joint_m'] > 10.}
    excluded = {r['id'] for r in exclusions}
    keep = np.array([r['id'] not in excluded for r in original])
    rows = [r for r, selected in zip(original, keep) if selected]
    kept_ids = [r['id'] for r in rows]
    assert len(rows) == 11892
    save(out/'excluded_cases.json',dict(predicate='max(abs(UEM decoded global body23 coordinates in meters)) >10',
        count=19,selection='Post-hoc UEM-prediction-dependent exclusion; exact same19 annotation-case IDs excluded from every model and GT',rows=exclusions))
    save(out/'cases.json',rows)
    configuration=dict(source=str(root),source_configuration_sha256=sha(root/'configuration.json'),
        source_cases_sha256=sha(root/'cases.json'),excluded_manifest_sha256=sha(out/'excluded_cases.json'),
        retained_cases_sha256=sha(out/'cases.json'),original_cases=11911,excluded_cases=19,retained_cases=len(rows),
        recordings=len({r['uuid'] for r in rows}),physical_windows=len({(r['uuid'],r['start']) for r in rows}),
        source_audit_sha256=sha(root/'UEM_SCORING_DTYPE_AUDIT.json'),implementation_sha256=sha(__file__),
        physical_metrics='Reaggregate original per-case metrics, including recording-balanced means. Original units/formulas unchanged.',
        semantic_metrics='Recompute FID against retained GT embeddings and R32 groups/ranks using retained11892 paired embeddings, seeds0/1/2. No averaging old ranks.',
        input_policy='Same DA3 camera + train-only gravity/initial-height prior; supplied shape/native samplers/statistics unchanged.',
        limitation='Post-hoc exclusion based on UniEgoMotion output; filtered-cohort results, not the full-test performance. Same exclusion in all comparison rows. Full11911-case report and all raw predictions preserved.')
    save(out/'configuration.json',configuration)
    with np.load(root/'scores/GT/embeddings.npz') as z:
        assert z['case_id'].tolist()==ids
        gt,text=z['motion'][keep].copy(),z['text'][keep].copy()
    expected_evaluator=read(root/'scores/GT/summary.json')['evaluator_sha256']
    sources={}
    for name,label in [('GT','GT'),*LABELS]:
        source=root/'scores'/label;target=out/'scores'/label;target.mkdir(parents=True,exist_ok=True)
        summary=read(source/'summary.json');completion=read(source/'COMPLETE.json')
        assert completion['cases']==summary['cases']==11911 and completion['summary_sha256']==sha(source/'summary.json')
        assert summary['evaluator_sha256']==expected_evaluator
        assert summary['input_configuration_sha256']==sha(root/'configuration.json')
        per_case=read(source/'per_case.json');assert [r['id'] for r in per_case]==ids
        selected=[r for r,k in zip(per_case,keep) if k]
        if label=='GT' and 'sjpe' not in selected[0]['metrics']:
            # The historical GT summary had extended metrics backfilled, while
            # its original per-case file predates those fields. Recompute GT
            # self-errors/jerk using the exact frozen body23 target cache.
            from nymeria_baselines.motion_error_metrics import extra_metrics
            cache=root.parent/'motion_metrics_20261105'
            receipt=read(cache/'gt_cache.json')
            assert receipt['cases_sha256']==sha(root/'cases.json')
            assert receipt['sha256']==sha(cache/'gt_body23.npy')
            targets=np.load(cache/'gt_body23.npy',mmap_mode='r')
            for row,index in zip(selected,np.flatnonzero(keep)):
                row['metrics'].update(extra_metrics(targets[index],targets[index],fps=20.))
        with np.load(source/'embeddings.npz') as z:
            assert z['case_id'].tolist()==ids
            motion=z['motion'][keep].copy()
            assert np.array_equal(z['text'][keep],text)
        assert motion.shape==gt.shape==text.shape==(11892,256)
        assert np.isfinite(motion).all() and np.isfinite(gt).all()
        retrieval,groups=retrieval_details(motion,text,kept_ids)
        assert retrieval['R32_cases_per_seed']==11872 and retrieval['R32_dropped_per_seed']==20
        physical={k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}
        recordings=defaultdict(list)
        for row in selected:recordings[row['uuid']].append(row['metrics'])
        balanced={k:float(np.mean([np.mean([v[k] for v in values]) for values in recordings.values()])) for k in physical}
        case_errors=np.array([r['metrics']['mpjpe_m']*100 for r in selected])
        result=dict(model=label,name=name,cases=len(rows),recordings=len(recordings),physical=physical,
            recording_balanced_physical=balanced,FID=fid(motion,gt),retrieval=retrieval,
            case_distribution=dict(mean_mpjpe_cm=float(case_errors.mean()),median_mpjpe_cm=float(np.median(case_errors)),
                p95_case_mpjpe_cm=float(np.percentile(case_errors,95)),max_case_mpjpe_cm=float(case_errors.max())),
            input_configuration_sha256=sha(out/'configuration.json'),source_summary_sha256=sha(source/'summary.json'),
            source_per_case_sha256=sha(source/'per_case.json'),source_embeddings_sha256=sha(source/'embeddings.npz'),
            evaluator_sha256=expected_evaluator,text_cache_sha256=summary['text_cache_sha256'],completed=time.time())
        assert all(np.isfinite(v) for v in physical.values())
        if label=='GT':assert result['FID']==physical['mpjpe_m']==physical['sjpe']==0
        npz(target/'embeddings.npz',motion=motion,text=text,case_id=np.array(kept_ids))
        save(target/'per_case.json',selected);save(target/'retrieval_groups.json',groups)
        save(target/'summary.json',result);save(target/'COMPLETE.json',dict(cases=len(rows),summary_sha256=sha(target/'summary.json')))
        sources[label]=dict(source_summary_sha256=result['source_summary_sha256'],summary_sha256=sha(target/'summary.json'))
        print(json.dumps(dict(model=name,cases=len(rows),MPJPE_cm=physical['mpjpe_m']*100,PA_cm=physical['pa_mpjpe_m']*100,
            Root_ADE_cm=physical['root_err_m']*100,FID=result['FID'],R3=retrieval['R32_R03_mean'])),flush=True)
    save(out/'SCORING_COMPLETE.json',dict(cases_per_model=len(rows),excluded=19,models=4,recordings=configuration['recordings'],sources=sources))


def publish(root, out):
    cfg=read(out/'configuration.json');done=read(out/'SCORING_COMPLETE.json')
    assert cfg['retained_cases']==done['cases_per_model']==11892
    rows=[['Model / DA3 camera','Cases',*[v[2] for v in FIELDS],'FID','R@3 %']]
    results=[]
    for name,label in [('GT','GT'),*LABELS]:
        p=out/'scores'/label;value=read(p/'summary.json')
        assert read(p/'COMPLETE.json')['summary_sha256']==sha(p/'summary.json')
        assert value['cases']==11892 and value['input_configuration_sha256']==sha(out/'configuration.json')
        physical=value['physical'];retrieval=value['retrieval']
        rows.append([name,str(value['cases']),*[f'{physical[k]*scale:.4f}' for k,scale,_ in FIELDS],f"{value['FID']:.6f}",
            f"{retrieval['R32_R03_mean']:.3f} ± {retrieval['R32_R03_std']:.3f}"])
        results.append(value)
    with (out/'comparison.csv').open('w') as f:csv.writer(f).writerows(rows)
    with (out/'all_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['Model','Aggregation','Metric','Value'])
        for r in results:
            writer.writerow([r['name'],'window_weighted','FID',r['FID']])
            for group in ['physical','recording_balanced_physical','retrieval','case_distribution']:
                for key,value in r[group].items():writer.writerow([r['name'],group,key,value])
    description=(f"Filtered matched comparison: {cfg['retained_cases']:,} annotation cases / {cfg['physical_windows']:,} physical windows / "
        f"{cfg['recordings']} recordings, T97/20FPS. User requested excluding the EXACT19 previously reported UniEgoMotion cases with max(abs(global body23 coordinates))>10m. "
        'The same19 annotation-case IDs are excluded from GT, EgoEgo and EgoAllo too. This is a POST-HOC UEM-output-dependent subset, not the full test cohort. '
        'Original11911-case report and raw predictions are preserved. No inference/model/normalization/statistics/placement changes. '
        'All physical means and recording-balanced aggregates recomputed from retained per-case scores; FID and R@1/2/3 recomputed from retained cached embeddings. '
        'R32 uses seeds0/1/2,11872 cases and20 dropped per seed; physical/FID use all11892. Same training-only initial gravity/height prior as the original DA3 experiment. '
        'Whole-clip similarity PA-MPJPE, UPE14/LPE9, DisCoRD mean sJPE; GT included.')
    save(out/'summary.json',dict(configuration=cfg,models=results,description=description))
    (out/'RESULTS.md').write_text('# '+TITLE+'\n\n'+description+'\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [rows[0],['---']*len(rows[0]),*rows[1:]])+'\n\nArtifacts: `'+str(out)+'`\n')
    helper=runpy.run_path('/scratch2/whwjdqls99/cosmos/operations/phase1_notext_fourway_55k_20261019/upload_notion.py')
    api=helper['Notion'](helper['read_token']());rt=helper['rich_text'];plain=helper['plain_text']
    def children(uid):
        result=[];cursor=None
        while True:
            q={'page_size':100}
            if cursor:q['start_cursor']=cursor
            d=api.request('GET','/blocks/'+uid+'/children?'+urllib.parse.urlencode(q))
            result.extend(b for b in d['results'] if not b.get('archived') and not b.get('in_trash'))
            if not d.get('has_more'):return result
            cursor=d['next_cursor']
    state=out/'notion_publication.json'
    if state.exists():ids=read(state)
    else:
        existing=children(PAGE)
        matches=[(i,b) for i,b in enumerate(existing) if b['type']=='heading_2' and plain(b['heading_2']['rich_text'])==TITLE]
        assert len(matches)<=1
        if matches:
            i,heading=matches[0];table=next(b for b in existing[i+1:] if b['type']=='table')
            ids=dict(heading=heading['id'],table=table['id'])
        else:
            blocks=[dict(object='block',type='heading_2',heading_2=dict(rich_text=rt(TITLE))),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(description))),
                dict(object='block',type='table',table=dict(table_width=len(rows[0]),has_column_header=True,has_row_header=False,
                    children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(x) for x in row])) for row in rows])),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt('Artifacts: '+str(out)+'; excluded_cases.json pins all19 IDs and source prediction hashes. comparison.csv and all_metrics.csv contain the recomputed results; per-case metrics, retained embeddings, new R32 groups/ranks and configuration hashes are saved. Original full-set section above remains archival.')))]
            answer=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=blocks))
            ids=dict(heading=answer['results'][0]['id'],table=answer['results'][2]['id'])
        save(state,ids)
    actual=children(ids['table']);assert len(actual)==len(rows)
    for block,expected in zip(actual,rows):
        if [plain(c) for c in block['table_row']['cells']]!=expected:
            api.request('PATCH','/blocks/'+block['id'],dict(table_row=dict(cells=[rt(v) for v in expected])))
    assert [[plain(c) for c in b['table_row']['cells']] for b in children(ids['table'])]==rows
    url='https://www.notion.so/'+PAGE.replace('-','')+'#'+ids['heading'].replace('-','')
    save(out/'NOTION_VERIFIED.json',dict(url=url,table=ids['table'],comparison_sha256=sha(out/'comparison.csv'),time=time.time()))
    save(out/'COMPLETE.json',dict(status='complete_verified',cases_per_model=11892,excluded=19,models=4,
        original_predictions_preserved=True,notion_verified=True,configuration_sha256=sha(out/'configuration.json'),comparison_sha256=sha(out/'comparison.csv')))
    print('PUBLISHED',url,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/scratch2/whwjdqls99/cosmos/operations/da3_c2m_fulltest_20261116'))
    a=p.parse_args();guard();out=a.root/'exclude19_20261116';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'SCORING_COMPLETE.json').exists():rescore(a.root,out)
        with (a.root/'notion_report.lock').open('a') as remote_lock:
            fcntl.flock(remote_lock,fcntl.LOCK_EX)
            publish(a.root,out)


if __name__=='__main__':main()
