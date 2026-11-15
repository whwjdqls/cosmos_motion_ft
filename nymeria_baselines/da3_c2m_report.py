"""Additive, read-back-verified Notion status/results for the DA3 experiment."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import runpy
import time
import urllib.parse
import numpy as np

from nymeria_baselines.fulltest_v2m import read, save, sha

PAGE = '40dd2a44-f44b-8080-aad3-cda6b5fea7c8'
TAG = 'DA3 camera-to-motion / full test / 2026-11-16 / training-prior gauge'
LABELS = [('EgoEgo', 'egoego_da3_trainprior'), ('EgoAllo', 'egoallo_da3_trainprior'), ('UniEgoMotion', 'uem_da3_trainprior')]
FIELDS = [('mpjpe_m',100,'MPJPE cm'),('pa_mpjpe_m',100,'PA-MPJPE cm'),('root_err_m',100,'Root ADE cm'),
          ('upe_m',100,'UPE cm'),('lpe_m',100,'LPE cm'),('sjpe',1,'sJPE'),
          ('foot_skate_cm_s',1,'Skate cm/s'),('foot_depth_mean_cm',1,'Penetration cm')]


def table(out):
    cfg = read(out/'configuration.json')
    header = ['Model / camera input','Cases',*[x[2] for x in FIELDS],'FID','R@3 %']
    rows = [header]; complete = 0; records = []
    for title,label,source in [('GT','GT',out/'scores/GT/summary.json')]+[
        (name+' / '+kind, spec['label'], path)
        for name,label in LABELS
        for spec in [next(s for s in cfg['models'] if s['label']==label)]
        for kind,path in [('measured camera',Path(spec['measured_reference'])/'scores'/spec['measured_label']/'summary.json'),
                          ('DA3 + train prior',out/'scores'/label/'summary.json')]]:
        scored = source.exists()
        if 'DA3 +' in title:
            receipt = source.parent/'COMPLETE.json'
            scored = scored and receipt.exists()
            if scored:
                assert read(receipt)['summary_sha256']==sha(source)
                assert read(receipt)['cases']==11911
        if scored:
            s=read(source)
            if 'DA3 +' in title:
                assert s['cases']==11911 and s['recordings']==71
                assert s['input_configuration_sha256']==sha(out/'configuration.json')
                complete+=1
            physical=s['physical'];retrieval=s['retrieval']
            rows.append([title,str(s['cases']),*[f'{physical[k]*scale:.4f}' if k in physical else 'unavailable' for k,scale,_ in FIELDS],
                f"{s['FID']:.6f}",f"{retrieval['R32_R03_mean']:.3f} ± {retrieval['R32_R03_std']:.3f}"])
            records.append(dict(name=title,source=str(source),summary_sha256=sha(source),summary=s))
        else:
            progress=[]
            for p in sorted((out/'models'/label).glob('progress_rank*.json')):
                d=read(p);progress.append(f"r{d['rank']} {d['completed']}/{d['total']}")
            reason='; '.join(progress) or 'Pending'
            block=out/'EGOALLO_BLOCKED.json'
            if label=='egoallo_da3_trainprior' and block.exists():reason='Blocked: checkpoint unreadable by this account'
            rows.append([title,reason,*['—']*(len(header)-2)])
    return rows,records,complete


def publish(out):
    rows,records,complete=table(out)
    description = ('Full accepted Nymeria test cohort: 11,911 annotation cases, 11,677 physical windows, 71 recordings; '
        '97 frames at 20 FPS. DA3NESTED-GIANT-LARGE-1.1 predicted metric camera poses; metric scale was already applied. '
        'DA3 did not estimate absolute gravity, floor or initial height: an equal-record TRAIN-ONLY initial orientation/height prior supplies these. '
        'No test-GT camera/body/floor/scale is used in conditioning or placement. Supplied shape and native samplers/seeds/weights are unchanged. '
        'Measured-camera results below are historical references with additional gravity/height information; they are not an equal-information comparison. '
        'Whole-clip similarity PA-MPJPE; UPE14/LPE9; DisCoRD four-end-effector mean sJPE. Original frozen Nymeria TMR statistics and GT; R32 with three grouping seeds. '
        'Text is used for retrieval scoring only, not motion sampling.')
    local='# '+TAG+'\n\n'+description+'\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [rows[0],['---']*len(rows[0]),*rows[1:]])
    local+='\n\nArtifacts: `'+str(out)+'`. Native metric keys, per-case scores and recording-balanced summaries are retained.\n'
    (out/'RESULTS.md').write_text(local)
    with (out/'comparison.csv').open('w') as f: csv.writer(f).writerows(rows)
    save(out/'comparison.json',dict(completed_models=complete,models=records,description=description))
    h=runpy.run_path('/scratch2/whwjdqls99/cosmos/operations/phase1_notext_fourway_55k_20261019/upload_notion.py')
    api=h['Notion'](h['read_token']());rt=h['rich_text'];plain=h['plain_text']
    def children(uid):
        values=[];cursor=None
        while True:
            query={'page_size':100}
            if cursor:query['start_cursor']=cursor
            d=api.request('GET','/blocks/'+uid+'/children?'+urllib.parse.urlencode(query))
            values.extend(b for b in d['results'] if not b.get('archived') and not b.get('in_trash'))
            if not d.get('has_more'):return values
            cursor=d['next_cursor']
    state=out/'notion_publication.json'
    if state.exists():
        ids=read(state)
    else:
        # Recover a successful but unacknowledged append before creating anything.
        existing=children(PAGE)
        matches=[(i,b) for i,b in enumerate(existing) if b['type']=='heading_2' and plain(b['heading_2']['rich_text'])==TAG]
        if matches:
            assert len(matches)==1
            i,b=matches[0];t=next(x for x in existing[i+1:] if x['type']=='table')
            ids=dict(heading=b['id'],table=t['id'],rows=[r['id'] for r in children(t['id'])])
        else:
            blocks=[dict(object='block',type='heading_2',heading_2=dict(rich_text=rt(TAG))),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(description))),
                dict(object='block',type='table',table=dict(table_width=len(rows[0]),has_column_header=True,has_row_header=False,
                    children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(v) for v in r])) for r in rows])),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt('Artifacts: '+str(out)+'; RESULTS.md, comparison.csv, per-case scores, immutable camera handoff, training_camera_prior.json, MEASURED_ADAPTER_PARITY.json. Pending rows are not evaluated results.')))]
            d=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=blocks))
            uid=d['results'][2]['id']
            ids=dict(heading=d['results'][0]['id'],table=uid,rows=[r['id'] for r in children(uid)])
        save(state,ids)
    assert len(ids['rows'])==len(rows)
    current=children(ids['table'])
    for uid,expected,actual in zip(ids['rows'],rows,current):
        if [plain(c) for c in actual['table_row']['cells']]!=expected:
            api.request('PATCH','/blocks/'+uid,dict(table_row=dict(cells=[rt(v) for v in expected])))
    actual=children(ids['table'])
    assert [[plain(c) for c in b['table_row']['cells']] for b in actual]==rows
    control=out/'EGOALLO_FRAMEWORK_PARITY.json'
    if control.exists():
        c=read(control);assert c['passed']
        tag=TAG+' | Environment acceptance: '
        note=(tag+'EgoEgo/UEM use Torch2.7.1+cu128/Python3.10. EgoAllo uses Torch2.10.0+cu128/Python3.13 '
            'with isolated jaxtyping0.3.2, rotary-embedding-torch0.8.6 and wadler-lindig0.1.7; original training environments are unchanged. '
            f"Same measured-camera/shape/seed control versus saved native EgoAllo: mean {c['mean_difference_cm']:.6f}cm, max {c['max_difference_cm']:.6f}cm. "
            'Initial acceptance cases passed native singleton replay bitwise and batch128 checks for EgoEgo/UEM; this is not a whole-cohort batch-parity guarantee. '
            'EgoEgo EMA555800 SHA6e19cec080e58b32186a49952a067fef365877f224701128b61febc173f4ffff; '
            'UEM final555800 SHAe45124576d72193bf91154ee7aab6710b6e6ec3762663070ae3080208d7d2d20; '
            'EgoAllo final200000 SHAf7a197c5d132fc49a96df1d56847240bc2c975779b4c92e5e4db7aa4d608c27f. '
            'Exact paths, converted-camera hashes, source pins and all original samplers/settings are in configuration.json and NATIVE_DEPENDENCIES_VERIFIED.json. '
            'DA3 floor/contact metrics refer to the assumed training-prior floor. Low skating alone can reflect feet above that floor; inspect saved foot heights and airborne fractions as well.')
        matches=[b for b in children(PAGE) if b['type']=='paragraph' and plain(b['paragraph']['rich_text']).startswith(tag)]
        assert len(matches)<=1
        if matches:
            uid=matches[0]['id']
            if plain(matches[0]['paragraph']['rich_text'])!=note:
                api.request('PATCH','/blocks/'+uid,dict(paragraph=dict(rich_text=rt(note))))
        else:
            d=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=[dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(note)))]))
            uid=d['results'][0]['id']
        block=api.request('GET','/blocks/'+uid)
        assert plain(block['paragraph']['rich_text'])==note
    audit=out/'UEM_SCORING_DTYPE_AUDIT.json'
    if audit.exists():
        a=read(audit)
        tag=TAG+' | Scoring dtype correction: '
        note=(tag+'The native UEM decoder saved float64 joint positions; the common TMR completion adapter uses float32. '
            'An exact-value assertion originally stopped scoring before any UEM batch was saved. '
            'The scorer now converts joints to the benchmark standard float32 BEFORE physical metrics and completion, matching the original prediction writer. '
            f"All {a['cases']} saved predictions were audited; maximum component-relative rounding error {a.get('maximum_component_relative_rounding_error',0):.9g}. "
            'Original prediction files, sampling, checkpoints, training/evaluator normalization, coordinate placement and camera conditions were unchanged. '
            'Existing EgoEgo/EgoAllo inputs were already float32 and their values are unchanged. Audit: UEM_SCORING_DTYPE_AUDIT.json.')
        matches=[b for b in children(PAGE) if b['type']=='paragraph' and plain(b['paragraph']['rich_text']).startswith(tag)]
        assert len(matches)<=1
        if matches:
            uid=matches[0]['id']
            if plain(matches[0]['paragraph']['rich_text'])!=note:
                api.request('PATCH','/blocks/'+uid,dict(paragraph=dict(rich_text=rt(note))))
        else:
            d=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=[dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(note)))]))
            uid=d['results'][0]['id']
        block=api.request('GET','/blocks/'+uid)
        assert plain(block['paragraph']['rich_text'])==note
        if a.get('large_position_cases_over10m',0):
            replay=out/'UEM_EXTREME_NATIVE_REPLAY.json'
            if replay.exists():
                d=read(replay)
                control=(f"Worst-case native singleton replay: max absolute joint {d['native_singleton_max_abs_joint_m']:.6g}m; "
                    f"saved batch128 {d['saved_batch128_max_abs_joint_m']:.6g}m. Both exceed10m: {d['both_extreme_over10m']}. "
                    'The native replay confirms catastrophic instability on this case; its magnitude differs from the batch result, so the precise raw mean is numerically unstable. ')
            else: control='Worst-case original singleton-sampler replay is pending. '
            statistics=[]
            for model,label in [('GT','GT'),*LABELS]:
                p=out/'scores'/label/'per_case.json'
                if not p.exists():continue
                records_case=read(p)
                values=np.array([r['metrics']['mpjpe_m']*100 for r in records_case])
                statistics.append(dict(model=model,cases=len(values),mean_mpjpe_cm=float(values.mean()),
                    median_mpjpe_cm=float(np.median(values)),p95_case_mpjpe_cm=float(np.percentile(values,95)),
                    maximum_case_mpjpe_cm=float(values.max()),cases_mpjpe_over10m=int((values>1000).sum())))
            save(out/'robust_case_distribution.json',dict(aggregation='All11911 cases; no filtering/replacement; percentiles across per-case mean-joint errors',models=statistics))
            tag=TAG+' | UEM extreme-output audit: '
            note=(tag+f"{a['large_position_cases_over10m']} of {a['cases']} cases already have absolute joint coordinates over10m before any FP32 conversion; "
                f"largest {a['absolute_joint_percentiles_m']['max']:.6g}m. The full-set mean is dominated by these divergent outputs. "
                'They are retained, not filtered, clipped, grounded or replaced. '+control+
                'Several largest-case camera inputs have horizontal translation jumps roughly40–100 training standard deviations; this is an association, not proof of cause. '+
                '; '.join(f"{d['model']}: median case MPJPE {d['median_mpjpe_cm']:.4f}cm, p95 {d['p95_case_mpjpe_cm']:.4f}cm" for d in statistics)+
                '. Full raw means remain in the comparison table. Per-case distribution and native replay are saved separately.')
            matches=[b for b in children(PAGE) if b['type']=='paragraph' and plain(b['paragraph']['rich_text']).startswith(tag)]
            assert len(matches)<=1
            if matches:
                uid=matches[0]['id']
                if plain(matches[0]['paragraph']['rich_text'])!=note:
                    api.request('PATCH','/blocks/'+uid,dict(paragraph=dict(rich_text=rt(note))))
            else:
                d=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=[dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(note)))]))
                uid=d['results'][0]['id']
            block=api.request('GET','/blocks/'+uid)
            assert plain(block['paragraph']['rich_text'])==note
    save(out/'NOTION_VERIFIED.json',dict(time=time.time(),completed_models=complete,table=ids['table'],
        comparison_sha256=sha(out/'comparison.csv'),url='https://www.notion.so/'+PAGE.replace('-','')+'#'+ids['heading'].replace('-','')))
    outlier_control=not audit.exists() or not read(audit).get('large_position_cases_over10m',0) or (out/'UEM_EXTREME_NATIVE_REPLAY.json').exists()
    if complete==3 and outlier_control:
        save(out/'COMPLETE.json',dict(status='complete_verified',models=3,cases_per_model=11911,recordings=71,
            comparison_sha256=sha(out/'comparison.csv'),notion_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with (a.out/'notion_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(a.out)
