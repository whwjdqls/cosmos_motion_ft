"""GA-MPJPE inventory for completed FULL test reconstruction evaluations only.

Reuse the already computed sequence-global alignment, verify per-case coverage
and geometry, distinguish ReViV's GA_SIM from its per-frame PA, and preserve
all earlier artifacts. Random20/fixed71 and unpaired generation are excluded.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,fcntl,json,runpy,time,urllib.parse
import numpy as np
from nymeria_baselines.fulltest_v2m import read,save,sha,guard
from nymeria_baselines.sequence_ga_mpjpe import align_sequence,selftest
from nymeria_baselines.da3_c2m_report import PAGE
from nymeria_baselines.fulltest_v2m_publish import LABELS

BASE=Path('/scratch2/whwjdqls99/cosmos/operations')
TITLE='GA-MPJPE — all saved FULL-test reconstruction evaluations — 2026-11-16'
ROOTS=[('V2M ablations',BASE/'fulltest_v2m_ablations_20261104'),
       ('Final supplied-shape UEM',BASE/'idle_gpu_followups_20261104/uem_final350'),
       ('GT camera',BASE/'egoego_fulltest_20261107'),
       ('GT camera',Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105')),
       ('DA3 camera / full',BASE/'da3_c2m_fulltest_20261116'),
       ('DA3 camera / exclude19',BASE/'da3_c2m_fulltest_20261116/exclude19_20261116')]


def local_results(out):
    reference=BASE/'fulltest_v2m_ablations_20261104'
    canonical=read(reference/'cases.json');lookup={r['id']:i for i,r in enumerate(canonical)}
    target_root=BASE/'motion_metrics_20261105'
    assert read(target_root/'gt_cache.json')['cases_sha256']==sha(reference/'cases.json')
    assert read(target_root/'gt_cache.json')['sha256']==sha(target_root/'gt_body23.npy')
    gt=np.load(target_root/'gt_body23.npy',mmap_mode='r')
    results=[];skipped=[];seen_gt=set()
    for group,root in ROOTS:
        cfg=read(root/'configuration.json');cases=read(root/'cases.json');ids=[r['id'] for r in cases]
        assert len(cases)>=11000 and len({r['uuid'] for r in cases})==71
        assert all(uid in lookup for uid in ids)
        excluded=set(read(root/'EXCLUDED_MODELS.json')['excluded_models']) if (root/'EXCLUDED_MODELS.json').exists() else set()
        for path in sorted((root/'scores').glob('*/summary.json')):
            key=path.parent.name
            if key in excluded:
                skipped.append(dict(source=str(path),reason='User-excluded incomplete evaluation'));continue
            marker=path.parent/'COMPLETE.json'
            if not marker.exists():
                skipped.append(dict(source=str(path),reason='No completed score receipt'));continue
            data=read(path);assert read(marker)['summary_sha256']==sha(path)
            assert data['cases']==len(cases) and data['recordings']==71
            if key=='GT':
                cohort=sha(root/'cases.json')
                if cohort in seen_gt:continue
                seen_gt.add(cohort)
            values=read(path.parent/'per_case.json');assert [v['id'] for v in values]==ids
            errors=np.array([r['metrics']['pa_mpjpe_m'] for r in values],dtype=np.float64)
            assert np.isfinite(errors).all() and abs(errors.mean()-data['physical']['pa_mpjpe_m'])<1e-10
            source_root=root.parent if group=='DA3 camera / exclude19' else root
            spec=next((s for s in read(source_root/'configuration.json').get('models',[]) if s['label']==key),{})
            display=LABELS.get(key,key)
            if key.startswith('uem_final_'):display='UEM final555800 / '+key.removeprefix('uem_final_')
            if key=='egoego_camera_shape_ema_final555800':display='EgoEgo EMA555800'
            if key=='egoallo_final200k_camera_shape':display='EgoAllo200k'
            checks=[]
            for index in sorted(set([0,len(cases)//2,len(cases)-1])):
                row=cases[index];truth=gt[lookup[row['id']]]
                if key=='GT':prediction=truth
                else:
                    with np.load(source_root/'models'/key/'predictions'/(row['id']+'.npz')) as z:prediction=z['joints'].copy()
                actual,*_=align_sequence(prediction,truth)
                difference=abs(actual-errors[index]);assert difference<1e-4,(group,key,row['id'],difference)
                checks.append(dict(id=row['id'],fresh_float64_ga_m=actual,cached_wholeclip_ga_m=float(errors[index]),difference_m=difference))
            by_record=defaultdict(list)
            for r,error in zip(values,errors):by_record[r['uuid']].append(error)
            entry=dict(group=group,model=display,model_key=key,step=spec.get('step'),joints=23,cases=len(cases),recordings=71,
                ga_mpjpe_m=float(errors.mean()),recording_balanced_ga_mpjpe_m=float(np.mean([np.mean(v) for v in by_record.values()])),
                median_case_ga_mpjpe_m=float(np.median(errors)),p95_case_ga_mpjpe_m=float(np.percentile(errors,95)),
                source=str(path),source_summary_sha256=sha(path),source_per_case_sha256=sha(path.parent/'per_case.json'),
                cohort_sha256=sha(root/'cases.json'),checkpoint_sha256=spec.get('checkpoint_sha256'),
                verification='Existing per-case whole-clip PA, all-case reaggregation; independent first/middle/last float64 alignment checks',
                independent_checks=checks,input=spec.get('inputs',spec.get('condition',group)))
            name=f'{len(results):02d}_{key}'
            save(out/'per_case'/f'{name}.json',[dict(id=r['id'],uuid=r['uuid'],ga_mpjpe_m=float(error)) for r,error in zip(values,errors)])
            results.append(entry);print('LOCAL',group,display,len(cases),entry['ga_mpjpe_m']*100,flush=True)
    # ReViV is a different joint protocol. Its original PA is PER-FRAME, so use
    # the explicitly computed GA_SIM (single whole-clip scale+R+t) instead.
    root=Path('/scratch2/minsu1206/cosmos/operations/reviv_zeroshot_20261105/full/scores_common16')
    cfg=read(root/'configuration.json');all_summary=read(root/'summary.json')
    assert cfg['metrics']['GA_SIM']=='one rotation + translation + scale over whole clip'
    assert cfg['code_sha256']==sha(root.parent.parent/'score.py')
    assert read(root/'COMPLETE.json')['summary_sha256']==sha(root/'summary.json')
    for label,data in all_summary.items():
        values=read(root/(label+'_per_case.json'));assert len(values)==11911 and [r['id'] for r in values]==[r['id'] for r in canonical]
        errors=np.array([r['metrics']['GA_SIM_MPJPE_mm']/1000 for r in values])
        assert abs(errors.mean()*1000-data['window_mean']['GA_SIM_MPJPE_mm'])<1e-9
        entry=dict(group='Common16 / ReViV comparison',model=label,model_key=label,joints=16,cases=11911,recordings=71,
            ga_mpjpe_m=float(errors.mean()),recording_balanced_ga_mpjpe_m=data['recording_balanced_mean']['GA_SIM_MPJPE_mm']/1000,
            source=str(root/'summary.json'),source_summary_sha256=sha(root/'summary.json'),source_per_case_sha256=sha(root/(label+'_per_case.json')),
            cohort_sha256=sha(reference/'cases.json'),verification='Existing per-case GA_SIM; verified full-clip fit source hash/configuration; NOT per-frame PA')
        results.append(entry);print('COMMON16',label,entry['ga_mpjpe_m']*100,flush=True)
    return results,skipped


def remote_results(out):
    """Preserve full-test published results separately from local verification.

    Source tables already use the shared validation_metrics.physical -> measure
    -> joint_metrics whole-clip scorer. These are reported values, not fresh
    local recomputations; Foundry prediction arrays are not mounted on Yonsei.
    """
    path=out/'remote_fulltest_sections.json'
    if not path.exists():return [],[]
    sources=read(path);records=[];skipped=[]
    for section in sources:
        title=section['title']
        if 'Predicted camera trajectories' in title:
            skipped.append(dict(source=title,reason='Camera pose extraction, no human-joint reconstruction'));continue
        if 'complete' not in title.lower():continue
        for table in section['tables']:
            rows=table['rows']
            if not rows:continue
            header=rows[0]
            if header[0]=='Motion condition' and 'PA-MPJPE cm ↓' in header:
                ga=header.index('PA-MPJPE cm ↓');raw=header.index('MPJPE cm ↓')
                for row in rows[1:]:
                    records.append(dict(group='Foundry published FULL-test / body23',model=title,condition=row[0],cases=11677,joints=23,
                        ga_mpjpe_m=float(row[ga])/100,raw_mpjpe_m=float(row[raw])/100,
                        source_page='412d2a44f44b80a2936aca27f96ac016',source_heading=section['id'],source_table=table['id'],
                        verification='Existing published whole-clip PA from shared validation scorer; source arrays unavailable locally, no fresh local recomputation'))
            elif header[0]=='Metric':
                mapping={r[0]:r[1:] for r in rows[1:]}
                key=next((k for k in ['pa_mpjpe_cm','body23_pa_mpjpe_m','pa_mpjpe_m'] if k in mapping),None)
                if key is None:continue
                for index,condition in enumerate(header[1:]):
                    if 'm2v' in condition.lower() or condition.lower().startswith(('t2m','ti2m')):continue
                    value=mapping[key][index]
                    if value in ('—','unavailable','N/A','NA',''):continue
                    records.append(dict(group='Foundry published FULL-test / body23',model=title,condition=condition,cases=11677,joints=23,
                        ga_mpjpe_m=float(value)/(100 if key.endswith('_cm') else 1),
                        source_page='412d2a44f44b80a2936aca27f96ac016',source_heading=section['id'],source_table=table['id'],
                        verification='Existing published whole-clip PA; source arrays unavailable locally, no fresh local recomputation'))
    return records,skipped


def publish(out,results,remote,skipped):
    table=[['Evaluation / input','Cases','Joints','GA-MPJPE cm ↓']]
    for d in results:table.append([d['group']+' / '+d['model'],str(d['cases']),str(d['joints']),f"{100*d['ga_mpjpe_m']:.4f}"])
    other=[['Published Foundry full-test model / condition','Cases','GA-MPJPE cm ↓','Verification']]
    for d in remote:
        other.append([d['model']+' / '+d['condition'],str(d['cases']),f"{100*d['ga_mpjpe_m']:.5f}",'Published whole-clip PA; no local re-fit'])
    description=('FULL all-window Nymeria reconstruction evaluations only:11911 annotation cases/11677 physical windows, plus the explicitly requested DA3 exclude19 subset11892 cases. '
        'No1420-window/random20, fixed71, partial evaluation, video-only output or unpaired Phase2 generation is included. '
        'GA-MPJPE uses ONE proper rotation+translation+uniform scale over the full97-frame sequence and all evaluated joints. '
        'Body23 evaluations already computed this as whole-clip PA: all saved per-case values are reaggregated, coverage/source hashes verified, and first/middle/last clips independently aligned in float64 per local run. '
        'The GT-camera11911-case runs were independently recomputed in full in the preceding section. '
        'ReViV uses a separate common16 mapping: use its GA_SIM, NOT its per-frame PA_SIM. Do not directly rank across joint sets, conditioning, checkpoints or cohorts. '
        'DA3 uses a train-only gravity/height prior; raw UEM divergent cases remain in the full row, and exclude19 is labeled separately. Low aligned error does not imply stable raw trajectories.')
    remote_note=('Foundry FULL11677-window reconstruction rows below are copied from completed, published source tables and labeled separately. '
        'They use the shared validation_metrics.physical→measure→joint_metrics whole-clip alignment path. Their source prediction arrays are not mounted on Yonsei, so these are NOT claimed as fresh independent local re-fits. '
        'A GT row is shown only when the source actually published the physical value; absent source GT physical metrics are not fabricated. Generation-only T2M/TI2M columns and fixed71 FiLM monitoring are omitted.')
    save(out/'summary.json',dict(description=description,local_results=results,remote_reported_results=remote,skipped=skipped,
        protocol=dict(frames=97,alignment='one proper Sim(3) per entire motion sequence',random20_included=False),
        source_checks=dict(geometry=sha(Path(__file__).with_name('sequence_ga_mpjpe.py')),
            legacy_alignment=sha(Path('motion_expert_joint_attention/eval_motion_recon.py')),
            validation_metrics=sha(Path('motion_expert_phase3_edge_joint/validation_metrics.py')))))
    with (out/'comparison.csv').open('w') as f:csv.writer(f).writerows(table)
    with (out/'remote_reported.csv').open('w') as f:csv.writer(f).writerows(other)
    (out/'RESULTS.md').write_text('# '+TITLE+'\n\n'+description+'\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [table[0],['---']*4,*table[1:]])+'\n\n'+remote_note+'\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [other[0],['---']*4,*other[1:]])+'\n')
    helper=runpy.run_path('/scratch2/whwjdqls99/cosmos/operations/phase1_notext_fourway_55k_20261019/upload_notion.py')
    api=helper['Notion'](helper['read_token']());rt=helper['rich_text'];plain=helper['plain_text']
    def children(uid):
        result=[];cursor=None
        while True:
            q={'page_size':100}
            if cursor:q['start_cursor']=cursor
            d=api.request('GET','/blocks/'+uid+'/children?'+urllib.parse.urlencode(q));result.extend(b for b in d['results'] if not b.get('archived') and not b.get('in_trash'))
            if not d.get('has_more'):return result
            cursor=d['next_cursor']
    def block(kind,text):return dict(object='block',type=kind,**{kind:dict(rich_text=rt(text))})
    def tblock(rows):return dict(object='block',type='table',table=dict(table_width=4,has_column_header=True,has_row_header=False,
        children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(v) for v in r])) for r in rows]))
    state=out/'notion_publication.json'
    if state.exists():ids=read(state)
    else:
        existing=children(PAGE);matches=[(i,b) for i,b in enumerate(existing) if b['type']=='heading_2' and plain(b['heading_2']['rich_text'])==TITLE]
        assert len(matches)<=1
        if matches:
            i,b=matches[0];tables=[x for x in existing[i+1:] if x['type']=='table'];ids=dict(heading=b['id'],local_table=tables[0]['id'],remote_table=tables[1]['id'])
        else:
            response=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=[block('heading_2',TITLE),block('paragraph',description),tblock(table),
                block('paragraph',remote_note),tblock(other),block('paragraph','Artifacts: '+str(out)+'; all per-case GA values, source summary/per-case/manifest hashes, independent spotchecks, remote source block IDs and exclusions are retained. No prediction, normalization or training was changed. All earlier result sections remain intact.')]))
            ids=dict(heading=response['results'][0]['id'],local_table=response['results'][2]['id'],remote_table=response['results'][4]['id'])
        save(state,ids)
    for key,expected in [('local_table',table),('remote_table',other)]:
        assert [[plain(c) for c in b['table_row']['cells']] for b in children(ids[key])]==expected
    url='https://www.notion.so/'+PAGE.replace('-','')+'#'+ids['heading'].replace('-','')
    save(out/'NOTION_VERIFIED.json',dict(url=url,local_rows=len(results),remote_reported_rows=len(remote),summary_sha256=sha(out/'summary.json'),time=time.time()))
    save(out/'COMPLETE.json',dict(status='complete_verified',full_test_only=True,random20_included=False,local_rows=len(results),
        remote_reported_rows=len(remote),remote_independently_recomputed=False,notion_verified=True,summary_sha256=sha(out/'summary.json')))
    print('PUBLISHED',url,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'ga_all_fulltest_20261116');a=p.parse_args();guard();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(a.out/'GEOMETRY_SELFTEST.json',selftest())
        results,skipped=local_results(a.out);remote,remote_skipped=remote_results(a.out)
        with (BASE/'da3_c2m_fulltest_20261116/notion_report.lock').open('a') as remote_lock:
            fcntl.flock(remote_lock,fcntl.LOCK_EX);publish(a.out,results,remote,skipped+remote_skipped)


if __name__=='__main__':main()
