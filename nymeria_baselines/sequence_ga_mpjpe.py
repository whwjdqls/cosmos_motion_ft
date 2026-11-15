"""One proper similarity transform per complete T97 motion sequence.

Recompute measured-GT-camera baseline GA-MPJPE from saved body23 predictions.
This names explicitly the whole-clip alignment historically called PA-MPJPE.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import fcntl
import runpy
import time
import urllib.parse

import numpy as np

from nymeria_baselines.fulltest_v2m import read,save,sha,npz,guard
from nymeria_baselines.da3_c2m_report import LABELS,PAGE

TITLE='GT camera-to-motion | sequence-global GA-MPJPE | full11911 | 2026-11-16'


def align_sequence(prediction, target):
    p,g=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if p.shape!=g.shape or p.ndim!=3 or p.shape[-1]!=3 or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Matched finite [T,J,3] positions required')
    # A single point cloud spans all frames AND all joints; no per-frame fit.
    flat,truth=p.reshape(-1,3),g.reshape(-1,3)
    pm,gm=flat.mean(0),truth.mean(0)
    centered,centered_gt=flat-pm,truth-gm
    variance=float(np.square(centered).sum()/len(flat))
    if variance<1e-20:raise ValueError('Degenerate constant prediction')
    u,singular,vt=np.linalg.svd(centered_gt.T@centered/len(flat))
    signs=np.ones(3)
    if np.linalg.det(u@vt)<0:signs[-1]=-1
    rotation=(u*signs)@vt
    scale=float((singular*signs).sum()/variance)
    translation=gm-scale*(rotation@pm)
    assert np.linalg.det(rotation)>0.999999999 and scale>=0
    aligned=(scale*(flat@rotation.T)+translation).reshape(p.shape)
    return float(np.linalg.norm(aligned-g,axis=-1).mean()),scale,rotation,translation


def selftest():
    rng=np.random.default_rng(31);p=rng.normal(size=(9,7,3))
    angle=.6;c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    target=2.3*(p@rotation.T)+[3,-1,.8]
    error,scale,_,_=align_sequence(p,target);assert error<1e-12 and abs(scale-2.3)<1e-12
    moving=target.copy();moving[:,:,0]+=np.arange(len(p))[:,None]*.4
    global_error,*_=align_sequence(p,moving)
    per_frame=np.mean([align_sequence(a[None],b[None])[0] for a,b in zip(p,moving)])
    assert global_error>.1 and per_frame<1e-12
    mirrored=p.copy();mirrored[:,:,0]*=-1
    reflected_error,_,r,_=align_sequence(p,mirrored)
    assert reflected_error>.1 and np.linalg.det(r)>.99999999
    return dict(passed=True,global_similarity_error_m=error,time_varying_translation_global_error_m=global_error,
        per_frame_fit_error_m=per_frame,reflection_error_m=reflected_error,reflection_not_allowed=True)


def compute(root,out):
    from motion_expert_joint_attention.eval_motion_recon import joint_metrics
    out.mkdir(parents=True,exist_ok=True);save(out/'GEOMETRY_SELFTEST.json',selftest())
    cfg=read(root/'configuration.json');rows=read(root/'cases.json');ids=[r['id'] for r in rows]
    cache=root.parent/'motion_metrics_20261105';cache_info=read(cache/'gt_cache.json')
    assert cache_info['cases_sha256']==sha(root/'cases.json') and cache_info['sha256']==sha(cache/'gt_body23.npy')
    target=np.load(cache/'gt_body23.npy',mmap_mode='r');assert target.shape==(11911,97,23,3)
    configuration=dict(cases=11911,physical_windows=len({(r['uuid'],r['start']) for r in rows}),recordings=71,
        input='MEASURED GT camera trajectories + original supplied shape; not DA3; original full test cohort',
        metric='Per T97 clip fit ONE proper rotation, translation and uniform scale jointly over97*23=2231 points, minimizing squared distance; then average Euclidean joint error. No per-frame alignment; reflection forbidden.',
        relationship='Same mathematical definition as historical whole-clip pa_mpjpe_m; independently recomputed in float64 and compared per case.',
        source_configuration_sha256=sha(root/'configuration.json'),cases_sha256=sha(root/'cases.json'),
        gt_cache_sha256=cache_info['sha256'],implementation_sha256=sha(__file__),units='meters in machine-readable per-case records, centimeters in tables')
    save(out/'configuration.json',configuration)
    models=[]
    for name,label in [('GT','GT'),*LABELS]:
        if label=='GT':source=root;source_label='GT'
        else:
            spec=next(s for s in cfg['models'] if s['label']==label)
            source=Path(spec['measured_reference']);source_label=spec['measured_label']
        summary=read(source/'scores'/source_label/'summary.json')
        per_case=read(source/'scores'/source_label/'per_case.json')
        assert len(per_case)==11911 and [r['id'] for r in per_case]==ids
        assert summary['cases']==11911 and sha(source/'cases.json')==sha(root/'cases.json')
        records=[];scales=[];rotations=[];translations=[];max_legacy_difference=0.;max_cached_difference=0.
        for i,row in enumerate(rows):
            if label=='GT':pred=np.array(target[i],copy=True)
            else:
                with np.load(source/'models'/source_label/'predictions'/(row['id']+'.npz')) as z:pred=z['joints'].copy()
            gt=target[i]
            ga,scale,r,t=align_sequence(pred,gt)
            legacy=joint_metrics(pred,gt)['pa_mpjpe_m']
            cached=per_case[i]['metrics']['pa_mpjpe_m']
            max_legacy_difference=max(max_legacy_difference,abs(ga-legacy))
            max_cached_difference=max(max_cached_difference,abs(legacy-cached))
            assert abs(ga-legacy)<1e-4 and abs(legacy-cached)<1e-4,(row['id'],ga,legacy,cached)
            records.append(dict(id=row['id'],uuid=row['uuid'],start=row['start'],ga_mpjpe_m=ga,
                old_sequence_pa_mpjpe_m=cached,raw_mpjpe_m=per_case[i]['metrics']['mpjpe_m']))
            scales.append(scale);rotations.append(r);translations.append(t)
            if i%2000==0:print(name,i,'/',len(rows),flush=True)
        by_record=defaultdict(list)
        for r in records:by_record[r['uuid']].append(r['ga_mpjpe_m'])
        values=np.array([r['ga_mpjpe_m'] for r in records])
        result=dict(model=name,source=str(source),source_label=source_label,cases=len(rows),recordings=len(by_record),
            ga_mpjpe_m=float(values.mean()),recording_balanced_ga_mpjpe_m=float(np.mean([np.mean(v) for v in by_record.values()])),
            median_case_ga_mpjpe_m=float(np.median(values)),p95_case_ga_mpjpe_m=float(np.percentile(values,95)),
            source_raw_mpjpe_m=summary['physical']['mpjpe_m'],old_whole_clip_pa_mpjpe_m=summary['physical']['pa_mpjpe_m'],
            max_recompute_vs_legacy_case_difference_m=max_legacy_difference,max_legacy_vs_cached_case_difference_m=max_cached_difference,
            source_summary_sha256=sha(source/'scores'/source_label/'summary.json'))
        if label=='GT':assert result['ga_mpjpe_m']<1e-12
        dest=out/name.replace(' ','_');dest.mkdir(exist_ok=True)
        save(dest/'per_case.json',records)
        npz(dest/'single_clip_transforms.npz',case_id=np.array(ids),scale=np.array(scales),rotation=np.stack(rotations),translation=np.stack(translations))
        save(dest/'summary.json',result);models.append(result)
        print('RESULT',result,flush=True)
    save(out/'summary.json',dict(configuration=configuration,models=models))
    save(out/'SCORING_COMPLETE.json',dict(cases_per_model=11911,models=4,summary_sha256=sha(out/'summary.json')))


def publish(out):
    summary=read(out/'summary.json');assert read(out/'SCORING_COMPLETE.json')['summary_sha256']==sha(out/'summary.json')
    rows=[['Model / measured GT camera','Cases','Raw MPJPE cm','GA-MPJPE cm','Prior whole-clip PA cm']]
    for d in summary['models']:
        rows.append([d['model'],str(d['cases']),f"{100*d['source_raw_mpjpe_m']:.4f}",f"{100*d['ga_mpjpe_m']:.4f}",f"{100*d['old_whole_clip_pa_mpjpe_m']:.4f}"])
    with (out/'comparison.csv').open('w') as f:csv.writer(f).writerows(rows)
    text=('All11911 T97/20FPS annotation cases,11677 physical windows,71 recordings. Original final EgoEgo, EgoAllo and UniEgoMotion checkpoints, supplied shapes and MEASURED GT CAMERA inference outputs. '
        'This uses the full original cohort; the19-case DA3 exclusion is not applied here. '
        'GA-MPJPE fits one proper rotation + translation + uniform scale across the ENTIRE97-frame body23 sequence (2231 points), then averages Euclidean joint error. No per-frame fit and no reflection. '
        'The historical metric labeled PA-MPJPE already used this same whole-clip Umeyama definition. Independent float64 recomputation, saved per-clip transforms and per-case comparison verify the equivalence; small FP32/FP64 roundoff is expected. '
        'GT self-error and synthetic tests of a global transform, a time-varying translation, and reflection passed. No inference/training/normalization/predictions were changed.')
    (out/'RESULTS.md').write_text('# '+TITLE+'\n\n'+text+'\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [rows[0],['---']*len(rows[0]),*rows[1:]])+'\n\nArtifacts: `'+str(out)+'`\n')
    h=runpy.run_path('/scratch2/whwjdqls99/cosmos/operations/phase1_notext_fourway_55k_20261019/upload_notion.py')
    api=h['Notion'](h['read_token']());rt=h['rich_text'];plain=h['plain_text']
    def children(uid):
        result=[];cursor=None
        while True:
            q={'page_size':100}
            if cursor:q['start_cursor']=cursor
            d=api.request('GET','/blocks/'+uid+'/children?'+urllib.parse.urlencode(q));result.extend(b for b in d['results'] if not b.get('archived') and not b.get('in_trash'))
            if not d.get('has_more'):return result
            cursor=d['next_cursor']
    state=out/'notion_publication.json'
    if state.exists():ids=read(state)
    else:
        existing=children(PAGE);matches=[(i,b) for i,b in enumerate(existing) if b['type']=='heading_2' and plain(b['heading_2']['rich_text'])==TITLE]
        assert len(matches)<=1
        if matches:
            i,b=matches[0];table=next(x for x in existing[i+1:] if x['type']=='table');ids=dict(heading=b['id'],table=table['id'])
        else:
            blocks=[dict(object='block',type='heading_2',heading_2=dict(rich_text=rt(TITLE))),dict(object='block',type='paragraph',paragraph=dict(rich_text=rt(text))),
                dict(object='block',type='table',table=dict(table_width=len(rows[0]),has_column_header=True,has_row_header=False,
                    children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(v) for v in row])) for row in rows])),
                dict(object='block',type='paragraph',paragraph=dict(rich_text=rt('Artifacts: '+str(out)+'; per-case GA errors, single scale/rotation/translation per entire clip, independent alignment tests, cached-PA agreement and source hashes are saved.')))]
            d=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=blocks));ids=dict(heading=d['results'][0]['id'],table=d['results'][2]['id']);save(state,ids)
    actual=children(ids['table']);assert len(actual)==len(rows)
    for b,expected in zip(actual,rows):
        if [plain(c) for c in b['table_row']['cells']]!=expected:api.request('PATCH','/blocks/'+b['id'],dict(table_row=dict(cells=[rt(v) for v in expected])))
    assert [[plain(c) for c in b['table_row']['cells']] for b in children(ids['table'])]==rows
    url='https://www.notion.so/'+PAGE.replace('-','')+'#'+ids['heading'].replace('-','')
    save(out/'NOTION_VERIFIED.json',dict(url=url,comparison_sha256=sha(out/'comparison.csv'),time=time.time()))
    save(out/'COMPLETE.json',dict(status='complete_verified',cases_per_model=11911,models=4,notion_verified=True))
    print('PUBLISHED',url,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/scratch2/whwjdqls99/cosmos/operations/da3_c2m_fulltest_20261116'));a=p.parse_args();guard()
    out=a.root/'gt_camera_ga_mpjpe_20261116';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'SCORING_COMPLETE.json').exists():compute(a.root,out)
        with (a.root/'notion_report.lock').open('a') as remote_lock:
            fcntl.flock(remote_lock,fcntl.LOCK_EX);publish(out)


if __name__=='__main__':main()
