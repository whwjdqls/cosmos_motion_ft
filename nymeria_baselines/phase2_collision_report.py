"""Aggregate the complete paired Phase2 collision sweep and publish its report."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

LABELS = ('GT', 'text_only', 'image_only', 'image_text')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize_collisions(out):
    out = Path(out)
    cfg = json.loads((out/'configuration.json').read_text())
    cases = json.loads((out/'cases.json').read_text())
    quality = json.loads((out/'summary.json').read_text())
    assert quality['cases'] == len(cases) == 11892
    groups = sorted({c['uuid'] for c in cases})
    group_index = {key:i for i,key in enumerate(groups)}
    counts = np.zeros(len(groups), dtype=np.int64)
    flags = np.zeros((4,len(groups)),dtype=np.int64)
    frame_flags = np.zeros((4,len(cases),97),dtype=bool)
    depths = {label:[] for label in LABELS}
    measures = {label:[] for label in LABELS}
    per_case = []
    physical_gt = {}
    source_qc_cases = 0; source_qc_frames = 0
    for index, row in enumerate(cases):
        path = out/'collision_cases'/row['case_id']
        result = json.loads(path.with_suffix('.json').read_text())
        assert result['case_id'] == row['case_id']
        qc_count = result['coverage'].get('gt_head_camera_outlier_frames', 0)
        source_qc_cases += int(qc_count > 0); source_qc_frames += qc_count
        group = group_index[row['uuid']]; counts[group] += 97
        with np.load(path.with_suffix('.npz')) as z:
            for vi,label in enumerate(LABELS):
                metrics = result['metrics'][label]
                f = z[label+'__frame_hits']
                assert f.shape == (97,) and int(f.sum()) == metrics['collision_frames']
                assert len(metrics['active_capsule_names']) == 16 and len(metrics['excluded_capsule_names']) == 6
                frame_flags[vi,index] = f; flags[vi,group] += int(f.sum())
                depths[label].append(z[label+'__depths'].copy())
                measures[label].append(metrics)
                per_case.append({'case_id':row['case_id'], 'window_id':row['window_id'],
                    'uuid':row['uuid'], 'start_frame':row['start_frame'], 'condition':label,
                    'collision_frame_percent':metrics['collision_frame_percent'],
                    'collision_frames':metrics['collision_frames'],
                    'body_part_frame_percent':metrics['colliding_body_part_frame_percent'],
                    'mean_capsule_intrusion_cm':metrics['intrusion_depth_mean_cm'],
                    'p95_capsule_intrusion_cm':metrics['intrusion_depth_p95_cm'],
                    'common_metric_voxels':result['coverage']['common_metric_voxels']})
                per_case[-1]['GT_source_head_camera_outlier_frames'] = qc_count
            # Duplicate-caption windows cannot change the GT scene measurement.
            gt = z['GT__frame_hits'].copy()
            if row['window_id'] in physical_gt:
                assert np.array_equal(physical_gt[row['window_id']],gt)
            else: physical_gt[row['window_id']] = gt
    assert len(physical_gt) == cfg['physical_windows'] == 11657
    draws = np.random.default_rng(20261030).integers(0,len(groups),size=(5000,len(groups)))
    bootstrap = flags[:,draws].sum(axis=-1)/counts[draws].sum(axis=-1)[None]*100
    results = {}
    for vi,label in enumerate(LABELS):
        observed = measures[label]; packed = np.concatenate(depths[label])*100
        results[label] = {'collision_frame_percent':float(frame_flags[vi].mean()*100),
            'collision_frames':int(frame_flags[vi].sum()), 'total_frames':len(cases)*97,
            'recording_bootstrap95_percent':np.percentile(bootstrap[vi],[2.5,97.5]).tolist(),
            'affected_case_percent':float(frame_flags[vi].any(axis=1).mean()*100),
            'body_part_frame_percent':float(np.mean([m['colliding_body_part_frame_percent'] for m in observed])),
            'pooled_intrusion_mean_cm':float(packed.mean()) if len(packed) else 0.,
            'pooled_intrusion_p95_cm':float(np.percentile(packed,95)) if len(packed) else 0.,
            'sensitivity':[{**s,'collision_frame_percent':float(np.mean([m['sensitivity'][i]['collision_frame_percent'] for m in observed]))}
                           for i,s in enumerate(observed[0]['sensitivity'])],
            'body_parts':{key:float(np.mean([m['body_parts'][key] for m in observed])) for key in observed[0]['body_parts']}}
    paired = {}
    for reference in ['text_only','image_only']:
        difference = bootstrap[3]-bootstrap[LABELS.index(reference)]
        paired['image_text_minus_'+reference] = {
            'difference_percentage_points':results['image_text']['collision_frame_percent']-results[reference]['collision_frame_percent'],
            'recording_bootstrap95_percentage_points':np.percentile(difference,[2.5,97.5]).tolist()}
    collision = {'status':'complete','cases':len(cases),'physical_windows':len(physical_gt),
        'recordings':len(groups),'frames_per_condition':len(cases)*97,'skipped':0,'results':results,
        'paired_comparisons':paired,'bootstrap':{'unit':'recording','resamples':5000,'seed':20261030},
        'physical_collision_validated':False,'configuration':cfg}
    collision['source_quality_diagnostic'] = {'cases_with_head_camera_outliers':source_qc_cases,
        'case_frame_observations_flagged':source_qc_frames,'anatomical_separation_bounds_m':[.05,.35],
        'cases_removed':0,'poses_modified':False,'not_a_physical_collision_label':True}
    save(out/'collision_summary.json',collision)
    with (out/'collision_per_case.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    np.savez_compressed(out/'collision_per_frame.npz',conditions=np.array(LABELS),
        case_id=np.array([r['case_id'] for r in cases]),frame_flags=frame_flags)
    table = [['Condition','Collision frames %','Body-part/frames %','Mean/P95 intrusion cm','R@3 %','FID']]
    for label in LABELS:
        c=results[label];q=quality['results'][label]
        table.append([label,f"{c['collision_frame_percent']:.3f}",f"{c['body_part_frame_percent']:.3f}",
            f"{c['pooled_intrusion_mean_cm']:.3f} / {c['pooled_intrusion_p95_cm']:.3f}",
            f"{q['retrieval']['R32_R03_mean']:.3f}",f"{q['FID']:.6f}"])
    report = ['# Phase2 full-test modalities and scene collision','',
        '11892 caption/window cases;11657physical windows;71recordings. GT and three actual input modalities. '
        'Same200k checkpoint, paired stable per-case noise, T97/20fps, UniPC35/shift3/CFG2 against native empty-text/no-image.',
        '', '| '+' | '.join(table[0])+' |','|'+'---|'*len(table[0]),
        *['| '+' | '.join(row)+' |' for row in table[1:]], '',
        'Collision proxy: approved SDK legacy cloud, no point-count sample;2cm voxels and at least3connected '
        'voxels more than2cm inside the capsule union. Both thighs, shins and feet are excluded;16capsules remain including hips. '
        'This measures capsule/cloud overlap, not signed physical scene penetration.', '',
        'Scene placement uses the fixed source-window yaw/origin and calibrated floor. GT is canonicalized for the evaluation frame; '
        'predictions retain all sampled features with no fitting, recentering or grounding. The shared scene crop covers GT AND all '
        'three generated trajectories. The previous71-window GT rate is not reused for this different full cohort.', '',
        'Recording-level paired95% intervals:', json.dumps(paired,indent=2), '',
        'Three-modality inputs: text only has no image; image only receives an empty caption; image+text receives both. '
        'Image-only retrieval is against withheld evaluation captions. Native T2M/TI2M prompt layouts differ. '
        'Collision intervals resample recordings, not generation seeds.', '', 'Artifacts: '+str(out)]
    report += ['', 'Source-quality diagnostic (no cases removed or poses modified): ' +
               json.dumps(collision['source_quality_diagnostic'])]
    (out/'COLLISION_RESULTS.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    means=np.array([results[k]['collision_frame_percent'] for k in LABELS])
    intervals=np.array([results[k]['recording_bootstrap95_percent'] for k in LABELS])
    ax.barh(LABELS,means,color=['#3e7cab','#e29b32','#419367','#9764ac'])
    ax.errorbar(means,np.arange(4),xerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),fmt='none',color='black',capsize=3)
    ax.set(xlabel='Collision-proxy frames (%)',title='GT and Phase2 input modalities; thighs/shins/feet excluded')
    fig.savefig(out/'collision_comparison.png',dpi=170);plt.close(fig)
    with zipfile.ZipFile(out/'collision_quantitative.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['collision_summary.json','collision_per_case.csv','collision_per_frame.npz','COLLISION_RESULTS.md',
                     'configuration.json','protocol_audit.json','first_batch_generation_parity.json','summary.json',
                     'executed_source.py','scene_engine_source.py','collision_comparison.png']:
            z.write(out/name,name)
        for name in ['source_qc_fix.json','head_camera_gate_diagnostic.json','source_tracking_jump_check.json',
                     'scene_engine_source_before_qc_fix.py']:
            if (out/name).exists():z.write(out/name,name)
    scope = report[2] + ' Source-quality audit: ' + str(source_qc_cases) + \
        ' cases contain head/camera separation outliers; all cases and poses are retained. ' + \
        'Independent full-source/window world reconstruction is checked for flagged windows.'
    publish(out,table,scope,paired)


def publish(out,table,scope,paired,heading='Completed full-test modality scene-collision results'):
    helper=Path('/scratch2/whwjdqls99/cosmos/operations/nymeria_test71_pointclouds_20261029/download.py')
    spec=importlib.util.spec_from_file_location('phase2_scene_notion',helper)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    page='407d2a44-f44b-81a5-ad53-dbb685cdbd9e'
    rich=lambda s:[{'type':'text','text':{'content':s}}]
    blocks=[{'object':'block','type':'heading_2','heading_2':{'rich_text':rich(heading)}},
        {'object':'block','type':'paragraph','paragraph':{'rich_text':rich(scope)}},
        {'object':'block','type':'table','table':{'table_width':len(table[0]),'has_column_header':True,'has_row_header':False,
          'children':[{'object':'block','type':'table_row','table_row':{'cells':[rich(s) for s in row]}} for row in table]}},
        {'object':'block','type':'paragraph','paragraph':{'rich_text':rich(
            'Approved legacy-filter point cloud;2cm voxels, >=3connected voxels >2cm inside16body capsules. Both thighs, shins '
            'and feet are excluded; hips retained. Fixed window coordinate transform; no prediction fitting, recentering or grounding. '
            'Common scene support covers all conditions. This is a capsule/cloud overlap proxy. Per-recording paired95% intervals: '+json.dumps(paired))}},
        {'object':'block','type':'paragraph','paragraph':{'rich_text':rich('All11892cases completed, zero skipped; GT is measured on the same full cohort. Artifacts: '+str(out))}}]
    ack=api.notion_request('PATCH','blocks/'+page+'/children',{'children':blocks})
    token=Path('/home/whwjdqls99/.config/notion/token').read_text().strip()
    def get(endpoint):
        request=urllib.request.Request('https://api.notion.com/v1/'+endpoint,headers={'Authorization':'Bearer '+token,'Notion-Version':'2026-03-11'})
        with urllib.request.urlopen(request,timeout=90) as response:return json.load(response)
    remote=get('blocks/'+ack['results'][2]['id']+'/children')
    actual=[[''.join(x.get('plain_text',x.get('text',{}).get('content','')) for x in cell) for cell in row['table_row']['cells']] for row in remote['results']]
    assert actual==table
    receipt={'state':'publishing','page_id':page,'table_readback_verified':True,'uploads':[]}
    for name,kind,mime in [('collision_comparison.png','image','image/png'),('collision_quantitative.zip','file','application/zip')]:
        path=out/name;data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        assert len(data)<20*1024*1024
        upload=api.notion_request('POST','file_uploads',{'mode':'single_part','filename':name,'content_type':mime})
        boundary='Phase2SceneMetrics'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
        request=urllib.request.Request('https://api.notion.com/v1/file_uploads/'+upload['id']+'/send',data=body,method='POST',
            headers={'Authorization':'Bearer '+token,'Notion-Version':'2026-03-11','Content-Type':'multipart/form-data; boundary='+boundary})
        with urllib.request.urlopen(request,timeout=120) as response:assert json.load(response)['status']=='uploaded'
        item=api.notion_request('PATCH','blocks/'+page+'/children',{'children':[{'object':'block','type':kind,kind:{
            'type':'file_upload','file_upload':{'id':upload['id']},'caption':rich('Phase2 scene collision | '+name)}}]})['results'][0]
        with urllib.request.urlopen(item[kind]['file']['url'],timeout=120) as response:assert hashlib.sha256(response.read()).hexdigest()==digest
        receipt['uploads'].append({'path':str(path),'block_id':item['id'],'remote_sha256_verified':True,'sha256':digest})
        save(out/'publication.json',receipt)
    receipt['state']='complete_verified';save(out/'publication.json',receipt)
    save(out/'COMPLETE.json',{'cases':11892,'recordings':71,'skipped':0,'state':'complete_verified',
        'report':'https://app.notion.com/p/'+page.replace('-',''),'physical_collision_validated':False})
