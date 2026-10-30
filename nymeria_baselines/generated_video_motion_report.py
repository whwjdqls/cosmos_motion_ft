"""Matched two-stage motion evaluation, selected qualitative comparisons and Notion."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from nymeria_baselines.phase2_collision_report import publish, save


def summarize_pipeline(out):
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text())
    direct=Path(cfg['paired_baseline_operation'])
    if not (direct/'COMPLETE.json').exists():raise RuntimeError('Matched direct-modality sweep is not complete yet')
    cases=json.loads((out/'cases.json').read_text());quality=json.loads((out/'summary.json').read_text())
    original=json.loads((direct/'collision_summary.json').read_text())
    original_quality=json.loads((direct/'summary.json').read_text())
    with np.load(direct/'collision_per_frame.npz') as z:
        assert z['case_id'].tolist()==[r['case_id'] for r in cases]
        original_gt=z['frame_flags'][0].copy()
        original_both=z['frame_flags'][3].copy()
    flags=[];depths=[];measures=[];rows=[]
    for i,row in enumerate(cases):
        path=out/'collision_cases'/row['case_id']
        m=json.loads(path.with_suffix('.json').read_text())['metrics']['generated_video_motion']
        with np.load(path.with_suffix('.npz')) as z:
            assert np.array_equal(z['GT__frame_hits'],original_gt[i]),row['case_id']
            flags.append(z['generated_video_motion__frame_hits'].copy())
            depths.append(z['generated_video_motion__depths'].copy())
        measures.append(m)
        rows.append({'case_id':row['case_id'],'uuid':row['uuid'],'start_frame':row['start_frame'],
            'collision_frame_percent':m['collision_frame_percent'],'body_part_frame_percent':m['colliding_body_part_frame_percent'],
            'intrusion_mean_cm':m['intrusion_depth_mean_cm'],'intrusion_p95_cm':m['intrusion_depth_p95_cm']})
    flags=np.stack(flags);packed=np.concatenate(depths)*100
    groups=sorted({r['uuid'] for r in cases});which=np.array([groups.index(r['uuid']) for r in cases])
    numerator=np.array([flags[which==i].sum() for i in range(len(groups))])
    denominator=np.array([(which==i).sum()*97 for i in range(len(groups))])
    baseline_n=np.array([original_both[which==i].sum() for i in range(len(groups))])
    draw=np.random.default_rng(20261030).integers(0,len(groups),size=(5000,len(groups)))
    b=numerator[draw].sum(1)/denominator[draw].sum(1)*100
    differences=(numerator[draw]-baseline_n[draw]).sum(1)/denominator[draw].sum(1)*100
    result={'collision_frame_percent':float(flags.mean()*100),'collision_frames':int(flags.sum()),
        'total_frames':flags.size,'recording_bootstrap95_percent':np.percentile(b,[2.5,97.5]).tolist(),
        'body_part_frame_percent':float(np.mean([m['colliding_body_part_frame_percent'] for m in measures])),
        'pooled_intrusion_mean_cm':float(packed.mean()) if len(packed) else 0.,
        'pooled_intrusion_p95_cm':float(np.percentile(packed,95)) if len(packed) else 0.,
        'affected_case_percent':float(flags.any(1).mean()*100),
        'sensitivity':[{**s,'collision_frame_percent':float(np.mean([m['sensitivity'][j]['collision_frame_percent'] for m in measures]))}
                       for j,s in enumerate(measures[0]['sensitivity'])]}
    combined={**original['results'],'generated_video_motion':result}
    all_quality={**original_quality['results'],'generated_video_motion':quality['results']['generated_video_motion']}
    comparison={'pipeline_minus_direct_image_text':{'percentage_points':result['collision_frame_percent']-combined['image_text']['collision_frame_percent'],
        'recording_bootstrap95_percentage_points':np.percentile(differences,[2.5,97.5]).tolist()}}
    summary={'status':'complete','cases':len(cases),'physical_windows':11657,'recordings':71,'skipped':0,
        'results':combined,'quality':all_quality,'paired_comparisons':comparison,'configuration':cfg,
        'GT_collision_flags_identical_to_direct_full_test':True,'physical_collision_validated':False}
    summary['source_quality_diagnostic']=original.get('source_quality_diagnostic',{})
    save(out/'collision_summary.json',summary)
    np.savez_compressed(out/'collision_per_frame.npz',case_id=np.array([r['case_id'] for r in cases]),
                        frame_flags=flags,GT_frame_flags=original_gt)
    with (out/'collision_per_case.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels=['GT','text_only','image_only','image_text','generated_video_motion']
    table=[['Condition','Collision frames %','R@3 %','FID','Foot skate','Foot depth cm','Feet >5cm %']]
    for label in labels:
        q=all_quality[label];p=q['physics_20fps'];c=combined[label]
        table.append([label,f"{c['collision_frame_percent']:.3f}",f"{q['retrieval']['R32_R03_mean']:.3f}",f"{q['FID']:.6f}",
            f"{p['foot_skate_from_height_mean']:.6f}",f"{p['foot_depth_mean_cm_mean']:.3f}",
            f"{p['all_feet_above5cm_frame_fraction_mean']*100:.3f}"])
    scope=('Matched11892cases/11657physical windows/71recordings. New condition: first image+caption generates97video frames '
        'using the same Phase1 EMA200k native video specialist, UniPC30/shift10/guidance1, with bridges disabled and no motion/camera input. '
        'Decode to lossless RGB, restore the observed first frame and Wan re-encode. Original Phase3 baseline200k reconstructs motion '
        'with UniPC35/shift3 and text CFG2; the generated video is identical in both CFG branches, with no video CFG. '
        'Same neutral shapes and paired initial motion noise as the direct-modality sweep. No GT motion or future GT video is provided. '
        'The reconstruction model is Phase3 rather than the direct Phase2 specialist; differences cannot be attributed solely to generated video. '
        'Stored source-quality outliers are retained and reported in the quantitative archive.')
    report=['# Generated video -> motion: matched full-test results','',scope,'',
        '| '+' | '.join(table[0])+' |','|'+'---|'*len(table[0]),*['| '+' | '.join(r)+' |' for r in table[1:]],'',
        'Collision: SDK legacy-filter cloud,2cm voxels, >=3connected voxels >2cm inside16capsules; thighs/shins/feet excluded, hips kept. '
        'Fixed evaluation-frame placement; no prediction fitting/recentering/grounding. GT collision flags reproduce the direct full-test GT exactly.',
        '',json.dumps(comparison,indent=2),'','Artifact root: '+str(out)]
    (out/'COLLISION_RESULTS.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.barh(labels,[combined[k]['collision_frame_percent'] for k in labels],color=['#397aaa','#df922b','#389266','#9163a5','#bd5849'])
    ax.set(xlabel='Collision-proxy frames (%)',title='Direct modalities versus generated-video motion pipeline; GT included')
    fig.savefig(out/'collision_comparison.png',dpi=170);plt.close(fig)
    with zipfile.ZipFile(out/'collision_quantitative.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['collision_summary.json','collision_per_case.csv','collision_per_frame.npz','COLLISION_RESULTS.md',
                     'configuration.json','executed_source.py','summary.json','collision_comparison.png',
                     'video_motion_independence.json','text_cfg_acceptance.json']:
            z.write(out/name,name)
        for name in ['handoff_output_audit.json','pro_first_case_acceptance.json',
                     'source_qc_case_acceptance.json','published_video_acceptance.json',
                     'runtime_throughput_snapshot.json']:
            if (out/name).exists():z.write(out/name,name)
        for path in sorted((out/'allocations').glob('*.gpu.json')):
            z.write(path,'execution_resources/'+path.name)
    publish(out,table,scope,comparison,heading='Completed generated-video -> motion: text CFG only, matched full test')


def render_example(out,row,index):
    from PIL import Image,ImageDraw,ImageFont
    from motion_expert_joint_attention.camera_head_recanonicalization import decode_uniego
    from motion_expert_joint_attention.uniego_layout import canonicalize_frame0,ground_features
    from render_motion import render_single,head_rotation_from_features,write_mp4
    from nymeria_camera_dataset import decode_window_pyav
    import av
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text())
    direct=Path(cfg['paired_baseline_operation']);folder=out/'case_outputs'/row['case_id']
    gallery=out/'qualitative';gallery.mkdir(exist_ok=True)
    path=gallery/(row['case_id']+'.mp4')
    if path.exists():return
    begin=index//4*4
    direct_sample=direct/'samples'/f'batch_{begin:07d}.npz'
    if not direct_sample.exists():return
    with np.load(direct_sample) as z:
        offset=index-begin
        assert z['case_id'][offset]==row['case_id']
        pred=z['raw_features'][:,offset].copy()
    with np.load(folder/'motion.npz') as z:pipeline=z['raw_features'].copy()
    features=np.load(Path(cfg['reference_request']['data_dir'])/row['features_file']).astype(np.float32)
    gt=canonicalize_frame0(ground_features(features,row['floor_offset']))
    with av.open(str(folder/'generated_video.mp4')) as container:
        video=np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    gt_video=decode_window_pyav(row['image_path'],row['image_start'],97,20.)
    observed=np.load(row['image_file']).transpose(1,2,0)
    observed_video=np.broadcast_to(observed,(97,)+observed.shape)
    shape=(384,384)
    panels=[]
    for label,raw,color in [('GT motion',gt,'tab:blue'),('Text only',pred[1],'tab:orange'),
            ('Image only',pred[0],'tab:green'),('Image + text',pred[2],'tab:purple'),
            ('Generated-video motion',pipeline,'tab:red')]:
        frames=render_single(decode_uniego(raw).world_positions,caption=label,color=color,width=384,height=384,
            frame_stride=1,max_frames=97,head_rotation0=head_rotation_from_features(raw))
        assert len(frames)==97
        panels.append(frames)
    outputs=[]
    for frame in range(97):
        canvas=Image.new('RGB',(1152,1216),'#f9f9f6');draw=ImageDraw.Draw(canvas)
        for xy,clip,label in [((0,64),gt_video,'GT video (reference)'),((384,64),video,'Generated video'),
                              ((768,64),observed_video,'Observed first image')]:
            image=Image.fromarray(clip[frame]).resize(shape,Image.Resampling.LANCZOS);canvas.paste(image,xy)
            border='#18754b' if label=='Observed first image' or (label=='Generated video' and frame==0) else '#b57423' if label=='Generated video' else '#2b669b'
            draw.rectangle((xy[0]+2,xy[1]+2,xy[0]+381,xy[1]+381),outline=border,width=5)
            draw.rectangle((xy[0],xy[1],xy[0]+383,xy[1]+23),fill=border);draw.text((xy[0]+7,xy[1]+6),label,fill='white')
        for i,clip in enumerate(panels):
            x=(i%3)*384;y=448+(i//3)*384;canvas.paste(Image.fromarray(clip[frame]),(x,y))
        import textwrap
        draw.multiline_text((10,7),'\n'.join(textwrap.wrap(row['text'],150))[:450],fill='#222222',spacing=3)
        draw.multiline_text((790,865),'Video generation: no CFG\nMotion: text CFG2 only\nSame video in both branches\nNo GT motion / future video',fill='#222222',spacing=8)
        outputs.append(np.asarray(canvas))
    write_mp4(np.stack(outputs),str(path),fps=20)
    Image.fromarray(outputs[0]).save(path.with_suffix('.png'))
    publish_video(out,path,'Generated-video motion pipeline | '+row['case_id']+' | only first image observed (green); generated video amber')


def publish_video(out,path,caption):
    helper=Path('/scratch2/whwjdqls99/cosmos/operations/nymeria_test71_pointclouds_20261029/download.py')
    spec=importlib.util.spec_from_file_location('pipeline_video_notion',helper);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    token=Path('/home/whwjdqls99/.config/notion/token').read_text().strip();data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    upload=api.notion_request('POST','file_uploads',{'mode':'single_part','filename':path.name,'content_type':'video/mp4'})
    boundary='GeneratedVideoMotionExample'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: video/mp4\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
    request=urllib.request.Request('https://api.notion.com/v1/file_uploads/'+upload['id']+'/send',data=body,method='POST',
        headers={'Authorization':'Bearer '+token,'Notion-Version':'2026-03-11','Content-Type':'multipart/form-data; boundary='+boundary})
    with urllib.request.urlopen(request,timeout=120) as response:assert json.load(response)['status']=='uploaded'
    block=api.notion_request('PATCH','blocks/407d2a44-f44b-81a5-ad53-dbb685cdbd9e/children',{'children':[
        {'object':'block','type':'video','video':{'type':'file_upload','file_upload':{'id':upload['id']},
         'caption':[{'type':'text','text':{'content':caption}}]}}]})['results'][0]
    with urllib.request.urlopen(block['video']['file']['url'],timeout=120) as response:assert hashlib.sha256(response.read()).hexdigest()==digest
    save(path.with_suffix('.publication.json'),{'block_id':block['id'],'sha256':digest,'remote_sha256_verified':True})
