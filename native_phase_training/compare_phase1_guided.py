"""Matched two-task Phase1 comparison with camera-guided forward dynamics.

All model/data work requires an owned compute allocation. Training is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


FIELDS = ('model_mode','num_frames','resolution','fps','seed','num_steps','shift',
          'guidance','prompt','vision_path','action_path','condition_frame_indexes_vision')
FORWARD = ('original_no_text', 'original_caption', 'two_task')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); os.replace(tmp,path)


def require_compute():
    assert os.environ.get('SLURM_JOB_ID')
    assert 'login' not in socket.gethostname() and socket.gethostname().split('.')[0] not in ('cs-gpu-01','node37')


def verify_output(path, expected, video):
    meta = read(Path(path)/'sample_outputs.json')
    assert meta['status'] == 'success', path
    for key in FIELDS:
        if key in expected:
            assert meta['args'].get(key) == expected[key], (path,key,meta['args'].get(key),expected[key])
    if video:
        assert (Path(path)/'vision.mp4').is_file()
    return meta


def prepare(args):
    require_compute()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'configuration.json').exists():
        old=read(out/'configuration.json')
        assert old['checkpoint']==str(args.checkpoint)
        return
    reference=read(args.reference/'configuration.json')
    cases=reference['cases'];assert len(cases)==len({c['uuid'] for c in cases})==71
    candidate=args.checkpoint;baseline=Path(reference['baseline_checkpoint'])
    assert candidate.name==baseline.name=='iter_000200000'
    for checkpoint in (candidate,baseline):
        for component in ('model','optim','scheduler','trainer'):
            assert (checkpoint/component/'.metadata').is_file()
    assert sha(baseline/'model/.metadata')==reference['baseline_metadata_sha256']
    repo=Path(__file__).resolve().parents[1]
    old_guidance=read(args.original_guidance/'configuration.json')
    old_impl=args.original_guidance.parent/'phase1_camera_cfg_20261021/camera_cfg_config.py'
    assert sha(old_impl)==sha(repo/'native_phase_training/camera_cfg_inference_config.py')
    sources={
        'original_no_text':args.reference/'models/baseline_no_text/runtime/fd_input.jsonl',
        'two_task':args.reference/'models/candidate/runtime/fd_input.jsonl',
        'original_caption':args.reference.parent/'phase1_notext_fourway_55k_20261019/inputs/original_200k/runtime/fd_input.jsonl',
    }
    config=dict(schema='phase1_twotask200k_guidance2_v1',checkpoint=str(candidate),baseline_checkpoint=str(baseline),
        checkpoint_metadata_sha256=sha(candidate/'model/.metadata'),baseline_metadata_sha256=sha(baseline/'model/.metadata'),
        contracts={key:read(path.parent.parent/'native_phase1_contract.json') for key,path in [('two_task',candidate),('original',baseline)]},
        contract_hashes={key:sha(path.parent.parent/'native_phase1_contract.json') for key,path in [('two_task',candidate),('original',baseline)]},
        reference=str(args.reference),reference_sha256=sha(args.reference/'configuration.json'),cases=cases,
        steps=30,shift=10.,fps=20,num_frames=97,seed=0,prefix=1,ema=True,
        forward_guidance=dict(camera_scale=2.,text_scale=1.,null='camera tokens absent; same first image and caption',
                              diffusion_cache=False,text_kv_reuse=False),
        inverse_guidance=1.,notion_page=args.notion_page,selected_indices=[0,7,14,28,42,63],
        scope='71 fixed T97 windows, one per held-out recording; not all windows',
        metrics='Forward: generated frames1-96, native spatial GT alignment, equal case weights. Inverse: unchanged scale-matched translation / Sim3 ATE convention.',
        caveats=['Camera-absent guidance is an inference intervention; two-task fine-tuning did not train camera dropout.',
                 'Matched primary comparison has empty captions for both checkpoints; original-caption is a separate input-rich reference.',
                 'Fixed six qualitative indices reused before measuring new results; no quality-based selection.',
                 'Both checkpoints are200k; training task mix and caption policy differ.'],
        source_sha256=sha(__file__),camera_cfg_sha256=sha(old_impl),inputs={},reused_forward={},reused_inverse={})
    for variant,source in sources.items():
        rows=lines(source);assert len(rows)==71
        records=[]
        for c,row in zip(cases,rows,strict=True):
            assert row['name']==c['name']+'_forward_dynamics'
            assert row['condition_frame_indexes_vision']==[0]
            assert row['vision_path']==c['first_frame'] and row['action_path']==c['action_path']
            assert bool(row['prompt'].strip()) == (variant=='original_caption')
            record={**row,'guidance':2.,'normalize_cfg':False}
            records.append(record)
        config['inputs'][variant]=records
        reuse={}
        if variant!='two_task':
            condition='captioned' if variant=='original_caption' else 'no_caption'
            for old in old_guidance['records']:
                if (old['model'],old['condition'],old['method'],old['scale'])!=('original200k',condition,'cfg',2.):continue
                i=next(i for i,c in enumerate(cases) if c['name']==old['case'])
                path=Path(old['path']);meta=verify_output(path.parent,records[i],True)
                metric_path=args.original_guidance/'metrics/cases'/f"original200k__{condition}__{old['case']}__cfg__{old['scale']}.json"
                metric=read(metric_path);assert sha(path)==metric['video_sha256']
                assert old['checkpoint']==str(baseline)
                reuse[cases[i]['name']]=dict(directory=str(path.parent),video_sha256=sha(path),metadata_sha256=sha(path.parent/'sample_outputs.json'),
                                           metric_source=str(metric_path),metric_sha256=sha(metric_path))
            assert len(reuse)==10
        config['reused_forward'][variant]=reuse
        pending=[r for c,r in zip(cases,records,strict=True) if c['name'] not in reuse]
        target=out/'inputs'/f'{variant}_fd.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in pending))
    inverse=lines(args.reference/'models/baseline_no_text/runtime/invdyn_input.jsonl')
    assert len(inverse)==71
    for c,row in zip(cases,inverse,strict=True):
        assert row['prompt']=='' and row['guidance']==1. and row['name']==c['name']+'_inverse_dynamics'
        directory=args.reference/'models/baseline_no_text/inference'/row['name']
        verify_output(directory,row,False)
        config['reused_inverse'][c['name']]=dict(directory=str(directory),metadata_sha256=sha(directory/'sample_outputs.json'))
        assert sha(c['gt'])==c['gt_sha256'] and sha(c['action_path'])==c['action_sha256']
    config['inputs']['inverse']=inverse
    (out/'inputs/two_task_inverse.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inverse))
    save(out/'configuration.json',config)
    save(out/'progress.json',dict(stage='prepared',cases=71,reused_forward=20,reused_inverse=71,
        new_forward=193,new_inverse=71,time=time.time(),job=os.environ['SLURM_JOB_ID']))
    print('Prepared: reuse20 guided videos and71 inverse outputs; generate193 guided videos and71 inverse predictions.',flush=True)


def output(cfg,out,variant,index,mode='forward'):
    case=cfg['cases'][index]
    if mode=='inverse':
        expected=cfg['inputs']['inverse'][index]
        if variant=='original':
            reused=cfg['reused_inverse'][case['name']]
            directory=Path(reused['directory']);assert sha(directory/'sample_outputs.json')==reused['metadata_sha256']
        else:directory=out/'models/two_task_inverse/inference'/expected['name']
        return directory,verify_output(directory,expected,False)
    expected=cfg['inputs'][variant][index]
    reused=cfg['reused_forward'][variant].get(case['name'])
    if reused:
        directory=Path(reused['directory'])
        assert sha(directory/'vision.mp4')==reused['video_sha256']
        assert sha(directory/'sample_outputs.json')==reused['metadata_sha256']
    else:directory=out/'models'/variant/'inference'/expected['name']
    return directory,verify_output(directory,expected,True)


def analyze(args):
    require_compute()
    import numpy as np
    import torch
    from native_phase_training.evaluate_inverse_forward import LPIPSAlex,_frame_metrics,_read_video_rgb,_resize_gt_like_native,HORIZONS
    from nymeria_world.eval_inverse_dynamics import eval_seq,gt_abs,rel_steps
    torch.set_num_threads(4)
    out=args.output;cfg=read(out/'configuration.json')
    assert sha(Path(cfg['checkpoint'])/'model/.metadata')==cfg['checkpoint_metadata_sha256']
    assert sha(Path(cfg['baseline_checkpoint'])/'model/.metadata')==cfg['baseline_metadata_sha256']
    prior=Path(cfg['reference']);lpips=LPIPSAlex('cuda:0',16)
    forward=[];inverse=[]
    for i,c in enumerate(cfg['cases']):
        file=out/'metrics/cases'/(c['name']+'.json')
        if file.exists():
            row=read(file);assert row['configuration_sha256']==sha(out/'configuration.json')
            forward+=row['forward'];inverse+=row['inverse'];continue
        assert sha(c['gt'])==c['gt_sha256']
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        assert gt.shape==(97,256,256,3)
        frows=[];irows=[]
        for variant in ('GT',*FORWARD):
            directory=None if variant=='GT' else output(cfg,out,variant,i)[0]
            pred=gt if variant=='GT' else _read_video_rgb(directory/'vision.mp4')
            assert pred.shape==gt.shape
            with torch.inference_mode():values=_frame_metrics(gt[1:],pred[1:],lpips)
            assert all(np.isfinite(v).all() for v in values.values())
            frows.append(dict(case=c['name'],variant=variant,path=c['gt'] if directory is None else str(directory/'vision.mp4'),
                sha256=sha(c['gt'] if directory is None else directory/'vision.mp4'),
                metrics={k:float(v.mean()) for k,v in values.items()},
                horizons={h:{k:float(v[s].mean()) for k,v in values.items()} for h,s in HORIZONS.items()},
                frame_metrics={k:v.tolist() for k,v in values.items()}))
        poses=gt_abs(Path(c['gt']).parent/'gt_camera_cosmos.npz')
        dt,dR=rel_steps(poses);gt9=np.concatenate([dt,dR[:,:,0],dR[:,:,1]],axis=1)
        previous=read(prior/'metrics/cases'/(c['name']+'.json'))
        for variant in ('GT','original','two_task'):
            if variant=='GT': action=gt9;source=None
            else:
                directory,meta=output(cfg,out,variant,i,'inverse')
                action=np.asarray(meta['outputs'][0]['content']['action'],dtype=np.float64)
                source=dict(path=str(directory/'sample_outputs.json'),sha256=sha(directory/'sample_outputs.json'))
            assert action.shape==(96,9) and np.isfinite(action).all()
            value=eval_seq(action,poses);assert all(np.isfinite(x) for x in value.values())
            if variant in ('GT','original'):
                pv='GT' if variant=='GT' else 'baseline_no_text'
                old=next(x for x in previous['inverse'] if x['variant']==pv)['metrics']
                assert max(abs(value[k]-old[k]) for k in value)<1e-10
            irows.append(dict(case=c['name'],variant=variant,metrics=value,source=source))
        prior_gt=next(x for x in previous['forward'] if x['variant']=='GT')['metrics']
        assert max(abs(frows[0]['metrics'][k]-prior_gt[k]) for k in prior_gt)<1e-7
        save(file,dict(configuration_sha256=sha(out/'configuration.json'),forward=frows,inverse=irows))
        forward+=frows;inverse+=irows
        save(out/'progress.json',dict(stage='scoring',completed=i+1,total=71,time=time.time()))
        print('Scored',i+1,71,flush=True)
    def aggregate(rows):
        result={}
        for variant in sorted({r['variant'] for r in rows}):
            selected=[r for r in rows if r['variant']==variant];assert len(selected)==71
            result[variant]={key:dict(mean=float(np.mean([r['metrics'][key] for r in selected])),
                median=float(np.median([r['metrics'][key] for r in selected]))) for key in selected[0]['metrics']}
        return result
    rng=np.random.default_rng(20261029);boots=rng.integers(0,71,size=(5000,71))
    paired={}
    for task,rows,baselines in [('forward',forward,('original_no_text','original_caption')),('inverse',inverse,('original',))]:
        for baseline in baselines:
            delta={}
            for key in rows[0]['metrics']:
                a={r['case']:r['metrics'][key] for r in rows if r['variant']=='two_task'}
                b={r['case']:r['metrics'][key] for r in rows if r['variant']==baseline}
                values=np.array([a[c['name']]-b[c['name']] for c in cfg['cases']])
                delta[key]=dict(mean=float(values.mean()),ci95=np.quantile(values[boots].mean(1),[.025,.975]).tolist())
            paired[task+'_two_task_minus_'+baseline]=delta
    summary=dict(cases=71,skipped=0,forward=aggregate(forward),inverse=aggregate(inverse),paired_differences=paired,
        configuration_sha256=sha(out/'configuration.json'),checkpoint=cfg['checkpoint'],original_checkpoint=cfg['baseline_checkpoint'],
        video_frames_scored=[1,96],GT_and_original_inverse_parity=True,
        inverse_note='Translation error is scale-matched; ATE is Sim(3)-aligned. Neither is raw metric trajectory error.',
        guidance_note='Camera CFG2, native text CFG1. Inverse uses native guidance1. Camera-free null removes camera tokens; no trained camera-dropout assumption.')
    save(out/'metrics/summary.json',summary)
    for name,rows in [('forward',forward),('inverse',inverse)]:
        flat=[dict(case=r['case'],variant=r['variant'],**r['metrics']) for r in rows]
        with (out/'metrics'/f'{name}_per_case.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    save(out/'METRICS_COMPLETE.json',dict(cases=71,forward_models=3,inverse_models=2,GT_measured=True,time=time.time()))
    print(json.dumps(summary,indent=2),flush=True)


def render(args):
    require_compute()
    import numpy as np
    from PIL import Image
    from native_phase_training.evaluate_inverse_forward import _read_video_rgb,_resize_gt_like_native
    from native_phase_training.visualize_checkpoint import _annotated_video_filter,_plot_camera_comparison
    out=args.output;cfg=read(out/'configuration.json');gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    ffmpeg=Path(os.environ['COSMOS_ENV_ROOT'])/'bin/ffmpeg'
    env={**os.environ,'LD_LIBRARY_PATH':str(Path(os.environ['COSMOS_ENV_ROOT'])/'lib')}
    assets=[]
    for i in cfg['selected_indices']:
        c=cfg['cases'][i]
        # Save exactly the spatially aligned GT used by metrics, without lossy
        # re-encoding before the final display composite.
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        raw=gallery/(c['name']+'_aligned_gt.mkv')
        subprocess.run([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','256x256',
            '-framerate','20','-i','pipe:0','-an','-c:v','ffv1',str(raw)],input=gt.tobytes(),env=env,check=True)
        for arm in ['original_no_text']+(['original_caption'] if i in cfg['selected_indices'][:2] else []):
            paths=[raw,output(cfg,out,arm,i)[0]/'vision.mp4',output(cfg,out,'two_task',i)[0]/'vision.mp4']
            labels=['GT reference','Original200k | camera CFG2'+(' | text' if arm=='original_caption' else ''),'Two-task200k | camera CFG2']
            filters=[_annotated_video_filter(input_index=j,output_label=f'v{j}',width=384,height=384,
                header_height=40,font_size=14,label=label,prefix_length=None if j==0 else 1) for j,label in enumerate(labels)]
            filters.append('[v0][v1][v2]hstack=inputs=3[out]')
            dest=gallery/(c['name']+'_'+arm+'_guidance2.mp4')
            command=[str(ffmpeg),'-y','-loglevel','error','-filter_complex_threads','1']
            for p in paths:command+=['-i',str(p)]
            command+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-r','20','-frames:v','97',
                      '-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
            subprocess.run(command,env=env,check=True)
            actual=_read_video_rgb(dest);assert actual.shape==(97,424,1152,3)
            # Check observed frame0 is green and generated frame1 is red.
            for x in (385,769):
                for frame,ch in ((0,1),(1,0)):
                    color=actual[frame,50,x].astype(float);assert color[ch]>color[(ch+1)%3]+40,(i,frame,color)
            preview=dest.with_suffix('.png');Image.fromarray(actual[32]).save(preview)
            assets.append(dict(kind='video',index=i,case=c['name'],path=str(dest),sha256=sha(dest),
                caption=('Matched empty-caption inputs' if arm=='original_no_text' else 'Original receives its usual caption; two-task remains no-text')+
                        '; GT / original200k cameraCFG2 / two-task200k cameraCFG2',preview=str(preview)))
        raw.unlink()
        if i in cfg['selected_indices'][:2]:
            images=[]
            for variant,label in [('original','Original200k'),('two_task','Two-task200k')]:
                _,meta=output(cfg,out,variant,i,'inverse');file=gallery/(c['name']+'_'+variant+'_inverse.png')
                _plot_camera_comparison(action=np.asarray(meta['outputs'][0]['content']['action']),
                    gt_camera=Path(c['gt']).parent/'gt_camera_cosmos.npz',output=file,
                    title=label+' | inverse dynamics | native guidance1',n_cameras=7)
                images.append(Image.open(file).convert('RGB'))
            canvas=Image.new('RGB',(sum(x.width for x in images),max(x.height for x in images)),'white')
            x=0
            for im in images:canvas.paste(im,(x,0));x+=im.width
            file=gallery/(c['name']+'_inverse_comparison.png');canvas.save(file)
            assets.append(dict(kind='image',index=i,case=c['name'],path=str(file),sha256=sha(file),
                caption='GT and inferred camera trajectories, original200k versus two-task200k; native inverse guidance1.'))
        print('Rendered fixed case',i,flush=True)
    save(out/'media.json',dict(assets=assets,video_count=8,image_count=2,
        audit='97frames; frame0 green observed; frames1-96 red generated; aligned GT; fixed indices'))
    save(out/'MEDIA_COMPLETE.json',dict(videos=8,images=2,time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','analyze','render'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--original-guidance',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--notion-page',default='408d2a44-f44b-80e8-82cf-cc8d8f4b7869')
    args=parser.parse_args()
    globals()[args.command](args)
