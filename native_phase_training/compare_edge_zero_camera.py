"""Matched native Edge zero-shot / two-task camera-CFG and I2V diagnostics."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from applications.habitat_gs.runtime import require_compute, checked_command, official_command, evaluation_environment
from native_phase_training.compare_phase1_guided import read, save, sha, verify_output

ROOT = Path('/home/whwjdqls99/cosmos_motion_ft')
OPS = Path('/scratch2/whwjdqls99/cosmos/operations')
PRIOR = OPS/'phase1_twotask200k_guidance2_20261029'
I2V = OPS/'phase1_twotask_i2v_20261031'
BASE = Path('/lustre/whwjdqls99/cosmos/weka/Cosmos3-Edge')
SCALES = [1.0, 1.5, 2.0]


def tag(value):
    return str(float(value)).replace('.', 'p')


def prepare(out):
    from native_phase_training.run_contract import resolve_eval_contract
    cfg = read(PRIOR/'configuration.json')
    previous_i2v = read(I2V/'configuration.json')
    assert read(PRIOR/'COMPLETE.json') and read(I2V/'COMPLETE.json')
    assert sha(ROOT/'native_phase_training/camera_cfg_inference_config.py') == cfg['camera_cfg_sha256']
    contract, _, checkpoint, _ = resolve_eval_contract(Path(cfg['checkpoint']))
    assert sha(checkpoint/'model/.metadata') == cfg['checkpoint_metadata_sha256']
    assert json.loads(json.dumps(contract.to_dict())) == cfg['contracts']['two_task']
    indices = [0,7,14,28,42,63]
    cases = [copy.deepcopy(cfg['cases'][i]) for i in indices]
    records, reuse = [], {}
    for c in cases:
        for name, digest in [('gt','gt_sha256'),('action_path','action_sha256')]:
            assert sha(c[name]) == c[digest]
        c['first_frame_sha256'] = sha(c['first_frame'])
        record = cfg['inputs']['two_task'][c['index']]
        assert record['prompt'] == '' and record['guidance'] == 2.0
        assert record['seed'] == 0 and record['num_steps'] == 30
        path = PRIOR/'models/two_task/inference'/record['name']
        verify_output(path, record, True)
        reuse[c['name']] = dict(directory=str(path), video_sha256=sha(path/'vision.mp4'),
                               metadata_sha256=sha(path/'sample_outputs.json'))
        for scale in SCALES:
            r = copy.deepcopy(record)
            r.update(guidance=scale, name=c['name']+'_cfg'+tag(scale)+'_forward_dynamics')
            records.append(r)
    inventory = []
    oldroot = OPS/'phase3_followups_20261013'
    for name in ['official_camera_nano_edge_20261014','official_camera_fiveway_scale10_20261014']:
        directory = oldroot/name
        items = list((directory/'Edge').rglob('sample_outputs.json'))
        sample = read(items[0]) if items else {}
        inventory.append(dict(directory=str(directory), edge_outputs=len(items), example_args=sample.get('args'),
            reusable=False, reason='Different fixed cohort/window ordering, captions, seeds and35steps; fiveway also scales camera translation10x. This sweep uses original camera units, empty text, seed0 and30steps.'))
    base_identity = {name:sha(BASE/name) for name in ['config.json','modular_model_index.json','transformer/diffusion_pytorch_model.safetensors.index.json']}
    frozen = dict(schema='edge_zs_camera_cfg_and_i2v_v1', cases=cases, scales=SCALES, indices=indices,
        base_checkpoint=str(BASE), base_identity=base_identity, checkpoint=str(checkpoint),
        checkpoint_metadata_sha256=cfg['checkpoint_metadata_sha256'], contract=cfg['contracts']['two_task'],
        camera_cfg_sha256=cfg['camera_cfg_sha256'], reference=str(PRIOR), reference_sha256=sha(PRIOR/'configuration.json'),
        reuse=reuse, prior_i2v=str(I2V), prior_i2v_configuration_sha256=sha(I2V/'configuration.json'),
        prior_i2v_input_sha256=sha(I2V/'inputs.jsonl'), i2v_cases=previous_i2v['cases'],
        camera_translation_gain=1.0, camera_text='', num_steps=30, shift=10, seed=0,
        cache='Diffusion cache and text-KV reuse disabled for every camera weight',
        notion_page='409d2a44-f44b-8089-9b2b-ec1fc10ba9bc', inventory=inventory)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'configuration.json').exists():
        assert read(out/'configuration.json') == frozen
    else:
        save(out/'configuration.json',frozen)
        save(out/'inventory.json',inventory)
        (out/'zero_camera.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'two_task_camera.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['guidance']!=2.))
        (out/'zero_i2v.jsonl').write_text((I2V/'inputs.jsonl').read_text())
    save(out/'PREPARED.json',dict(new_camera_predictions=30,reused_camera_predictions=6,
        new_i2v_predictions=10,reused_i2v_predictions=20,configuration_sha256=sha(out/'configuration.json')))


def base_command(inputs, destination, camera):
    cmd=[os.sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        '-m',('native_phase_training.zero_camera_inference' if camera else 'cosmos_framework.scripts.inference'),
        '--checkpoint-path',str(BASE)]
    cmd += ['--experiment-overrides','model.config.tokenizer.bucket_name=""',
        'model.config.tokenizer.vae_path='+os.environ['WAN_VAE_PATH'],
        '--sampler','unipc','--use-ema-weights','--parallelism-preset','latency',
        '--dp-shard-size','1','--dp-replicate-size','1','--cp-size','1','--cfgp-size','1',
        '--no-use-torch-compile','--no-use-cuda-graphs','--no-guardrails',
        '-o',str(destination),'-i',str(inputs)]
    if camera:cmd += ['--no-diffusion-cache']
    return cmd


def infer(out):
    cfg=read(out/'configuration.json')
    for arm in ['zero_camera','two_task_camera','zero_i2v']:
        marker=out/(arm+'_VERIFIED.json')
        if marker.exists():continue
        destination=out/'models'/arm;inputs=out/(arm+'.jsonl')
        env=os.environ.copy()
        if arm.startswith('zero'):
            env['ZERO_CAMERA_BASE_MODEL']='1'
            command=base_command(inputs,destination,arm=='zero_camera')
        else:
            profile=copy.deepcopy(read(I2V/'configuration.json')['profiles']['two_task'])
            profile.update(config_file='native_phase_training/zero_camera_cfg_config.py',diffusion_cache=False)
            command=official_command(profile,inputs,destination)
            env=evaluation_environment(profile);env['ZERO_CAMERA_BASE_MODEL']='0'
        save(out/'progress.json',dict(stage='inference',arm=arm,time=time.time()))
        records=[json.loads(x) for x in inputs.read_text().splitlines()]
        # Recover completed inference after a reporting-only failure without
        # overwriting its sampling log (which contains the guidance audit).
        complete = all((destination/r['name']/'sample_outputs.json').exists()
                       and (destination/r['name']/'vision.mp4').exists() for r in records)
        if not complete:
            checked_command(command,out/(arm+'.log'),cwd=os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'],env=env)
        audit=[]
        for record in records:
            path=destination/record['name'];meta=read(path/'sample_outputs.json')
            assert meta['status']=='success'
            for k,v in record.items():assert meta['args'].get(k)==v,(arm,k)
            audit.append(dict(name=record['name'],video_sha256=sha(path/'vision.mp4'),metadata_sha256=sha(path/'sample_outputs.json')))
        if arm.endswith('camera'):
            text=(out/(arm+'.log')).read_text()
            calls=[ast.literal_eval(x.split('CAMERA_CFG_END ',1)[1]) for x in text.splitlines() if x.startswith('CAMERA_CFG_END ')]
            assert len(calls)==len(records),(arm,len(calls))
            for r,a in zip(records,calls,strict=True):
                assert a==dict(scale=r['guidance'],conditional=30,camera_free=0 if r['guidance']==1 else 30)
            parity=[json.loads(x.split('ZERO_CAMERA_NATIVE_PARITY ',1)[1]) for x in text.splitlines() if x.startswith('ZERO_CAMERA_NATIVE_PARITY ')]
            assert len(parity)==1 and parity[0] and all(r['equal'] for r in parity[0])
            save(out/(arm+'_guidance_audit.json'),dict(calls=calls,native_scale1_parity=parity))
        import yaml
        if (destination/'config.yaml').exists():
            model_cfg=yaml.safe_load((destination/'config.yaml').read_text())['model']['config']
        else:
            # Official HF loading logs the resolved runtime config but does
            # not emit the DCP-only config.yaml artifact.
            configs=[ast.literal_eval(line.split('OmniMoTModel: config ',1)[1])
                     for line in (destination/'debug.log').read_text().splitlines()
                     if 'OmniMoTModel: config ' in line]
            assert len(configs)==1,(arm,len(configs))
            model_cfg=configs[0]
        enabled=model_cfg['lora_enabled']
        assert bool(enabled)==(arm=='two_task_camera'),(arm,enabled)
        save(out/(arm+'_resolved_model_config.json'),model_cfg)
        save(marker,dict(samples=audit,lora_enabled=enabled,input_sha256=sha(inputs)))


def camera_path(out,cfg,c,model,scale):
    if model=='two_task' and scale==2.:
        reuse=cfg['reuse'][c['name']];path=Path(reuse['directory'])/'vision.mp4'
        assert sha(path)==reuse['video_sha256']
        return path
    return out/'models'/(model+'_camera')/(c['name']+'_cfg'+tag(scale)+'_forward_dynamics')/'vision.mp4'


def analyze(out):
    import numpy as np
    import torch
    from native_phase_training.evaluate_inverse_forward import LPIPSAlex,_read_video_rgb,_resize_gt_like_native,_frame_metrics
    cfg=read(out/'configuration.json');metric=LPIPSAlex('cuda:0',16)
    rows=[];temporal=[]
    for c in cfg['cases']:
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        for model,scale in [('GT',None)]+[(m,s) for m in ['zero','two_task'] for s in SCALES]:
            path=Path(c['gt']) if model=='GT' else camera_path(out,cfg,c,model,scale)
            pred=gt if model=='GT' else _read_video_rgb(path)
            assert pred.shape==gt.shape==(97,256,256,3)
            with torch.inference_mode():values=_frame_metrics(gt[1:],pred[1:],metric)
            assert all(np.isfinite(v).all() for v in values.values())
            rows.append(dict(case=c['name'],model=model,scale=scale,path=str(path),sha256=sha(path),
                metrics={k:float(v.mean()) for k,v in values.items()},frames={k:v.tolist() for k,v in values.items()}))
    summary=[]
    for model,scale in [('GT',None)]+[(m,s) for m in ['zero','two_task'] for s in SCALES]:
        rr=[r for r in rows if (r['model'],r['scale'])==(model,scale)]
        summary.append(dict(model=model,scale=scale,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']}))
    old=read(I2V/'metrics.json');i2v_rows=[]
    for c in cfg['i2v_cases']:
        if c['kind']!='held_out_caption':continue
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        pred=_read_video_rgb(out/'models/zero_i2v'/c['record_name']/'vision.mp4')
        assert pred.shape==gt.shape==(97,256,256,3)
        with torch.inference_mode():values=_frame_metrics(gt[1:],pred[1:],metric)
        i2v_rows.append(dict(case=c['id'],variant='zero',**{k:float(v.mean()) for k,v in values.items()}))
    i2v_summary={**old['summary'],'zero':{k:float(np.mean([r[k] for r in i2v_rows])) for k in ['psnr_db','ssim','lpips_alex']}}
    save(out/'metrics.json',dict(camera_summary=summary,camera_per_case=rows,i2v_summary=i2v_summary,
        i2v_zero_per_case=i2v_rows,i2v_prior_metrics_sha256=sha(I2V/'metrics.json')))
    save(out/'METRICS_COMPLETE.json',dict(cases=6,scored_generated_frames=96))


def render(out):
    import html
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from native_phase_training.evaluate_inverse_forward import _read_video_rgb,_resize_gt_like_native
    cfg=read(out/'configuration.json');gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',14)
    ffmpeg=str(Path(os.environ['COSMOS_ENV_ROOT'])/'bin/ffmpeg')
    env={**os.environ,'LD_LIBRARY_PATH':str(Path(os.environ['COSMOS_ENV_ROOT'])/'lib')}
    assets=[]

    def encode(identifier,panels,labels,positions,width,height,caption,kind):
        lines=[];line=''
        for word in caption.split():
            candidate=(line+' '+word).strip()
            if line and font.getlength(candidate)>width-24:lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        bottom=height;height+=len(lines)*22+18;height+=height%2
        path=gallery/(identifier+'.mp4')
        cmd=[ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
             '-framerate','20','-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','fast','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,env=env);snaps=[]
        for t in range(97):
            canvas=Image.new('RGB',(width,height),(17,23,31));draw=ImageDraw.Draw(canvas)
            for frames,label,(x,y,size),is_gt in zip(panels,labels,positions,[x.startswith('GT') for x in labels],strict=True):
                canvas.paste(Image.fromarray(frames[t]).resize((size,size)),(x,y+38))
                color=(91,158,245) if is_gt else ((34,220,90) if t==0 else (245,65,65))
                draw.text((x+7,y+2),label,font=font,fill='white')
                draw.text((x+7,y+21),'REFERENCE' if is_gt else ('OBSERVED' if t==0 else 'GENERATED'),font=small,fill=color)
                draw.rectangle((x+1,y+39,x+size-2,y+size+36),outline=color,width=4)
            for n,line in enumerate(lines):draw.text((10,bottom+6+n*22),line,font=font,fill='white')
            proc.stdin.write(np.asarray(canvas).tobytes())
            if t in [0,24,48,72,96]:snaps.append(canvas.copy())
        proc.stdin.close();assert proc.wait()==0
        decoded=_read_video_rgb(path);assert decoded.shape==(97,height,width,3)
        for label,(x,y,size) in zip(labels,positions):
            if label.startswith('GT'):continue
            for t,ch in [(0,1),(1,0)]:
                px=decoded[t,y+55,x+2].astype(float);assert px[ch]>px[(ch+1)%3]+40
        contact=Image.new('RGB',(width,height*len(snaps)))
        for n,im in enumerate(snaps):contact.paste(im,(0,n*height))
        cp=path.with_suffix('.png');contact.save(cp)
        assets.append(dict(kind=kind,id=identifier,path=str(path),sha256=sha(path),caption=caption,contact=str(cp)))

    for c in cfg['cases']:
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        panels=[gt];labels=['GT reference'];positions=[(320,0,256)]
        for row,(model,title) in enumerate([('zero','Zero-shot Edge'),('two_task','Two-task200k')]):
            for col,scale in enumerate(SCALES):
                panels.append(_read_video_rgb(camera_path(out,cfg,c,model,scale)))
                labels.append(f'{title} | camera {scale:g}')
                positions.append((col*320,300+row*358,320))
        encode('camera_'+str(c['index']).zfill(2),panels,labels,positions,960,1016,
            f'Case {c["index"]} | Same image + original camera trajectory; no text. Columns: no CFG (1), 1.5, 2. Rows: zero-shot / two-task200k. UniPC30, shift10, seed0; camera translation gain1.', 'camera')
    prior=read(I2V/'configuration.json')
    for c in cfg['i2v_cases']:
        panels=[];labels=[]
        for model,title in [('zero','Zero-shot Edge'),('original','Original Phase1 200k'),('two_task','Two-task Phase1 200k')]:
            path=out/'models/zero_i2v'/c['record_name']/'vision.mp4' if model=='zero' else I2V/'models'/model/c['record_name']/'vision.mp4'
            panels.append(_read_video_rgb(path));labels.append(title)
        encode('i2v_'+c['id'],panels,labels,[(i*320,0,320) for i in range(3)],960,358,
            c['formatted_caption']+' | Same image/seed0; UniPC35, text CFG6, shift10; no camera input.','i2v')
    save(out/'media.json',dict(assets=assets))
    cards=[]
    for a in assets:
        cards.append('<article><h2>'+html.escape(a['id'])+'</h2><p>'+html.escape(a['caption'])+
                     '</p><video controls preload="metadata" src="'+html.escape(Path(a['path']).name)+
                     '"></video></article>')
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Edge zero-shot comparisons</title>'
        '<style>body{background:#111820;color:#eee;font:16px sans-serif;max-width:1100px;margin:30px auto}'
        'video{width:100%;max-height:85vh}article{margin:35px 0}</style>'
        '<h1>Edge zero-shot / Phase1 comparisons</h1><p>Green: observed first frame. Red: generated. Blue: GT. '
        'Camera CFG sweep and separate text + first-image generation; six fixed windows, one seed.</p>'+''.join(cards))
    save(out/'MEDIA_COMPLETE.json',dict(camera_videos=6,i2v_videos=10,fully_decoded=True,borders_verified=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','infer','analyze','render','all']);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();require_compute()
    for name in (['prepare','infer','analyze','render'] if args.command=='all' else [args.command]):globals()[name](args.output)


if __name__=='__main__':main()
