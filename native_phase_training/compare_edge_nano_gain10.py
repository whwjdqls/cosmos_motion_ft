"""Matched zero-shot Edge/Nano camera translation x10 and text/image study.

Official CLI, official model YAMLs and existing isolated camera-CFG hook.
No training edits. All data/model/render work requires allocated compute.
"""
from __future__ import annotations
import argparse, ast, copy, html, json, os, subprocess, time
from pathlib import Path
from applications.habitat_gs.runtime import require_compute, checked_command
from native_phase_training.compare_phase1_guided import read, save, sha, lines
from native_phase_training import compare_edge_zero_camera as prior

ROOT=Path(__file__).resolve().parents[1]
OLD=prior.OPS/'edge_zeroshot_camera_cfg_20261031'
SCALES=[1.,1.5,2.]


def prepare(out):
    import numpy as np
    import yaml
    out.mkdir(parents=True,exist_ok=True)
    cfg=read(OLD/'configuration.json')
    assert read(OLD/'COMPLETE.json')['verified_assets']==17
    assert sha(ROOT/'native_phase_training/camera_cfg_inference_config.py')==cfg['camera_cfg_sha256']
    source=lines(OLD/'zero_camera.jsonl');records=[];actions=[]
    for c in cfg['cases']:
        original=Path(c['action_path']);assert sha(original)==c['action_sha256']
        a=np.asarray(read(original),dtype=np.float32)
        assert a.shape==(96,9) and np.isfinite(a).all()
        b=a.copy();b[:,:3]*=10
        assert np.array_equal(b[:,3:],a[:,3:])
        dest=out/'inputs'/c['name']/'camera_translation_x10.json'
        save(dest,b.tolist())
        check=np.asarray(read(dest),dtype=np.float32)
        assert np.array_equal(check[:,:3],a[:,:3]*10) and np.array_equal(check[:,3:],a[:,3:])
        assert sha(c['gt'])==c['gt_sha256'] and sha(c['first_frame'])==c['first_frame_sha256']
        actions.append(dict(case=c['name'],original=str(original),original_sha256=sha(original),scaled=str(dest),scaled_sha256=sha(dest),translation_gain=10,rotation_unchanged=True))
        rr=[r for r in source if r['name'].startswith(c['name']+'_cfg')];assert len(rr)==3
        for r in rr:
            r=copy.deepcopy(r);r['action_path']=str(dest);records.append(r)
    for arm in ['Edge_camera','Nano_camera']:
        (out/(arm+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'Nano_i2v.jsonl').write_text((OLD/'zero_i2v.jsonl').read_text())
    models={}
    framework=Path(os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'])
    for family in ['Edge','Nano']:
        base=prior.BASE.parent/('Cosmos3-'+family)
        template=framework/'cosmos_framework/inference/configs/model'/('Cosmos3-'+family+'.yaml')
        y=yaml.safe_load(template.read_text());m=y['model']['config']
        m['ema']['enabled']=False;m['lora_enabled']=False
        m['tokenizer']['bucket_name']='';m['tokenizer']['vae_path']=os.environ['WAN_VAE_PATH']
        # Generation uses Wan for the image prefix; these understanding/audio
        # modules are unused, as in the prior official five-way Nano study.
        m['sound_gen']=False;m['sound_dim']=None;m['sound_tokenizer']=None
        if family=='Nano':m['vlm_config']['model_instance']['config']['include_visual']=False
        path=out/('model_'+family+'.yaml');path.write_text(yaml.safe_dump(y,sort_keys=False))
        identity={str(p.relative_to(base)):sha(p) for p in [base/'config.json',base/'model.safetensors.index.json',base/'modular_model_index.json',base/'transformer/diffusion_pytorch_model.safetensors.index.json'] if p.is_file()}
        assert identity
        models[family]=dict(checkpoint=str(base),identity=identity,official_template=str(template),template_sha256=sha(template),config=str(path),config_sha256=sha(path))
    # Recheck all reused MP4 and metadata receipts rather than assume the prior
    # completion marker alone guarantees artifact integrity.
    reused=[]
    for arm in ['zero_camera','two_task_camera','zero_i2v']:
        for r in read(OLD/(arm+'_VERIFIED.json'))['samples']:
            p=OLD/'models'/arm/r['name']
            assert sha(p/'vision.mp4')==r['video_sha256'] and sha(p/'sample_outputs.json')==r['metadata_sha256']
            reused.append(dict(path=str(p/'vision.mp4'),sha256=r['video_sha256']))
    for c in cfg['cases']:
        p=prior.camera_path(OLD,cfg,c,'two_task',2.);reused.append(dict(path=str(p),sha256=sha(p)))
    # Previous original/two-task image generation was also independently audited.
    for arm in ['original','two_task']:
        manifest=read(prior.I2V/'models'/arm/'VERIFIED.json')
        assert manifest['configuration_sha256']==sha(prior.I2V/'configuration.json')
        for r in manifest['records']:
            p=prior.I2V/'models'/arm/r['name']
            assert sha(p/'vision.mp4')==r['video_sha256'] and sha(p/'sample_outputs.json')==r['metadata_sha256']
            reused.append(dict(path=str(p/'vision.mp4'),sha256=r['video_sha256']))
    config=dict(schema='edge_nano_gain10_v1',prior=str(OLD),prior_configuration_sha256=sha(OLD/'configuration.json'),prior_metrics_sha256=sha(OLD/'metrics.json'),cases=cfg['cases'],i2v_cases=cfg['i2v_cases'],camera_scales=SCALES,zero_shot_translation_gain=10,phase1_translation_gain=1,models=models,actions=actions,reused=reused,camera_cfg_sha256=cfg['camera_cfg_sha256'],protocol=dict(camera='same six windows, initial images, no text, UniPC30/shift10/seed0/T97/20fps/256; cameraCFG1/1.5/2; no diffusion or text-KV cache',i2v='same10 text+first-image inputs; UniPC35/textCFG6/shift10/seed0/native diffusion cache; no camera',scored_frames='1-96, equal window weights, MP4-decoded RGB as prior study',zero_shot='untouched official HF weights, generation only; EMA copy/LoRA/audio disabled; unused Nano understanding ViT omitted'),notion_page=cfg['notion_page'])
    if (out/'configuration.json').exists():assert read(out/'configuration.json')==config
    else:save(out/'configuration.json',config)
    save(out/'PREPARED.json',dict(camera_predictions=36,i2v_predictions=10,scaled_translation_only=True,reused_artifacts=len(reused)))


def infer(out):
    cfg=read(out/'configuration.json')
    for arm in ['Edge_camera','Nano_camera','Nano_i2v']:
        marker=out/(arm+'_VERIFIED.json')
        if marker.exists():continue
        family,mode=arm.split('_');model=cfg['models'][family]
        assert sha(model['config'])==model['config_sha256']
        for rel,digest in model['identity'].items():assert sha(Path(model['checkpoint'])/rel)==digest
        inputs=out/(arm+'.jsonl');destination=out/'models'/arm;records=lines(inputs)
        cmd=[os.sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',('native_phase_training.zero_camera_inference' if mode=='camera' else 'cosmos_framework.scripts.inference'),'--checkpoint-path',model['checkpoint'],'--config-file',model['config'],'--sampler','unipc','--use-ema-weights','--parallelism-preset','latency','--dp-shard-size','1','--dp-replicate-size','1','--cp-size','1','--cfgp-size','1','--no-use-torch-compile','--no-use-cuda-graphs','--no-guardrails','-o',str(destination),'-i',str(inputs)]
        if mode=='camera':cmd+=['--no-diffusion-cache']
        save(out/'progress.json',dict(stage='inference',arm=arm,time=time.time()))
        if not all((destination/r['name']/'sample_outputs.json').exists() for r in records):
            checked_command(cmd,out/(arm+'.log'),cwd=os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'],env=os.environ.copy())
        audit=[]
        for r in records:
            p=destination/r['name'];meta=read(p/'sample_outputs.json');assert meta['status']=='success'
            for k,v in r.items():assert meta['args'].get(k)==v,(arm,k)
            audit.append(dict(name=r['name'],video_sha256=sha(p/'vision.mp4'),metadata_sha256=sha(p/'sample_outputs.json')))
        configs=[ast.literal_eval(s.split('OmniMoTModel: config ',1)[1]) for s in (destination/'debug.log').read_text().splitlines() if 'OmniMoTModel: config ' in s]
        assert len(configs)==1 and configs[0]['lora_enabled'] is False
        save(out/(arm+'_runtime_config.json'),configs[0])
        if mode=='camera':
            log=(out/(arm+'.log')).read_text()
            calls=[ast.literal_eval(s.split('CAMERA_CFG_END ',1)[1]) for s in log.splitlines() if s.startswith('CAMERA_CFG_END ')]
            assert len(calls)==len(records)
            for r,c in zip(records,calls,strict=True):assert c==dict(scale=r['guidance'],conditional=30,camera_free=0 if r['guidance']==1 else 30)
            parity=[json.loads(s.split('ZERO_CAMERA_NATIVE_PARITY ',1)[1]) for s in log.splitlines() if s.startswith('ZERO_CAMERA_NATIVE_PARITY ')]
            assert len(parity)==1 and parity[0] and all(c['equal'] for c in parity[0])
            save(out/(arm+'_guidance_audit.json'),dict(calls=calls,scale1_native_parity=parity))
        save(marker,dict(samples=audit,input_sha256=sha(inputs),lora_enabled=False))


def path_camera(out,c,model,scale):
    return out/'models'/(model+'_camera')/(c['name']+'_cfg'+prior.tag(scale)+'_forward_dynamics')/'vision.mp4'


def analyze(out):
    import numpy as np
    import torch
    from native_phase_training.evaluate_inverse_forward import LPIPSAlex,_read_video_rgb,_resize_gt_like_native,_frame_metrics
    cfg=read(out/'configuration.json');old=read(OLD/'metrics.json');lpips=LPIPSAlex('cuda:0',16)
    rows=[]
    for c in cfg['cases']:
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        for model,scale in [('GT',None)]+[(m,s) for m in ['Edge','Nano'] for s in SCALES]:
            p=Path(c['gt']) if model=='GT' else path_camera(out,c,model,scale)
            pred=gt if model=='GT' else _read_video_rgb(p);assert pred.shape==gt.shape==(97,256,256,3)
            with torch.inference_mode():v=_frame_metrics(gt[1:],pred[1:],lpips)
            assert all(np.isfinite(a).all() for a in v.values())
            rows.append(dict(case=c['name'],model=model,scale=scale,gain=None if model=='GT' else 10,metrics={k:float(a.mean()) for k,a in v.items()},frames={k:a.tolist() for k,a in v.items()},path=str(p),sha256=sha(p)))
    for r in old['camera_per_case']:
        if r['model']=='GT':continue
        r=copy.deepcopy(r);r['model']='Edge' if r['model']=='zero' else 'Two-task200k';r['gain']=1;rows.append(r)
    summary=[]
    for model,gain in [('GT',None),('Edge',10),('Nano',10),('Edge',1),('Two-task200k',1)]:
        for scale in ([None] if model=='GT' else SCALES):
            rr=[r for r in rows if (r['model'],r['gain'],r['scale'])==(model,gain,scale)];assert len(rr)==6
            summary.append(dict(model=model,gain=gain,scale=scale,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']}))
    irows=[]
    for c in cfg['i2v_cases']:
        if c['kind']!='held_out_caption':continue
        gt=_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        p=out/'models/Nano_i2v'/c['record_name']/'vision.mp4';pred=_read_video_rgb(p);assert pred.shape==gt.shape==(97,256,256,3)
        with torch.inference_mode():v=_frame_metrics(gt[1:],pred[1:],lpips)
        assert all(np.isfinite(a).all() for a in v.values())
        irows.append(dict(case=c['id'],path=str(p),sha256=sha(p),**{k:float(a.mean()) for k,a in v.items()}))
    isum=copy.deepcopy(old['i2v_summary']);isum['Nano']={k:float(np.mean([r[k] for r in irows])) for k in ['psnr_db','ssim','lpips_alex']}
    save(out/'metrics.json',dict(camera_summary=summary,camera_per_case=rows,i2v_summary=isum,i2v_nano_per_case=irows))
    save(out/'METRICS_COMPLETE.json',dict(cases=6,scored_frames_per_case=96))


def render(out):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from native_phase_training.evaluate_inverse_forward import _read_video_rgb,_resize_gt_like_native
    cfg=read(out/'configuration.json');oldcfg=read(OLD/'configuration.json');gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',14)
    assets=[]
    def encode(identifier,panels,labels,positions,width,height,caption,kind):
        text=[];line=''
        for word in caption.split():
            cand=(line+' '+word).strip()
            if line and font.getlength(cand)>width-24:text.append(line);line=word
            else:line=cand
        if line:text.append(line)
        bottom=height;height+=22*len(text)+18;height+=height%2
        p=gallery/(identifier+'.mp4');env={**os.environ,'LD_LIBRARY_PATH':str(Path(os.environ['COSMOS_ENV_ROOT'])/'lib')}
        cmd=[str(Path(os.environ['COSMOS_ENV_ROOT'])/'bin/ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','20','-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','fast','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(p)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,env=env);snaps=[]
        for t in range(97):
            canvas=Image.new('RGB',(width,height),(17,23,31));d=ImageDraw.Draw(canvas)
            for frames,label,(x,y,size) in zip(panels,labels,positions,strict=True):
                gt=label.startswith('GT');color=(91,158,245) if gt else ((34,220,90) if t==0 else (245,65,65))
                canvas.paste(Image.fromarray(frames[t]).resize((size,size)),(x,y+38));d.text((x+6,y+2),label,font=font,fill='white')
                d.text((x+6,y+21),'REFERENCE' if gt else ('OBSERVED' if t==0 else 'GENERATED'),font=small,fill=color)
                d.rectangle((x+1,y+39,x+size-2,y+size+36),outline=color,width=4)
            for i,s in enumerate(text):d.text((10,bottom+6+22*i),s,font=font,fill='white')
            proc.stdin.write(np.asarray(canvas).tobytes())
            if t in [0,24,48,72,96]:snaps.append(canvas.copy())
        proc.stdin.close();assert proc.wait()==0
        video=_read_video_rgb(p);assert video.shape==(97,height,width,3)
        for label,(x,y,size) in zip(labels,positions):
            if label.startswith('GT'):continue
            for t,ch in [(0,1),(1,0)]:
                px=video[t,y+55,x+2].astype(float);assert px[ch]>px[(ch+1)%3]+40
        im=Image.new('RGB',(width,height*5))
        for i,s in enumerate(snaps):im.paste(s,(0,height*i))
        contact=p.with_suffix('.png');im.save(contact)
        assets.append(dict(kind=kind,id=identifier,path=str(p),sha256=sha(p),caption=caption,contact=str(contact)))
    for c in cfg['cases']:
        panels=[_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)];labels=['GT reference'];positions=[(352,0,256)]
        for row,(model,title) in enumerate([('Nano','Nano ZS x10'),('Edge','Edge ZS x10'),('two_task','Two-task200k x1')]):
            for col,scale in enumerate(SCALES):
                p=prior.camera_path(OLD,oldcfg,c,model,scale) if model=='two_task' else path_camera(out,c,model,scale)
                panels.append(_read_video_rgb(p));labels.append(f'{title} | CFG {scale:g}');positions.append((320*col,300+358*row,320))
        encode('gain10_camera_'+str(c['index']).zfill(2),panels,labels,positions,960,1374,f'Case {c["index"]} | Same image and camera rotations; zero-shot translation x10, two-task x1. No text. Columns camera CFG1 (none),1.5,2. UniPC30, shift10, seed0.','camera')
    for c in cfg['i2v_cases']:
        paths=[out/'models/Nano_i2v'/c['record_name']/'vision.mp4',OLD/'models/zero_i2v'/c['record_name']/'vision.mp4',prior.I2V/'models/original'/c['record_name']/'vision.mp4',prior.I2V/'models/two_task'/c['record_name']/'vision.mp4']
        encode('nano_i2v_'+c['id'],[_read_video_rgb(p) for p in paths],['Nano zero-shot','Edge zero-shot','Original Phase1 200k','Two-task Phase1 200k'],[(320*i,0,320) for i in range(4)],1280,358,c['formatted_caption']+' | Same image/seed0; UniPC35, textCFG6, shift10. No camera input (trajectory gain does not apply).','i2v')
    save(out/'media.json',dict(assets=assets))
    content='<meta charset="utf-8"><title>Nano/Edge camera gain10</title><style>body{background:#111820;color:#eee;font:16px sans-serif;max-width:1280px;margin:20px auto}video{width:100%;max-height:90vh}</style><h1>Nano and Edge: camera translation x10</h1>'
    for a in assets:content+='<h2>'+html.escape(a['id'])+'</h2><p>'+html.escape(a['caption'])+'</p><video controls preload="metadata" src="'+Path(a['path']).name+'"></video>'
    (gallery/'index.html').write_text(content)
    save(out/'MEDIA_COMPLETE.json',dict(camera_videos=6,i2v_videos=10,decoded_frames=97,borders_verified=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','infer','analyze','render','all']);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();require_compute()
    for stage in (['prepare','infer','analyze','render'] if a.command=='all' else [a.command]):
        if stage!='prepare':save(a.output/'progress.json',dict(stage=stage,time=time.time()))
        globals()[stage](a.output)
if __name__=='__main__':main()
