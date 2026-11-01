"""Controlled inference-only two-task versus original Phase-1 I2V comparison.

Uses the existing native inference entrypoint and saved evaluation profile.
All preparation, model execution, scoring and rendering require compute.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import time

from applications.habitat_gs.camera import sha256
from applications.habitat_gs.profile import make_record, verify
from applications.habitat_gs.runtime import require_compute, official_command, evaluation_environment, checked_command
from native_phase_training.compare_phase1_guided import save, read


def prepare(out: Path):
    reference = Path('/scratch2/whwjdqls99/cosmos/operations/phase1_twotask200k_guidance2_20261029/configuration.json')
    profile_path = Path('/scratch2/whwjdqls99/cosmos/applications/habitat_gs/evaluation_profile.json')
    reference_data, profile = read(reference), read(profile_path)
    verify(profile)
    assert profile['checkpoint'] == reference_data['baseline_checkpoint']
    from native_phase_training.run_contract import resolve_eval_contract
    profiles = {}
    for variant, checkpoint in [('original', reference_data['baseline_checkpoint']), ('two_task', reference_data['checkpoint'])]:
        contract, source, checkpoint, _ = resolve_eval_contract(Path(checkpoint))
        assert checkpoint.name == 'iter_000200000'
        p = copy.deepcopy(profile)
        p.update(checkpoint=str(checkpoint), contract=json.loads(json.dumps(contract.to_dict())), contract_source=source,
                 checkpoint_metadata_sha256=sha256(checkpoint/'model/.metadata'))
        verify(p)
        assert p['contract'] == reference_data['contracts'][variant]
        profiles[variant] = p
    assert profiles['two_task']['contract']['active_modes'] == ['forward_dynamics','inverse_dynamics']
    indices = [0, 7, 14, 28, 42, 63]
    cases = []
    for i in indices:
        c = copy.deepcopy(reference_data['cases'][i])
        c.update(kind='held_out_caption', id=f'case_{i:02d}', seed=0)
        cases.append(c)
    for i in [0, 14]:
        for direction in ['left', 'right']:
            c = copy.deepcopy(reference_data['cases'][i])
            c.update(kind='controlled_turn', id=f'turn_{i:02d}_{direction}', seed=0,
                     caption=f'The camera wearer slowly turns {direction} while standing in the same place.')
            cases.append(c)
    records = []
    for c in cases:
        c['first_frame_sha256'] = sha256(c['first_frame'])
        assert sha256(c['gt']) == c['gt_sha256']
        record = make_record(profile, 'image2video', c['id'], c['first_frame'], c['caption'], c['seed'])
        assert 'action_path' not in record and record.get('condition_frame_indexes_vision', [0]) == [0]
        assert (record['num_steps'], record['guidance'], record['shift']) == (35, 6.0, 10.0)
        c['formatted_caption'] = json.loads(record['prompt'])['actions'][0]['description']
        c['record_name'] = record['name']
        records.append(record)
    cfg = dict(schema='phase1_i2v_comparison_v1', profiles=profiles, cases=cases,
               reference=str(reference), reference_sha256=sha256(reference),
               profile_source=str(profile_path), profile_source_sha256=sha256(profile_path),
               fixed_indices=indices, seed=0, inputs='one image + text; no camera trajectory',
               task='image2video', matched_input_file='inputs.jsonl',
               intended_scope='6 held-out cases + 2 same-image left/right pairs; not full-test evaluation',
               notion_page='409d2a44-f44b-8005-9af9-df4535fd8897')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'configuration.json').exists():
        assert read(out/'configuration.json') == cfg, 'Configuration drift'
    else:
        save(out/'configuration.json', cfg)
        (out/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    save(out/'PREPARED.json', dict(cases=len(cases), predictions=2*len(cases),
                                 input_sha256=sha256(out/'inputs.jsonl'), configuration_sha256=sha256(out/'configuration.json')))


def infer(out: Path):
    cfg = read(out/'configuration.json')
    assert sha256(out/'inputs.jsonl') == read(out/'PREPARED.json')['input_sha256']
    for variant, profile in cfg['profiles'].items():
        verify(profile)
        destination = out/'models'/variant
        if (destination/'VERIFIED.json').exists():
            continue
        save(out/'progress.json', dict(stage='inference', variant=variant, time=time.time()))
        checked_command(official_command(profile, out/'inputs.jsonl', destination), out/(variant+'.log'),
                        cwd=os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'], env=evaluation_environment(profile))
        audit = []
        records = [json.loads(x) for x in (out/'inputs.jsonl').read_text().splitlines()]
        for record in records:
            path = destination/record['name']
            meta = read(path/'sample_outputs.json')
            assert meta['status'] == 'success', meta
            for key, val in record.items():
                assert meta['args'].get(key) == val, (variant,key)
            assert not meta['args'].get('action_path'), 'Unexpected camera conditioning'
            audit.append(dict(name=record['name'], video_sha256=sha256(path/'vision.mp4'),
                              metadata_sha256=sha256(path/'sample_outputs.json')))
        save(destination/'VERIFIED.json', dict(records=audit, checkpoint=profile['checkpoint'],
                                             configuration_sha256=sha256(out/'configuration.json')))


def analyze(out: Path):
    import numpy as np
    import torch
    from native_phase_training.evaluate_inverse_forward import LPIPSAlex, _read_video_rgb, _resize_gt_like_native, _frame_metrics
    cfg = read(out/'configuration.json')
    metric = LPIPSAlex('cuda:0',16)
    rows, frames, diagnostics = [], [], []
    for c in cfg['cases']:
        gt = _resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256)
        assert gt.shape == (97,256,256,3)
        for variant in ['GT','original','two_task']:
            path = Path(c['gt']) if variant == 'GT' else out/'models'/variant/c['record_name']/'vision.mp4'
            pred = gt if variant == 'GT' else _read_video_rgb(path)
            assert pred.shape == gt.shape
            diagnostics.append(dict(case=c['id'],variant=variant,
                temporal_rgb_mae_0_1=float(np.abs(np.diff(pred.astype(np.float32)/255,axis=0)).mean()),
                video_sha256=sha256(path)))
            if c['kind'] == 'held_out_caption':
                with torch.inference_mode(): scores = _frame_metrics(gt[1:],pred[1:],metric)
                assert all(np.isfinite(x).all() for x in scores.values())
                rows.append(dict(case=c['id'],variant=variant,**{k:float(v.mean()) for k,v in scores.items()}))
                frames.append(dict(case=c['id'],variant=variant,**{k:v.tolist() for k,v in scores.items()}))
    sensitivity = []
    for i in [0,14]:
        for variant in ['original','two_task']:
            videos = [_read_video_rgb(out/'models'/variant/f'turn_{i:02d}_{d}_image2video'/'vision.mp4') for d in ['left','right']]
            sensitivity.append(dict(case=i,variant=variant,
                left_vs_right_rgb_mae_0_1=float(np.abs(videos[0][1:].astype(np.float32)-videos[1][1:].astype(np.float32)).mean()/255)))
    summary = {variant:{k:float(np.mean([r[k] for r in rows if r['variant']==variant]))
                       for k in ['psnr_db','ssim','lpips_alex']} for variant in ['GT','original','two_task']}
    save(out/'metrics.json',dict(summary=summary,per_case=rows,per_frame=frames,
        temporal_diagnostics=diagnostics,prompt_swap_sensitivity=sensitivity,
        note='Six fixed windows, one seed; GT resemblance is not text-following accuracy. Prompt-swap changes do not establish correct turn direction. Controlled prompts have no matching GT future.'))
    with (out/'per_case.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'METRICS_COMPLETE.json',dict(cases=6,generated_frames_per_case=96))


def render(out: Path):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from native_phase_training.evaluate_inverse_forward import _read_video_rgb, _resize_gt_like_native
    cfg = read(out/'configuration.json')
    gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    ffmpeg=str(Path(os.environ['COSMOS_ENV_ROOT'])/'bin/ffmpeg')
    env={**os.environ,'LD_LIBRARY_PATH':str(Path(os.environ['COSMOS_ENV_ROOT'])/'lib')}
    font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',14)
    assets=[]
    for c in cfg['cases']:
        is_gt=c['kind']=='held_out_caption'
        panels=[_read_video_rgb(out/'models'/v/c['record_name']/'vision.mp4') for v in ['original','two_task']]
        labels=['Original Phase 1 | 200k','Two-task Phase 1 | 200k']
        if is_gt:
            panels.insert(0,_resize_gt_like_native(_read_video_rgb(Path(c['gt'])),256,256));labels.insert(0,'GT reference')
        width=384*len(panels)
        # Wrap using measured font advances, not an estimated character width.
        lines=[];line=''
        for word in c['formatted_caption'].split():
            proposed=(line+' '+word).strip()
            if line and font.getlength(proposed)>width-24:
                lines.append(line);line=word
            else:line=proposed
        if line:lines.append(line)
        assert all(font.getlength(line)<=width-24 for line in lines)
        height=424+max(2,len(lines))*20+34;height+=(height%2)
        video=gallery/(c['id']+'_labeled.mp4')
        command=[ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
                 '-framerate','20','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,env=env)
        snapshots=[]
        for t in range(97):
            canvas=Image.new('RGB',(width,height),(17,23,31));draw=ImageDraw.Draw(canvas)
            for j,(data,label) in enumerate(zip(panels,labels)):
                x=j*384;canvas.paste(Image.fromarray(data[t]).resize((384,384)),(x,40))
                ground=is_gt and j==0
                color=(91,158,245) if ground else ((34,220,90) if t==0 else (245,65,65))
                draw.rectangle((x+1,41,x+382,422),outline=color,width=4)
                draw.text((x+9,5),label,font=font,fill='white')
                draw.text((x+9,24),'REFERENCE' if ground else ('OBSERVED INPUT' if t==0 else 'GENERATED'),font=small,fill=color)
            for n,line in enumerate(lines):draw.text((10,434+n*20),line,font=font,fill='white')
            draw.text((10,height-23),f'{c["id"]} | seed 0 | UniPC35 / CFG6 / shift10 | frame {t}/96',font=small,fill=(165,180,195))
            proc.stdin.write(np.asarray(canvas).tobytes())
            if t in [0,24,48,72,96]:snapshots.append(canvas.copy())
        proc.stdin.close();assert proc.wait()==0
        decoded=_read_video_rgb(video);assert decoded.shape==(97,height,width,3)
        for j in range(1 if is_gt else 0,len(panels)):
            for t,ch in [(0,1),(1,0)]:
                px=decoded[t,60,j*384+2].astype(float);assert px[ch]>px[(ch+1)%3]+40
        contact=Image.new('RGB',(width,height*5));[contact.paste(im,(0,n*height)) for n,im in enumerate(snapshots)]
        contact.save(gallery/(c['id']+'_contact.png'))
        assets.append(dict(case=c['id'],path=str(video),sha256=sha256(video),frames=97,fps=20,
                           kind=c['kind'],caption=c['formatted_caption'],contact=str(gallery/(c['id']+'_contact.png'))))
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Phase 1 I2V comparison</title>'+
        '<style>body{background:#151b24;color:white;font:16px system-ui;max-width:1200px;margin:auto}video{width:100%}</style><h1>Original versus two-task Phase 1, 200k</h1>'+''.join(
        '<h2>'+html.escape(a['case'])+'</h2><p>'+html.escape(a['caption'])+'</p><video controls src="'+Path(a['path']).name+'"></video>' for a in assets))
    save(out/'media.json',dict(assets=assets))
    save(out/'MEDIA_COMPLETE.json',dict(videos=len(assets),fully_decoded=True,observed_generated_borders_verified=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','infer','analyze','render','all'])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();require_compute()
    for name in (['prepare','infer','analyze','render'] if args.command=='all' else [args.command]):
        globals()[name](args.output)


if __name__=='__main__':main()
