"""Build an explicit provisional/final local report without external credentials."""
import argparse
import csv
import io
import json
from pathlib import Path
import time
import zipfile
from native_motion_phase1.evaluate_fulltest import save_json


def build(out):
    cfg=json.loads((out/'configuration.json').read_text())
    summaries={}
    for task in ['motion2video','video2motion','text2motion','textimg2motion']:
        p=out/'metrics'/(task+'_summary.json')
        if p.exists():summaries[task]=json.loads(p.read_text())
    complete=len(summaries)==4 and all(s.get('complete',False) for s in summaries.values())
    lines=['# Naeive adaptation — full Nymeria held-out evaluation','',
        '**Status: '+('reconstruction/video scoring complete; Nymeria TMR still pending.' if complete else 'RUNNING — provisional results; full-test scoring is incomplete.')+'**','',
        f"Final EMA200k. {cfg['eligible_cases']} annotation cases / {cfg['physical_windows']} unique windows / {cfg['recordings']} recordings per task.",
        'Image+camera→video and video→camera: unsupported by the replaced motion heads.','',
        'UniPC30, shift10, CFG1, T97,20fps,256²; first image retained for M2V/TI2M. All video metrics exclude observed RGB frame0.','',
        'Each task table reports its currently scored coverage; these partial cohorts may differ. Do not interpret a partial task comparison as the full-test result.','']
    for task,s in summaries.items():
        lines += ['## '+task,'',f"Scored {s['n']} / {s['expected']} cases. Complete: {s['complete']}.",'',
                  '| Metric | Generated mean | Matched GT mean |','| --- | ---: | ---: |']
        stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['metric','model_mean','GT_mean','model_median','model_std'])
        for key,value in s['aggregate']['model'].items():
            if not isinstance(value,dict) or 'mean' not in value:continue
            gt=s['aggregate']['GT'].get(key,'unavailable')
            gt=gt['mean'] if isinstance(gt,dict) else gt
            lines.append(f"| {key} | {value['mean']:.6g} | {gt:.6g} |" if isinstance(gt,(int,float)) else f"| {key} | {value['mean']:.6g} | {gt} |")
            writer.writerow([key,value['mean'],gt,value.get('median'),value.get('std')])
        if task=='motion2video':
            for key in ['FVD_I3D','CD_FVD_VideoMAE_v2']:
                lines.append(f"| {key} | {s.get(key,'pending')} | 0 (identity reference) |")
                writer.writerow([key,s.get(key),'0 (identity reference)','',''])
        else:lines.append('| Nymeria TMR FID / R@1,2,3 / matching distance / diversity | Pending evaluator bundle | Pending measured GT retrieval |')
        (out/(task+'_metrics.csv')).write_text(stream.getvalue())
        lines.append('')
    lines += ['## Interpretation and limitations','',
        'Motion metrics use raw decoded SOMA30 coordinates and additionally the fixed Body23 subset, with no generated grounding or trajectory reset. PA-MPJPE alone uses a whole-clip similarity fit. Text-conditioned reconstruction distances are paired diagnostics, since many motions can satisfy a caption.',
        'GT physics is measured on the identical cases. GT-self FVD=0 is an identity reference; GT PSNR is mathematically infinite. Nymeria TMR FID and retrieval require the selected Nymeria evaluator, statistics and cached captions; no BONES substitute is used.',
        'Full coverage, rejection reasons, per-case seeds, exact checkpoint/source fingerprints and per-case metrics are retained locally.','']
    (out/'report.md').write_text('\n'.join(lines))
    receipt=dict(reconstruction_video_complete=complete,tmr_complete=False,coverage={k:dict(n=s['n'],expected=s['expected'],complete=s['complete']) for k,s in summaries.items()},time=time.time())
    save_json(out/'report_status.json',receipt)
    temp=out/'quantitative_evidence.tmp'
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['report.md','report_status.json','configuration.json','rejections.json','video_evaluator.json','metric_copy_verified.json','environment.freeze.txt']:
            p=out/name
            if p.exists():z.write(p,name)
        for p in out.glob('*_metrics.csv'):z.write(p,p.name)
        for p in (out/'metrics').glob('*_summary.json'):z.write(p,'summaries/'+p.name)
    temp.replace(out/'quantitative_evidence.zip')
    print(json.dumps(receipt),flush=True)
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    while True:
        done=build(a.out)
        if done or not a.follow:break
        time.sleep(60)
