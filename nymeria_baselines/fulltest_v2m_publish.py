"""Update only this sweep's blocks on the user-selected Notion Ablations page."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import hashlib
import runpy
import time
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

PAGE='40dd2a44-f44b-8080-aad3-cda6b5fea7c8'
LABELS={'GT':'GT','p3_baseline':'Phase3 original baseline','p3_joint':'Phase3 joint generation',
 'p3_camera':'Phase3 camera-conditioned tasks','p3_fourtask':'Phase3 four-task control',
 'p3_hidden_film':'Phase3 hidden-state FiLM','p3_floor18_film':'Phase3 floor-aware18D FiLM',
 'uem_rgb':'UEM RGB-only reconstruction','uem_recon':'UEM RGB+trajectory reconstruction: RGB input only',
 'uem_all':'UEM all-task: RGB input only','uem_shape':'UEM supplied skeleton, three-mode reconstruction: RGB input only'}

def main(a):
 publication_lock=(a.out/'publication.lock').open('a')
 fcntl.flock(publication_lock,fcntl.LOCK_EX)
 h=runpy.run_path('/scratch2/whwjdqls99/cosmos/operations/phase1_notext_fourway_55k_20261019/upload_notion.py')
 api=h['Notion'](h['read_token']());rt=h['rich_text'];plain=h['plain_text']
 cfg=read(a.out/'configuration.json') if (a.out/'configuration.json').exists() else None
 specs={r['label']:r for r in cfg['models']} if cfg else {}
 excluded=set(read(a.out/'EXCLUDED_MODELS.json')['excluded_models']) if (a.out/'EXCLUDED_MODELS.json').exists() else set()
 header=['Model','Step','Shape','Cases scored','MPJPE cm ↓','PA-MPJPE cm ↓','Root ADE cm ↓','Skate cm/s ↓','Foot penetration cm ↓','FID ↓','R@3 % ↑','Status']
 rows=[header];all_complete=True
 for key,label in LABELS.items():
  spec=specs.get(key,{})
  summary=a.out/'scores'/key/'summary.json';done=a.out/'scores'/key/'COMPLETE.json'
  if key in excluded:
   n=0;vals=['—']*7;status='Excluded by user; partial outputs retained'
  elif done.exists():
   result=read(summary);assert read(done)['summary_sha256']==sha(summary)
   v=result['physical'];r=result['retrieval'];n=result['cases']
   vals=[f'{v["mpjpe_m"]*100:.3f}',f'{v["pa_mpjpe_m"]*100:.3f}',f'{v["root_err_m"]*100:.3f}',
       f'{v["foot_skate_cm_s"]:.3f}',f'{v["foot_depth_mean_cm"]:.3f}',f'{result["FID"]:.6f}',
       f'{r["R32_R03_mean"]:.2f} ± {r["R32_R03_std"]:.2f}'];status='Complete'
  else:
   all_complete=False;n=0;vals=['—']*7;status='Queued'
   for p in [a.out/'models'/key/'progress.json',a.out/'scores'/key/'progress.json']:
    if p.exists():
     x=read(p);status=x['stage']+': '+str(x.get('completed',0))+'/'+str(x.get('total',11911))
     if x['stage'].startswith('sampling'):
      # Limited acceptance runs count only new cases. Count saved outputs for
      # the published status so a two-case probe cannot look like completion.
      count=sum(1 for _ in (a.out/'models'/key/'predictions').glob('*.npz'))
      status=x['stage']+': '+str(count)+'/'+str(x.get('total',11911))
   if (a.out/(key+'.failed')).exists():status='Failed; see worker logs (no final metric claimed)'
   if cfg is None:status='Preparing frozen checkpoints and all-window manifest'
  shape='Known' if key=='GT' or key.startswith('p3_') or key=='uem_shape' else 'Predicted'
  rows.append([label,str(spec.get('step','—')),shape,str(n),*vals,status])
 status=('COMPLETE' if all_complete else 'IN PROGRESS')+' — full-test V2M comparison, updated '+time.strftime('%Y-%m-%d %H:%M:%S %Z')+'. '
 status+='11,911 caption/window cases; 11,677 physical windows across all71 held-out recordings. No 71-window subsampling. Partial rows are not final metrics.'
 if excluded:status+=' Excluded by user: '+', '.join(LABELS.get(k,k) for k in sorted(excluded))+'.'
 if (a.out/'MINSU_RESERVED.json').exists():
  remote_done=sum((a.out/'scores'/key/'COMPLETE.json').exists() for key in ('uem_recon','uem_all'))
  status+=f' Two UEM models (recon/all) are assigned to minsu; {remote_done}/2 have completed scoring. Batches are benchmarked without changing sampling steps or per-case noise streams.'
 details='Video-only inference, empty captions and no camera trajectories or head inpainting. Phase3 retains its provided neutral skeleton; original UEM predicts shape; new UEM receives the full centered SOMA30 skeleton. Phase3 UniPC35/shift3/guidance1; UEM native1000-step diffusion with its saved guidance setting. Checkpoint steps and task exposures differ. Training continues independently.'
 methods='Common23 body joints; raw motion, no prediction-dependent floor shift or alignment except PA-MPJPE. Nymeria-trained TMR and its own pinned normalization; the existing training-only Body23→SOMA30 completion applies to all models and GT. R@1/2/3 uses32 paired candidates and three grouping seeds0/1/2; FID uses all cases. Both window-weighted and recording-balanced physical metrics are saved. Trajectory-only UEM cannot perform trained V2M and is excluded.'
 statefile=a.out/'notion_publication.json';state=read(statefile) if statefile.exists() else {}
 def block(kind,text):return dict(object='block',type=kind,**{kind:dict(rich_text=rt(text))})
 if not state:
  blocks=[block('heading_2','Full test V2M: Phase3 and UniEgoMotion ablations (2026-11-04)'),block('paragraph',status),block('paragraph',details),block('paragraph',methods),
      block('paragraph','Artifacts, checkpoint/data identities, predictions, per-case metrics and retrieval groups: '+str(a.out)),
      dict(object='block',type='table',table=dict(table_width=len(header),has_column_header=True,has_row_header=False,
          children=[dict(object='block',type='table_row',table_row=dict(cells=[rt(x) for x in r])) for r in rows]))]
  response=api.request('PATCH','/blocks/'+PAGE+'/children',dict(children=blocks))
  ids=[b['id'] for b in response['results']]
  actual=api.request('GET','/blocks/'+ids[-1]+'/children?page_size=100')['results']
  state=dict(page=PAGE,url='https://app.notion.com/p/binne/Ablations-'+PAGE.replace('-',''),status_block=ids[1],table=ids[-1],rows=[b['id'] for b in actual])
  save(statefile,state)
 else:
  api.request('PATCH','/blocks/'+state['status_block'],dict(paragraph=dict(rich_text=rt(status))))
  current_ids={b['id'] for b in api.request('GET','/blocks/'+state['table']+'/children?page_size=100')['results']}
  for uid,row in zip(state['rows'],rows,strict=True):
   if uid not in current_ids:continue  # Preserve rows removed manually in Notion.
   api.request('PATCH','/blocks/'+uid,dict(table_row=dict(cells=[rt(x) for x in row])))
 actual=api.request('GET','/blocks/'+state['table']+'/children?page_size=100')['results']
 assert len(state['rows'])==len(rows) and len(set(state['rows']))==len(rows)
 by_id={b['id']:b for b in actual}
 expected_by_id=dict(zip(state['rows'],rows,strict=True))
 for uid in set(state['rows'])&set(by_id):
  assert [plain(c) for c in by_id[uid]['table_row']['cells']]==expected_by_id[uid]
 state['manually_removed_row_ids']=[uid for uid in state['rows'] if uid not in by_id]
 # Other page editors may append rows; preserve those unrelated blocks.
 state['unmanaged_table_row_ids']=[b['id'] for b in actual if b['id'] not in set(state['rows'])]
 assert plain(api.request('GET','/blocks/'+state['status_block'])['paragraph']['rich_text'])==status
 state.update(verified_time=time.time(),complete=all_complete);save(statefile,state)
 with (a.out/'comparison.csv').open('w') as f:csv.writer(f).writerows(rows)
 if all_complete:save(a.out/'COMPLETE.json',dict(models=len(LABELS)-1-len(excluded&set(LABELS)),excluded_models=sorted(excluded),cases=11911,notion_verified=True))
 print(json.dumps(dict(url=state['url'],complete=all_complete,verified=True)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
