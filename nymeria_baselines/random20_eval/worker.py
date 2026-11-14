"""Finite sharded random20 input preparation, native sampling and reusable scoring."""
from pathlib import Path
import argparse,gc,hashlib,importlib,json,os,sys,time
import numpy as np
import torch
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('shard',type=int);parser.add_argument('--operation',type=Path,required=True);ARGS=parser.parse_args()
R=ARGS.operation.resolve();P=json.loads((R/'plan.json').read_text());RUN=P['run'];SNAP=Path(RUN['snapshot']);STEP=int(RUN['step'])
for suffix in reversed(('motion_expert_t2m_edge','motion_expert_joint_attention','nymeria_world','')):sys.path.insert(0,str(SNAP/suffix))
from motion_expert_phase3_edge_joint.settings import RECIPE,atomic_json,digest_json
from motion_expert_phase3_edge_image_joint.train import configure_local_paths
from motion_expert_phase3_edge_joint.validation import load_case,read_configuration
from motion_expert_phase3_edge_joint import validation_metrics as vm
from whole_cfg import Guided
from motion_text import MotionTextScope

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mapped(value):
 for old,new in RUN['path_map'].items():
  if str(value).startswith(old+'/'):return str(value).replace(old,new,1)
 return str(value)
def progress(stage,**kw):
 x=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),shard=SHARD,**kw);atomic_json(OUT/'status.json',x);print(json.dumps(x),flush=True)
def npz(path,**values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.npz');np.savez(temp,**values);temp.replace(path)

def main():
 global SHARD,OUT
 SHARD=ARGS.shard;OUT=R/'shards'/str(SHARD);OUT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ['PHASE3_IMAGE_JOINT_PATH_MAP']=json.dumps(RUN['path_map']);configure_local_paths()
 manifest=read(P['manifest']);assert sha(P['manifest'])==P['manifest_sha256'];selected=manifest['rows'][SHARD::P['shards']];assert len(selected)==355
 contract=read(R/'phase2_contract.json');training=read(Path(RUN['run_dir'])/'contract.json');payload=torch.load(P['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
 assert payload['step']==STEP and payload['fingerprint']==digest_json(training) and sha(P['checkpoint'])==P['checkpoint_sha256'];assert payload['ema']['num_updates']+payload['ema'].get('start_step',0)==STEP
 for name,h in training['implementation'].items():assert sha(SNAP/'motion_expert_phase3_edge_joint'/name)==h
 assert sha(SNAP/'motion_expert_joint_attention/modality_bridge.py')==training['shared_bridge'];assert sha(SNAP/'scripts/phase3_performance.py')==training['execution_policy']
 if 'image_joint_model_sha256' in training:assert sha(SNAP/'motion_expert_phase3_edge_image_joint/model.py')==training['image_joint_model_sha256']
 RECIPE.clear();RECIPE.update(payload['scheduler']['recipe']);assert RECIPE['tasks'].get('joint',0)==0 and all(x=='empty' for x in RECIPE['caption_policy'].values())
 if RUN.get('policy_module'):importlib.import_module(RUN['policy_module']).configure_checkpoint(payload)
 if any(token in str(RECIPE.get('experiment','')) for token in ('cross_only','crossonly')) and not (RUN.get('policy_module') and RUN.get('sample_module')):raise ValueError('Cross-only checkpoints require their own policy_module and sample_module')
 for field,package in [('image_joint_implementation','motion_expert_phase3_edge_image_joint'),('camera_prediction_implementation','motion_expert_phase3_edge_image_joint_camera'),('camera_directional_nojoint_implementation','motion_expert_phase3_edge_camera_directional_nojoint'),('camera_cross_only_implementation','motion_expert_phase3_edge_camera_cross_only'),('camera_crossonly_implementation','motion_expert_phase3_edge_camera_cross_only')]:
  for name,h in training.get(field,{}).items():assert sha(SNAP/package/name)==h
 old_plan=read(P['source_old_plan']);config,_,asset_identity=read_configuration(old_plan['validation_config']);expected_assets=RECIPE['validation']['assets'];ignored={'config','cases'} if P.get('allow_asset_path_relocation',False) else set();assert {k:v for k,v in asset_identity.items() if k not in ignored}=={k:v for k,v in expected_assets.items() if k not in ignored}
 atomic_json(OUT/'configuration.json',dict(checkpoint_step=STEP,weights='ema',checkpoint_sha256=P['checkpoint_sha256'],manifest_sha256=P['manifest_sha256'],native_metric_assets=asset_identity,runner_sha256=sha(__file__),gpu=os.environ['CUDA_VISIBLE_DEVICES'],training_fingerprint=payload['fingerprint']))
 import config as native_config
 for field,attr in [('motion_stats_mean','MOTION_STATS_MEAN'),('motion_stats_std','MOTION_STATS_STD'),('nymeria_uniego_root','NYMERIA_UNIEGO_ROOT')]:setattr(native_config,attr,mapped(contract[field]))
 from nymeria_joint_dataset import NymeriaJointDataset,decode_window_pyav
 from motion_expert_phase3_edge.visualize import make_pipeline,load_vae,encode_window
 root=Path(P['nymeria_root']);dataset=NymeriaJointDataset(manifest_path=str(root/'video/manifest_video.jsonl'),split_file=str(root/'train_test_split.json'),split='test',num_frames=97,aligned_num_frames=97,fps=20,task_weights={'video2motion':.5,'motimg2video':.5},bones_text2motion_frac=0,cfg_dropout=0,prefer_latents=True,latent_root=str(root/'joint_latents_T97_256tier_256'),train=False,uniego_root=native_config.NYMERIA_UNIEGO_ROOT,floor_calibration_json=str(root/'metadata/floor_calibration.json'),seed=0)
 native=read(P['native_rows']);parent=read(P['parent_cohort']) if P.get('parent_cohort') else {'rows':[]};parent_by_id={r['id']:r for r in parent['rows']};prepared=read(Path(P['prepared_inputs_operation'])/'cases.json') if P.get('prepared_inputs_operation') else [];prepared={r['id']:r for r in prepared};pipe=make_pipeline();progress('loading_vae');vae=load_vae(Path(config['vae']),'256',97,'cuda');model=None;text_scope=None;video_metric=None;embedder=None;all_records=[];versions=None;last_camera=None
 for chunk_index,start in enumerate(range(0,len(selected),64)):
  chunk=OUT/'chunks'/f'{chunk_index:03d}';chunk.mkdir(parents=True,exist_ok=True);rows=selected[start:start+64];records=[]
  if (chunk/'COMPLETE.json').exists():all_records.extend(read(chunk/'cases.json'));continue
  for index,row in enumerate(rows):
   progress('preparing_inputs',chunk=chunk_index,prepared=start+index,total=len(selected))
   if prepared:
    record=prepared[row['id']];assert all(record[k]==row[k] for k in ('id','uuid','start','seed','caption'));load_case(record,contract);records.append(record);continue
   n=native[row['id']];motion,neutral,pad=dataset._load_motion(n['uni'],n['s'],n['off'],avail=97);assert not pad.any() and np.isfinite(motion).all() and np.isfinite(neutral).all()
   delta=0.
   if parent_by_id:
    prior=parent_by_id[row['source_physical_id']];reference=np.load(prior['array'],mmap_mode='r')[prior['row']];delta=float(np.abs(motion-reference).max());assert delta<2e-5,('motion preprocessing mismatch',row['id'],delta)
   folder=R/'inputs'/row['id'];folder.mkdir(parents=True,exist_ok=True);bundle=folder/'input.npz'
   if not bundle.exists():
    frames=decode_window_pyav(n['vis'],n['s'],97,20.);transformed=pipe(dict(video=torch.from_numpy(frames).permute(3,0,1,2).contiguous(),mode='image2video',ai_caption=''),'256');rgb=transformed['video'].permute(1,2,3,0).contiguous().numpy();video,_=encode_window(vae,pipe,'256',frames,'cuda')
    assert rgb.shape==(97,256,256,3) and rgb.dtype==np.uint8 and video.shape==(48,25,16,16) and np.isfinite(video).all()
    npz(bundle,motion=motion,neutral=neutral,video=video,rgb=rgb,motion_mean_sha256=contract['motion_mean'],motion_std_sha256=contract['motion_std'])
   record=dict(row,bundle=str(bundle),bundle_sha256=sha(bundle),floor_offset=n['off'],motion_source=n['uni'],raw_video=n['vis'],motion_preprocessing_max_abs_difference=delta);load_case(record,contract);records.append(record)
  atomic_json(chunk/'cases.json',records);all_records.extend(records);atomic_json(OUT/'cases.json',all_records)
  if model is None:
   progress('loading_ema_model',chunk=chunk_index)
   if RUN.get('camera'):
    import motion_expert_phase3_edge_image_joint_camera.model as camera_module
    from motion_expert_phase3_edge_image_joint_camera.model import EdgePhase3
   else:
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
   from motion_expert_phase3_edge_joint.adapters import load_trainable_state
   sample=importlib.import_module(RUN.get('sample_module','motion_expert_phase3_edge_camera_directional_nojoint.sample' if RUN.get('camera') else 'motion_expert_phase3_edge_joint.sample')).sample
   model=EdgePhase3(RUN['phase1'],RUN['phase2'],device='cuda',activation_checkpointing=False);assert model.provenance['phase2_contract']==contract and model.provenance['phase1_metadata_sha256']==payload['sources']['phase1_metadata_sha256'];load_trainable_state(model,payload,weights='ema');model.requires_grad_(False).eval()
   for k,v in model.bridges.state_dict().items():torch.testing.assert_close(v.detach().cpu(),payload['ema']['state']['bridges'][k],rtol=0,atol=0)
   versions={n:p._version for n,p in model.bridges.named_parameters()};text_scope=MotionTextScope(model.motion);original_pack=camera_module.pack_predictions if RUN.get('camera') else model.edge.pack_video
   def blank_pack(*args,**kwargs):
    captions=args[1] if RUN.get('camera') else args[0];assert not any(captions);return original_pack(*args,**kwargs)
   if RUN.get('camera'):camera_module.pack_predictions=blank_pack
   else:model.edge.pack_video=blank_pack
   del payload
  def generate(batch,mode,guidance,text=False,wrapped=True):
   nonlocal last_camera
   arrays=[load_case(r,contract) for r in batch];inputs={name:torch.from_numpy(np.stack([a[i] for a in arrays])).cuda().float() for i,name in enumerate(('motion','neutral','video'))};target=Guided(model,mode,guidance) if wrapped else model
   try:
    with text_scope.use([r['caption'] for r in batch] if text else None),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     output=sample(target,mode=mode,caption=['']*len(batch),seed=[r['seed'] for r in batch],steps=35,guidance=1.,**inputs,**({'return_all':True} if RUN.get('camera') else {}))
     if RUN.get('camera'):
      last_camera=output['camera'].float().cpu().numpy();assert last_camera.shape==(len(batch),96,64) and np.isfinite(last_camera).all();output=output['motion' if mode=='video2motion' else 'video']
    if wrapped:
     assert target.counters['conditioned_calls']==35 and target.counters['reference_calls']==(35 if guidance!=1 else 0);assert target.counters['reference_bridge_calls']==0 and target.counters['conditioned_bridge_calls']==35*7*len(batch)
    if mode=='motimg2video':torch.testing.assert_close(output[:,:,0],inputs['video'][:,:,0],rtol=0,atol=0)
    assert torch.isfinite(output).all();return output.float().cpu().numpy()
   finally:
    if wrapped:target.close()
  if not (OUT/'ACCEPTANCE.json').exists():
   progress('native_cfg1_parity',chunk=chunk_index)
   for mode in ('video2motion','motimg2video'):np.testing.assert_array_equal(generate(records[:1],mode,1.,wrapped=False),generate(records[:1],mode,1.))
   atomic_json(OUT/'ACCEPTANCE.json',dict(passed=True,step=STEP,ema_tensors_exact=True,cfg1_native_parity_both_modes=True,whole_bridge_disabled_reference=True,reference_bridge_calls=0,all_generator_captions_empty=True))
  for task in P['tasks']:
   folder=chunk/task['id'];(folder/task['mode']).mkdir(parents=True,exist_ok=True);pending=[r for r in records if not (folder/task['mode']/(r['id']+'.npz')).exists()];offset=0;batch_size=P['sampling_batch'][task['mode']]
   while offset<len(pending):
    batch=pending[offset:offset+batch_size];progress('sampling',task=task['id'],chunk=chunk_index,complete=offset,total=len(pending),batch=len(batch))
    try:generated=generate(batch,task['mode'],task['guidance'],task['text'])
    except torch.cuda.OutOfMemoryError:
     gc.collect();torch.cuda.empty_cache()
     if batch_size==1:raise
     batch_size=max(1,batch_size//2);progress('oom_backoff',task=task['id'],batch=batch_size);continue
    for i,(row,value) in enumerate(zip(batch,generated,strict=True)):npz(folder/task['mode']/(row['id']+'.npz'),generated=value,seed=row['seed'],guidance=task['guidance'],caption=row['caption'] if task['text'] else '',checkpoint_step=STEP,weights='ema',manifest_sha256=P['manifest_sha256'],**({'generated_camera':last_camera[i],'camera_guidance':'conditioned_readout_no_CFG_scaling'} if RUN.get('camera') else {}))
    offset+=len(batch)
  gc.collect();torch.cuda.empty_cache();hub=torch.hub.get_dir()
  for task in P['tasks']:
   folder=chunk/task['id'];progress('scoring',task=task['id'],chunk=chunk_index,windows=len(records))
   if (folder/'COMPLETE.json').exists():continue
   if task['mode']=='motimg2video':
    if video_metric is None:video_metric=vm.VideoMetrics(config)
    offset=[0];original_paired=video_metric.paired
    def paired(gt,pred):
     values,a,b=original_paired(gt,pred);row=records[offset[0]];npz(folder/'video_metrics'/(row['id']+'.npz'),**values,fvd_gt=a,fvd_generated=b);offset[0]+=1;return values,a,b
    video_metric.paired=paired;original_class=vm.VideoMetrics;vm.VideoMetrics=lambda _:video_metric
    try:
     with torch.inference_mode():result={'m2v':vm.video_scores(config,records,folder,contract,dict(decode_batch_size=P['decode_batch']),vae=vae)}
    finally:vm.VideoMetrics=original_class;video_metric.paired=original_paired;torch.hub.set_dir(hub)
    assert offset[0]==len(records)
   else:
    sys.path.insert(0,str(SNAP/'shape_aware_TMR'));import nymeria_eval
    if embedder is None:embedder=nymeria_eval.NymeriaTMREmbedder(config['tmr_checkpoint'],config['tmr_stats'],config['text_cache'],'cuda')
    parts=[];texts=[];original_motion=embedder.embed_motion;original_text=embedder.embed_text;original_class=nymeria_eval.NymeriaTMREmbedder
    def capture_motion(*args,**kwargs):
     value=original_motion(*args,**kwargs);parts.append(value.detach().cpu().numpy());return value
    def capture_text(*args,**kwargs):
     value=original_text(*args,**kwargs);texts.append(value.detach().cpu().numpy());return value
    embedder.embed_motion=capture_motion;embedder.embed_text=capture_text;nymeria_eval.NymeriaTMREmbedder=lambda *a,**k:embedder
    try:
     with torch.inference_mode():values=vm.motion_scores(config,records,folder,contract)
    finally:embedder.embed_motion=original_motion;embedder.embed_text=original_text;nymeria_eval.NymeriaTMREmbedder=original_class
    count=(len(records)+31)//32;assert len(parts)==2*count and len(texts)==1;npz(folder/'motion_embeddings.npz',ids=np.array([r['id'] for r in records]),pred=np.concatenate(parts[:count]),gt=np.concatenate(parts[count:]),text=texts[0])
    from metrics_extra import extra_motion_metrics
    values.update(extra_motion_metrics(records,folder,contract,[]));result={'v2m':values}
   atomic_json(folder/'summary.json',result);atomic_json(folder/'COMPLETE.json',dict(windows=len(records),checkpoint_step=STEP,weights='ema',task=task['id'],manifest_sha256=P['manifest_sha256']))
  assert all(p._version==versions[n] for n,p in model.bridges.named_parameters());atomic_json(chunk/'COMPLETE.json',dict(windows=len(records),tasks=3,weights_unchanged=True));progress('chunk_complete',chunk=chunk_index,complete=start+len(records),total=len(selected))
 assert len(all_records)==355;text_scope.close();atomic_json(OUT/'COMPLETE.json',dict(windows=355,tasks=3,predictions=1065,checkpoint_step=STEP,manifest_sha256=P['manifest_sha256']));progress('complete',windows=355)

if __name__=='__main__':
 try:main()
 except BaseException as error:
  if 'OUT' in globals():atomic_json(OUT/'ERROR.json',dict(type=type(error).__name__,error=str(error),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
  raise
