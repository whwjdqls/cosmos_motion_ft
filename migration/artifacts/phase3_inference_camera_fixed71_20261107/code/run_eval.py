"""Isolated fixed71 non-text CFG evaluation; no writes to training outputs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=[r['id'] for r in PLAN['runs']])
    parser.add_argument('partition',choices=['all','v2m','m2v'])
    args = parser.parse_args()
    run = next(r for r in PLAN['runs'] if r['id'] == args.model)
    snapshot, model_out = Path(run['snapshot']), Path(run['out'])
    out=model_out/('part_'+args.partition)
    variants=[v for v in PLAN['variants'] if args.partition=='all' or v['id'].startswith(args.partition)]
    assert out.is_relative_to('/mnt/local')
    out.mkdir(exist_ok=True)
    for suffix in reversed(('motion_expert_t2m_edge', 'motion_expert_joint_attention', 'nymeria_world', '')):
        sys.path.insert(0, str(snapshot / suffix))
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['PHASE3_IMAGE_JOINT_PATH_MAP'] = json.dumps(run['path_map'])
    from motion_expert_phase3_edge_joint.settings import RECIPE, atomic_json, digest_json
    from motion_expert_phase3_edge_image_joint.train import configure_local_paths
    from camera_model import EdgePhase3
    from motion_expert_phase3_edge_joint.adapters import load_trainable_state
    from motion_expert_phase3_edge_joint.sample import sample
    from motion_expert_phase3_edge_joint.validation import read_configuration, load_case
    from motion_expert_phase3_edge_joint.validation_metrics import video_scores, motion_scores
    from whole_cfg import Guided
    from camera_sample import sample as camera_sample, CameraSwitch
    from motion_text import MotionTextScope

    def progress(stage, **extra):
        record = dict(stage=stage, model_id=args.model,partition=args.partition,attention=run['attention'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        atomic_json(out / 'status.json', record)
        print(json.dumps(record), flush=True)

    progress('verifying_inputs')
    training = json.loads((Path(run['run_dir']) / 'contract.json').read_text())
    payload = torch.load(run['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    assert payload['step'] == run['step'] and payload['fingerprint'] == digest_json(training)
    assert payload['scheduler']['recipe'] == training['recipe']
    recipe = payload['scheduler']['recipe']
    assert recipe['caption_policy'] == {'video2motion':'empty', 'motimg2video':'empty'}
    assert recipe['bridge_execution'] == {'backend':'grouped', 'precision':'bf16_bridge'}
    for name, expected in training['implementation'].items():
        assert sha(snapshot / 'motion_expert_phase3_edge_joint' / name) == expected
    assert sha(snapshot / 'motion_expert_joint_attention/modality_bridge.py') == training['shared_bridge']
    assert sha(snapshot / 'scripts/phase3_performance.py') == training['execution_policy']
    assert sha(snapshot / 'motion_expert_phase3_edge_image_joint/model.py') == training['image_joint_model_sha256']
    RECIPE.clear()
    RECIPE.update(recipe)
    configure_local_paths()
    config, records, identity = read_configuration(PLAN['validation_config'])
    assert identity == recipe['validation']['assets']
    baseline = Path(run['run_dir']) / 'validation' / f"step_{run['step']:09d}"
    assert (baseline / 'COMPLETE.json').exists()
    baseline_identity = json.loads((baseline/'identity.json').read_text())
    assert baseline_identity['policy']['weights'] == 'ema'
    ema_metadata = {k:v for k,v in payload['ema'].items() if k != 'state'}
    assert ema_metadata['decay'] == .999
    assert ema_metadata.get('start_step', 0) + ema_metadata['num_updates'] == run['step']
    configuration = dict(run=run, checkpoint_sha256=sha(run['checkpoint']), policy=PLAN,
                         evaluator_assets=identity, training_fingerprint=payload['fingerprint'],
                         sources=payload['sources'], ema_metadata=ema_metadata, selected_weights='ema',
                         physical_gpu=os.environ['EVAL_PHYSICAL_GPU'], baseline_identity=baseline_identity,
                         runner_sha256=sha(__file__), cfg_sha256=sha(ROOT/'whole_cfg.py'),camera_model_sha256=sha(ROOT/'camera_model.py'),camera_native_sha256=sha(ROOT/'camera_native.py'),camera_sampler_sha256=sha(ROOT/'camera_sample.py'),text_scope_sha256=sha(ROOT/'motion_text.py'))
    atomic_json(out / 'configuration.json', configuration)
    acceptance = dict(reference_policy='all_seven_phase3_bridges_disabled', modes={})
    model = EdgePhase3(run['phase1'], run['phase2'], device='cuda', activation_checkpointing=False)
    assert model.provenance['phase1_metadata_sha256'] == payload['sources']['phase1_metadata_sha256']
    assert model.provenance['phase2_contract'] == payload['sources']['phase2_contract']
    load_trainable_state(model, payload, weights='ema')
    for name,module in [('bridges',model.bridges)]:
        for key,value in module.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(),payload['ema']['state'][name][key],rtol=0,atol=0)
    contract = model.provenance['phase2_contract']
    model.requires_grad_(False).eval()
    versions = {n:p._version for n,p in model.bridges.named_parameters()}
    del payload
    text_scope=MotionTextScope(model.motion)
    assert all(not p.requires_grad for p in model.native.parameters())
    camera_frozen_versions={n:p._version for n,p in model.native.named_parameters() if any(x in n for x in ['action2llm','llm2action','action_modality_embed'])}
    assert camera_frozen_versions
    arrays = {r['id']:load_case(r, contract) for r in records}

    def inputs_for(rows):
        return {name:torch.from_numpy(np.stack([arrays[r['id']][i] for r in rows])).cuda().float()
                for i,name in enumerate(('motion','neutral','video'))}

    def generate(rows,mode,scale,*,camera=True,text=False,audit=False,native=False):
        values=inputs_for(rows)
        switched=CameraSwitch(model,camera)
        target=Guided(switched,mode,scale)
        try:
            with text_scope.use([r['caption'] for r in rows] if text else None), torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                if native:
                    assert not camera
                    result=sample(target,mode=mode,caption=['']*len(rows),seed=[r['seed'] for r in rows],steps=35,guidance=1.,**values)
                    result=dict(motion=result if mode=='video2motion' else values['motion'],video=result if mode=='motimg2video' else values['video'],camera=None,audit=None)
                else:
                    result=camera_sample(target,mode=mode,seed=[r['seed'] for r in rows],steps=35,camera_enabled=camera,audit=audit,**values)
        finally:target.close()
        assert target.counters['conditioned_calls']==35
        assert target.counters['reference_calls']==(35 if scale!=1 else 0)
        assert target.counters['reference_bridge_calls']==0
        assert target.counters['conditioned_bridge_calls']==35*7*len(rows)
        return result

    progress('acceptance_checks')
    acceptance['camera_initialization']='independent Gaussian; no GT camera input'
    for mode in sorted({v['mode'] for v in variants}):
        key='motion' if mode=='video2motion' else 'video'
        native=generate(records[:1],mode,1.,camera=False,native=True)[key]
        controlled=generate(records[:1],mode,1.,camera=False)[key]
        torch.testing.assert_close(native,controlled,rtol=0,atol=0)
        del native,controlled
        values=inputs_for(records[:1]);g=torch.Generator(device='cuda').manual_seed(777)
        inputs=dict(**values,motion_sigma=torch.tensor([.7 if mode=='video2motion' else 0.],device='cuda'),
            video_sigma=torch.tensor([.7 if mode=='motimg2video' else 0.],device='cuda'),captions=[''],modes=[mode],
            camera_state=torch.randn(1,96,64,device='cuda',generator=g),camera_sigma=torch.tensor([.7],device='cuda'))
        inputs[key]=torch.randn(values[key].shape,device='cuda',generator=g)
        if mode=='motimg2video':inputs['video'][:,:,0]=values['video'][:,:,0]
        from motion_expert_phase3_edge_joint.runtime import execution_context
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),execution_context(recipe['bridge_execution']):
            on=model(**inputs)
            off=model(**inputs,camera_enabled=False)
            other=model(**{**inputs,'camera_state':-inputs['camera_state']})
            reference=model(**inputs,bridges_enabled=False)
            changed=({**inputs,'video':torch.flip(inputs['video'],dims=[2]),'camera_state':-inputs['camera_state']} if mode=='video2motion'
                else {**inputs,'motion':torch.zeros_like(inputs['motion']),'neutral':torch.zeros_like(inputs['neutral'])})
            reference_changed=model(**changed,bridges_enabled=False)
            torch.testing.assert_close(reference[key],reference_changed[key],rtol=0,atol=0)
            differences=dict(camera_on_vs_off=(on[key].float()-off[key].float()).abs().max().item(),
                camera_state_sensitivity=(on[key].float()-other[key].float()).abs().max().item(),
                bridge_on_vs_off=(on[key].float()-reference[key].float()).abs().max().item())
            assert all(v>0 for v in differences.values()),differences
            assert on['camera_velocity'].shape==(1,96,64) and torch.isfinite(on['camera_velocity']).all()
        selected=records[0::2][:32]
        progress('camera_off_cached_native_parity',mode=mode,batch=32)
        reproduced=generate(selected,mode,1.,camera=False)[key].float().cpu().numpy()
        for row,prediction in zip(selected,reproduced,strict=True):
            with np.load(baseline/mode/(row['id']+'.npz')) as archive:
                assert int(archive['seed'])==row['seed'];np.testing.assert_array_equal(prediction,archive['generated'])
        del reproduced
        acceptance['modes'][mode]=dict(camera_off_sampler_native_bit_exact=True,camera_off_cached32_native_bit_exact=True,
            reference_source_invariance_bit_exact=True,differences=differences,camera_velocity_shape=[1,96,64])
        del values,inputs,on,off,other,reference,reference_changed,changed
    acceptance['passed']=True
    atomic_json(out/'acceptance.json',acceptance)

    generation={}
    for variant in variants:
        folder=model_out/variant['id'];(folder/variant['mode']).mkdir(parents=True,exist_ok=True)
        history=[];completed=0
        shards=([records] if variant['sharding']=='contiguous' else [records[0::2],records[1::2]])
        for shard_index,shard in enumerate(shards):
            offset,current=0,variant['batch']
            while offset<len(shard):
                rows=shard[offset:offset+current]
                progress('sampling',variant=variant['id'],complete=completed,total=71,batch=len(rows))
                start=time.monotonic()
                try:
                    result=generate(rows,variant['mode'],variant['guidance'],text=variant['text'],audit=True)
                    target=result['motion' if variant['mode']=='video2motion' else 'video'].float().cpu().numpy()
                    cameras=result['camera'].float().cpu().numpy();noise=result['audit']
                except torch.cuda.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache()
                    if current==1:raise
                    current=max(1,current//2);progress('oom_backoff',batch=current);continue
                for i,row in enumerate(rows):
                    destination=folder/variant['mode']/(row['id']+'.npz');assert not destination.exists()
                    np.savez(destination,generated=target[i],camera=cameras[i],seed=row['seed'],guidance=variant['guidance'],
                        caption=row['caption'] if variant['text'] else '',initial_noise_sha256=noise['target_initial_sha256'][i],initial_camera_sha256=noise['camera_initial_sha256'][i])
                history.append(dict(shard=shard_index,offset=offset,batch=len(rows),seconds=time.monotonic()-start))
                completed+=len(rows);offset+=len(rows);del result,target,cameras,noise
        assert completed==71
        generation[variant['id']]=dict(predictions=71,batches=history)
        atomic_json(folder/'GENERATION_COMPLETE.json',generation[variant['id']])
    assert all(p._version==versions[n] for n,p in model.bridges.named_parameters())
    assert all(p._version==camera_frozen_versions[n] for n,p in model.native.named_parameters() if n in camera_frozen_versions)
    atomic_json(out/'GENERATION_COMPLETE.json',dict(predictions=71*len(variants),variants=generation,weights_unchanged=True,
        camera_frozen_versions=camera_frozen_versions,motion_text_calls=text_scope.calls,text_audit=text_scope.audit))
    text_scope.close();del text_scope,model,arrays
    gc.collect();torch.cuda.empty_cache()
    results={}
    for variant in variants:
        folder=model_out/variant['id'];progress('scoring',variant=variant['id'])
        with torch.inference_mode():
            if variant['mode']=='video2motion':metrics=motion_scores(config,records,folder,contract)
            else:
                policy={**recipe['validation'],'weights':'ema','guidance':variant['guidance'],'decode_batch_size':4}
                from metric_hub import pinned_metric_hub
                with pinned_metric_hub(config):
                    metrics=video_scores(config,records,folder,contract,policy)
        atomic_json(folder/'summary.json',metrics);results[variant['id']]=metrics
        atomic_json(folder/'COMPLETE.json',dict(predictions=71,metrics=metrics,camera_enabled=True))
        progress('variant_complete',variant=variant['id'],metrics=metrics)
        gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'COMPLETE.json',dict(predictions=71*len(variants),checkpoint_step=run['step'],metrics=results))
    progress('complete')

if __name__=='__main__':main()
