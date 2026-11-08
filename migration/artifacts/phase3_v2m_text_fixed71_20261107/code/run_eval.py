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
    args = parser.parse_args()
    run = next(r for r in PLAN['runs'] if r['id'] == args.model)
    snapshot, out = Path(run['snapshot']), Path(run['out'])
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
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
    from motion_expert_phase3_edge_joint.adapters import load_trainable_state
    from motion_expert_phase3_edge_joint.sample import sample
    from motion_expert_phase3_edge_joint.validation import read_configuration, load_case
    from motion_expert_phase3_edge_joint.validation_metrics import motion_scores, physical
    from whole_cfg import Guided
    from motion_text import MotionTextScope

    def progress(stage, **extra):
        record = dict(stage=stage, model_id=args.model, attention=run['attention'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
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
                         runner_sha256=sha(__file__), cfg_sha256=sha(ROOT/'whole_cfg.py'),text_scope_sha256=sha(ROOT/'motion_text.py'))
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
    text_scope = MotionTextScope(model.motion)
    video_pack = model.edge.pack_video
    caption_calls = dict(video=0)
    def empty_video(captions, modes, latents, sigma):
        assert not any(captions), 'Video expert must remain caption-free'
        caption_calls['video'] += 1
        return video_pack(captions, modes, latents, sigma)
    model.edge.pack_video = empty_video
    arrays = {r['id']:load_case(r, contract) for r in records}

    def inputs_for(rows):
        return {name:torch.from_numpy(np.stack([arrays[r['id']][i] for r in rows])).cuda().float()
                for i,name in enumerate(('motion','neutral','video'))}

    def generate(rows, mode, scale, wrapped=True, text=False):
        values = inputs_for(rows)
        target = Guided(model, mode, scale) if wrapped else model
        try:
            with text_scope.use([r['caption'] for r in rows] if text else None), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = sample(target, mode=mode, caption=['']*len(rows), seed=[r['seed'] for r in rows],
                                steps=35, guidance=1., **values)
        finally:
            if wrapped: target.close()
        if wrapped:
            assert target.counters['conditioned_calls'] == 35
            assert target.counters['reference_calls'] == (35 if scale != 1 else 0)
            assert target.counters['reference_bridge_calls'] == 0
            assert target.counters['conditioned_bridge_calls'] == 35*7*len(rows)
        if mode == 'motimg2video':
            torch.testing.assert_close(result[:, :, 0], values['video'][:, :, 0], rtol=0, atol=0)
        assert torch.isfinite(result).all()
        return result

    progress('acceptance_checks')
    for mode in ('video2motion',):
        native, wrapped = generate(records[:1], mode, 1., False), generate(records[:1], mode, 1.)
        torch.testing.assert_close(native, wrapped, rtol=0, atol=0)
        del native, wrapped
        values = inputs_for(records[:1])
        generator = torch.Generator(device='cuda').manual_seed(19431)
        ms = .7 if mode == 'video2motion' else 0.
        vs = .7 if mode == 'motimg2video' else 0.
        inputs = dict(**values, motion_sigma=torch.tensor([ms],device='cuda'),
                      video_sigma=torch.tensor([vs],device='cuda'), captions=[''], modes=[mode])
        key = 'motion' if mode == 'video2motion' else 'video'
        inputs[key] = torch.randn(values[key].shape, device='cuda', generator=generator)
        if mode == 'motimg2video':
            inputs['video'][:, :, 0] = values['video'][:, :, 0]
        changed = ({**inputs, 'video':torch.flip(inputs['video'], dims=[2])} if mode == 'video2motion'
                   else {**inputs, 'motion':torch.zeros_like(inputs['motion']), 'neutral':torch.zeros_like(inputs['neutral'])})
        from motion_expert_phase3_edge_joint.runtime import execution_context
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), execution_context(recipe['bridge_execution']):
            conditional = model(**inputs)[key]
            null = model(**inputs, bridges_enabled=False)[key]
            null_changed = model(**changed, bridges_enabled=False)[key]
            torch.testing.assert_close(null, null_changed, rtol=0, atol=0)
            difference = (conditional.float()-null.float()).abs().max().item()
            assert difference > 0
        acceptance['modes'][mode] = dict(cfg1_native_bit_exact=True, reference_source_invariance_bit_exact=True,
                                        condition_reference_max_abs_difference=difference)
        del inputs, changed, conditional, null, null_changed, values
        # Reproduce the first native validation batch at identical batch positions.
        selected = records[0::2][:32]
        progress('cfg1_cached_native_parity', mode=mode, batch=len(selected))
        reproduced = generate(selected, mode, 1.).float().cpu().numpy()
        for row,prediction in zip(selected,reproduced,strict=True):
            with np.load(baseline/mode/(row['id']+'.npz')) as archive:
                assert int(archive['seed']) == row['seed']
                np.testing.assert_array_equal(prediction,archive['generated'])
        acceptance['modes'][mode]['cached_native_cfg1_batch32_bit_exact'] = True
        acceptance['modes'][mode]['cached_native_cases_checked'] = len(selected)
        del reproduced
    # Confirm captions affect the frozen motion reasoner while generator captions stay empty.
    values = inputs_for(records[:1])
    values['motion'] = torch.randn_like(values['motion'])
    inputs = dict(**values, motion_sigma=torch.tensor([.7],device='cuda'),
                  video_sigma=torch.tensor([0.],device='cuda'),captions=[''],modes=['video2motion'])
    from motion_expert_phase3_edge_joint.runtime import execution_context
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16), execution_context(recipe['bridge_execution']):
        plain = model(**inputs)['motion'].float()
        with text_scope.use([records[0]['caption']]):
            textual = model(**inputs)['motion'].float()
            text_reference = model(**inputs,bridges_enabled=False)['motion'].float()
        with text_scope.use(['A person is swimming underwater.']):
            alternate = model(**inputs)['motion'].float()
        blank_reference = model(**inputs,bridges_enabled=False)['motion'].float()
    differences = dict(text_vs_empty=(textual-plain).abs().max().item(),
                       text_vs_alternate=(textual-alternate).abs().max().item(),
                       reference_text_vs_empty=(text_reference-blank_reference).abs().max().item())
    assert all(value>0 for value in differences.values()), differences
    assert text_scope.current is None
    acceptance['text_conditioning'] = dict(route='frozen_motion_expert.prepare_conditions only',
        both_cfg_passes_keep_same_caption=True,video_captions_empty=True,differences=differences,
        encoded=text_scope.audit)
    del values,inputs,plain,textual,text_reference,alternate,blank_reference
    acceptance['passed'] = True
    atomic_json(out/'acceptance.json', acceptance)

    # Identical batches and case seeds for each paired CFG2 variant.
    offset,current=0,PLAN['batch_size']
    batch_history=[]
    while offset<len(records):
        selected=records[offset:offset+current]
        generated={}
        start=time.monotonic()
        try:
            for variant,text_enabled in [('video_cfg2',False),('video_text_cfg2',True)]:
                progress('sampling',variant=variant,complete=offset,total=71,batch=len(selected))
                generated[variant]=generate(selected,'video2motion',2.,text=text_enabled).float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            generated.clear();gc.collect();torch.cuda.empty_cache()
            if current==1:raise
            current=max(1,current//2)
            progress('oom_backoff',batch=current)
            continue
        for variant,results in generated.items():
            folder=out/variant/'video2motion';folder.mkdir(parents=True,exist_ok=True)
            for row,pred in zip(selected,results,strict=True):
                temporary=folder/(row['id']+'.tmp.npz')
                np.savez(temporary,generated=pred,seed=row['seed'],guidance=2.,
                         caption=row['caption'] if variant=='video_text_cfg2' else '')
                temporary.replace(folder/(row['id']+'.npz'))
        batch_history.append(dict(offset=offset,batch=len(selected),both_variants_seconds=time.monotonic()-start))
        offset+=len(selected)
        del generated,results
    assert all(p._version==versions[n] for n,p in model.bridges.named_parameters())
    assert text_scope.current is None
    atomic_json(out/'GENERATION_COMPLETE.json',dict(windows=71,predictions=142,
        caption_calls=caption_calls,motion_text_calls=text_scope.calls,encoded=text_scope.audit,
        weights_unchanged=True,batches=batch_history))
    # Measure GT physics with exactly the same reconstruction conventions.
    from runtime_paths import resolve_legacy_path
    from motion_expert_phase3_edge_joint.validation import local_path
    mean,std = [np.load(local_path(resolve_legacy_path(contract[k]))).astype(np.float32)
                for k in ('motion_stats_mean','motion_stats_std')]
    gt_rows=[]
    for row in records:
        gt=arrays[row['id']][0]*std+mean
        gt_rows.append(dict(id=row['id'],**physical(gt,gt)))
    atomic_json(out/'gt_physical_per_case.json',gt_rows)
    atomic_json(out/'gt_physical_summary.json',{key:float(np.mean([r[key] for r in gt_rows]))
                for key in gt_rows[0] if key!='id'})
    text_scope.close()
    model.edge.pack_video=video_pack
    del model,arrays,video_pack,text_scope
    gc.collect();torch.cuda.empty_cache()
    progress('scoring')
    all_metrics={}
    for variant in PLAN['variants']:
        folder=out/variant
        with torch.inference_mode():
            results=motion_scores(config,records,folder,contract)
        atomic_json(folder/'summary.json',results)
        all_metrics[variant]=results
        progress('variant_scored',variant=variant,metrics=results)
        gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'cfg1_native_summary.json',json.loads((baseline/'summary.json').read_text()))
    atomic_json(out/'COMPLETE.json',dict(windows=71,predictions=142,checkpoint_step=run['step'],
        weights='ema',guidance=2.,metrics=all_metrics))
    progress('complete', windows=71)


if __name__ == '__main__':
    main()
