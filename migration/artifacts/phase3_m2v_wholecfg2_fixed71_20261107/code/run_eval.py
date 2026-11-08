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
    from motion_expert_phase3_edge_joint.validation_metrics import video_scores
    from whole_cfg import Guided

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
                         runner_sha256=sha(__file__), cfg_sha256=sha(ROOT/'whole_cfg.py'))
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
    motion_prepare, video_pack = model.motion.prepare_conditions, model.edge.pack_video
    caption_calls = dict(motion=0, video=0)

    def empty_motion(captions):
        assert not any(captions)
        caption_calls['motion'] += 1
        return motion_prepare(captions)

    def empty_video(captions, modes, latents, sigma):
        assert not any(captions)
        caption_calls['video'] += 1
        return video_pack(captions, modes, latents, sigma)

    model.motion.prepare_conditions, model.edge.pack_video = empty_motion, empty_video
    arrays = {r['id']:load_case(r, contract) for r in records}

    def inputs_for(rows):
        return {name:torch.from_numpy(np.stack([arrays[r['id']][i] for r in rows])).cuda().float()
                for i,name in enumerate(('motion','neutral','video'))}

    def generate(rows, mode, scale, wrapped=True):
        values = inputs_for(rows)
        target = Guided(model, mode, scale) if wrapped else model
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
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
    for mode in ('motimg2video',):
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
        selected = records[0::2][:PLAN['batch_size']]
        progress('cfg1_cached_native_parity', mode=mode, batch=len(selected))
        reproduced = generate(selected, mode, 1.).float().cpu().numpy()
        for row,prediction in zip(selected,reproduced,strict=True):
            with np.load(baseline/mode/(row['id']+'.npz')) as archive:
                assert int(archive['seed']) == row['seed']
                np.testing.assert_array_equal(prediction,archive['generated'])
        acceptance['modes'][mode]['cached_native_cfg1_batch32_bit_exact'] = True
        acceptance['modes'][mode]['cached_native_cases_checked'] = len(selected)
        del reproduced
    acceptance['passed'] = True
    atomic_json(out/'acceptance.json', acceptance)

    batch_history=[]
    for mode in ('motimg2video',):
        completed = 0
        # Match the two-rank native CFG1 validation's per-case batch positions.
        for virtual_rank in (0,1):
            shard = records[virtual_rank::2]
            offset, current = 0, PLAN['batch_size']
            while offset < len(shard):
                selected = shard[offset:offset+current]
                progress('sampling', mode=mode, complete=completed, total=71, batch=len(selected))
                start=time.monotonic()
                try:
                    result = generate(selected, mode, 2.).float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if current == 1: raise
                    current = max(1,current//2)
                    continue
                folder = out/mode
                folder.mkdir(exist_ok=True)
                for row,pred in zip(selected,result,strict=True):
                    tmp = folder/(row['id']+'.tmp.npz')
                    np.savez(tmp, generated=pred, seed=row['seed'], guidance=2.)
                    tmp.replace(folder/(row['id']+'.npz'))
                batch_history.append(dict(virtual_rank=virtual_rank,offset=offset,batch=len(selected),seconds=time.monotonic()-start))
                offset += len(selected)
                completed += len(selected)
                del result
        assert completed == 71
    assert all(p._version == versions[n] for n,p in model.bridges.named_parameters())
    atomic_json(out/'GENERATION_COMPLETE.json',dict(windows=71,predictions=71,caption_calls=caption_calls,weights_unchanged=True,batches=batch_history))
    model.motion.prepare_conditions, model.edge.pack_video = motion_prepare, video_pack
    del model, arrays, motion_prepare, video_pack, empty_motion, empty_video
    gc.collect()
    torch.cuda.empty_cache()
    progress('scoring')
    policy = {**recipe['validation'], 'weights':'ema', 'guidance':2.0, 'decode_batch_size':PLAN['decode_batch_size']}
    with torch.inference_mode():
        results = video_scores(config, records, out, contract, policy)
    atomic_json(out/'summary.json',dict(m2v=results))
    atomic_json(out/'cfg1_summary.json', json.loads((baseline/'summary.json').read_text()))
    atomic_json(out/'COMPLETE.json', dict(windows=71, predictions=71, checkpoint_step=run['step'], weights='ema', guidance=2., metrics=results))
    progress('complete', windows=71)


if __name__ == '__main__':
    main()
