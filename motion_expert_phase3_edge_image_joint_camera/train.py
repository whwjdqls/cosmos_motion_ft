"""Reuse the exact joint trainer, adding camera inputs without a camera loss."""
import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch

from motion_expert_phase3_edge_image_joint.train import configure_local_paths
from motion_expert_phase3_edge_joint.settings import RECIPE, atomic_json, sha256
from .data import Phase3Dataset, collate
from .model import EdgePhase3
from .objectives import noise_batch, specialist_loss
from .policy import configure
from .execution import topology, require_acceptance


def prepare_trainer(probe_updates=0):
    from motion_expert_phase3_edge_joint import train
    configure()
    mapping = configure_local_paths()
    # Retain the existing caption-safe specialist parity reference and bounded
    # disposable probes, without changing the source used by the current run.
    source = inspect.getsource(train.preflight)
    old = 'reasoner_inputs=model.motion.prepare_conditions(batch["captions"])'
    assert source.count(old) == 1
    exec(compile(source.replace(old, 'reasoner_inputs=model.motion.prepare_conditions(inputs["captions"])'),
                 __file__ + ':native_preflight', 'exec'), train.__dict__)
    native_preflight = train.preflight
    source = inspect.getsource(train.main)
    assert source.count('int(STOP)') == 1
    # Single GPU executes the same two 16-sample loss groups sequentially.
    # Require a fresh preflight with the actual GPU count and microbatch;
    # the earlier microbatch4 diagnostic receipt cannot authorize production.
    old_topology = '''    if not args.preflight and world not in (2, 4):
        raise ValueError("Production uses two or four allocated GPUs")'''
    old_acceptance = '''    if (not acceptance.get("passed") or acceptance["fingerprint"] != fingerprint
            or acceptance["world_size"] < 2):
        raise RuntimeError("Missing, stale or non-DDP acceptance gate")'''
    assert source.count(old_topology) == source.count(old_acceptance) == 1
    source = source.replace(old_topology, '    camera_topology(world, args.microbatch)')
    source = source.replace(old_acceptance,
        '    camera_require_acceptance(acceptance, fingerprint, world, args.microbatch)')
    train.camera_topology = topology
    train.camera_require_acceptance = require_acceptance
    train.CAMERA_PROBE_UPDATES = probe_updates
    source = source.replace('int(STOP)', 'int(STOP or (CAMERA_PROBE_UPDATES > 0 and step >= start_step + CAMERA_PROBE_UPDATES))')
    exec(compile(source, __file__ + ':bounded_probe', 'exec'), train.__dict__)
    train.Phase3Dataset, train.collate, train.EdgePhase3 = Phase3Dataset, collate, EdgePhase3
    train.noise_batch, train.specialist_loss = noise_batch, specialist_loss
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        contract = original_identity(args, dataset, model)
        repo = Path(__file__).parents[1]
        contract.update(
            image_joint_implementation={p.name: sha256(p) for p in sorted((repo/'motion_expert_phase3_edge_image_joint').glob('*.py'))},
            camera_prediction_implementation={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            execution_policy=sha256(repo/'scripts/phase3_performance.py'),
            camera_cache=dict(path=str(dataset.camera_root), fingerprint=dataset.camera_index['fingerprint'],
                              index_sha256=sha256(dataset.camera_root/'index.json')),
            local_cache_path_translation=mapping,
            runtime_overlays=dict(preflight_reference='actual_sanitized_captions',
                                  bounded_probe='optional_stop_after_updates_full_schedule_preserved',
                                  execution='one_or_two_GPUs_exact_topology_preflight_required'),
            execution_topology=topology(int(os.environ.get('WORLD_SIZE', '1')), args.microbatch),
        )
        return contract

    train.source_identity = source_identity

    def preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint):
        native_preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint)
        train.barrier()
        path = args.out/'acceptance.json'
        receipt = json.loads(path.read_text())
        train.barrier()
        if rank == 0:
            path.unlink()
        train.barrier()
        rows = [dataset[(rank, rank % len(dataset))]]
        raw = collate(rows)
        model.eval()
        audits = []
        # All three tasks must predict all96 native camera transitions. Their
        # camera outputs are deliberately detached and excluded from the loss.
        for mode in RECIPE['tasks']:
            raw['modes'], raw['captions'] = [mode], ['PRIVATE_CAPTION']
            inputs, target = noise_batch(raw, device)
            with torch.no_grad():
                output = model(**inputs)
                assert output['camera_velocity'].shape == (1,96,64)
                assert torch.isfinite(output['camera_velocity']).all()
                assert not output['camera_velocity'].requires_grad
                assert all(not bool(mask.any()) for mask in output['packed'].action.condition_mask)
                assert sum(len(x) for x in output['packed'].action.noisy_frame_indexes) == 96
                expected_time = (inputs['camera_sigma']*1000).repeat_interleave(96)
                torch.testing.assert_close(output['packed'].action.timesteps.float(), expected_time, rtol=0, atol=0)
                first, terms = specialist_loss(output, target, inputs, mean, std)
                changed = {**output, 'camera_velocity':torch.full_like(output['camera_velocity'],float('nan')),
                           'preds_action':[torch.full_like(x,float('nan')) for x in output['preds_action']]}
                second, _ = specialist_loss(changed, target, inputs, mean, std)
                torch.testing.assert_close(first,second,rtol=0,atol=0)
                assert set(target) == {'motion','velocity'}
                if mode == 'joint':
                    assert inputs['captions'] == ['']
                    poisoned = model(**{**inputs,'captions':['PRIVATE_TARGET_CAPTION']})
                    for key in ['motion','video','camera_velocity']:
                        torch.testing.assert_close(output[key],poisoned[key],rtol=0,atol=0)
                # Disabling only the added camera path is an exact reference to
                # the unchanged image-joint implementation for these inputs.
                disabled = model(**inputs, camera_enabled=False)
                assert disabled['motion'].shape == output['motion'].shape
                audits.append(dict(task=mode,camera_tokens=96,clean_camera_tokens=0,
                    camera_sigma=inputs['camera_sigma'].item(),camera_loss_ignored_bitwise=True,
                    camera_readout_detached=True,camera_prediction_finite=True,
                    motion_target_count=terms['motion_count'].item(),video_target_count=terms['video_count'].item(),
                    video_motion_change_with_camera={k:float((output[k]-disabled[k]).abs().max()) for k in ['motion','video']}))
                del output, disabled, changed
        model.train()
        # A real backward must not acquire a loss edge to a camera prediction.
        raw['modes']=['joint'];raw['captions']=['']
        inputs,target=noise_batch(raw,device)
        output=model(**inputs)
        branch_norms={}
        for key, branch in [('motion',(output['motion']-target['motion']).square().mean()),
                            ('video',(output['video'][:,:,1:]-target['velocity'][:,:,1:]).square().mean())]:
            grads=torch.autograd.grad(branch,tuple(model.bridges.parameters()),allow_unused=True,retain_graph=True)
            norm=sum(g.float().square().sum() for g in grads if g is not None).sqrt()
            assert torch.isfinite(norm) and norm.item()>0,key
            branch_norms[key]=norm.item()
        camera_leaf=output['camera_velocity'].clone().requires_grad_(True)
        loss,_=specialist_loss({**output,'camera_velocity':camera_leaf},target,inputs,mean,std)
        assert torch.autograd.grad(loss,camera_leaf,allow_unused=True,retain_graph=False)[0] is None
        train.assert_frozen(model, train.frozen_versions(model))
        reports=train.all_objects(audits,world)
        branch_reports=train.all_objects(branch_norms,world)
        if rank==0:
            receipt['camera_prediction_audits_by_rank']=reports
            receipt['joint_both_target_bridge_gradient_norms']=branch_reports
            receipt['recipe']=RECIPE.copy()
            receipt['checks'].extend(['camera_predictions_all_three_tasks_no_clean_camera_tokens',
                                     'zero_camera_loss_and_no_camera_readout_gradient',
                                     'joint_caption_invariant_camera_motion_video_outputs'])
            atomic_json(path,receipt)
            print('[phase3/camera-prediction] CAMERA_ACCEPTANCE_PASSED',flush=True)
        train.barrier()

    train.preflight=preflight
    return train


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates',type=int,default=0)
    args,rest=parser.parse_known_args()
    if args.probe_updates<0:
        raise ValueError('Probe updates must be nonnegative')
    if '--conditioning-attention' not in rest:
        rest+=['--conditioning-attention','bidirectional']
    if '--clean-motion-timestep' not in rest:
        rest+=['--clean-motion-timestep','zero']
    train=prepare_trainer(args.probe_updates)
    sys.argv=[sys.argv[0],*rest]
    from scripts.phase3_performance import execution_policy
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with execution_policy(cache_bridge_masks=True,checkpoint_motion=True):
        train.main()


if __name__=='__main__':
    main()
