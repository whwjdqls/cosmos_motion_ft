"""Same camera training objectives, with trainable freshly initialized motion."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from motion_expert_phase3_edge_image_joint.train import configure_local_paths
from motion_expert_phase3_edge_joint.settings import RECIPE, atomic_json, sha256
from motion_expert_phase3_edge_image_joint_camera import train as camera_train
from motion_expert_phase3_edge_image_joint_camera.execution import topology, require_acceptance
from motion_expert_phase3_edge_image_joint_camera.objectives import noise_batch, specialist_loss
from .data import Phase3Dataset, collate
from .model import EdgePhase3
from .overlays import function_source, adapt_main, adapt_preflight
from .policy import INITIALIZATION, configure
from .state import trainable_parameters, frozen_versions, assert_frozen, new_optimizer
from .state import checkpoint_state, load_checkpoint_state


def scratch_require_acceptance(receipt, fingerprint, world, microbatch):
    require_acceptance(receipt, fingerprint, world, microbatch)
    if 'scratch_motion_updates_and_full_optimizer_roundtrip' not in receipt.get('checks', []):
        raise RuntimeError('Scratch-motion training requires its own complete-state acceptance')


def prepare_trainer(probe_updates=0):
    from motion_expert_phase3_edge_joint import train
    configure()
    mapping = configure_local_paths()
    # Every overlay is recorded in this package's hashed source. Shared camera
    # forward, noising, losses, dataset and training-loop source stay unchanged.
    original_identity = train.source_identity
    native_source = adapt_preflight(function_source(train, 'preflight'))
    main_source = adapt_main(function_source(train, 'main'))
    base_configure_training = train.configure_training

    def configure_training(args):
        if args.phase3_lora_rank or args.ema_decay or args.bridge_parameterization not in (None, 'shared'):
            raise ValueError('Additional LoRA/EMA/private bridge parameters are for frozen-specialist Phase3, not the scratch recipe')
        if args.bridge_attention_backend not in (None, 'dense') or args.bridge_precision not in (None, 'fp32'):
            raise ValueError('Historical scratch overlays do not support the new bridge execution flags')
        base_configure_training(args)

    train.__dict__.update(
        Phase3Dataset=Phase3Dataset, collate=collate, EdgePhase3=EdgePhase3,
        noise_batch=noise_batch, specialist_loss=specialist_loss,
        trainable_parameters=trainable_parameters, trainable_state=checkpoint_state,
        load_trainable_state=load_checkpoint_state, frozen_versions=frozen_versions,
        assert_frozen=assert_frozen, new_optimizer=new_optimizer,
        configure_training=configure_training,
        camera_topology=topology, scratch_require_acceptance=scratch_require_acceptance,
        SCRATCH_PROBE_UPDATES=probe_updates,
    )
    exec(compile(native_source, __file__+':shared_preflight', 'exec'), train.__dict__)
    native_preflight = train.preflight
    exec(compile(main_source, __file__+':shared_main', 'exec'), train.__dict__)

    def source_identity(args, dataset, model):
        contract = original_identity(args, dataset, model)
        repo = Path(__file__).resolve().parents[1]
        for field, folder in [('image_joint_implementation', 'motion_expert_phase3_edge_image_joint'),
                              ('camera_prediction_implementation', 'motion_expert_phase3_edge_image_joint_camera'),
                              ('scratch_motion_implementation', Path(__file__).parent.name)]:
            contract[field] = {p.name:sha256(p) for p in sorted((repo/folder).glob('*.py'))}
        constructor_modules = ['model', 'layer', 'self_cross', 'attention', 'config', 'ablation_settings', 'motion_heads']
        contract['motion_constructor_implementation'] = {
            str(Path(sys.modules[name].__file__).resolve().relative_to(repo)): sha256(sys.modules[name].__file__)
            for name in constructor_modules}
        contract.update(
            execution_policy=sha256(repo/'scripts/phase3_performance.py'),
            phase2=dict(weights_loaded=False, initialization=INITIALIZATION.copy(),
                architecture_artifact_sha256=sha256(args.phase2),
                contract=model.provenance['phase2_contract']),
            camera_cache=dict(path=str(dataset.camera_root),fingerprint=dataset.camera_index['fingerprint'],
                index_sha256=sha256(dataset.camera_root/'index.json')),
            local_cache_path_translation=mapping,
            execution_topology=topology(int(os.environ.get('WORLD_SIZE', '1')), args.microbatch),
            runtime_overlays=dict(preflight_reference='actual_sanitized_captions',
                bounded_probe='optional_stop_after_updates_full_schedule_preserved',
                training='motion_plus_bridges_optimizer_clip_checkpoint_full_state',
                frozen_checks='same_cosmos_reasoner_and_generator_only',
                acceptance='same_camera_audits_plus_scratch_initialization_updates_and_complete_state'),
        )
        return contract
    train.source_identity = source_identity

    # Reuse the camera experiment's complete audits without maintaining a
    # second, diverging copy. Only their native preflight dependency differs.
    namespace = dict(vars(camera_train), train=train, native_preflight=native_preflight)
    exec(compile(function_source(camera_train, 'preflight', parent='prepare_trainer'),
                 __file__+':unchanged_camera_audits', 'exec'), namespace)
    camera_preflight = namespace['preflight']

    def preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint):
        assert not model.provenance['phase2_weights_loaded']
        initial_motion = model.provenance['initial_motion_sha256']
        camera_preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint)
        train.barrier()
        path = args.out/'acceptance.json'
        receipt = json.loads(path.read_text())
        if rank == 0:
            path.unlink()
        train.barrier()
        from .model import state_digest
        trained_motion = state_digest(model.motion.motion_state_dict())
        assert initial_motion != trained_motion, 'Fresh motion parameters did not update'
        assert model.motion.heads.llm2motion.weight.count_nonzero().item() > 0
        assert all(p.requires_grad for p in model.motion.parameters())
        assert_frozen(model, frozen_versions(model))
        states = train.all_objects(dict(initial_motion=initial_motion, trained_motion=trained_motion,
            phase2_weights_loaded=False, initialization=model.provenance['motion_initialization']), world)
        assert len({s['initial_motion'] for s in states}) == len({s['trained_motion'] for s in states}) == 1
        if rank == 0:
            receipt['scratch_motion_audits_by_rank'] = states
            receipt['preflight_lr_schedule'] = {
                'policy': 'unchanged_production_1000_step_warmup_then_cosine',
                'optimizer_update_lrs': [train.lr_for_step(i) for i in range(1,14)]}
            receipt['checks'].append('scratch_motion_updates_and_full_optimizer_roundtrip')
            receipt['production_weights'] = 'fresh_motion_and_zero_gated_bridges_not_disposable_preflight_weights'
            atomic_json(path, receipt)
            print('[phase3/scratch] SCRATCH_MOTION_ACCEPTANCE_PASSED', flush=True)
        train.barrier()
    train.preflight = preflight
    return train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates', type=int, default=0)
    args, rest = parser.parse_known_args()
    if args.probe_updates < 0:
        raise ValueError('Probe updates must be nonnegative')
    for flag, value in [('--conditioning-attention', 'bidirectional'), ('--clean-motion-timestep', 'zero')]:
        if flag not in rest:
            rest += [flag, value]
    trainer = prepare_trainer(args.probe_updates)
    sys.argv = [sys.argv[0], *rest]
    from scripts.phase3_performance import execution_policy
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with execution_policy(cache_bridge_masks=True, checkpoint_motion=True):
        trainer.main()


if __name__ == '__main__':
    main()
