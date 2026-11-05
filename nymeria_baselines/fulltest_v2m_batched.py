"""Inference-only batching for the frozen full-test V2M experiment.

The frozen evaluator/configuration remain unchanged. UEM keeps every native
ancestral step, with an independent original random stream for each case.
Phase3 keeps the native UniPC scheduler, timesteps, x0 conversion and seeds.
"""
from collections import OrderedDict
from contextlib import contextmanager
import argparse
import fcntl
import gc
import importlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from nymeria_baselines import fulltest_v2m as frozen


@contextmanager
def independent_uem_noise(model, seeds):
    """Redirect only diffusion's noise draws; never mutate torch/upstream files.

    Drawing a (1,T,D) tensor per generator preserves the B=1 CUDA Philox
    sequence at initialization and at every ancestral step, regardless of B.
    Model construction and the reasoner's/training computation are untouched.
    """
    import torch
    namespace = model.diffusion.p_sample_loop_progressive.__func__.__globals__
    original = namespace['th']
    generators = [torch.Generator(device='cuda').manual_seed(int(s)) for s in seeds]

    class NoiseProxy:
        def __getattr__(self, key):
            return getattr(original, key)

        def randn(self, *shape, **kwargs):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                shape = tuple(shape[0])
            assert shape[0] == len(generators), (shape, len(generators))
            return torch.cat([original.randn((1, *shape[1:]), generator=g, **kwargs)
                              for g in generators], dim=0)

        def randn_like(self, tensor, **kwargs):
            assert tensor.shape[0] == len(generators)
            assert tensor.is_contiguous(), 'Native UEM noise must retain contiguous layout'
            return self.randn(tuple(tensor.shape), device=tensor.device,
                              dtype=tensor.dtype, **kwargs)

    namespace['th'] = NoiseProxy()
    try:
        yield
    finally:
        namespace['th'] = original


def p3_batch(model, *, motion, neutral, video, seeds):
    import torch
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1., use_dynamic_shifting=False)
    scheduler.set_timesteps(35, device=motion.device, shift=3.)
    generators = [torch.Generator(device=motion.device).manual_seed(int(s)) for s in seeds]
    state = torch.cat([torch.randn((1, *motion.shape[1:]), device=motion.device,
                                  dtype=torch.float32, generator=g) for g in generators])
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = timestep.float().reshape(1).expand(len(seeds)) / 1000.
        prediction = model(motion=state, video=video, neutral=neutral,
                           motion_sigma=sigma, video_sigma=sigma * 0,
                           captions=[''] * len(seeds), modes=['video2motion'] * len(seeds))['motion'].float()
        actual_sigma = scheduler.sigmas[index].to(device=motion.device, dtype=torch.float32).clamp_min(1e-6)
        velocity = (state - prediction) / actual_sigma
        # The official UniPC step is deterministic; generator is unused by it.
        state = scheduler.step(velocity, timestep, state, return_dict=False, generator=generators[0])[0].float()
    assert torch.isfinite(state).all()
    return state


class Runner:
    def __init__(self, args):
        import torch
        from nymeria_baselines.body_shape import MOTION_INDICES
        self.args = args
        self.cfg, self.rows = frozen.load_common(args)
        self.spec = next(s for s in self.cfg['models'] if s['label'] == args.model)
        self.root = args.out / 'models' / args.model
        self.root.mkdir(parents=True, exist_ok=True)
        assert frozen.sha(self.spec['checkpoint']) == self.spec['checkpoint_sha256']
        ck = torch.load(self.spec['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        if self.spec['family'] == 'phase3':
            cls = importlib.import_module(self.spec['module'] + '.model').EdgePhase3
            source = self.spec['sources']
            self.model = cls(source['phase1'], source['phase2'], device='cuda', activation_checkpointing=False)
            assert self.model.provenance['phase1_metadata_sha256'] == source['phase1_metadata_sha256']
            assert self.model.provenance['phase2_contract'] == source['phase2_contract']
            self.model.bridges.load_state_dict(ck['bridges'], strict=True)
            contract = self.cfg['phase2_contract']
            self.mean, self.std = np.load(contract['motion_stats_mean']), np.load(contract['motion_stats_std'])
        else:
            sys.path.insert(0, str(frozen.OVERLAY / 'UniEgoMotion' if self.spec['variant'] == 'shape'
                                   else frozen.OLD / 'upstream/UniEgoMotion'))
            from yacs.config import CfgNode
            from module.uem_module import UEM_Module
            hp = CfgNode(ck['hyper_parameters'])
            hp.TRAIN.ONLY_VALIDATE = True
            self.model = UEM_Module(hp)
            self.model.load_state_dict(ck['state_dict'], strict=True)
            self.model.cuda()
            self.guidance = hp.TRAIN.COND_SCALE
            with np.load(frozen.CACHE / 'train_stats.npz') as z:
                self.mean, self.std = z['motion_mean'][MOTION_INDICES], z['motion_std'][MOTION_INDICES]
        self.model.eval().requires_grad_(False)
        del ck
        gc.collect()
        self.features = OrderedDict()
        self.vae = None

    def inputs(self, rows):
        import torch
        from nymeria_baselines.uem_visual_cache import frame_ids
        neutral, conditions = [], []
        for row in rows:
            assert frozen.sha(row['bundle']) == row['bundle_sha256']
            with np.load(row['bundle']) as z:
                neutral.append(z['neutral'].copy())
            if self.spec['family'] == 'phase3':
                path = Path(row['latent'])
                if not path.exists():
                    from motion_expert_phase3_edge.visualize import load_vae, make_pipeline, encode_window
                    from nymeria_camera_dataset import decode_window_pyav
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.with_suffix('.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        if not path.exists():
                            if self.vae is None:
                                self.vae = load_vae(os.environ['WAN_VAE_PATH'], '256', 97, 'cuda')
                            frames = decode_window_pyav(row['rgb'], row['start'], 97, 20.)
                            latent, _ = encode_window(self.vae, make_pipeline(), '256', frames, 'cuda')
                            frozen.npz(path, latents=latent)
                with np.load(path) as z:
                    video = z['latents'].copy()
                assert video.shape == (48, 25, 16, 16) and np.isfinite(video).all()
                conditions.append(video)
            else:
                key = row['array']
                if key not in self.features:
                    self.features[key] = tuple(np.load(frozen.FEATURES / 'records' / f'{key}.{suffix}.npy', mmap_mode='r')
                                               for suffix in ('frame_ids', 'features'))
                while len(self.features) > 4:
                    self.features.popitem(last=False)
                ids, features = self.features[key]
                wanted = frame_ids(row['start'])
                slots = np.searchsorted(ids, wanted)
                assert (slots < len(ids)).all() and np.array_equal(ids[slots], wanted)
                conditions.append(np.array(features[slots], dtype=np.float32))
        shape = torch.from_numpy(np.stack(neutral)).cuda().float()
        condition = torch.from_numpy(np.stack(conditions)).cuda().float()
        if self.spec['family'] == 'phase3':
            return dict(motion=torch.zeros(len(rows), 97, 283, device='cuda'), neutral=shape, video=condition)
        y = dict(valid_frames=torch.ones(len(rows), 97, dtype=torch.long, device='cuda'),
                 img_embs=condition, valid_img_embs=torch.ones(len(rows), 97, dtype=torch.long, device='cuda'),
                 img_mask=torch.zeros(len(rows), 97, dtype=torch.long, device='cuda'))
        if self.spec['variant'] == 'shape':
            y['neutral_joints'] = shape
        return y

    def sample(self, rows, inputs, native=False):
        import torch
        seeds = [r['seed'] for r in rows]
        with torch.inference_mode():
            if self.spec['family'] == 'phase3':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if native:
                        from motion_expert_phase3_edge.sample import sample
                        assert len(rows) == 1
                        return sample(self.model, mode='video2motion', caption='', seed=seeds[0],
                                      steps=35, guidance=1., **inputs)
                    return p3_batch(self.model, seeds=seeds, **inputs)
            if native:
                assert len(rows) == 1
                torch.manual_seed(seeds[0])
                torch.cuda.manual_seed_all(seeds[0])
                return self.model.sample(inputs, B=1, cond_scale=self.guidance)
            with independent_uem_noise(self.model, seeds):
                return self.model.sample(inputs, B=len(rows), cond_scale=self.guidance)

    def decode(self, generated):
        from nymeria_baselines.body_shape import BODY_JOINTS
        from nymeria_baselines.visualize_official_baselines import decode
        phase3 = self.spec['family'] == 'phase3'
        raw = generated * self.std + self.mean if phase3 else generated[:, :220] * self.std + self.mean
        joints, rotations = decode(raw, 30 if phase3 else 23)
        return (joints[:, BODY_JOINTS], rotations[:, BODY_JOINTS]) if phase3 else (joints, rotations)

    def benchmark(self):
        import torch
        # Compare the adapter with the native B=1 sampler before choosing B.
        rows = self.rows[:max(self.args.batch_sizes)]
        inputs = self.inputs(rows[:1])
        native = self.sample(rows[:1], inputs, native=True).float().cpu().numpy()
        adapted = self.sample(rows[:1], inputs).float().cpu().numpy()
        assert np.array_equal(native, adapted), 'B=1 adapter must reproduce the native sampler bit-for-bit'
        receipt = dict(model=self.args.model, device=torch.cuda.get_device_name(),
                       checkpoint_sha256=self.spec['checkpoint_sha256'], source_sha256=frozen.sha(__file__),
                       native_b1_bitwise=True, cases=[r['id'] for r in rows], results=[])
        refs = []
        for row in rows[:self.args.parity_cases]:
            refs.append(self.sample([row], self.inputs([row]), native=True).float().cpu().numpy()[0])
        for size in self.args.batch_sizes:
            try:
                inputs = self.inputs(rows[:size])
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.monotonic()
                output = self.sample(rows[:size], inputs).float().cpu().numpy()
                torch.cuda.synchronize()
                seconds = time.monotonic() - start
                comparisons = []
                for i, ref in enumerate(refs[:size]):
                    refj, _ = self.decode(ref)
                    actualj, _ = self.decode(output[i])
                    comparisons.append(dict(case_id=rows[i]['id'],
                                            max_feature_difference=float(np.max(np.abs(ref-output[i]))),
                                            mean_joint_difference_cm=float(np.linalg.norm(refj-actualj, axis=-1).mean()*100),
                                            max_joint_difference_cm=float(np.linalg.norm(refj-actualj, axis=-1).max()*100)))
                result = dict(batch_size=size, seconds=seconds, seconds_per_case=seconds/size,
                              peak_memory_gib=torch.cuda.max_memory_allocated()/2**30, parity=comparisons)
                # Reject material batch-dependent motion changes. Keep the full
                # differences in the receipt rather than claiming bitwise B>1.
                result['accepted'] = all(x['mean_joint_difference_cm'] <= 0.1 and
                                         x['max_joint_difference_cm'] <= 1.0 for x in comparisons)
                receipt['results'].append(result)
                print(result, flush=True)
                del output, inputs
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                receipt['results'].append(dict(batch_size=size, accepted=False, oom=True))
                break
        accepted = [r for r in receipt['results'] if r['accepted']]
        assert accepted
        best = min(accepted, key=lambda r:r['seconds_per_case'])
        receipt['selected_batch_size'] = best['batch_size']
        frozen.save(self.root / 'batch_benchmark.json', receipt)
        print('SELECTED_BATCH_SIZE=' + str(best['batch_size']), flush=True)

    def run(self):
        import torch
        from nymeria_baselines.body_shape import BODY_JOINTS
        lock = (self.root/'worker.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.root/'INFERENCE_COMPLETE.json').exists():
            return
        batch_size = self.args.batch_size
        if batch_size == 0:
            receipt = frozen.read(self.root/'batch_benchmark.json')
            assert receipt['checkpoint_sha256'] == self.spec['checkpoint_sha256']
            assert receipt['source_sha256'] == frozen.sha(__file__)
            batch_size = receipt['selected_batch_size']
        pending = []
        existing = 0
        for row in self.rows:
            dest = self.root/'predictions'/(row['id']+'.npz')
            if dest.exists():
                with np.load(dest) as z:
                    assert str(z['checkpoint_sha256']) == self.spec['checkpoint_sha256'] and np.isfinite(z['joints']).all()
                existing += 1
            else:
                pending.append(row)
        if self.args.limit:
            pending = pending[:self.args.limit]
        started = time.monotonic()
        new = 0
        for offset in range(0, len(pending), batch_size):
            rows = pending[offset:offset+batch_size]
            output = self.sample(rows, self.inputs(rows)).float().cpu().numpy()
            for row, generated in zip(rows, output):
                joints, rotations = self.decode(generated)
                with np.load(row['bundle']) as z:
                    neutral = z['neutral'][list(BODY_JOINTS)].copy()
                if self.spec['family'] == 'uem' and self.spec['variant'] != 'shape':
                    from nymeria_baselines.body23_tmr_adapter import neutral_from_pca
                    with np.load(frozen.CACHE/'body_shape16.npz') as basis:
                        neutral = neutral_from_pca(generated[:,220:], basis)
                assert joints.shape == (97,23,3) and np.isfinite(joints).all() and np.isfinite(rotations).all()
                frozen.npz(self.root/'predictions'/(row['id']+'.npz'), generated=generated,
                           joints=joints.astype(np.float32), rotations=rotations.astype(np.float32),
                           neutral23=neutral.astype(np.float32), checkpoint_sha256=np.array(self.spec['checkpoint_sha256']),
                           case_id=np.array(row['id']), seed=row['seed'], inference_batch_size=len(rows),
                           batch_adapter_sha256=np.array(frozen.sha(__file__)))
            new += len(rows)
            frozen.progress(self.root, 'sampling_batched', completed=existing+new,
                            total=len(self.rows), new_cases=new, batch_size=batch_size,
                            seconds_per_case=(time.monotonic()-started)/new, step=self.spec['step'])
        if all((self.root/'predictions'/(row['id']+'.npz')).exists() for row in self.rows):
            frozen.save(self.root/'INFERENCE_COMPLETE.json', dict(cases=len(self.rows), step=self.spec['step'],
                        checkpoint_sha256=self.spec['checkpoint_sha256'], batch_size=batch_size,
                        batch_adapter_sha256=frozen.sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['benchmark', 'sample'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=int, default=0, help='0 reads this model GPU benchmark receipt')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--parity-cases', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    frozen.guard()
    runner = Runner(args)
    runner.benchmark() if args.command == 'benchmark' else runner.run()


if __name__ == '__main__':
    main()
