"""Two-stage text/image/video/body rollout, exported as a portable demo bundle."""
import argparse
import datetime as dt
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from .camera import sha256, validate_pose
from .profile import make_record, verify
from .rollout import atomic_json, load_output, find_video
from .runtime import checked_command, evaluation_environment, official_command, require_compute

NAMES = ['Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head','Jaw','LeftEye','RightEye',
         'LeftShoulder','LeftArm','LeftForeArm','LeftHand','LeftHandThumbEnd','LeftHandMiddleEnd',
         'RightShoulder','RightArm','RightForeArm','RightHand','RightHandThumbEnd','RightHandMiddleEnd',
         'LeftLeg','LeftShin','LeftFoot','LeftToeBase','RightLeg','RightShin','RightFoot','RightToeBase']
PARENTS = [-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28]


def progress(out, stage, **values):
    record = dict(stage=stage, at=dt.datetime.now().astimezone().isoformat(), **values)
    atomic_json(out / 'progress.json', record)
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    require_compute()
    cfg = json.loads(os.path.expandvars(args.config.read_text()))
    out = Path(cfg['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'rollout.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if cfg['prompts'] != ['The camera wearer walks forward.', 'The camera wearer sits down on a chair.']:
        if not cfg['prompts'] or any(not text.strip() for text in cfg['prompts']):
            raise ValueError('Each segment needs a prompt')
    profile = json.loads(Path(cfg['profile']).read_text())
    verify(profile)
    framework_revision = subprocess.check_output(
        ['git', '-C', os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'], 'rev-parse', 'HEAD'], text=True).strip()
    if framework_revision != profile['framework_revision']:
        raise ValueError('Framework revision differs from evaluated profile')
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    configuration = out / 'configuration.json'
    if configuration.exists() and json.loads(configuration.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Choose a fresh directory for a changed configuration')
    assets = out / 'assets'
    assets.mkdir(exist_ok=True)
    for key, name in [('scene','scene.gs.ply'), ('navmesh','scene.navmesh'), ('camera','camera_rays.npz'),
                      ('head_calibration','head_camera_calibration_train.json'), ('profile','phase1_evaluation_profile.json'),
                      ('motion_reference','phase3_evaluation_reference.json'), ('caption_implementation','caption_conditioning_reference.py')]:
        destination = assets / name
        if not destination.exists():
            shutil.copy2(cfg[key], destination)
        if sha256(destination) != sha256(cfg[key]):
            raise ValueError('Archived asset differs: ' + key)
    camera_metadata = Path(cfg['camera']).with_suffix('.json')
    shutil.copy2(camera_metadata, assets / 'camera_calibration_metadata.json')
    # Equal recording weight, using only the immutable training-cache shapes.
    shape = assets / 'neutral_shape.npy'
    if not shape.exists():
        cache = Path(cfg['shape_cache'])
        records = json.loads((cache / 'index.json').read_text())['rows']
        sources = sorted({row['array'] for row in records})
        shapes = [np.load(cache / (name + '.neutral.npy')).astype(np.float64) for name in sources]
        if any(value.shape != (30, 3) or not np.isfinite(value).all() for value in shapes):
            raise ValueError('Invalid training shape')
        mean = np.stack(shapes).mean(0)
        mean -= mean.mean(0, keepdims=True)
        np.save(shape, mean.astype(np.float32))
        atomic_json(assets / 'neutral_shape_provenance.json', dict(source='mean centered SOMA30 training-cache neutral joints',
            cache=str(cache), cache_index_sha256=sha256(cache / 'index.json'), recordings=len(sources),
            source_arrays=sources, weighting='one vote per cached recording; no held-out motion or shape'))
    atomic_json(assets / 'skeleton.json', dict(joint_names=NAMES, parents=PARENTS, head_index=6,
        root_index=0, foot_indices=[24,25,28,29], handedness='right', world_up='+Y', neutral_shape='neutral_shape.npy'))
    source_hashes = {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    source_version = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:16]
    source_dir = assets / 'application_sources' / source_version
    source_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        if not (source_dir / path.name).exists():
            shutil.copy2(path, source_dir / path.name)
    runtime = dict(cfg, scene=str(assets / 'scene.gs.ply'), navmesh=str(assets / 'scene.navmesh'),
        camera=str(assets / 'camera_rays.npz'), neutral_shape=str(shape), skeleton=str(assets / 'skeleton.json'),
        head_calibration=str(assets / 'head_camera_calibration_train.json'))
    runtime_path = out / 'runtime_config.json'
    atomic_json(runtime_path, runtime)
    previous_configuration = json.loads(configuration.read_text()) if configuration.exists() else None
    if previous_configuration is not None:
        history = out / 'configuration_history'
        history.mkdir(exist_ok=True)
        backup = history / (hashlib.sha256(configuration.read_bytes()).hexdigest()[:16] + '.json')
        if not backup.exists():
            shutil.copy2(configuration, backup)
    atomic_json(configuration, dict(fingerprint=fingerprint, config=cfg,
        created=dt.datetime.now().astimezone().isoformat(), source_revision=subprocess.check_output(
            ['git','-C',os.environ['REPO_ROOT'],'rev-parse','HEAD'], text=True).strip(),
        framework_revision=framework_revision, application_sources=str(source_dir.relative_to(out)),
        application_sha256=source_hashes,
        limitation='Synthetic application, no GT; raw predictions, no collision/floor correction; body transitions are not inpainted.'))
    render_env = os.environ.copy()
    cuda = Path(os.environ['CUDA_HOME'])
    render_env['LD_LIBRARY_PATH'] = f'{cuda}/lib64:{cuda}/targets/x86_64-linux/lib'
    renderer = [cfg['habitat_python'], '-m', 'applications.habitat_gs.render', '--scene', runtime['scene'],
        '--navmesh', runtime['navmesh'], '--camera', runtime['camera'], '--face-size', str(cfg.get('cube_face_size',384))]
    initial = out / 'initial.png'
    pose = validate_pose(cfg['start_pose'])
    atomic_json(out / 'initial.pose.json', pose.tolist())
    if not initial.exists():
        checked_command(renderer + ['--pose', str(out / 'initial.pose.json'), '--out', str(initial)], out / 'render.log', env=render_env)
    model_env = evaluation_environment(profile)
    motion_env = os.environ.copy()
    repo = Path(os.environ['REPO_ROOT'])
    motion_env['PYTHONPATH'] = ':'.join(map(str, [repo/'motion_expert_t2m_edge', repo/'motion_expert_joint_attention',
        repo/'motion_expert', repo/'nymeria_world', repo.parent/'kimodo_open', repo, os.environ['COSMOS_FRAMEWORK_EDGE_ROOT']]))
    motion_env['LD_LIBRARY_PATH'] = ''
    motion_env['EDGE_MOTION_REPRESENTATION'] = 'camera_head_recanonicalization_v1'
    conditioning = initial
    segments = []
    for i, prompt in enumerate(cfg['prompts']):
        root = out / f'segment_{i:03d}'
        root.mkdir(exist_ok=True)
        seed = int(cfg['seed']) + i
        atomic_json(root / 'start.pose.json', pose.tolist())
        shutil.copy2(conditioning, root / 'conditioning.png')
        record = make_record(profile, 'image2video', f'human_{i:03d}', root/'conditioning.png', prompt, seed)
        inputs = root / 'image2video.jsonl'
        inputs.write_text(json.dumps(record) + '\n')
        progress(out, 'generating_video', segment=i, prompt=prompt)
        if not (root / 'inference' / record['name'] / 'sample_outputs.json').exists():
            checked_command(official_command(profile, inputs, root/'inference'), root/'image2video.log',
                cwd=os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'], env=model_env)
        result, _ = load_output(root / 'inference', record)
        video = find_video(result, root / 'inference' / record['name'])
        shutil.copy2(video, root / 'generated.mp4')
        progress(out, 'reconstructing_motion_with_text', segment=i, prompt=prompt)
        if not (root / 'inference.json').exists():
            checked_command([sys.executable, '-m', 'applications.habitat_gs.human_motion', '--config', str(runtime_path),
                '--video', str(root/'generated.mp4'), '--start', str(root/'start.pose.json'), '--prompt', prompt,
                '--seed', str(seed), '--out', str(root)], root/'motion.log', env=motion_env,
                cwd=os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'])
        progress(out, 'rendering_body_and_camera_replay', segment=i)
        if not (root / 'rendered/COMPLETE.json').exists():
            checked_command([cfg['habitat_python'], '-m', 'applications.habitat_gs.human_replay',
                '--config', str(runtime_path), '--segment', str(root)], root/'replay.log', env=render_env)
        # Independently verify that the next observation contains no debug lines.
        checked_command(renderer + ['--pose', str(root/'end.pose.json'), '--out', str(root/'next_condition_independent.png')],
                        root/'replay.log', env=render_env)
        from PIL import Image
        if not np.array_equal(np.asarray(Image.open(root/'next_condition.png')),
                              np.asarray(Image.open(root/'next_condition_independent.png'))):
            raise ValueError('Replay last frame differs from independent overlay-free render')
        replay = json.loads((root/'replay.json').read_text())
        replay['segment'] = i
        replay['frame_times_global_s'] = (i*4.8 + np.arange(97)/20).tolist()
        atomic_json(root/'replay.json', replay)
        poses = np.asarray(replay['camera_to_world_optical'])
        if not np.allclose(poses[0], pose, atol=1e-9):
            raise ValueError('Segment start camera is discontinuous')
        info = dict(index=i, prompt=prompt, seed=seed, frames=97, fps=20, start_time_s=i*4.8,
            end_time_s=(i+1)*4.8, replay=f'segment_{i:03d}/replay.json',
            video=f'segment_{i:03d}/generated.mp4', comparison=f'segment_{i:03d}/rendered/comparison.mp4',
            camera_source='Phase3 reconstructed Head, not Phase1 inverse dynamics',
            next_condition=f'segment_{i:03d}/next_condition.png')
        atomic_json(root/'COMPLETE.json', info)
        segments.append(info)
        pose, conditioning = validate_pose(poses[-1]), root/'next_condition.png'
    # Concatenate display clips, retaining observed boundary frames in each.
    import imageio_ffmpeg
    listing = out/'comparison_clips.txt'
    listing.write_text(''.join("file '" + str(out/s['comparison']).replace("'", "'\\''") + "'\n" for s in segments))
    checked_command([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel','error','-f','concat','-safe','0',
        '-i',str(listing),'-c','copy','-movflags','+faststart',str(out/'rollout_comparison.mp4')], out/'presentation.log')
    manifest = dict(schema='habitat_human_rollout_v1', segments=segments, frames_per_segment=97, fps=20,
        scene='assets/scene.gs.ply', navmesh='assets/scene.navmesh', camera_rays='assets/camera_rays.npz',
        skeleton='assets/skeleton.json', frames='PNG filenames are zero-based six-digit frame indices',
        coordinates='right-handed Habitat scene, +Y up, explicit scene_units_per_meter',
        scene_units_per_meter=cfg['scene_units_per_meter'], matrices='row-major arrays; multiply column vectors on the left',
        optical_axes='+X right, +Y down, +Z forward; T_world_gl = T_world_optical @ diag(1,-1,-1,1)',
        timing='global_t = 4.8*segment + frame/20; adjacent segments share their boundary time',
        boundary_policy='raw 97-frame segments retained; for unique-time playback keep segment0 and drop frame0 of each later segment',
        conditioning='Each next I2V image is a new GS render at the last body-derived camera pose',
        motion_continuity='Camera boundary is exact. Bodies are independently reconstructed; no pose inpainting or hidden interpolation.',
        preview='rollout_comparison.mp4', replay_instructions='REPLAY.md',
        assets_sha256={p.name:sha256(p) for p in assets.iterdir() if p.is_file()})
    atomic_json(out/'manifest.json', manifest)
    instructions = Path(__file__).with_name('HUMAN_REPLAY.md')
    shutil.copy2(instructions, out/'REPLAY.md')
    cards = ''.join(f'<h2>{s["index"]+1}. {html.escape(s["prompt"])}</h2><video controls width="768" src="{s["comparison"]}"></video>' for s in segments)
    (out/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Human motion rollout</title>'
        '<body style="background:#161616;color:white;font:16px sans-serif"><h1>Habitat-GS human rollout</h1>'
        '<p>Generated RGB | body-derived camera GS replay; below: reconstructed skeleton in scene.</p>'
        '<video controls width="768" src="rollout_comparison.mp4"></video>'+cards+'<p><a href="REPLAY.md">Replay instructions</a> | <a href="manifest.json">Data manifest</a></p></body></html>')
    inventory = {str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p))
                 for p in sorted(out.rglob('*')) if p.is_file() and str(p.relative_to(out)) not in {'inventory.json','rollout.lock','progress.json','COMPLETE.json'}}
    atomic_json(out/'inventory.json', inventory)
    atomic_json(out/'COMPLETE.json', dict(segments=len(segments), physical_duration_s=4.8*len(segments),
        display_frames=97*len(segments), unique_time_frames=1+96*len(segments), manifest='manifest.json',
        inventory_sha256=sha256(out/'inventory.json'), gallery='index.html'))
    progress(out, 'complete', segments=len(segments), output=str(out))


if __name__ == '__main__':
    main()
