"""Render -> evaluated I2V -> evaluated inverse dynamics -> SE(3) -> render."""
import argparse
import datetime as dt
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import imageio_ffmpeg
from .camera import integrate_actions, sha256, validate_pose
from .profile import make_record, verify
from .runtime import checked_command, official_command, require_compute, evaluation_environment


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_output(root, record):
    path = root / record['name'] / 'sample_outputs.json'
    data = json.loads(path.read_text())
    if data['status'] != 'success' or data['args']['model_mode'] != record['model_mode']:
        raise RuntimeError('Official inference failed/skipped: ' + str(path))
    for key, value in record.items():
        if key in data['args'] and data['args'][key] != value:
            raise ValueError('Saved inference args differ from requested profile: ' + key)
    return data, path


def find_video(data, sample_dir):
    files = [Path(p) for item in data['outputs'] for p in item.get('files', []) if str(p).endswith('.mp4')]
    existing = []
    for path in files:
        candidates = [path] if path.is_absolute() else [sample_dir/path, sample_dir.parent/path]
        existing += [p.resolve() for p in candidates if p.is_file()]
    existing = list(dict.fromkeys(existing))
    if len(existing) != 1:
        raise ValueError('Expected one generated official MP4, found ' + str(existing))
    return existing[0]


def validate_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if len(frames) != 97 or frames[0].shape != (256,256,3) or abs(fps-20) > .01:
        raise ValueError(f'Expected97 RGB256 frames at20fps; got {len(frames)} / {fps}')
    return frames


def preview(frames, path, camera_mode, prompt):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    # Keep the prompt outside the image, visible throughout the whole clip.
    lines, line = [], ''
    for word in ('Prompt: ' + prompt).split():
        candidate = (line + ' ' + word).strip()
        if font.getlength(candidate) <= 240:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = ''
        for char in word:
            if font.getlength(line + char) > 240:
                lines.append(line)
                line = ''
            line += char
    if line:
        lines.append(line)
    height = 284 + 16 + 16 * len(lines)
    height += height % 2
    background = Image.new('RGB', (256, height), (16,16,16))
    draw = ImageDraw.Draw(background)
    for row, line in enumerate(lines):
        draw.text((8, 292 + 16*row), line, fill='white', font=font)
    background = np.array(background)
    with subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'256x{height}','-r','20','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(path)],stdin=subprocess.PIPE) as proc:
        for i, frame in enumerate(frames):
            panel = background.copy()
            panel[28:284] = frame
            color = (0,255,0) if i == 0 else (255,70,70)
            cv2.rectangle(panel,(1,29),(254,282),color,3)
            label = 'GS CONDITION' if i == 0 else 'GENERATED'
            cv2.putText(panel,label,(6,19),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
            proc.stdin.write(panel.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Preview encoding failed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    require_compute()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    out = Path(config['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'rollout.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile_path = Path(config['profile'])
    profile = json.loads(profile_path.read_text())
    verify(profile)
    framework = Path(os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'])
    revision = subprocess.check_output(['git','-C',str(framework),'rev-parse','HEAD'],text=True).strip()
    if revision != profile['framework_revision']:
        raise ValueError('Framework revision differs from exported evaluation profile')
    camera_path, scene = Path(config['camera']), Path(config['scene'])
    with np.load(camera_path, allow_pickle=False) as data:
        camera_mode = str(data['mode'])
    if camera_mode.startswith('pinhole_approximation') and not config.get('allow_pinhole_approximation',False):
        raise ValueError('Fisheye calibration unavailable: explicitly allow the documented pinhole approximation or export VRS rays')
    if int(config['segments']) < 1 or not config['prompt'].strip():
        raise ValueError('Provide a prompt and at least one segment')
    app_sources = {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity = dict(config=config, profile_sha256=sha256(profile_path), camera_sha256=sha256(camera_path),
                    scene_sha256=sha256(scene), framework_revision=revision, application_sources=app_sources)
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest_path = out/'configuration.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Output belongs to another configuration; choose a fresh directory')
    else:
        atomic_json(manifest_path, dict(identity, fingerprint=fingerprint, camera_mode=camera_mode,
            created=dt.datetime.now().astimezone().isoformat(),
            limitation='Generated camera is an estimate; no metric scale recovery, registration, collision correction, or guaranteed scene consistency.'))
    renderer_python = config['habitat_python']
    model_env = evaluation_environment(profile)
    render_env = os.environ.copy()
    # CUDA12.8 Cosmos-specific shared libraries must not override simulator libraries.
    cuda_home = Path(os.environ.get('CUDA_HOME', '/opt/ohpc/pub/apps/cuda/12.8'))
    render_env['LD_LIBRARY_PATH'] = os.pathsep.join(str(p) for p in
        (cuda_home/'lib64', cuda_home/'targets/x86_64-linux/lib') if p.is_dir())
    renderer_prefix = [renderer_python,'-m','applications.habitat_gs.render','--scene',str(scene),
        '--camera',str(camera_path),'--face-size',str(config.get('cube_face_size',384))]
    if config.get('navmesh'):
        renderer_prefix += ['--navmesh',config['navmesh']]
    first = out/'initial.png'
    if not first.exists():
        cmd = renderer_prefix + ['--out',str(first),'--seed',str(config['seed']),
            '--camera-height',str(config.get('initial_camera_height_m',1.5)*config['scene_units_per_meter'])]
        if config.get('start_pose'):
            start = out/'provided_start.pose.json'
            atomic_json(start,validate_pose(config['start_pose']).tolist())
            cmd += ['--pose',str(start)]
        checked_command(cmd,out/'renderer.log',env=render_env)
    current_pose = validate_pose(json.loads(first.with_suffix('.pose.json').read_text()))
    conditioning = first
    completed = []
    for segment in range(config['segments']):
        root = out/f'segment_{segment:03d}'
        root.mkdir(exist_ok=True)
        receipt_path = root/'COMPLETE.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['fingerprint'] != fingerprint or not np.allclose(current_pose,receipt['start_pose'],atol=1e-8):
                raise ValueError('Saved segment does not match continuation pose/configuration')
            for artifact in receipt['artifacts']:
                if sha256(artifact['path']) != artifact['sha256']:
                    raise ValueError('Saved segment artifact changed: ' + artifact['path'])
            current_pose = validate_pose(receipt['end_pose'])
            conditioning = Path(receipt['next_condition'])
            completed.append(receipt)
            continue
        atomic_json(out/'progress.json',dict(stage='generating',segment=segment,total=config['segments']))
        seed = int(config['seed']) + segment
        generated = make_record(profile,'image2video',f'rollout_{segment:03d}',conditioning,config['prompt'],seed)
        input_path = root/'image2video.jsonl'
        input_path.write_text(json.dumps(generated)+'\n')
        inference = root/'inference'
        output_json = inference/generated['name']/'sample_outputs.json'
        if not output_json.exists():
            checked_command(official_command(profile,input_path,inference),root/'image2video.log',cwd=framework,env=model_env)
        data, _ = load_output(inference,generated)
        video = find_video(data,inference/generated['name'])
        frames = validate_video(video)
        inverse = make_record(profile,'inverse_dynamics',f'rollout_{segment:03d}',video,'',seed)
        inverse_path = root/'inverse_dynamics.jsonl'
        inverse_path.write_text(json.dumps(inverse)+'\n')
        atomic_json(out/'progress.json',dict(stage='estimating_camera',segment=segment,total=config['segments']))
        inverse_json = inference/inverse['name']/'sample_outputs.json'
        if not inverse_json.exists():
            checked_command(official_command(profile,inverse_path,inference),root/'inverse_dynamics.log',cwd=framework,env=model_env)
        prediction, _ = load_output(inference,inverse)
        actions = np.asarray(prediction['outputs'][0]['content']['action'])
        poses = integrate_actions(actions,current_pose,config['scene_units_per_meter'])
        np.savez(root/'camera_trajectory.npz',actions=actions,camera_to_world_optical=poses,
                 frame_times_s=segment*96/20 + np.arange(97)/20)
        end_pose_path = root/'end.pose.json'
        atomic_json(end_pose_path,poses[-1].tolist())
        next_condition = root/'next_condition.png'
        if not next_condition.exists():
            checked_command(renderer_prefix+['--pose',str(end_pose_path),'--out',str(next_condition)],
                            out/'renderer.log',env=render_env)
        from PIL import Image, ImageDraw
        anchor = np.asarray(Image.open(next_condition).convert('RGB'))
        comparison = Image.new('RGB',(512,284))
        comparison.paste(Image.fromarray(frames[-1]),(0,28))
        comparison.paste(Image.fromarray(anchor),(256,28))
        draw = ImageDraw.Draw(comparison)
        draw.text((5,5),'GENERATED LAST FRAME',fill='red')
        draw.text((261,5),'GS AT ESTIMATED CAMERA',fill='lime')
        comparison.save(root/'boundary.png')
        preview(frames,root/'preview.mp4',camera_mode,config['prompt'])
        receipt = dict(segment=segment,seed=seed,fingerprint=fingerprint,start_pose=current_pose.tolist(),
            end_pose=poses[-1].tolist(), next_condition=str(next_condition), generated_video=str(video),
            camera_mode=camera_mode, elapsed_scene_time_s=(segment+1)*96/20,
            boundary_rgb_mse=float(np.square(frames[-1].astype(float)/255-anchor.astype(float)/255).mean()),
            boundary_metric_note='Self-consistency diagnostic only; neither panel is independent GT for generated motion.',
            artifacts=[dict(path=str(p),sha256=sha256(p)) for p in
                [video, inverse_json, root/'camera_trajectory.npz', next_condition, root/'boundary.png', root/'preview.mp4']])
        atomic_json(receipt_path,receipt)
        completed.append(receipt)
        current_pose,conditioning = poses[-1],next_condition
    # Presentation concatenation keeps each observed re-anchor visible. Shared
    # boundary instants occur twice; metric physical time is in each pose NPZ.
    concat = out/'preview_list.txt'
    def escaped(path):
        return str(path).replace("'", "'\\''")
    concat.write_text(''.join("file '"+escaped(out/f'segment_{i:03d}/preview.mp4')+"'\n" for i in range(config['segments'])))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
                    '-c','copy','-movflags','+faststart',str(out/'rollout_preview.mp4')],check=True)
    gallery = '<html><body><h1>Habitat-GS video rollout</h1><p>Green: simulator observation; red: generated. Camera: '+camera_mode+'.</p>'
    gallery += '<video controls width="768" src="rollout_preview.mp4"></video>'
    gallery += '<p>Prompt: '+html.escape(config['prompt'])+'</p>'
    for i in range(config['segments']):
        gallery += f'<h2>Segment {i}</h2><img width="768" src="segment_{i:03d}/boundary.png">'
    (out/'index.html').write_text(gallery+'</body></html>')
    atomic_json(out/'COMPLETE.json',dict(fingerprint=fingerprint,segments=config['segments'],
        elapsed_scene_time_s=config['segments']*96/20,preview_shared_boundary_duplicated=True,
        gallery=str(out/'index.html'),camera_mode=camera_mode))
    atomic_json(out/'progress.json',dict(stage='complete',segments=config['segments']))
    print(json.dumps(dict(status='complete',gallery=str(out/'index.html'))),flush=True)


if __name__ == '__main__':
    main()
