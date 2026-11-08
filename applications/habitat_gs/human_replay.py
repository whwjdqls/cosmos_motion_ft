"""Save full fisheye replay and depth-tested body/camera inspection frames."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .camera import CV_FROM_GL, validate_pose
from .inspect_worker import Inspector
from .runtime import require_compute


def observer_pose(target):
    origin = target + np.array([2.8, 1.8, 2.8])
    forward = target - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = origin
    return validate_pose(pose)


def encoder(path, width, height):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '20', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
        stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--segment', type=Path, required=True)
    args = p.parse_args()
    require_compute()
    config = json.loads(args.config.read_text())
    replay = json.loads((args.segment / 'replay.json').read_text())
    skeleton = json.loads(Path(config['skeleton']).read_text())
    poses = np.asarray(replay['camera_to_world_optical'])
    joints = np.asarray(replay['joints_scene'])
    rgb = np.load(args.segment / 'generated_rgb.npy', mmap_mode='r')
    out = args.segment / 'rendered'
    for folder in ('generated_rgb', 'gs_egocentric', 'gs_body_camera'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    inspector = Inspector(config)
    import quaternion
    target = np.mean(joints[:, 0], axis=0)
    observer = observer_pose(target)
    (out / 'observer.pose.json').write_text(json.dumps(observer.tolist(), indent=2) + '\n')
    scene_video, body_video, combined = encoder(out / 'gs_egocentric.mp4', 256, 256), \
        encoder(out / 'gs_body_camera.mp4', 768, 512), encoder(out / 'comparison.mp4', 768, 960)
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    try:
        for i, pose in enumerate(poses):
            # No debug geometry is submitted until the model-camera RGB exists.
            frame = inspector.renderer.render(pose, inspector.rays, inspector.valid)
            gl = observer @ CV_FROM_GL
            state = inspector.renderer.hs.AgentState()
            state.position = gl[:3, 3]
            state.rotation = quaternion.from_rotation_matrix(gl[:3, :3])
            inspector.renderer.sim.get_agent(1).set_state(state, reset_sensors=True)
            debug = inspector.renderer.sim.get_debug_line_render()
            debug.set_line_width(2.)
            for child, parent in enumerate(skeleton['parents']):
                if parent >= 0:
                    inspector.line(joints[i, parent], joints[i, child], (1., .45, .08, 1.))
            inspector.glyph(pose, (1., .25, 1., 1.))
            for a, b in zip(poses[:i, :3, 3], poses[1:i + 1, :3, 3]):
                inspector.line(a, b, (.1, .9, 1., 1.))
            view = np.asarray(inspector.renderer.sim.get_sensor_observations(agent_ids=1)['inspection'])[..., :3].copy()
            for folder, data in [('generated_rgb', rgb[i]), ('gs_egocentric', frame), ('gs_body_camera', view)]:
                Image.fromarray(data).save(out / folder / f'{i:06d}.png')
            scene_video.stdin.write(frame.tobytes())
            body_video.stdin.write(view.tobytes())
            canvas = Image.new('RGB', (768, 960), (18, 18, 18))
            canvas.paste(Image.fromarray(rgb[i]).resize((384, 384)), (0, 30))
            canvas.paste(Image.fromarray(frame).resize((384, 384)), (384, 30))
            canvas.paste(Image.fromarray(view), (0, 430))
            d = ImageDraw.Draw(canvas)
            color = 'lime' if i == 0 else '#ff6666'
            d.text((8, 7), 'Observed first frame' if i == 0 else 'Generated video', fill=color, font=font)
            d.text((392, 7), 'GS replay from reconstructed head', fill='lime', font=font)
            d.rectangle((1, 31, 382, 412), outline=color, width=3)
            d.text((8, 413), 'Orange: motion | magenta: camera | cyan: path', fill='white', font=font)
            d.text((8, 942), replay['prompt'], fill='white', font=font)
            combined.stdin.write(np.asarray(canvas).tobytes())
        for proc in (scene_video, body_video, combined):
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('Replay encoding failed')
    finally:
        for proc in (scene_video, body_video, combined):
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        inspector.renderer.sim.close()
    Image.open(out / 'gs_egocentric/000096.png').save(args.segment / 'next_condition.png')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(frames=97, fps=20, observer_pose=observer.tolist(),
        skeleton=skeleton, camera_source='decoded Head * train calibration', comparison='comparison.mp4')) + '\n')


if __name__ == '__main__':
    main()
