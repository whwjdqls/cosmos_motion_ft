"""Render every estimated camera pose, exposing live frames and a GS replay."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import imageio_ffmpeg
import numpy as np
from PIL import Image
from .render import Renderer
from .runtime import require_compute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require_compute()
    config = json.loads(args.config.read_text())
    with np.load(config['camera'], allow_pickle=False) as data:
        rays, valid = data['rays'], data['valid']
    with np.load(args.trajectory, allow_pickle=False) as data:
        poses = data['camera_to_world_optical']
    if poses.shape != (97,4,4) or not np.isfinite(poses).all():
        raise ValueError('Expected all 97 finite camera poses')
    args.out.mkdir(parents=True,exist_ok=True)
    renderer = Renderer(config['scene'],config.get('navmesh'),config.get('cube_face_size',384))
    process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
        '-f','rawvideo','-pix_fmt','rgb24','-s','256x256','-r','20','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
        str(args.out/'camera_replay.mp4')],stdin=subprocess.PIPE)
    start = time.monotonic()
    try:
        for index,pose in enumerate(poses):
            frame = renderer.render(pose,rays,valid)
            process.stdin.write(frame.tobytes())
            temporary = args.out/'latest.partial.png'
            Image.fromarray(frame).save(temporary)
            temporary.replace(args.out/'latest.png')
            temporary = args.out/'progress.partial.json'
            temporary.write_text(json.dumps(dict(frame=index+1,total=len(poses))))
            temporary.replace(args.out/'progress.json')
            time.sleep(max(0,(index+1)/20-(time.monotonic()-start)))
        process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError('GS replay encoding failed')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        renderer.sim.close()
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(frames=97,fps=20,
        trajectory=str(args.trajectory),final_pose=poses[-1].tolist()))+'\n')


if __name__=='__main__':
    main()
