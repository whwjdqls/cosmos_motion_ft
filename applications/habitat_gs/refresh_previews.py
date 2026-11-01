"""Add prompt banners to completed rollouts without rerunning inference."""
import argparse
import fcntl
import html
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from .camera import sha256
from .rollout import preview, validate_video, atomic_json
from .runtime import require_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require_compute()
    root = args.output.resolve()
    lock = (root/'rollout.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completion = json.loads((root/'COMPLETE.json').read_text())
    config = json.loads((root/'configuration.json').read_text())['config']
    sources, outputs = [], []
    for i in range(completion['segments']):
        segment = root/f'segment_{i:03d}'
        receipt = json.loads((segment/'COMPLETE.json').read_text())
        video = Path(receipt['generated_video'])
        digest = sha256(video)
        expected = next(a['sha256'] for a in receipt['artifacts'] if a['path']==str(video))
        if digest != expected:
            raise ValueError('Generated video has changed')
        output = segment/'preview_prompt.mp4'
        preview(validate_video(video), output, completion['camera_mode'], config['prompt'])
        sources.append(dict(path=str(video),sha256=digest))
        outputs.append(output)
    listing = root/'preview_prompt_list.txt'
    listing.write_text(''.join("file '"+str(p).replace("'", "'\\''")+"'\n" for p in outputs))
    combined = root/'rollout_preview_prompt.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','concat',
                    '-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(combined)],check=True)
    page = (root/'index.html').read_text().replace('src="rollout_preview.mp4"','src="rollout_preview_prompt.mp4"')
    if '<p>Prompt:' not in page:
        page = page.replace('</h1>', '</h1><p>Prompt: '+html.escape(config['prompt'])+'</p>',1)
    (root/'index.html').write_text(page)
    atomic_json(root/'prompt_preview.json',dict(prompt=config['prompt'],sources=sources,
        outputs=[dict(path=str(p),sha256=sha256(p)) for p in outputs+[combined]],
        inference_rerun=False,original_generation_fingerprint=completion['fingerprint']))
    print(combined,flush=True)


if __name__=='__main__':
    main()
