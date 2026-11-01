"""Download one pinned HF scene, its navmesh and optional navigation episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scene', default='val/interior_0516_840045')
    args = ap.parse_args()
    info = get_json('https://huggingface.co/api/datasets/RukawaY/gs_scenes')
    revision = info['sha']
    base = f'https://huggingface.co/api/datasets/RukawaY/gs_scenes/tree/{revision}/'
    entries = get_json(base + args.scene + '?recursive=true')
    selected = [e for e in entries if e['type'] == 'file' and e['path'].endswith(('.gs.ply', '.navmesh'))]
    if not any(e['path'].endswith('.gs.ply') for e in selected):
        raise RuntimeError('No Gaussian scene in selected directory')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'asset_manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior['revision'] != revision or prior['scene'] != args.scene:
            raise RuntimeError('Use a fresh output directory for a changed asset revision/scene')
    records = []
    for entry in selected:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid asset path')
        dest = args.out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/RukawaY/gs_scenes/resolve/{revision}/{entry["path"]}'
        if not dest.exists():
            temporary = dest.with_suffix(dest.suffix + '.partial')
            print('Downloading', entry['path'], entry['size'], flush=True)
            with urllib.request.urlopen(url, timeout=120) as src, temporary.open('wb') as target:
                while data := src.read(4 * 1024 * 1024):
                    target.write(data)
            temporary.replace(dest)
        if dest.stat().st_size != entry['size']:
            raise RuntimeError('Asset size mismatch: ' + str(dest))
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        lfs_sha = entry.get('lfs', {}).get('oid')
        if lfs_sha and digest.hexdigest() != lfs_sha.removeprefix('sha256:'):
            raise RuntimeError('HF LFS SHA256 mismatch')
        records.append(dict(path=str(dest.resolve()), source=entry['path'], bytes=entry['size'], sha256=digest.hexdigest()))
    manifest_path.write_text(json.dumps(dict(repository='RukawaY/gs_scenes', revision=revision,
        scene=args.scene, files=records), indent=2) + '\n')
    print(manifest_path, flush=True)


if __name__ == '__main__':
    main()
