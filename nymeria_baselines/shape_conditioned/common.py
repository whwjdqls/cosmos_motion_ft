"""Run identities and compute-only guards for the shape-input ablations."""
from pathlib import Path
import hashlib
import json
import os
import socket


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname():
        raise RuntimeError('Model/data work requires an owned compute allocation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def contract(cache, upstream, model, batch, horizon):
    audit = json.loads((cache/'acceptance.json').read_text())
    assert audit['phase3_matched_windows'] == 112503
    assert audit['phase3_max_normalized_difference'] <= 1e-4
    manifest=json.loads((upstream/'source_manifest.json').read_text())
    for relative, expected in manifest['output_files'].items():
        if sha(upstream/relative)!=expected:
            raise RuntimeError('Isolated baseline source changed: '+relative)
    return dict(version='supplied_shape_v1', model=model, global_batch=batch,
                horizon=horizon, frames=97, fps=20, hands=False,
                conditioning='measured camera trajectory + clean actor shape',
                shape_prediction=False, shape_loss=False,
                shape='UEM: Phase2 full centered SOMA30 neutral skeleton; EgoAllo: supplied16D train-fitted SOMA PCA, not SMPL betas',
                cache=str(cache), index_sha256=sha(cache/'index.json'),
                statistics_sha256=sha(cache/'train_stats.npz'),
                shape_basis_sha256=sha(cache/'body_shape16.npz'),
                upstream=str(upstream), source_manifest_sha256=sha(upstream/'source_manifest.json'),
                runner_source_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                initialization='fresh random weights; resume only this exact contract')


def save_contract(out, value):
    out.mkdir(parents=True, exist_ok=True)
    path = out/'experiment.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Refusing a changed shape-input recipe/source on resume')
    atomic_json(path, value)
