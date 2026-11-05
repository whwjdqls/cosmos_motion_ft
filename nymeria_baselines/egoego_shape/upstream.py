"""Load the original MIT-licensed network without its unused SMPL/renderer stack.

Definitions are executed verbatim from upstream commit 4dae8cb. Only imports
used exclusively by AMASS/SMPL visualization and sliding-window tools are omitted.
No upstream source file, model layer, attention implementation or schedule is edited.
"""
import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

COMMIT = '4dae8cb67b453f0a0042809bf37aafad5604615b'
DEFAULT = Path('/scratch2/minsu1206/cosmos/operations/egoego_shape_20261104/upstream/egoego_release')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(root=DEFAULT):
    root = Path(root)
    return {'repository': 'https://github.com/lijiaman/egoego_release', 'commit': COMMIT,
            'root': str(root), 'files': {name: sha(root/name) for name in
                ['egoego/model/transformer_module.py',
                 'egoego/model/transformer_cond_diffusion_model.py', 'LICENSE']}}


def load(root=DEFAULT):
    root = Path(root)
    key = '_egoego_native_' + sha(root/'egoego/model/transformer_cond_diffusion_model.py')[:16]
    if key in sys.modules:
        return sys.modules[key]
    sys.path.insert(0, str(root))
    path = root/'egoego/model/transformer_cond_diffusion_model.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    omitted = ('pytorch3d', 'egoego.vis', 'body_model', 'egoego.data', 'egoego.lafan1')
    tree.body = [node for node in tree.body if not (
        (isinstance(node, ast.ImportFrom) and (node.module or '').startswith(omitted)) or
        (isinstance(node, ast.Import) and any(a.name.startswith(omitted) for a in node.names)))]
    module = types.ModuleType(key)
    module.__file__ = str(path)
    sys.modules[key] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module
