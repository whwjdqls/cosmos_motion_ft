"""Explicit storage boundaries for Foundary and scheduler-backed sites."""
import os
from pathlib import Path


def storage_roots():
    site = os.environ.get('COSMOS_SITE', 'custom')
    if site == 'custom':
        return (Path('/mnt/local'),)
    if site == 'yonsei':
        return (Path('/scratch2/whwjdqls99/cosmos'), Path('/lustre/whwjdqls99/cosmos'))
    if site == 'grasp':
        return (Path(os.environ.get('COSMOS_STORAGE_ROOT', '/mnt/projects/ll/jungbinc')),)
    raise ValueError(f'Unsupported Phase3 site: {site}')


def storage_path(value, *, must_exist=True):
    path = Path(os.path.expandvars(str(value)))
    if not path.is_absolute():
        raise ValueError(f'Phase3 paths must be absolute: {path}')
    path = path.resolve()
    if not any(path.is_relative_to(root.resolve()) for root in storage_roots()):
        raise ValueError(f'Phase3 path is outside the selected site storage: {path}')
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path
