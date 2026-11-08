"""Keep repeated video-scorer construction on the checksum-pinned LPIPS path."""
from contextlib import contextmanager
from pathlib import Path
import torch

@contextmanager
def pinned_metric_hub(config):
    original=torch.hub.get_dir()
    expected=str(Path(config['lpips_alexnet']).parent.parent)
    torch.hub.set_dir(expected)
    try:
        yield
    finally:
        torch.hub.set_dir(original)
