"""Install only the requested camera-CFG hook, then run official inference.

HF checkpoint/config resolution is performed by the unmodified official CLI.
"""
import runpy
from native_phase_training.zero_camera_cfg_config import install_checked_camera_cfg


if __name__ == '__main__':
    install_checked_camera_cfg()
    runpy.run_module('cosmos_framework.scripts.inference', run_name='__main__')
