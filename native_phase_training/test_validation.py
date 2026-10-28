"""CPU checks for preview conditioning and explicit Foundary device selection."""
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from native_phase_training.compute_access import require_compute_access
from native_phase_training.validation import make_batch


class PreviewConditioningTests(unittest.TestCase):
    def test_prefix_masks_and_inverse_observations(self):
        latent = torch.zeros(1, 48, 25, 16, 16)
        action = np.zeros((96, 9), dtype=np.float32)
        for prefix, count in ((1, 1), (17, 5)):
            forward = make_batch(latent, action, 'forward_dynamics', prefix)
            inverse = make_batch(latent, action, 'inverse_dynamics', prefix)
            self.assertEqual(forward['sequence_plan'][0].condition_frame_indexes_vision, list(range(count)))
            self.assertEqual(inverse['sequence_plan'][0].condition_frame_indexes_vision, list(range(25)))
            self.assertEqual(forward['action'][0].shape, (96, 64))


class ComputeAccessTests(unittest.TestCase):
    def test_admission_launcher_uses_its_device_and_ancestry_checks(self):
        host = 'lg-gwangmo-4xh200-sep-1'
        gpu = 'GPU-fd07fdaf-5f61-6211-5c78-adf4aa85d7b4'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            admission = dict(host=host, uid=os.getuid(), launcher_pid=os.getpid(),
                             cuda_visible_devices='7')
            path.write_text(json.dumps(admission))
            env = dict(COSMOS_SITE='custom', NATIVEP1_FOUNDRY_ADMISSION=str(path),
                       CUDA_VISIBLE_DEVICES='7')
            with mock.patch.dict(os.environ, env, clear=True), mock.patch('socket.gethostname', return_value=host):
                require_compute_access()
                with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6'}):
                    with self.assertRaisesRegex(RuntimeError, 'Visible GPUs differ'):
                        require_compute_access()
                admission.update(launcher_pid=999999999, cuda_visible_devices=gpu)
                path.write_text(json.dumps(admission))
                # An invalid admission must not fall back to the UUID launcher.
                with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': gpu,
                        'FOUNDRY_GPU_UUID': gpu, 'COSMOS_STORAGE_ROOT': '/mnt/local/test'}):
                    with self.assertRaisesRegex(RuntimeError, 'not an active ancestor'):
                        require_compute_access()

    def test_foundary_requires_exact_single_gpu_selection(self):
        gpu = 'GPU-fd07fdaf-5f61-6211-5c78-adf4aa85d7b4'
        env = {'COSMOS_SITE': 'custom', 'COSMOS_STORAGE_ROOT': '/mnt/local/test',
               'FOUNDRY_GPU_UUID': gpu, 'CUDA_VISIBLE_DEVICES': gpu}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('socket.gethostname', return_value='lg-gwangmo-4xh200-sep-2'):
            require_compute_access()
            for visible in ('0', gpu + ',1', ''):
                with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}):
                    with self.assertRaises(RuntimeError):
                        require_compute_access()

    def test_unassigned_node_and_foreign_allocation_are_rejected(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch('socket.gethostname', return_value='node37'):
            with self.assertRaises(RuntimeError):
                require_compute_access()
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_USER': 'someone_else'}, clear=True), mock.patch('socket.gethostname', return_value='compute'):
            with self.assertRaises(RuntimeError):
                require_compute_access()

    def test_foundary_two_gpu_selection_must_match_exactly(self):
        gpu0 = 'GPU-fd07fdaf-5f61-6211-5c78-adf4aa85d7b4'
        gpu5 = 'GPU-89744c72-7e10-f5b6-3ab2-b92d6366520b'
        selection = gpu0 + ',' + gpu5
        env = {'COSMOS_SITE': 'custom', 'COSMOS_STORAGE_ROOT': '/mnt/local/test',
               'FOUNDRY_GPU_UUIDS': selection, 'CUDA_VISIBLE_DEVICES': selection}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('socket.gethostname', return_value='lg-gwangmo-4xh200-sep-2'):
            require_compute_access()
            for visible in (gpu0, gpu5 + ',' + gpu0, '0,5', selection + ',1'):
                with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}):
                    with self.assertRaises(RuntimeError):
                        require_compute_access()
            for invalid in (gpu0 + ',' + gpu0, selection + ',', ''):
                with mock.patch.dict(os.environ, {'FOUNDRY_GPU_UUIDS': invalid, 'CUDA_VISIBLE_DEVICES': invalid}):
                    with self.assertRaises(RuntimeError):
                        require_compute_access()


if __name__ == '__main__':
    unittest.main()
