"""CPU-only regressions for Foundary GPU selection and compute admission."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from native_phase_training.compute_admission import require_owned_compute
from native_phase_training.train_phase1_foundary import choose_gpus, validate_resume_checkpoint


def gpu(index, processes=(), memory=0, utilization=0):
    return dict(index=index, processes=list(processes), memory_used_mib=memory,
                utilization=utilization, uuid=f'GPU-{index}')


class FoundaryLaunchTests(unittest.TestCase):
    def test_process_free_gpu_selection(self):
        inventory = [gpu(0, [123], 20000, 0), gpu(1), gpu(2), gpu(3)]
        self.assertEqual([g['index'] for g in choose_gpus(inventory)], [1, 2])

    def test_one_gpu_gives_batch64(self):
        selected = choose_gpus([gpu(0, [123]), gpu(1)])
        self.assertEqual(64 // len(selected), 64)

    def test_explicit_gpus_do_not_substitute_other_free_devices(self):
        inventory = [gpu(3), gpu(5, [123], 518), gpu(7)]
        with self.assertRaises(RuntimeError):
            choose_gpus(inventory, [5, 7])
        self.assertEqual([g['index'] for g in choose_gpus(inventory, [7, 3])], [7, 3])
        with self.assertRaises(ValueError):
            choose_gpus(inventory, [7, 7])

    def test_busy_or_reserved_memory_not_selected(self):
        with self.assertRaises(RuntimeError):
            choose_gpus([gpu(0, memory=500), gpu(1, utilization=10), gpu(2, [123])])

    def test_resume_requires_all_training_state_components(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            checkpoint = storage / 'iter_000000207'
            checkpoint.mkdir()
            for component in ('model', 'optim', 'scheduler'):
                (checkpoint / component).mkdir()
                (checkpoint / component / '.metadata').write_bytes(b'test')
                (checkpoint / component / '__0_0.distcp').write_bytes(b'test')
            with self.assertRaises(RuntimeError):
                validate_resume_checkpoint(checkpoint, storage)
            (checkpoint / 'trainer').mkdir()
            (checkpoint / 'trainer/.metadata').write_bytes(b'test')
            (checkpoint / 'trainer/__0_0.distcp').write_bytes(b'test')
            self.assertEqual(validate_resume_checkpoint(checkpoint, storage), checkpoint)

    def test_owned_slurm_job_is_accepted(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_USER': 'researcher'}, clear=True), \
             patch('socket.gethostname', return_value='node25'), \
             patch('getpass.getuser', return_value='researcher'):
            require_owned_compute()

    def test_foreign_slurm_job_is_rejected(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_USER': 'someone_else'}, clear=True), \
             patch('socket.gethostname', return_value='node25'), \
             patch('getpass.getuser', return_value='researcher'):
            with self.assertRaises(RuntimeError):
                require_owned_compute()

    def test_local_admission_requires_live_launcher_and_same_devices(self):
        host = 'lg-gwangmo-4xh200-sep-1'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            value = dict(host=host, uid=os.getuid(), launcher_pid=os.getpid(), cuda_visible_devices='7')
            path.write_text(json.dumps(value))
            env = dict(COSMOS_SITE='custom', NATIVEP1_FOUNDRY_ADMISSION=str(path), CUDA_VISIBLE_DEVICES='7')
            with patch.dict(os.environ, env, clear=True), patch('socket.gethostname', return_value=host):
                require_owned_compute()
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6'}):
                    with self.assertRaises(RuntimeError):
                        require_owned_compute()
                value['launcher_pid'] = 999999999
                path.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError):
                    require_owned_compute()

    def test_local_launch_without_admission_is_rejected(self):
        with patch.dict(os.environ, {'COSMOS_SITE': 'custom'}, clear=True), \
             patch('socket.gethostname', return_value='lg-gwangmo-4xh200-sep-1'):
            with self.assertRaises(RuntimeError):
                require_owned_compute()


if __name__ == '__main__':
    unittest.main()
