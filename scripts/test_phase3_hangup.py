"""Exercise real process signals and TorchElastic without allocating a GPU."""
from pathlib import Path
import os
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts.phase3_agent_entry import guarded_handler

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/phase3_worker_entry.py"
AGENT = ROOT / "scripts/phase3_agent_entry.py"


def test_only_hangup_is_ignored():
    native = Mock(side_effect=RuntimeError("native termination"))
    handler = guarded_handler(native)
    handler(signal.SIGHUP, None)
    native.assert_not_called()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        with pytest.raises(RuntimeError, match="native termination"):
            handler(signum, None)
        native.assert_called_with(signum, None)


@pytest.mark.parametrize("target", ["--script", "--module"])
def test_worker_survives_hangup_and_forwards_arguments(tmp_path, target):
    probe = tmp_path / "hangup_probe.py"
    probe.write_text("import os,signal,sys\n"
                     "assert sys.argv[1:]==['--probe-argument','kept']\n"
                     "assert signal.getsignal(signal.SIGHUP)==signal.SIG_IGN\n"
                     "os.kill(os.getpid(),signal.SIGHUP)\n"
                     "print('WORKER_OK')\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(tmp_path)}
    result = subprocess.run([sys.executable, str(WORKER), target,
                             str(probe) if target == "--script" else "hangup_probe",
                             "--probe-argument", "kept"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "WORKER_OK" in result.stdout


def test_worker_keeps_explicit_term_behavior(tmp_path):
    probe = tmp_path / "stop_probe.py"
    probe.write_text("import os,signal\nos.kill(os.getpid(),signal.SIGTERM)\n")
    result = subprocess.run([sys.executable, str(WORKER), "--script", str(probe)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == -signal.SIGTERM


def test_actual_two_worker_agent_survives_hangup(tmp_path):
    probe = tmp_path / "elastic_probe.py"
    probe.write_text("import os,signal\n"
                     "assert signal.getsignal(signal.SIGHUP)==signal.SIG_IGN\n"
                     "if os.environ['LOCAL_RANK']=='0':os.kill(os.getppid(),signal.SIGHUP)\n"
                     "os.kill(os.getpid(),signal.SIGHUP)\n"
                     "print('ELASTIC_WORKER_OK',flush=True)\n")
    result = subprocess.run([sys.executable, str(AGENT), "--standalone", "--nnodes=1",
                             "--nproc-per-node=2", str(WORKER), "--script", str(probe)],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ELASTIC_WORKER_OK") == 2
    assert "SIGHUP ignored" in result.stdout


def test_launcher_snapshot_contains_both_signal_wrappers(tmp_path):
    from motion_expert_phase3_edge_image_joint.launch import freeze_sources
    manifest = freeze_sources(ROOT, tmp_path / "snapshot")
    for name in ["scripts/phase3_agent_entry.py", "scripts/phase3_worker_entry.py"]:
        assert name in manifest
        assert (tmp_path / "snapshot" / name).read_bytes() == (ROOT / name).read_bytes()
