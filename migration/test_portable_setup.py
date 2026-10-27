"""Path and launcher regression checks; no models, CUDA, or dataset reads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class PortablePathsTest(unittest.TestCase):
    def env(self, root):
        # Do not inherit server/model paths from the invoking shell.
        return {"PATH": os.defpath, "HOME": os.environ["HOME"],
                "COSMOS_SITE": "custom", "COSMOS_STORAGE_ROOT": str(root / "storage with spaces"),
                "COSMOS_ENV_ROOT": str(root / "env"), "COSMOS_RUNS_ROOT": str(root / "runs")}

    def test_shell_python_agree_and_preserve_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.env(root)
            env["WAN_VAE_PATH"] = str(root / "separate weights/vae.pth")
            command = ('source ./restored_env.sh; source ./restored_env.sh; '
                       'printf "%s\\n" "$REPO_ROOT" "$WEKA_ROOT" "$RUN_ROOT" "$WAN_VAE_PATH" "$COSMOS_FRAMEWORK_EDGE_ROOT"')
            shell = subprocess.check_output(["bash", "-e", "-c", command], cwd=REPO, env=env, text=True).splitlines()
            code = ("import json,runtime_paths as p; print(json.dumps([str(p.REPO_ROOT),str(p.WEKA_ROOT),"
                    "str(p.RUN_ROOT),str(p.WAN_VAE_PATH)]))")
            python = json.loads(subprocess.check_output([sys.executable, "-S", "-c", code], cwd=REPO, env=env, text=True))
            self.assertEqual(shell[:4], python)
            self.assertEqual(shell[0], str(REPO))
            self.assertEqual(shell[2], str(root / "runs"))
            self.assertEqual(shell[3], env["WAN_VAE_PATH"])
            self.assertEqual(shell[4], str(REPO / ".deps/cosmos-framework-edge"))

    def test_missing_custom_roots_fail_instead_of_falling_back(self):
        result = subprocess.run(["bash", "-c", "source ./restored_env.sh"], cwd=REPO,
                                env={"PATH": os.defpath, "COSMOS_SITE": "custom"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_edge_launcher_uses_custom_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.env(root)
            env["COSMOS_FRAMEWORK_EDGE_ROOT"] = str(root / "framework")
            weka = Path(env["COSMOS_STORAGE_ROOT"]) / "weka"
            stats = root / "runs/nymeria_camera_head_recanonicalization_v1/stats"
            for directory in (root / "framework/cosmos_framework", weka / "nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1"):
                directory.mkdir(parents=True)
            for path in (weka / "Cosmos3-Edge/model.safetensors.index.json",
                         weka / "cosmos3_edge_dcp/model/.metadata",
                         stats / "clean_calibrated_uniego283_mean.npy",
                         stats / "clean_calibrated_uniego283_std.npy"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            # Replace Python with a shell-only recorder. No training is run.
            recorder = root / "recorder"
            recorder.write_text('#!/bin/bash\nprintf "CHECK:%s|%s|%s\\n" "$PWD" "$BASE_CHECKPOINT_PATH" "$MOTION_STATS_MEAN"\n')
            recorder.chmod(0o700)
            env["COSMOS_PYTHON"] = str(recorder)
            result = subprocess.check_output(["bash", "motion_expert_t2m_edge/run.sh"], cwd=REPO, env=env, text=True)
            self.assertIn(f"CHECK:{root / 'framework'}|{weka / 'cosmos3_edge_dcp'}|{stats / 'clean_calibrated_uniego283_mean.npy'}", result)


if __name__ == "__main__":
    unittest.main()
