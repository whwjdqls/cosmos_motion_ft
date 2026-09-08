"""Regression tests for explicit Yonsei/Grasp runtime path selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class SiteStorageTest(unittest.TestCase):
    def _python_paths(self, site: str) -> dict[str, str]:
        env = os.environ.copy()
        env["COSMOS_SITE"] = site
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, runtime_paths as p; "
                    "print(json.dumps({"
                    "'repo': str(p.REPO_ROOT), "
                    "'storage': str(p.PROJECT_ROOT), "
                    "'weka': str(p.WEKA_ROOT), "
                    "'runs': str(p.RUN_ROOT), "
                    "'vae': str(p.WAN_VAE_PATH), "
                    "'legacy': p.resolve_legacy_path('/weka/jungbin/example.bin')"
                    "}))"
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def _shell_paths(self, site: str) -> list[str]:
        result = subprocess.run(
            [
                "bash",
                "-c",
                (
                    f"COSMOS_SITE={site}; export COSMOS_SITE; "
                    "source ./restored_env.sh; "
                    "printf '%s\\n' \"$REPO_ROOT\" \"$PROJECT_ROOT\" "
                    "\"$WEKA_ROOT\" \"$RUN_ROOT\" \"$WAN_VAE_PATH\""
                ),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()

    def test_grasp_python_paths_keep_weka_artifact_root(self) -> None:
        self.assertEqual(
            self._python_paths("grasp"),
            {
                "repo": "/home/jungbinc/cosmos_motion_ft",
                "storage": "/mnt/projects/ll/jungbinc",
                "weka": "/mnt/projects/ll/jungbinc/weka",
                "runs": "/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs",
                "vae": "/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth",
                "legacy": "/mnt/projects/ll/jungbinc/weka/example.bin",
            },
        )

    def test_grasp_shell_paths_keep_weka_artifact_root(self) -> None:
        self.assertEqual(
            self._shell_paths("grasp"),
            [
                "/home/jungbinc/cosmos_motion_ft",
                "/mnt/projects/ll/jungbinc",
                "/mnt/projects/ll/jungbinc/weka",
                "/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs",
                "/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth",
            ],
        )

    def test_yonsei_python_and_shell_paths(self) -> None:
        expected = {
            "repo": "/home/whwjdqls99/cosmos_motion_ft",
            "storage": "/lustre/whwjdqls99/cosmos",
            "weka": "/lustre/whwjdqls99/cosmos/weka",
            "runs": "/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs",
            "vae": "/lustre/whwjdqls99/cosmos/weka/wan22_vae/Wan2.2_VAE.pth",
            "legacy": "/lustre/whwjdqls99/cosmos/weka/example.bin",
        }
        self.assertEqual(self._python_paths("yonsei"), expected)
        self.assertEqual(
            self._shell_paths("yonsei"),
            [expected[key] for key in ("repo", "storage", "weka", "runs", "vae")],
        )


if __name__ == "__main__":
    unittest.main()
