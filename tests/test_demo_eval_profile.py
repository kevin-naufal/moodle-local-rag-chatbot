import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DemoEvalProfileTest(unittest.TestCase):
    def test_dev_profile_uses_mini_dataset_in_dry_run(self):
        env = os.environ.copy()
        env["DEMO_EVAL_PROFILE"] = "dev"

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PROJECT_ROOT / "scripts" / "run_demo_eval.ps1"),
                "-DryRun",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Profile : dev", result.stdout)
        self.assertIn("ch03_running_time_in_scope_3q.json", result.stdout)

    def test_prod_profile_uses_configured_full_dataset_in_dry_run(self):
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PROJECT_ROOT / "scripts" / "run_demo_eval.ps1"),
                "-Profile",
                "prod",
                "-DryRun",
            ],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Profile : prod", result.stdout)
        self.assertIn("ch03_running_time_in_scope_40q.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
