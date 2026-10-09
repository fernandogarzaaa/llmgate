import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_smoke_script_against_real_processes():
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "smoke_test.py"),
            "--config",
            os.path.join(ROOT, "examples", "config.mock.yaml"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "smoke test passed" in r.stdout
