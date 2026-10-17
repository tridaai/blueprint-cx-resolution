"""The eval gate itself, run as a test: it must pass on the tree it
ships with. (Slower than a unit test by design — it plays all 31
scripted conversations.)"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_gate_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals" / "run_evals.py")],
        capture_output=True, text=True, cwd=ROOT)
    assert "PASS" in proc.stdout, proc.stdout[-2000:]
    assert proc.returncode == 0, proc.stdout[-2000:]
