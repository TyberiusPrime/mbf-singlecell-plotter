import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


def test_ruff_check():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(SRC)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check failed:\n{result.stdout}{result.stderr}"


def test_ty_check():
    result = subprocess.run(
        [sys.executable, "-m", "ty", "check", str(SRC)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ty check failed:\n{result.stdout}{result.stderr}"
