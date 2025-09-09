import os
import subprocess
import sys
from pathlib import Path


def test_validate_regime_distributions_script():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "validate_regime_distributions.py"
    ref = repo_root / "configs" / "reference_regime_distributions.json"
    env = os.environ.copy()
    env["USE_DUMMY_ENV"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), "--ref", str(ref), "--steps", "200", "--tolerance", "0.3"],
        env=env,
    )
    assert result.returncode == 0
