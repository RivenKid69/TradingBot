import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Load stdlib logging before project paths are added
sys.path = [p for p in sys.path if p not in {str(ROOT), str(TESTS)}]
import logging  # noqa: F401
sys.path.extend([str(ROOT), str(TESTS)])
