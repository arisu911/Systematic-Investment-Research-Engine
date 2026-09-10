"""Pytest configuration and path setup."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in [str(_ROOT), str(_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
