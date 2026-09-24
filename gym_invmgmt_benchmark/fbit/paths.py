from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "gym_invmgmt_paper"


def install_vendor_path() -> Path:
    """Make the locked upstream snapshot importable without installing it."""
    value = str(VENDOR_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return VENDOR_ROOT
