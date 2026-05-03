"""
Small helper utilities used across the project.
Includes reproducibility, timestamping, run ID generation,
and safe CSV header creation.
"""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set Python + NumPy seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def now_ts() -> float:
    """
    Return current Unix timestamp.
    """
    return time.time()


def make_run_id(payload: dict[str, Any]) -> str:
    """
    Create a short deterministic run ID from a dictionary payload.
    """
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def safe_write_header_if_needed(csv_path: Path, header: list[str]) -> None:
    """
    Write CSV header only if file does not already exist.
    """
    if not csv_path.exists():
        csv_path.write_text(",".join(header) + "\n", encoding="utf-8")
