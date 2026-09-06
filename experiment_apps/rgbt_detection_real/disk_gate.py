"""Disk space gates for formal / long-budget RGB-T runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def disk_usage_gb(path: Path | str) -> dict[str, float]:
    usage = shutil.disk_usage(str(path))
    return {
        "total_gb": usage.total / (1024**3),
        "used_gb": usage.used / (1024**3),
        "free_gb": usage.free / (1024**3),
    }


def assert_disk_for_formal_train(
    path: Path | str,
    *,
    min_free_gb_start: float = 20.0,
    min_free_gb_hard: float = 10.0,
) -> dict[str, Any]:
    """Refuse to start formal training when free space is too low.

    - free < min_free_gb_start → reject start
    - free < min_free_gb_hard → hard block (also used before large checkpoint writes)
    """
    info = disk_usage_gb(path)
    free = float(info["free_gb"])
    status = {
        **info,
        "min_free_gb_start": float(min_free_gb_start),
        "min_free_gb_hard": float(min_free_gb_hard),
        "path": str(path),
        "ok_to_start": free >= float(min_free_gb_start),
        "ok_to_write_checkpoint": free >= float(min_free_gb_hard),
    }
    if free < float(min_free_gb_hard):
        raise RuntimeError(
            f"disk_hard_block: free={free:.2f}GB < {min_free_gb_hard}GB on {path}"
        )
    if free < float(min_free_gb_start):
        raise RuntimeError(
            f"disk_start_rejected: free={free:.2f}GB < {min_free_gb_start}GB on {path}; "
            "formal candidate training refused"
        )
    return status
