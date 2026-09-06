from __future__ import annotations

import re
import subprocess
from typing import Any


def _inspect_via_torch() -> dict[str, Any] | None:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_total_mb": int(props.total_memory / (1024 * 1024)),
                }
            )
        return {
            "count": len(devices),
            "devices": devices,
            "available": True,
            "source": "torch",
        }
    except Exception:  # noqa: BLE001
        return None


def _inspect_via_nvidia_smi() -> dict[str, Any] | None:
    """Fallback when host torch is CPU-only but NVIDIA driver / Docker GPU works."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    devices: list[dict[str, Any]] = []
    for line in (completed.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(re.sub(r"\D", "", parts[0]) or "0")
            memory = int(float(parts[2]))
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mb": memory,
            }
        )
    if not devices:
        return None
    return {
        "count": len(devices),
        "devices": devices,
        "available": True,
        "source": "nvidia-smi",
    }


def inspect_gpu() -> dict[str, Any]:
    """Best-effort GPU inspection; mock-safe when CUDA/tools are absent."""
    via_torch = _inspect_via_torch()
    if via_torch is not None:
        return via_torch
    via_smi = _inspect_via_nvidia_smi()
    if via_smi is not None:
        return via_smi
    return {
        "count": 0,
        "devices": [],
        "available": False,
        "note": "GPU not available (torch CUDA and nvidia-smi both unavailable)",
    }
