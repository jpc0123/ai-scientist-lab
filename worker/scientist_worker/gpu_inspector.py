from __future__ import annotations

from typing import Any


def inspect_gpu() -> dict[str, Any]:
    """Best-effort GPU inspection; mock-safe when CUDA/tools are absent."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
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
            return {"count": len(devices), "devices": devices, "available": True}
    except Exception:  # noqa: BLE001
        pass
    return {
        "count": 0,
        "devices": [],
        "available": False,
        "note": "GPU not available or torch CUDA unavailable",
    }
