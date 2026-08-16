"""Parse D-FINE / RGB-T artifacts into Freeze canonical metrics.

Does not invent APS from mAP50_95. APS is taken only from explicit APS / AP_small keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.services.training_monitor import parse_dfine_log_txt, parse_training_signals


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    nested = raw.get("metrics")
    if isinstance(nested, dict) and any(
        k in nested for k in ("mAP50_95", "mAP50", "APS", "AP_small", "accuracy")
    ):
        merged = dict(nested)
        for key in ("params_m", "flops_g", "gpu_memory_gb"):
            if key in raw and key not in merged:
                merged[key] = raw[key]
        return merged
    return dict(raw)


def parse_metrics_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return {}
    return _flatten_metrics(data)


def canonicalize_metrics(raw: Mapping[str, Any]) -> dict[str, float | None]:
    src = _flatten_metrics(raw)
    aps = _as_float(src.get("APS") if src.get("APS") is not None else src.get("AP_small"))
    return {
        "APS": aps,
        "mAP50_95": _as_float(src.get("mAP50_95") if src.get("mAP50_95") is not None else src.get("mAP")),
        "mAP50": _as_float(src.get("mAP50")),
        "params_m": _as_float(src.get("params_m")),
        "flops_g": _as_float(src.get("flops_g")),
        "gpu_memory_gb": _as_float(src.get("gpu_memory_gb")),
    }


def parse_metrics(output_dir: Path | str) -> dict[str, Any]:
    """Read metrics.json, then fall back to D-FINE log.txt / combined logs."""
    root = Path(output_dir)
    sources: list[str] = []
    raw: dict[str, Any] = {}

    metrics_path = root / "metrics.json"
    if metrics_path.is_file():
        raw = parse_metrics_file(metrics_path)
        sources.append(str(metrics_path))

    log_txt = root / "log.txt"
    if log_txt.is_file():
        log_metrics = parse_dfine_log_txt(log_txt)
        sources.append(str(log_txt))
        if raw.get("mAP50_95") is None and log_metrics.get("mAP50_95") is not None:
            raw["mAP50_95"] = log_metrics["mAP50_95"]

    combined = root / "combined.log"
    if combined.is_file():
        signals = parse_training_signals(
            combined.read_text(encoding="utf-8", errors="replace")
        )
        sources.append(str(combined))
        if raw.get("mAP50_95") is None and signals.get("mAP50_95") is not None:
            raw["mAP50_95"] = signals["mAP50_95"]

    canonical = canonicalize_metrics(raw)
    return {
        "metrics": canonical,
        "raw": raw,
        "sources": sources,
    }
