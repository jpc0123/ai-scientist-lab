"""Select ExperimentAdapter from Protocol.baseline.adapter. Not a fifth Agent."""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.adapters.base import ExperimentAdapter
from scientist_lab.instrumentation.appender import EventAppender


def adapter_key_from_protocol(protocol: Mapping[str, Any] | None) -> str:
    raw = str(((protocol or {}).get("baseline") or {}).get("adapter") or "dfine")
    token = raw.strip().lower().replace("-", "_")
    if token in {"rtdetr", "rt_detr"}:
        return "rtdetr"
    return "dfine"


def adapter_for_protocol(
    protocol: Mapping[str, Any] | None,
    events: EventAppender | None = None,
) -> ExperimentAdapter:
    key = adapter_key_from_protocol(protocol)
    if key == "rtdetr":
        from scientist_lab.adapters.rtdetr.adapter import RTDETRAdapter

        return RTDETRAdapter(events=events)
    from scientist_lab.adapters.dfine.adapter import DFINEAdapter

    return DFINEAdapter(events=events)
