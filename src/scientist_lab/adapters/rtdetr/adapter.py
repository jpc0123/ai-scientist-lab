"""RTDETRAdapter: HOW translation for RT-DETR. Not a fifth Agent."""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.rtdetr.how import resolve_adapter_how


class RTDETRAdapter(DFINEAdapter):
    adapter_key = "rtdetr"

    def _resolve_how(
        self, plan: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> dict[str, Any]:
        return resolve_adapter_how(plan, protocol)
