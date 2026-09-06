"""ExperimentAdapter contract (HOW only). Planner owns WHAT/WHY."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class MaterializeRejected(ValueError):
    """Plan semantics insufficient; Adapter must not invent a hypothesis."""


class ExperimentAdapter(Protocol):
    adapter_key: str

    def materialize_contract(
        self, plan: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Translate Plan+Protocol into a candidate ExperimentContract."""

    def validate_contract(self, candidate: Mapping[str, Any]) -> None:
        ...

    def execute(self, contract: Mapping[str, Any], protocol: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        ...

    def parse_metrics(self, output_dir: Any) -> dict[str, Any]:
        ...
