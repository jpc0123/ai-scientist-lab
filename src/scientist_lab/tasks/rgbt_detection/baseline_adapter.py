"""Detection baseline adapter protocol and registry (v0.8.1)."""

from __future__ import annotations

from typing import Any, Protocol

from scientist_lab.domain.contracts import ExperimentContract


class DetectionBaselineAdapter(Protocol):
    baseline_key: str

    def validate_parameters(self, parameters: dict[str, Any]) -> None: ...

    def build_native_config(self, contract: ExperimentContract) -> dict[str, Any]: ...

    def build_command_notes(self, contract: ExperimentContract) -> list[str]: ...

    def expected_checkpoint_names(self) -> list[str]: ...


_REGISTRY: dict[str, DetectionBaselineAdapter] = {}


def register_baseline(cls_or_adapter: Any) -> Any:
    """Register a baseline adapter class or instance."""
    if isinstance(cls_or_adapter, type):
        instance = cls_or_adapter()
        _REGISTRY[instance.baseline_key] = instance
        return cls_or_adapter
    _REGISTRY[cls_or_adapter.baseline_key] = cls_or_adapter
    return cls_or_adapter


def get_baseline_adapter(baseline_key: str) -> DetectionBaselineAdapter:
    key = (baseline_key or "").strip()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown baseline_key={baseline_key!r}; "
            f"registered={sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_baseline_keys() -> list[str]:
    return sorted(_REGISTRY)


def resolve_baseline_key(contract: ExperimentContract | dict[str, Any]) -> str:
    if isinstance(contract, ExperimentContract):
        params = contract.parameters or {}
    else:
        params = (contract.get("parameters") or {}) if contract else {}
    raw = params.get("baseline") or params.get("model") or "tiny_detector"
    return str(raw).strip() or "tiny_detector"


def is_real_baseline(baseline_key: str) -> bool:
    """True for non-debug baselines (torch / DFINE stand-in / future vendors)."""
    key = (baseline_key or "").strip()
    return key not in {"", "tiny_detector"}


def ensure_baselines_loaded() -> None:
    # Import side-effects register adapters.
    from scientist_lab.tasks.rgbt_detection.baselines import dfine_s as _dfine  # noqa: F401
    from scientist_lab.tasks.rgbt_detection.baselines import rtdetr_s as _rtdetr  # noqa: F401
    from scientist_lab.tasks.rgbt_detection.baselines import (  # noqa: F401
        tiny_detector as _tiny,
    )
