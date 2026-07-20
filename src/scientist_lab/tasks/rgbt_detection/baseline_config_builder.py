"""Build native configs for real detection baselines."""

from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baseline_adapter import (
    ensure_baselines_loaded,
    get_baseline_adapter,
    resolve_baseline_key,
)


def build_native_config(contract: ExperimentContract) -> dict[str, Any]:
    ensure_baselines_loaded()
    key = resolve_baseline_key(contract)
    adapter = get_baseline_adapter(key)
    adapter.validate_parameters(dict(contract.parameters or {}))
    return adapter.build_native_config(contract)
