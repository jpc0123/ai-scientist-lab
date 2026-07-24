"""Hard gates for real Digits execution in research loops (v2.1.4)."""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.research_loop.errors import RealLoopValidationError


FORBIDDEN_ENTRYPOINTS = frozenset(
    {
        "run_mock_experiment.py",
        "mock_experiment.py",
        "run_smoke.py",
    }
)

FORBIDDEN_ENVIRONMENT_KEYS = frozenset(
    {
        "scientist-experiment-v1",  # smoke / mock image family
        "mock",
        "fake",
    }
)

REQUIRED_DIGITS_FIELDS = {
    "environment_key": "digits-mlp-v1",
    "entrypoint": "run_experiment.py",
    "dataset_reference": "sklearn:digits",
}


def digits_real_contract_defaults(
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal Digits CPU contract fields for parent enrichment."""
    proto = dict(protocol or {})
    return {
        "schema_version": "1.0",
        "runner_profile": "local",
        "environment_key": str(
            proto.get("environment_key") or REQUIRED_DIGITS_FIELDS["environment_key"]
        ),
        "code_reference": str(
            proto.get("code_reference") or "local:experiment_app"
        ),
        "dataset_reference": str(
            proto.get("dataset_reference")
            or REQUIRED_DIGITS_FIELDS["dataset_reference"]
        ),
        "entrypoint": REQUIRED_DIGITS_FIELDS["entrypoint"],
        "execution_mode": str(proto.get("execution_mode") or "fast_eval"),
        "protocol_id": proto.get("protocol_id"),
        "parameters": {
            "learning_rate": 0.001,
            "epochs": 30,
            "hidden_units": 64,
            "batch_size": 64,
            "test_size": 0.2,
            **dict(proto.get("fixed_parameters") or {}),
        },
        "resources": {
            "gpu_count": 0,
            "cpu_count": 2,
            "memory_gb": 2,
            "timeout_seconds": 300,
        },
        "expected_outputs": [
            "metrics.json",
            "model.joblib",
            "training_curve.csv",
            "confusion_matrix.csv",
            "execution.json",
            "artifact_manifest.json",
            "combined.log",
        ],
    }


def enrich_parent_contract_for_digits(
    parent_contract: Mapping[str, Any] | None,
    *,
    protocol: Mapping[str, Any] | None = None,
    project_id: str,
    node_id: str,
    title: str | None = None,
    research_goal: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Fill missing Digits real fields on a baseline parent contract."""
    base = dict(parent_contract or {})
    defaults = digits_real_contract_defaults(protocol=protocol)
    # Defaults first, then parent overrides — except safety fields forced below.
    merged = {**defaults, **base}
    merged["project_id"] = str(base.get("project_id") or project_id)
    merged["node_id"] = str(base.get("node_id") or node_id)
    if title and not merged.get("title"):
        merged["title"] = title
    if research_goal and not merged.get("research_goal"):
        merged["research_goal"] = research_goal
    if hypothesis and not merged.get("hypothesis"):
        merged["hypothesis"] = hypothesis

    parent_params = dict(base.get("parameters") or {})
    merged["parameters"] = {
        **dict(defaults.get("parameters") or {}),
        **parent_params,
    }
    if protocol and protocol.get("protocol_id"):
        merged["protocol_id"] = protocol.get("protocol_id")

    # Force Digits real execution surface (never inherit mock smoke defaults).
    for key, value in REQUIRED_DIGITS_FIELDS.items():
        if key == "environment_key":
            proto_env = (protocol or {}).get("environment_key")
            merged[key] = str(proto_env or value)
        else:
            merged[key] = value
    merged["runner_profile"] = str(merged.get("runner_profile") or "local")
    if str(merged.get("execution_mode") or "") in {"smoke_test", "mock"}:
        merged["execution_mode"] = "fast_eval"
    return merged


def assert_real_digits_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Reject mock/smoke execution paths for real-loop Digits runs."""
    entrypoint = str(contract.get("entrypoint") or "").strip()
    env_key = str(contract.get("environment_key") or "").strip()
    dataset = str(contract.get("dataset_reference") or "").strip()

    if entrypoint in FORBIDDEN_ENTRYPOINTS or entrypoint.endswith("mock_experiment.py"):
        raise RealLoopValidationError(
            f"real loop rejects mock entrypoint={entrypoint!r}; "
            f"require {REQUIRED_DIGITS_FIELDS['entrypoint']!r}"
        )
    if env_key in FORBIDDEN_ENVIRONMENT_KEYS:
        raise RealLoopValidationError(
            f"real loop rejects mock/smoke environment_key={env_key!r}; "
            f"require {REQUIRED_DIGITS_FIELDS['environment_key']!r}"
        )
    if entrypoint != REQUIRED_DIGITS_FIELDS["entrypoint"]:
        raise RealLoopValidationError(
            f"real Digits execution requires entrypoint="
            f"{REQUIRED_DIGITS_FIELDS['entrypoint']!r}, got {entrypoint!r}"
        )
    if env_key != REQUIRED_DIGITS_FIELDS["environment_key"]:
        raise RealLoopValidationError(
            f"real Digits execution requires environment_key="
            f"{REQUIRED_DIGITS_FIELDS['environment_key']!r}, got {env_key!r}"
        )
    if dataset and dataset != REQUIRED_DIGITS_FIELDS["dataset_reference"]:
        raise RealLoopValidationError(
            f"real Digits execution requires dataset_reference="
            f"{REQUIRED_DIGITS_FIELDS['dataset_reference']!r}, got {dataset!r}"
        )
    return {
        "entrypoint": entrypoint,
        "environment_key": env_key,
        "dataset_reference": dataset or REQUIRED_DIGITS_FIELDS["dataset_reference"],
        "mock_execution": False,
    }
