from __future__ import annotations

from typing import Any


ENVIRONMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "mock-detection-v1": {
        "image": "scientist-mock-detection:v1",
        "allowed_entrypoints": ["run_detection_experiment.py", "run_mock_job.py"],
        "network_mode": "none",
        "max_gpu_count": 0,
        "cuda_required": False,
    },
    "rgbt-detection-v2": {
        "image": "scientist-rgbt-detection:v2",
        "allowed_entrypoints": ["run_detection_experiment.py"],
        "network_mode": "none",
        "max_gpu_count": 1,
        # CPU torch image; prefer GPU when host runtime has it (may still be CPU torch).
        "cuda_required": False,
        "prefer_gpu": True,
    },
    "rgbt-detection-v2-cuda": {
        "image": "scientist-rgbt-detection:v2-cuda",
        "allowed_entrypoints": ["run_detection_experiment.py"],
        "network_mode": "none",
        "max_gpu_count": 1,
        "cuda_required": True,
        "prefer_gpu": True,
    },
}


class SecurityError(ValueError):
    pass


def require_token(provided: str | None, expected: str | None, *, require_auth: bool) -> None:
    if not require_auth and not expected:
        return
    if not expected:
        return
    if not provided or provided.strip() != expected.strip():
        raise SecurityError("remote_auth_failed")


def validate_submit_payload(
    *,
    contract: dict[str, Any],
    environment: dict[str, Any],
    supported_environment_keys: list[str],
) -> str:
    env_key = str(
        (environment or {}).get("environment_key")
        or contract.get("environment_key")
        or ""
    ).strip()
    if not env_key:
        raise SecurityError("invalid_contract: missing environment_key")
    if env_key not in supported_environment_keys:
        raise SecurityError(
            f"remote_capability_mismatch: unsupported environment_key={env_key}"
        )
    if env_key not in ENVIRONMENT_REGISTRY:
        raise SecurityError(
            f"remote_capability_mismatch: environment not registered={env_key}"
        )

    # Reject client-supplied dangerous fields if present at top-level.
    forbidden_keys = {
        "docker_image",
        "shell",
        "host_path",
        "network_mode",
        "privileged",
        "entrypoint_override",
    }
    bad = forbidden_keys.intersection(contract.keys())
    if bad:
        raise SecurityError(f"invalid_contract: forbidden fields {sorted(bad)}")

    entrypoint = str(contract.get("entrypoint") or "")
    allowed = ENVIRONMENT_REGISTRY[env_key]["allowed_entrypoints"]
    if entrypoint and entrypoint not in allowed:
        raise SecurityError(
            f"invalid_contract: entrypoint {entrypoint!r} not allowed for {env_key}"
        )
    return env_key
