from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.runners.profile_repository import RunnerProfileRepository
from scientist_lab.runners.profiles import RunnerProfile


class RunnerProfileRegistry:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._repo = RunnerProfileRepository(session_factory)

    def register(
        self,
        *,
        profile_key: str,
        runner_type: str,
        endpoint: str | None = None,
        auth_token_env: str | None = None,
        allowed_environment_keys: list[str] | None = None,
        default_timeout_seconds: int = 7200,
        metadata: dict | None = None,
        enabled: bool = True,
    ) -> RunnerProfile:
        existing = self._repo.get(profile_key)
        if existing is not None:
            raise ValueError(f"runner profile already exists: {profile_key}")

        if runner_type == "remote_docker":
            if not endpoint:
                raise ValueError("remote_docker profile requires endpoint")
            if not allowed_environment_keys:
                raise ValueError(
                    "remote_docker profile requires allowed_environment_keys"
                )
        elif runner_type == "local_docker":
            endpoint = None
        else:
            raise ValueError(f"unsupported runner_type: {runner_type}")

        now = utc_now_iso()
        profile = RunnerProfile(
            profile_key=profile_key,
            runner_type=runner_type,  # type: ignore[arg-type]
            enabled=enabled,
            endpoint=endpoint,
            auth_token_env=auth_token_env,
            allowed_environment_keys=list(allowed_environment_keys or []),
            default_timeout_seconds=int(default_timeout_seconds),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        return self._repo.upsert(profile)

    def get(self, profile_key: str) -> RunnerProfile | None:
        return self._repo.get(profile_key)

    def require(self, profile_key: str, *, require_enabled: bool = True) -> RunnerProfile:
        profile = self.get(profile_key)
        if profile is None:
            raise KeyError(f"runner profile not found: {profile_key}")
        if require_enabled and not profile.enabled:
            raise ValueError(f"runner profile disabled: {profile_key}")
        return profile

    def list_profiles(self) -> list[RunnerProfile]:
        return self._repo.list_profiles()

    def check(self, profile_key: str) -> dict:
        profile = self.require(profile_key, require_enabled=False)
        result: dict = {
            "profile_key": profile.profile_key,
            "runner_type": profile.runner_type,
            "enabled": profile.enabled,
            "endpoint": profile.endpoint,
            "ok": False,
            "details": {},
        }
        if profile.runner_type == "local_docker":
            result["ok"] = True
            result["details"] = {"message": "local_docker profile is configuration-only"}
            return result
        if not profile.enabled:
            result["details"] = {"error": "profile disabled"}
            return result
        from scientist_lab.runners.remote_client import WorkerHttpClient

        token = None
        if profile.auth_token_env:
            token = os.environ.get(profile.auth_token_env)
        client = WorkerHttpClient(profile.endpoint or "", auth_token=token)
        try:
            health = client.health()
            caps = client.capabilities()
            result["ok"] = health.get("status") == "ok"
            result["details"] = {"health": health, "capabilities": caps}
        except Exception as exc:  # noqa: BLE001
            result["details"] = {"error": str(exc)}
        return result

    def ensure_defaults(self) -> None:
        if self.get("local") is None:
            self.register(
                profile_key="local",
                runner_type="local_docker",
                allowed_environment_keys=[],
                default_timeout_seconds=3600,
                metadata={"builtin": True},
            )
