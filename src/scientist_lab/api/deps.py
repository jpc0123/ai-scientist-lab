"""API dependency helpers (v1.7.1)."""

from __future__ import annotations

from typing import Callable

from scientist_lab.services.experiment_service import ExperimentService

ServiceFactory = Callable[[], ExperimentService]


def bind_service(factory: ServiceFactory) -> ServiceFactory:
    """Return a FastAPI-compatible dependency that yields ExperimentService."""

    def _dep() -> ExperimentService:
        return factory()

    return _dep
