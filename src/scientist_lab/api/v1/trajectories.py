"""Read-only trajectory workspace API: ledger events + ATDP six-tuple map."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends

from scientist_lab.api.errors import http_error
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.trajectory_workspace import list_trajectories, show_trajectory


def build_trajectories_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/trajectories", tags=["trajectories"])

    def service_dep() -> ExperimentService:
        return get_service()

    def project_root(service: ExperimentService) -> Any:
        return service.settings.project_root

    @router.get("")
    @router.get("/")
    def list_items(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return list_trajectories(project_root(service))

    @router.get("/{run_id}")
    def inspect_item(
        run_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return show_trajectory(project_root(service), run_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(403, code="permission_denied", message=str(exc)) from exc

    return router
