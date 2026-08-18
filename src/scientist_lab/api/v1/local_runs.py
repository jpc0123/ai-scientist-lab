"""Read-only local run inspector + fail-closed replay/dry-run (Demo UI)."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query

from scientist_lab.api.errors import http_error
from scientist_lab.api.schemas import LocalRunActionBody
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.local_run_inspector import (
    inspect_local_run,
    list_local_runs,
    read_local_run_file,
    run_local_action,
)


def build_local_runs_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/local-runs", tags=["local-runs"])

    def service_dep() -> ExperimentService:
        return get_service()

    def project_root(service: ExperimentService) -> Any:
        return service.settings.project_root

    @router.get("")
    @router.get("/")
    def list_runs(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = list_local_runs(project_root(service))
        payload["doctor"] = _doctor_snapshot(service, probe=False)
        payload["llm_status"] = _llm_snapshot(service)
        return payload

    @router.get("/{run_id}")
    def inspect_run(
        run_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            payload = inspect_local_run(project_root(service), run_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(403, code="permission_denied", message=str(exc)) from exc
        payload["doctor"] = _doctor_snapshot(service, probe=False)
        payload["llm_status"] = _llm_snapshot(service)
        return payload

    @router.get("/{run_id}/file")
    def read_file(
        run_id: str,
        name: str = Query(..., min_length=1, max_length=120),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return read_local_run_file(project_root(service), run_id, name)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(403, code="permission_denied", message=str(exc)) from exc

    @router.post("/{run_id}/actions")
    def run_action(
        run_id: str,
        body: LocalRunActionBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        doctor = _doctor_snapshot(service, probe=bool(body.execute), probe_runtime=bool(body.execute))
        llm = _llm_snapshot(service)
        try:
            return run_local_action(
                project_root(service),
                run_id,
                action=body.action,
                live=body.live,
                execute=body.execute,
                confirm_live=body.confirm_live,
                confirm_execute=body.confirm_execute,
                live_ready=bool(doctor.get("live_ready")),
                llm_ready=bool(llm.get("ready_for_real_calls")),
                provider=body.provider or "mock",
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(
                403,
                code="unsafe_operation",
                message=str(exc),
                suggested_action="保持 dry-run / mock replay。真 --live / --execute 必须二次确认且 doctor 就绪。",
            ) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    return router


def _doctor_snapshot(
    service: ExperimentService, *, probe_runtime: bool = False, probe: bool = False
) -> dict[str, Any]:
    if not probe:
        return {
            "live_ready": False,
            "overall": "not_probed_on_read",
            "note": "打开 run 目录不探测 GPU。勾选 --execute 并确认后才会跑 CUDA doctor；未就绪则 fail-closed，不会伪造 metrics。",
        }
    live_ready = False
    overall = "unknown"
    try:
        report = service.dfine_cuda_doctor(probe_runtime=probe_runtime)
        live_ready = bool(report.get("live_ready"))
        overall = str(report.get("overall") or "unknown")
    except Exception as exc:  # noqa: BLE001 — UI must fail closed, not crash
        return {
            "live_ready": False,
            "overall": "error",
            "error": str(exc),
            "note": "doctor 未就绪时禁止把 GPU 失败画成成功。",
        }
    return {
        "live_ready": live_ready,
        "overall": overall,
        "note": (
            "live_ready=false 时禁止 --execute，也不会伪造 metrics。"
            if not live_ready
            else "CUDA doctor 认为可以 live；UI 仍默认关闭 --execute。"
        ),
    }


def _llm_snapshot(service: ExperimentService) -> dict[str, Any]:
    try:
        status = service.get_llm_config_status()
    except Exception as exc:  # noqa: BLE001
        return {
            "ready_for_real_calls": False,
            "api_key_present": False,
            "error": str(exc),
        }
    return {
        "ready_for_real_calls": bool(status.get("ready_for_real_calls")),
        "api_key_present": bool(status.get("api_key_present")),
        "allow_network": bool(status.get("allow_network")),
        "provider": status.get("provider"),
        "model": status.get("model"),
        "missing_for_real": status.get("missing_for_real") or [],
    }
