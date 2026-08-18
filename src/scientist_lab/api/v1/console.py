"""Command-center snapshot + chat. GPU stays off from this router."""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, Depends

from scientist_lab.api.errors import http_error
from scientist_lab.api.schemas import (
    ConsoleChatBody,
    ConsoleMemoryBody,
    ConsoleSessionCreateBody,
    ConsoleSessionPatchBody,
    ConsoleWorkspaceCreateBody,
    ConsoleWorkspacePatchBody,
)
from scientist_lab.api.v1.local_runs import _llm_snapshot
from scientist_lab.services.console_chat import build_console_snapshot, handle_console_turn
from scientist_lab.services.console_sessions import (
    add_memory_note,
    add_workspace_memory,
    build_agent_board,
    create_session,
    create_workspace,
    delete_memory_note,
    delete_session,
    delete_workspace,
    delete_workspace_memory,
    enrich_workspace,
    get_session,
    get_workspace,
    list_sessions,
    list_workspaces,
    patch_session,
    patch_workspace,
    session_chat,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.local_run_inspector import list_local_runs, run_local_action


def build_console_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/console", tags=["console"])

    def service_dep() -> ExperimentService:
        return get_service()

    def project_root(service: ExperimentService) -> Any:
        return service.settings.project_root

    def dispatch_for(service: ExperimentService):
        def dispatch(*, run_id: str, action: str) -> dict[str, Any]:
            return run_local_action(
                project_root(service),
                run_id,
                action=action,
                live=False,
                execute=False,
                confirm_live=False,
                confirm_execute=False,
                live_ready=False,
                llm_ready=False,
                provider="mock",
            )

        return dispatch

    def catalogs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            projects = list(service_dep().list_projects())
        except Exception:  # noqa: BLE001
            projects = []
        packs = list(list_local_runs(project_root(service_dep())).get("items") or [])
        return projects, packs

    def labeled(payload: dict[str, Any]) -> dict[str, Any]:
        projects, packs = catalogs()
        if isinstance(payload.get("items"), list):
            out = dict(payload)
            out["items"] = [
                enrich_workspace(row, projects=projects, packs=packs)
                for row in list(payload.get("items") or [])
                if isinstance(row, dict)
            ]
            return out
        return enrich_workspace(payload, projects=projects, packs=packs)

    @router.get("/snapshot")
    def snapshot(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        llm = _llm_snapshot(service)
        payload = build_console_snapshot(project_root(service), llm_status=llm)
        payload["ok"] = True
        return payload

    @router.get("/agents")
    def agents(
        workspace_id: str | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        llm = _llm_snapshot(service)
        pack_id = None
        scoped = False
        if workspace_id:
            scoped = True
            try:
                pack_id = get_workspace(service.settings.runtime_dir, workspace_id).get("pack_id")
            except FileNotFoundError:
                pack_id = None
        return build_agent_board(
            project_root(service),
            llm_status=llm,
            pack_id=str(pack_id) if pack_id else None,
            scoped=scoped,
        )

    @router.get("/workspaces")
    def workspaces(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return labeled(list_workspaces(service.settings.runtime_dir))

    @router.post("/workspaces")
    def new_workspace(
        body: ConsoleWorkspaceCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return labeled(
            create_workspace(
                service.settings.runtime_dir,
                title=body.title,
                kind=body.kind,
                project_id=body.project_id,
                pack_id=body.pack_id,
            )
        )

    @router.get("/workspaces/{workspace_id}")
    def workspace_detail(
        workspace_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return labeled(get_workspace(service.settings.runtime_dir, workspace_id))
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.patch("/workspaces/{workspace_id}")
    def workspace_patch(
        workspace_id: str,
        body: ConsoleWorkspacePatchBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return labeled(
                patch_workspace(
                service.settings.runtime_dir,
                workspace_id,
                title=body.title,
                kind=body.kind,
                project_id=body.project_id,
                pack_id=body.pack_id,
                active=body.active,
                )
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.delete("/workspaces/{workspace_id}")
    def workspace_delete(
        workspace_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return delete_workspace(service.settings.runtime_dir, workspace_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/workspaces/{workspace_id}/memory")
    def workspace_memory_add(
        workspace_id: str,
        body: ConsoleMemoryBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return labeled(add_workspace_memory(service.settings.runtime_dir, workspace_id, body.text))
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.delete("/workspaces/{workspace_id}/memory/{note_id}")
    def workspace_memory_delete(
        workspace_id: str,
        note_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return labeled(delete_workspace_memory(service.settings.runtime_dir, workspace_id, note_id))
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/sessions")
    def sessions(
        workspace_id: str | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return list_sessions(service.settings.runtime_dir, workspace_id=workspace_id)

    @router.post("/sessions")
    def new_session(
        body: ConsoleSessionCreateBody = ConsoleSessionCreateBody(),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        llm = _llm_snapshot(service)
        snap = build_console_snapshot(project_root(service), llm_status=llm)
        try:
            return create_session(
                service.settings.runtime_dir,
                snapshot=snap,
                title=body.title,
                workspace_id=body.workspace_id,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/sessions/{session_id}")
    def session_detail(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return get_session(service.settings.runtime_dir, session_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.patch("/sessions/{session_id}")
    def session_patch(
        session_id: str,
        body: ConsoleSessionPatchBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return patch_session(
                service.settings.runtime_dir,
                session_id,
                title=body.title,
                pinned=body.pinned,
                active_agent=body.active_agent,
                workspace_id=body.workspace_id,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.delete("/sessions/{session_id}")
    def session_delete(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return delete_session(service.settings.runtime_dir, session_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/sessions/{session_id}/memory")
    def session_memory_add(
        session_id: str,
        body: ConsoleMemoryBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return add_memory_note(service.settings.runtime_dir, session_id, body.text)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.delete("/sessions/{session_id}/memory/{note_id}")
    def session_memory_delete(
        session_id: str,
        note_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return delete_memory_note(service.settings.runtime_dir, session_id, note_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/sessions/{session_id}/chat")
    def session_turn(
        session_id: str,
        body: ConsoleChatBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        llm = _llm_snapshot(service)
        snapshot_payload = build_console_snapshot(
            project_root(service),
            llm_status=llm,
        )
        try:
            result = session_chat(
                service.settings.runtime_dir,
                session_id,
                message=body.message,
                agent=body.agent,
                live=bool(body.live),
                snapshot=snapshot_payload,
                root=project_root(service),
                dispatch=dispatch_for(service),
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(
                403,
                code="unsafe_operation",
                message=str(exc),
                suggested_action="聊天框只发令与解释。GPU 必须在实验闭环页二次确认。",
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        result["ok"] = True
        return result

    @router.post("/chat")
    def chat(
        body: ConsoleChatBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        llm = _llm_snapshot(service)
        snapshot_payload = build_console_snapshot(
            project_root(service),
            llm_status=llm,
        )
        try:
            reply = handle_console_turn(
                message=body.message,
                history=list(body.history or []),
                snapshot=snapshot_payload,
                runtime_dir=service.settings.runtime_dir,
                live=bool(body.live),
                root=project_root(service),
                dispatch=dispatch_for(service),
                agent=body.agent or "planner",
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except PermissionError as exc:
            raise http_error(
                403,
                code="unsafe_operation",
                message=str(exc),
                suggested_action="聊天框只发令与解释。GPU 必须在实验闭环页二次确认。",
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise http_error(
                400,
                code="bad_request",
                message=str(exc),
                suggested_action="普通对话不是 JSON。若反复出现，打开模型配置确认联网后重试，或先用 /status。",
            ) from exc
        reply["ok"] = True
        reply["snapshot"] = {
            "llm_ready": bool(llm.get("ready_for_real_calls")),
            "c1_allowed": bool(snapshot_payload.get("c1", {}).get("allowed")),
        }
        return reply

    return router
