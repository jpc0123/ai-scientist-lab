"""Registered object-detection experiments. One protocol per id. No GPU."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends

from scientist_lab.api.errors import http_error
from scientist_lab.api.schemas import (
    IdeaInterviewChatBody,
    RegisteredExperimentCreateBody,
    RegisteredExperimentProposeBody,
)
from scientist_lab.api.v1.autonomous_campaigns import _campaigns
from scientist_lab.services.experiment_propose import ExperimentProposeError, propose_experiment_draft
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.idea_interview import IdeaInterviewError, IdeaInterviewService
from scientist_lab.services.registered_experiments import RegisteredExperimentError


def build_registered_experiments_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/registered-experiments", tags=["registered-experiments"])

    def service_dep() -> ExperimentService:
        return get_service()

    def _interviews(service: ExperimentService) -> IdeaInterviewService:
        root = _campaigns(service).project_root
        return IdeaInterviewService(root)

    def _resolve_provider(service: ExperimentService, *, live: bool):
        from scientist_lab.llm.fake_provider import FakeProvider
        from scientist_lab.llm.gateway import resolve_gateway_provider

        if live:
            try:
                status = service.get_llm_config_status()
            except Exception as exc:  # noqa: BLE001
                raise http_error(400, code="bad_request", message=str(exc)) from exc
            if not status.get("ready_for_real_calls"):
                raise http_error(
                    400,
                    code="bad_request",
                    message="LLM 未就绪：拒绝 Idea Interview / 起草；不会启动 GPU",
                )
            return resolve_gateway_provider(live=True)
        return FakeProvider()

    @router.get("")
    @router.get("/")
    def list_experiments(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return _campaigns(service).experiments.list()

    @router.post("/idea-interview")
    @router.post("/idea-interview/")
    def create_idea_interview(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return _interviews(service).create()

    @router.get("/idea-interview/{interview_id}")
    def get_idea_interview(
        interview_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _interviews(service).get(interview_id)
        except IdeaInterviewError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/idea-interview/{interview_id}/chat")
    def chat_idea_interview(
        interview_id: str,
        body: IdeaInterviewChatBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        live = bool(body.live)
        try:
            provider = _resolve_provider(service, live=live)
            return _interviews(service).chat(
                interview_id,
                body.message,
                live=live,
                provider=provider,
            )
        except IdeaInterviewError as exc:
            code = "not_found" if "unknown" in str(exc) else "bad_request"
            status = 404 if code == "not_found" else 400
            raise http_error(status, code=code, message=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — fail closed, no GPU
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/idea-interview/{interview_id}/clear")
    def clear_idea_interview(
        interview_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _interviews(service).clear(interview_id)
        except IdeaInterviewError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/{experiment_id}")
    def get_experiment(
        experiment_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        row = _campaigns(service).experiments.get(experiment_id)
        if row is None:
            raise http_error(404, code="not_found", message=f"unknown experiment: {experiment_id}")
        bound = [
            item
            for item in (_campaigns(service).list().get("items") or [])
            if str(item.get("experiment_id") or "") == experiment_id
        ]
        row["campaigns"] = bound
        return row

    @router.post("/propose")
    def propose_experiment(
        body: RegisteredExperimentProposeBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        live = bool(body.live)
        interview_id = str(body.interview_id or "").strip() or None
        user_intent = str(body.user_intent or "").strip() or None
        dialogue = list(body.dialogue or []) if body.dialogue else None
        idea_brief = None
        try:
            if interview_id:
                session = _interviews(service).get(interview_id)
                idea_brief = session.get("brief")
                user_intent = user_intent or str(session.get("user_intent") or "").strip() or None
                if not dialogue:
                    dialogue = [
                        {"role": row.get("role"), "text": row.get("text")}
                        for row in (session.get("turns") or [])
                        if str(row.get("text") or "").strip()
                    ]
            provider = _resolve_provider(service, live=live)
            result = propose_experiment_draft(
                provider=provider,
                live=live,
                user_intent=user_intent,
                idea_brief=idea_brief if isinstance(idea_brief, dict) else None,
                dialogue=dialogue,
                interview_id=interview_id,
            )
            if interview_id and result.get("ok"):
                try:
                    _interviews(service).mark_drafted(
                        interview_id,
                        str(result.get("experiment_id") or "") or None,
                    )
                except IdeaInterviewError:
                    pass
            return result
        except ExperimentProposeError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        except IdeaInterviewError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — fail closed, no GPU
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("")
    @router.post("/")
    def register_experiment(
        body: RegisteredExperimentCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).experiments.register(
                protocol=body.protocol,
                seed_plan=body.seed_plan,
                experiment_id=body.experiment_id,
                title=body.title,
                idea_brief=body.idea_brief,
                user_intent=body.user_intent,
                interview_id=body.interview_id,
            )
        except RegisteredExperimentError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    return router
