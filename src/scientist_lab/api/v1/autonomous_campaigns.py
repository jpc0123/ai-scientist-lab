"""P0 autonomous campaign API: start / poll / stop. No Cursor."""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, Depends

from scientist_lab.api.errors import http_error
from scientist_lab.api.schemas import (
    AutonomousCampaignExtendBody,
    AutonomousCampaignStartBody,
    CampaignSteerBody,
    HowCandidateAuthorBody,
    HowCandidateDecideBody,
    ScoutDisplayBody,
    ScoutIntentChatBody,
    ScoutIntentDecideBody,
)
from scientist_lab.services.autonomous_campaign import (
    AutonomousCampaignError,
    AutonomousCampaignService,
)
from scientist_lab.services.live_gpu_mutex import LiveGpuBusyError
from scientist_lab.services.experiment_service import ExperimentService


def _campaigns(service: ExperimentService) -> AutonomousCampaignService:
    if service._autonomous_campaigns is None:
        service._autonomous_campaigns = AutonomousCampaignService(
            project_root=service.settings.project_root,
        )
    return service._autonomous_campaigns


def build_autonomous_campaigns_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/autonomous-campaigns", tags=["autonomous-campaigns"])

    def service_dep() -> ExperimentService:
        return get_service()

    @router.get("")
    @router.get("/")
    def list_campaigns(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = _campaigns(service).list()
        payload.setdefault("experiments", [])
        try:
            status = service.get_llm_config_status()
            payload["llm"] = {
                "ready_for_real_calls": bool(status.get("ready_for_real_calls")),
                "api_key_present": bool(status.get("api_key_present")),
                "allow_network": bool(status.get("allow_network")),
                "provider": status.get("provider"),
                "model": status.get("model"),
                "missing_for_real": status.get("missing_for_real") or [],
            }
        except Exception as exc:  # noqa: BLE001
            payload["llm"] = {
                "ready_for_real_calls": False,
                "error": str(exc),
            }
        return payload

    @router.post("/probe-gpu")
    def probe_gpu(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            report = service.dfine_cuda_doctor(probe_runtime=True)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "live_ready": False,
                "overall": "error",
                "error": str(exc),
                "metrics_forged": False,
            }
        live_ready = bool(report.get("live_ready"))
        return {
            "ok": live_ready,
            "live_ready": live_ready,
            "overall": report.get("overall"),
            "metrics_forged": False,
            "note": (
                "live_ready=false 时禁止启动真实 GPU，也不会伪造 metrics。"
                if not live_ready
                else "CUDA doctor 认为可以 live。"
            ),
        }

    @router.post("")
    @router.post("/")
    def start_campaign(
        body: AutonomousCampaignStartBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        live_ready: bool | None = None
        llm_ready: bool | None = None
        campaigns = _campaigns(service)
        stub_runner = campaigns._live_runner is not None
        if body.execute and not stub_runner:
            try:
                report = service.dfine_cuda_doctor(probe_runtime=True)
                live_ready = bool(report.get("live_ready"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "status": "blocked",
                    "gpu": False,
                    "error": f"cuda doctor error: {exc}",
                }
        if body.llm_live:
            try:
                status = service.get_llm_config_status()
                llm_ready = bool(status.get("ready_for_real_calls"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "status": "blocked",
                    "error": f"LLM config error: {exc}",
                }
        try:
            return campaigns.start(
                experiment_id=body.experiment_id,
                confirm_human_gate=body.confirm_human_gate,
                execute=body.execute,
                llm_live=body.llm_live,
                max_extra_rounds=body.max_extra_rounds,
                planner_backend=body.planner_backend,
                reviewer_backend=body.reviewer_backend,
                plugin_worker=body.plugin_worker,
                llm_may_invent_how=body.llm_may_invent_how,
                live_ready=live_ready,
                llm_ready=llm_ready,
                background=True,
            )
        except PermissionError as exc:
            raise http_error(
                403,
                code="human_gate_required",
                message=str(exc),
                suggested_action="一次授权 confirm_human_gate 后，本实验内不再逐条 Allow。",
            ) from exc
        except LiveGpuBusyError as exc:
            raise http_error(
                409,
                code="live_gpu_busy",
                message=str(exc),
                details=exc.holder,
                suggested_action="等当前实验结束或点停止；不要并行再开一场真实 GPU。",
            ) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.get("/{campaign_id}")
    def get_campaign(
        campaign_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).get(campaign_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/{campaign_id}/stop")
    def stop_campaign(
        campaign_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).request_stop(campaign_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/{campaign_id}/resume")
    def resume_campaign(
        campaign_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        campaigns = _campaigns(service)
        stub_runner = campaigns._live_runner is not None
        doc_path = (
            service.settings.project_root
            / ".run"
            / "autonomous"
            / campaign_id
            / "campaign.json"
        )
        live_ready: bool | None = None
        llm_ready: bool | None = None
        execute = False
        llm_live = False
        if doc_path.is_file():
            try:
                row = json.loads(doc_path.read_text(encoding="utf-8"))
                execute = bool(row.get("execute"))
                llm_live = bool(row.get("llm_live"))
            except (OSError, json.JSONDecodeError):
                pass
        if execute and not stub_runner:
            try:
                report = service.dfine_cuda_doctor(probe_runtime=True)
                live_ready = bool(report.get("live_ready"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "campaign_id": campaign_id,
                    "status": "paused",
                    "gpu": False,
                    "error": f"cuda doctor error: {exc}",
                }
        if llm_live:
            try:
                status = service.get_llm_config_status()
                llm_ready = bool(status.get("ready_for_real_calls"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "campaign_id": campaign_id,
                    "status": "paused",
                    "error": f"LLM config error: {exc}",
                }
        try:
            return campaigns.resume(
                campaign_id,
                live_ready=live_ready,
                llm_ready=llm_ready,
                background=True,
            )
        except PermissionError as exc:
            raise http_error(
                403,
                code="human_gate_required",
                message=str(exc),
                suggested_action="实验启动时需 confirm_human_gate；续跑沿用该授权。",
            ) from exc
        except LiveGpuBusyError as exc:
            raise http_error(
                409,
                code="live_gpu_busy",
                message=str(exc),
                details=exc.holder,
                suggested_action="等当前 GPU 实验结束，或先暂停占用方。",
            ) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/{campaign_id}/extend")
    def extend_campaign(
        campaign_id: str,
        body: AutonomousCampaignExtendBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        campaigns = _campaigns(service)
        stub_runner = campaigns._live_runner is not None
        live_ready: bool | None = None
        llm_ready: bool | None = None
        doc_path = (
            service.settings.project_root
            / ".run"
            / "autonomous"
            / campaign_id
            / "campaign.json"
        )
        execute = False
        llm_live = False
        if doc_path.is_file():
            try:
                row = json.loads(doc_path.read_text(encoding="utf-8"))
                execute = bool(row.get("execute"))
                llm_live = bool(row.get("llm_live"))
            except (OSError, json.JSONDecodeError):
                pass
        if body.resume and execute and not stub_runner:
            try:
                report = service.dfine_cuda_doctor(probe_runtime=True)
                live_ready = bool(report.get("live_ready"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "campaign_id": campaign_id,
                    "status": "paused",
                    "gpu": False,
                    "error": f"cuda doctor error: {exc}",
                }
        if body.resume and llm_live:
            try:
                status = service.get_llm_config_status()
                llm_ready = bool(status.get("ready_for_real_calls"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "fail_closed": True,
                    "metrics_forged": False,
                    "campaign_id": campaign_id,
                    "status": "paused",
                    "error": f"LLM config error: {exc}",
                }
        try:
            return campaigns.extend_round_budget(
                campaign_id,
                add_rounds=body.add_rounds,
                confirm_protocol_amendment=body.confirm_protocol_amendment,
                resume=body.resume,
                live_ready=live_ready,
                llm_ready=llm_ready,
                background=True,
            )
        except PermissionError as exc:
            raise http_error(
                403,
                code="protocol_amendment_required",
                message=str(exc),
                suggested_action="勾选确认 Protocol Amendment 后再提高额度。",
            ) from exc
        except LiveGpuBusyError as exc:
            raise http_error(
                409,
                code="live_gpu_busy",
                message=str(exc),
                details=exc.holder,
                suggested_action="等当前 GPU 实验结束，或先暂停占用方。",
            ) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/{campaign_id}/sota-pursuit")
    def sota_pursuit(
        campaign_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).run_sota_pursuit(campaign_id)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/how-candidates/{candidate_id}/author-patch")
    def author_how_candidate(
        campaign_id: str,
        candidate_id: str,
        body: HowCandidateAuthorBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).author_how_candidate(
                campaign_id,
                candidate_id,
                live=body.live,
                plugin_source=body.plugin_source,
                unified_diff=body.unified_diff,
                confirm_human_gate=body.confirm_human_gate,
                plugin_worker=body.plugin_worker,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/how-candidates/{candidate_id}/decide")
    def decide_how_candidate(
        campaign_id: str,
        candidate_id: str,
        body: HowCandidateDecideBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).decide_how_candidate(
                campaign_id,
                candidate_id,
                decision=body.decision,
                confirm_human_gate=body.confirm_human_gate,
                note=body.note,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/steer")
    def set_campaign_steer(
        campaign_id: str,
        body: CampaignSteerBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).set_campaign_steer(
                campaign_id,
                action=body.action,
                text=body.text,
                why=body.why,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/scout-intent/chat")
    def chat_scout_intent(
        campaign_id: str,
        body: ScoutIntentChatBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).chat_scout_intent(
                campaign_id,
                body.message,
                live=body.live,
                locale=body.locale,
                action=body.normalized_action()
                if body.normalized_action() in {"llm_search", "library_search"}
                else None,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/scout-intent")
    def decide_scout_intent(
        campaign_id: str,
        body: ScoutIntentDecideBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).decide_scout_intent(
                campaign_id,
                action=body.action,
                query=body.query,
                why=body.why,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except AutonomousCampaignError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/{campaign_id}/scout-display")
    def localize_scout_display(
        campaign_id: str,
        body: ScoutDisplayBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return _campaigns(service).localize_scout(
                campaign_id,
                locale=body.locale,
                live=body.live,
            )
        except FileNotFoundError as orig:
            raise http_error(404, code="not_found", message=str(orig)) from orig
        except AutonomousCampaignError as orig:
            raise http_error(400, code="bad_request", message=str(orig)) from orig

    return router
