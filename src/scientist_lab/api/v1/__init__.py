"""REST API v1 routers for Scientist Lab Web Console (v1.7.1)."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query

from scientist_lab import API_VERSION
from scientist_lab.api.errors import http_error
from scientist_lab.api.pagination import clamp_limit, clamp_offset, page_response
from scientist_lab.api.schemas import (
    AuditBuildBody,
    AuditExportBody,
    CandidateRejectBody,
    ClaimMatrixBuildBody,
    CodeContextBuildBody,
    CompareExecutionsBody,
    CompareNodesBody,
    CompareTriadBody,
    DemoCreateBody,
    IterationFinalizeBody,
    LlmConfigUpdateBody,
    LiteratureConfigUpdateBody,
    LiteratureProbeBody,
    LlmProfileRegisterBody,
    MergeApproveBody,
    MergeFinalizeBody,
    MergePrepareBody,
    MergeRejectBody,
    MergeRollbackBody,
    MergeTestBody,
    PatchApplySandboxBody,
    PatchCheckSealBody,
    PatchExportReplayBody,
    PatchProposeRealBody,
    PlanNextBody,
    ProjectCreateBody,
    ProjectExportBody,
    ProjectImportBody,
    RealLoopApproveBody,
    RealLoopCreateBody,
    RealLoopExecuteBody,
    RealLoopExportBody,
    RealLoopPlanBody,
    RealLoopRejectBody,
    RealLoopReviewBody,
    RealLoopVerifyBody,
    ReleaseCandidateCreateBody,
    PatchDecideMergeBody,
    PatchRecordEvidenceBody,
    PatchTestBody,
    ReasonBody,
    RecoverBody,
    ReleaseCreateBody,
    ReleaseDiscardBody,
    ReleaseFreezeBody,
    ReportBuildBody,
    TreeAdvanceBody,
    TreeApproveBody,
    TreeCreateBody,
    TreePlanNextBody,
    TreeStopBody,
)
from scientist_lab.services.experiment_service import ExperimentService


def build_v1_router(get_service: Callable[[], ExperimentService]) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["v1"])

    def service_dep() -> ExperimentService:
        return get_service()

    # --- health / system -------------------------------------------------
    @router.get("/health")
    def health(service: ExperimentService = Depends(service_dep)) -> dict[str, Any]:
        docker_ok = False
        docker_error = None
        try:
            service.runner.client.ping()
            docker_ok = True
        except Exception as exc:  # noqa: BLE001
            docker_error = str(exc)
        return {
            "ok": True,
            "version": API_VERSION,
            "api": "v1",
            "docker_ok": docker_ok,
            "docker_error": docker_error,
            "db_path": str(service.settings.db_path),
        }

    @router.get("/system/summary")
    def system_summary(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.system_summary()

    @router.get("/system/path-policy")
    def path_policy() -> dict[str, Any]:
        from scientist_lab.patching.path_policy import PathPolicy

        policy = PathPolicy()
        return {
            "allowed_prefixes": list(policy.allowed_prefixes),
            "denied_prefixes": list(policy.denied_prefixes),
            "denied_names": list(policy.denied_names),
            "denied_suffixes": list(policy.denied_suffixes),
            "denied_substrings": list(policy.denied_substrings),
            "extra_denied_paths": list(policy.extra_denied_paths),
            "note": "Frontend display only; enforcement remains server-side.",
        }

    @router.get("/system/security")
    def system_security(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.security_posture()

    @router.get("/system/llm-config")
    def get_llm_config(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.get_llm_config_status()

    @router.post("/system/llm-config")
    def update_llm_config(
        body: LlmConfigUpdateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.update_llm_config(
                provider=body.provider,
                base_url=body.base_url,
                model=body.model,
                timeout_seconds=body.timeout_seconds,
                api_key=body.api_key,
                allow_network=body.allow_network,
                clear_api_key=body.clear_api_key,
            )
        except Exception as exc:  # noqa: BLE001 — map config errors
            raise http_error(
                400, code="llm_config_invalid", message=str(exc)
            ) from exc

    @router.get("/system/literature-config")
    def get_literature_config(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.get_literature_config_status()

    @router.post("/system/literature-config")
    def update_literature_config(
        body: LiteratureConfigUpdateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.update_literature_config(
                api_key=body.api_key,
                base_url=body.base_url,
                clear_api_key=body.clear_api_key,
            )
        except Exception as exc:  # noqa: BLE001
            raise http_error(
                400, code="literature_config_invalid", message=str(exc)
            ) from exc

    @router.post("/system/literature-config/probe")
    def probe_literature_config(
        body: LiteratureProbeBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        import re

        from scientist_lab.llm.config import redact_secrets
        from scientist_lab.llm.errors import LLMError, MissingAPIKeyError
        from scientist_lab.literature.errors import LiteratureError, LiteratureProviderError

        def _probe_message(exc: BaseException) -> str:
            text = redact_secrets(str(exc) or type(exc).__name__)
            return re.sub(r"\bs2-[A-Za-z0-9\-._]{6,}\b", "[REDACTED]", text)

        try:
            return service.probe_literature_search(query=body.query, limit=body.limit)
        except MissingAPIKeyError as exc:
            raise http_error(
                409,
                code="missing_api_key",
                message=_probe_message(exc),
                suggested_action="在文献配置页填写并保存 Semantic Scholar Key，再点「试搜一篇」。",
            ) from exc
        except (LiteratureProviderError, LiteratureError, LLMError, ValueError) as exc:
            raise http_error(
                502,
                code="literature_probe_failed",
                message=_probe_message(exc),
                retryable=bool(getattr(exc, "retryable", False)),
                details={"provider": "semantic_scholar"},
                suggested_action="确认 Key 有效且本机可访问 api.semanticscholar.org，然后重试。",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface probe failures to the UI
            raise http_error(
                502,
                code="literature_probe_failed",
                message=_probe_message(exc),
                retryable=True,
                details={"provider": "semantic_scholar"},
                suggested_action="可稍后重试；若持续失败，检查网络与 Semantic Scholar 服务状态。",
            ) from exc

    @router.get("/llm-profiles")
    def list_llm_profiles(
        enabled_only: bool = Query(False),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_llm_profiles(enabled_only=enabled_only)
        return {
            "items": items,
            "default_profile_id": service.llm_evals.get_default_profile_id(),
            "total": len(items),
        }

    @router.post("/llm-profiles")
    def register_llm_profile(
        body: LlmProfileRegisterBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        meta = dict(body.metadata or {})
        for key in list(meta.keys()):
            low = str(key).lower()
            if any(tok in low for tok in ("api_key", "authorization", "token", "password")):
                raise http_error(
                    400,
                    code="llm_profile_secret_forbidden",
                    message="profile metadata must not contain secrets",
                )
        try:
            profile = service.register_llm_profile(profile=body.model_dump())
        except Exception as exc:  # noqa: BLE001
            raise http_error(
                400, code="llm_profile_invalid", message=str(exc)
            ) from exc
        return {"profile": profile}

    @router.get("/llm-profiles/{profile_id}")
    def show_llm_profile(
        profile_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return {"profile": service.show_llm_profile(profile_id)}
        except KeyError as exc:
            raise http_error(
                404, code="llm_profile_not_found", message=str(exc)
            ) from exc

    @router.post("/llm-profiles/{profile_id}/select")
    def select_llm_profile(
        profile_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.select_llm_profile(profile_id)
        except KeyError as exc:
            raise http_error(
                404, code="llm_profile_not_found", message=str(exc)
            ) from exc

    @router.get("/system/doctor")
    def system_doctor(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.system_doctor()

    @router.post("/system/recover")
    def system_recover(
        body: RecoverBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = body or RecoverBody()
        return service.recover(dry_run=payload.dry_run)

    @router.post("/demo/seed-patch")
    def seed_demo_patch(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        """Create a mock PatchProposal so first-time users can explore the UI."""
        from datetime import datetime, timezone

        from scientist_lab.patching.service import build_mock_unified_diff

        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        relative = (
            "experiment_apps/rgbt_detection_real/adapters/"
            f"console_demo_note_{stamp}.md"
        )
        return service.patches.propose_mock(
            "project_demo_console",
            title="演示补丁：记录融合消融证据缺口",
            rationale=(
                "Web Console 演示用 Mock 补丁。仅文档变更，"
                "须人工审批后才能沙箱应用；不会修改主工作区。"
            ),
            unified_diff=build_mock_unified_diff(relative_path=relative),
        )

    @router.get("/demo/catalog")
    def demo_catalog(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return {"items": service.demos.list_demos()}

    @router.post("/demo/create")
    def demo_create(
        body: DemoCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.demos.create(body.kind, force=body.force)
        except ValueError as exc:
            raise http_error(400, code="invalid_request", message=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise http_error(500, code="demo_create_failed", message=str(exc)) from exc

    # --- projects --------------------------------------------------------
    @router.get("/projects")
    def list_projects(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_projects()
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/projects/{project_id}")
    def get_project(
        project_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.get_project(project_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/projects")
    def create_project(
        body: ProjectCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.create_project(
                title=body.title,
                research_question=body.research_question,
                research_goal=body.research_goal,
                description=body.description,
                task_type=body.task_type,
                dataset_keys=body.dataset_keys,
                protocol_ids=body.protocol_ids,
                runner_profile_keys=body.runner_profile_keys,
                default_llm_profile_id=body.default_llm_profile_id,
                expected_metrics=body.expected_metrics,
                constraints=body.constraints,
                protocol_draft=body.protocol_draft,
                project_id=body.project_id,
                mark_ready=body.mark_ready,
            )
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/projects/{project_id}/archive")
    def archive_project(
        project_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.archive_project(project_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/projects/{project_id}/export")
    def export_project(
        project_id: str,
        body: ProjectExportBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.export_project(project_id, output_dir=body.output_dir)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/projects/import")
    def import_project(
        body: ProjectImportBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.import_project(body.path, force=body.force)
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/projects/{project_id}/budget")
    def get_project_budget(
        project_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_budget(project_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/projects/{project_id}/plan-next")
    def project_plan_next(
        project_id: str,
        body: PlanNextBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = body or PlanNextBody()
        try:
            return service.plan_next(
                project_id,
                protocol_id=payload.protocol_id,
                current_best_node_id=payload.current_best_node_id,
                max_new_nodes=payload.max_new_nodes,
                max_gpu_hours=payload.max_gpu_hours,
                provider=payload.provider,
                allow_network=payload.allow_network,
                model_profile=payload.model_profile,
                require_quality_gate=payload.require_quality_gate,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/datasets")
    def list_datasets_api(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return {"items": service.list_datasets()}

    @router.get("/protocols")
    def list_protocols_api(
        project_id: str | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return {"items": service.list_protocols(project_id=project_id)}

    @router.get("/runner-profiles")
    def list_runner_profiles_api(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return {"items": service.list_runner_profiles()}

    # --- executions ------------------------------------------------------
    @router.get("/executions")
    def list_executions(
        project_id: str | None = None,
        node_id: str | None = None,
        status: str | None = None,
        runner_profile: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        # Fetch a larger window then page in-memory for stable contract.
        attempts = service.list_executions(
            limit=max(limit + offset, 200),
            project_id=project_id,
            node_id=node_id,
        )
        items: list[dict[str, Any]] = []
        for attempt in attempts:
            payload = attempt.model_dump(mode="json")
            if not payload.get("project_id"):
                node = service.repo.get_node(attempt.node_id)
                if node is not None:
                    payload["project_id"] = node.project_id
            items.append(payload)
        if status:
            wanted = {part.strip() for part in status.split(",") if part.strip()}
            items = [item for item in items if str(item.get("status") or "") in wanted]
        if runner_profile:
            items = [
                item
                for item in items
                if str(item.get("runner_profile") or "") == runner_profile
            ]
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/executions/{execution_id}")
    def get_execution(
        execution_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.get_execution(execution_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/executions/{execution_id}/logs")
    def get_execution_logs(
        execution_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            text = service.get_log_text(execution_id)
            return {"execution_id": execution_id, "log": text}
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/training/monitor")
    def training_monitor(
        project_id: str | None = None,
        limit: int = Query(40, ge=1, le=100),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        """Live training progress: active runs, failures, gate campaigns, epoch/mAP."""
        return service.training_monitor(project_id=project_id, limit=limit)

    @router.get("/executions/{execution_id}/failure-classification")
    def get_failure_classification(
        execution_id: str,
        persist: bool = Query(False),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        """Classify an incomplete/failed execution (engineering vs scientific)."""
        try:
            return service.classify_execution_failure(
                execution_id, persist=persist
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/nodes")
    def list_nodes(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_nodes(project_id=project_id)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.post("/comparisons/executions")
    def compare_executions_api(
        body: CompareExecutionsBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.compare_executions(
                body.execution_id_a, body.execution_id_b
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/comparisons/nodes")
    def compare_nodes_api(
        body: CompareNodesBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.compare_nodes(body.node_id_a, body.node_id_b)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/comparisons/node-groups")
    def compare_node_groups_api(
        body: CompareNodesBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.compare_node_groups(body.node_id_a, body.node_id_b)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/comparisons/fast-eval-triad")
    def compare_fast_eval_triad_api(
        body: CompareTriadBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = body or CompareTriadBody()
        try:
            return service.compare_fast_eval_triad(
                rgb_node_id=payload.rgb_node_id,
                thermal_node_id=payload.thermal_node_id,
                fusion_node_id=payload.fusion_node_id,
                write_report=payload.write_report,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- trees -----------------------------------------------------------
    @router.get("/trees")
    def list_trees(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_trees(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/trees/{tree_id}")
    def get_tree(
        tree_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.tree_show(tree_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/trees")
    def create_tree(
        body: TreeCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.tree_create(
                body.project_id,
                root_node_id=body.root_node_id,
                protocol_id=body.protocol_id,
                max_depth=body.max_depth,
                max_nodes=body.max_nodes,
                max_children=body.max_children,
                tree_id=body.tree_id,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/trees/{tree_id}/plan-next")
    def tree_plan_next(
        tree_id: str,
        body: TreePlanNextBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = body or TreePlanNextBody()
        try:
            return service.tree_plan_next(
                tree_id,
                rescore=payload.rescore,
                max_gpu_hours=payload.max_gpu_hours,
                provider=payload.provider,
                allow_network=payload.allow_network,
                model_profile=payload.model_profile,
                require_quality_gate=payload.require_quality_gate,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/trees/{tree_id}/approve")
    def tree_approve(
        tree_id: str,
        body: TreeApproveBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.tree_approve(
                tree_id,
                body.candidate_id,
                seeds=body.seeds or None,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/trees/{tree_id}/advance")
    def tree_advance(
        tree_id: str,
        body: TreeAdvanceBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        payload = body or TreeAdvanceBody()
        try:
            return service.tree_advance(tree_id, tree_node_id=payload.tree_node_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/trees/{tree_id}/stop")
    def tree_stop(
        tree_id: str,
        body: TreeStopBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.tree_stop(tree_id, reason=body.reason)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/trees/{tree_id}/evidence")
    def tree_evidence(
        tree_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.tree_evidence(tree_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/trees/{tree_id}/nodes")
    def get_tree_nodes(
        tree_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            nodes = service.tree_nodes(tree_id)
            return {"tree_id": tree_id, "items": nodes, "count": len(nodes)}
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/trees/{tree_id}/mermaid")
    def get_tree_mermaid(
        tree_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            payload = service.tree_export(tree_id, format="mermaid")
            return {
                "tree_id": tree_id,
                "format": "mermaid",
                "mermaid": payload.get("mermaid") or payload.get("content") or "",
                "export": payload,
            }
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    # --- plans -----------------------------------------------------------
    @router.get("/plans")
    def list_plans(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_plans(project_id=project_id)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/plans/{plan_id}")
    def get_plan(
        plan_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_plan(plan_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/plans/{plan_id}/review")
    def review_plan(
        plan_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.review_plan(plan_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/plans/{plan_id}/rank")
    def rank_plan(
        plan_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            ranked = service.rank_candidates(plan_id)
            return service.show_plan(plan_id) | {"rank_result": ranked}
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/plans/{plan_id}/candidates/{candidate_id}/generate-contract")
    def generate_contract_from_plan(
        plan_id: str,
        candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.generate_contract_from_plan(plan_id, candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/plans/{plan_id}/candidates/{candidate_id}/approve")
    def approve_candidate(
        plan_id: str,
        candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.approve_candidate(plan_id, candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/plans/{plan_id}/candidates/{candidate_id}/reject")
    def reject_candidate(
        plan_id: str,
        candidate_id: str,
        body: CandidateRejectBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        reason = (body.reason if body else None) or None
        try:
            return service.reject_candidate(plan_id, candidate_id, reason=reason)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- iterations ------------------------------------------------------
    @router.get("/iterations")
    def list_iterations(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_iterations(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/iterations/{iteration_id}")
    def get_iteration(
        iteration_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_iteration(iteration_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/iterations/{iteration_id}/approve")
    def approve_iteration(
        iteration_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.approve_iteration(iteration_id, wait=False)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/iterations/{iteration_id}/advance")
    def advance_iteration(
        iteration_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.advance_iteration(iteration_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/iterations/{iteration_id}/finalize")
    def finalize_iteration(
        iteration_id: str,
        body: IterationFinalizeBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.finalize_iteration(
                iteration_id,
                selected_node_id=body.selected_node_id,
                reason=body.reason,
                decision_type=body.decision_type,
                evidence_strength=body.evidence_strength,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- evidence / claims -----------------------------------------------
    @router.get("/evidence")
    def list_evidence(
        project_id: str | None = None,
        protocol_id: str | None = None,
        evidence_type: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_evidence(
            project_id=project_id,
            protocol_id=protocol_id,
            evidence_type=evidence_type,
        )
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/evidence/{evidence_id}")
    def get_evidence(
        evidence_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_evidence(evidence_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/claims")
    def list_claims(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_claims(project_id=project_id)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/claims/matrix")
    def get_claim_matrix(
        project_id: str = Query(...),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_claim_matrix(project_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/claims/matrix/build")
    def build_claim_matrix(
        body: ClaimMatrixBuildBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.build_claim_matrix(
                body.project_id, protocol_id=body.protocol_id
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    # --- reports / audits ------------------------------------------------
    @router.get("/reports")
    def list_reports(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_reports(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/reports/{report_id}")
    def get_report(
        report_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_report(report_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/reports/{report_id}/markdown")
    def get_report_markdown(
        report_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from pathlib import Path

        try:
            report = service.show_report(report_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        md_path = report.get("markdown_path")
        summary_path = report.get("summary_path")
        markdown = ""
        summary = ""
        if md_path and Path(md_path).is_file():
            markdown = Path(md_path).read_text(encoding="utf-8")
        if summary_path and Path(summary_path).is_file():
            summary = Path(summary_path).read_text(encoding="utf-8")
        return {
            "report_id": report_id,
            "project_id": report.get("project_id"),
            "markdown_path": md_path,
            "summary_path": summary_path,
            "markdown": markdown,
            "summary": summary,
            "has_markdown": bool(markdown),
        }

    @router.post("/reports/{report_id}/verify")
    def verify_report(
        report_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.verify_report(report_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/reports/{report_id}/export.json")
    def export_report_json(
        report_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            report = service.show_report(report_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        return {
            "report_id": report_id,
            "format": "json",
            "report": report,
        }

    @router.post("/reports/build")
    def build_report(
        body: ReportBuildBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.build_report(
                body.project_id,
                tree_id=body.tree_id,
                protocol_id=body.protocol_id,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.get("/audits")
    def list_audits(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_audits(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/audits/{bundle_id}")
    def get_audit(
        bundle_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_audit(bundle_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/audits/build")
    def build_audit(
        body: AuditBuildBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.build_audit(
                body.project_id,
                tree_id=body.tree_id,
                protocol_id=body.protocol_id,
                report_id=body.report_id,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/audits/{bundle_id}/verify")
    def verify_audit(
        bundle_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.verify_audit(bundle_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/audits/{bundle_id}/export")
    def export_audit(
        bundle_id: str,
        body: AuditExportBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from pathlib import Path

        try:
            result = service.export_audit(bundle_id, body.output_dir)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        if body.release_id:
            try:
                release = service.releases.mark_exported(
                    body.release_id,
                    export_path=str(result.get("exported_to") or body.output_dir),
                )
                result = {**result, "release": release}
            except KeyError as exc:
                raise http_error(404, code="not_found", message=str(exc)) from exc
            except ValueError as exc:
                raise http_error(409, code="conflict", message=str(exc)) from exc
        # Safety: export must not write into project_root source tree accidentally
        # beyond the caller-chosen output_dir.
        _ = Path(body.output_dir)
        return result

    # --- releases / workspaces (v1.8) ------------------------------------
    @router.get("/workspaces/summary")
    def workspaces_summary(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.workspace_summary()

    @router.get("/releases")
    def list_releases(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_releases(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/releases/{release_id}")
    def get_release(
        release_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_release(release_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/releases")
    def create_release(
        body: ReleaseCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.create_release(
                project_id=body.project_id,
                title=body.title,
                tree_id=body.tree_id,
                report_id=body.report_id,
                audit_bundle_id=body.audit_bundle_id,
                patch_ids=body.patch_ids,
            )
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/releases/{release_id}/freeze")
    def freeze_release(
        release_id: str,
        body: ReleaseFreezeBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        notes = body.notes if body else ""
        try:
            return service.freeze_release(release_id, notes=notes)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/releases/{release_id}/discard")
    def discard_release(
        release_id: str,
        body: ReleaseDiscardBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        reason = body.reason if body else ""
        try:
            return service.discard_release(release_id, reason=reason)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- merges (v1.9) ---------------------------------------------------
    @router.get("/merges")
    def list_merges(
        project_id: str | None = None,
        patch_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_merge_candidates(
            project_id=project_id, patch_id=patch_id, limit=500
        )
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/merges/profiles")
    def list_merge_profiles(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return {"items": service.list_merge_profiles()}

    @router.get("/merges/{merge_candidate_id}")
    def get_merge(
        merge_candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_show(merge_candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/merges/prepare")
    def prepare_merge(
        body: MergePrepareBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_prepare(
                body.patch_id, target_branch=body.target_branch
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/apply")
    def apply_merge_workspace(
        merge_candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_apply(merge_candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/test")
    def test_merge_workspace(
        merge_candidate_id: str,
        body: MergeTestBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        profile = body.profile if body else "smoke"
        try:
            return service.merge_test(merge_candidate_id, profile_id=profile)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/approve")
    def approve_merge(
        merge_candidate_id: str,
        body: MergeApproveBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_approve(
                merge_candidate_id,
                reason=(body.reason if body else "") or "",
                approved_by=(body.approved_by if body else "human") or "human",
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/reject")
    def reject_merge(
        merge_candidate_id: str,
        body: MergeRejectBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_reject(
                merge_candidate_id,
                reason=(body.reason if body else "") or "",
                approved_by=(body.approved_by if body else "human") or "human",
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/commit")
    def commit_merge(
        merge_candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_commit(merge_candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/finalize")
    def finalize_merge(
        merge_candidate_id: str,
        body: MergeFinalizeBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        profile = body.post_merge_profile if body else "syntax"
        auto_rb = body.auto_rollback_on_failure if body else True
        try:
            return service.merge_finalize(
                merge_candidate_id,
                post_merge_profile=profile,
                auto_rollback_on_failure=auto_rb,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/merges/{merge_candidate_id}/rollback")
    def rollback_merge(
        merge_candidate_id: str,
        body: MergeRollbackBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.merge_rollback(
                merge_candidate_id,
                reason=(body.reason if body else "") or "",
                trigger=(body.trigger if body else "human") or "human",
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- release candidates (v1.9.8) -------------------------------------
    @router.get("/release-candidates")
    def list_release_candidates(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_release_candidates(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/release-candidates/{release_candidate_id}")
    def get_release_candidate(
        release_candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_release_candidate(release_candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/release-candidates")
    def create_release_candidate(
        body: ReleaseCandidateCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.create_release_candidate(
                version=body.version,
                project_id=body.project_id,
                base_tag=body.base_tag,
                merge_candidate_ids=body.merge_candidate_ids,
                notes=body.notes,
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/release-candidates/{release_candidate_id}/verify")
    def verify_release_candidate(
        release_candidate_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.verify_release_candidate(release_candidate_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- patches ---------------------------------------------------------
    @router.get("/patches")
    def list_patches(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.list_patches(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/patches/sandbox-profiles")
    def list_patch_sandbox_profiles(
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.patches.list_sandbox_test_profiles()

    @router.post("/patches/propose-real")
    def propose_patch_real(
        body: PatchProposeRealBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.patching.real_mode import PatchRealModeError
        from scientist_lab.patching.real_patch_planner import RealPatchPlannerError

        try:
            return service.patches.propose_real(
                body.bundle_id,
                requested_provider=body.provider,
                allow_network=bool(body.allow_network),
                real_only=bool(body.real_only),
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (
            PatchRealModeError,
            RealPatchPlannerError,
            ValueError,
        ) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/patches/{patch_id}")
    def get_patch(
        patch_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.show_patch(patch_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/approve")
    def approve_patch(
        patch_id: str,
        body: ReasonBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.patches.approve(
                patch_id, reason=(body.reason if body else "") or ""
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/reject")
    def reject_patch(
        patch_id: str,
        body: ReasonBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.patches.reject(
                patch_id, reason=(body.reason if body else "") or ""
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/apply-sandbox")
    def apply_patch_sandbox(
        patch_id: str,
        body: PatchApplySandboxBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        force = bool(body.force) if body else False
        try:
            return service.patches.apply_sandbox(patch_id, force=force)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/test-sandbox")
    def test_patch_sandbox(
        patch_id: str,
        body: PatchTestBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        profile = body.profile if body else "smoke"
        try:
            return service.patches.test_sandbox(patch_id, profile=profile)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/record-evidence")
    def record_patch_evidence(
        patch_id: str,
        body: PatchRecordEvidenceBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        require_tests = bool(body.require_tests) if body else False
        try:
            return service.patches.record_evidence(
                patch_id, require_tests=require_tests
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/decide-merge")
    def decide_patch_merge(
        patch_id: str,
        body: PatchDecideMergeBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.patches.decide_merge(
                patch_id, decision=body.decision, reason=body.reason
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/check-seal")
    def check_patch_approval_seal(
        patch_id: str,
        body: PatchCheckSealBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.patching.approval_seal import ApprovalSealError

        persist = bool(body.persist) if body else True
        try:
            return service.patches.check_approval_seal(patch_id, persist=persist)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (ApprovalSealError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/patches/{patch_id}/export-replay")
    def export_patch_replay(
        patch_id: str,
        body: PatchExportReplayBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from pathlib import Path

        from scientist_lab.patching.replay_bundle import PatchReplayError

        payload = body or PatchExportReplayBody()
        out = (
            Path(payload.output_dir)
            if payload.output_dir
            else (
                Path(service.patches.outputs_root)
                / "_patch_replays"
                / patch_id
            )
        )
        try:
            return service.patches.export_patch_replay(
                patch_id,
                output_dir=out,
                label=payload.label or "patch_replay",
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (PatchReplayError, ValueError, OSError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    # --- code contexts (v2.2.1 / Web v2.2.9) ------------------------------
    @router.post("/code-contexts/build")
    def build_code_context(
        body: CodeContextBuildBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.patching.context_bundle import ContextBundleError
        from scientist_lab.patching.context_models import PatchRequest

        payload = body or CodeContextBuildBody()
        try:
            request = None
            if not payload.digits_demo:
                if not payload.project_id:
                    raise ValueError(
                        "project_id required when digits_demo=false"
                    )
                request = PatchRequest(
                    request_id=f"preq_web_{payload.project_id}",
                    project_id=payload.project_id,
                    goal="Web code context",
                    failure_summary="Built from Web Patch workbench",
                )
            return service.patches.build_code_context(
                request,
                persist=bool(payload.persist),
                bundle_id=payload.bundle_id,
                digits_demo=bool(payload.digits_demo),
            )
        except (ContextBundleError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/code-contexts")
    def list_code_contexts(
        project_id: str = Query(...),
        limit: int = Query(50, ge=1, le=200),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.patches.list_code_contexts(
            project_id, limit=clamp_limit(limit)
        )
        return {
            "items": [b.model_dump(mode="json") for b in items],
            "total": len(items),
            "project_id": project_id,
        }

    @router.get("/code-contexts/{bundle_id}")
    def get_code_context(
        bundle_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.patches.show_code_context(bundle_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/system/patch-provider-doctor")
    def patch_provider_doctor(
        allow_network: bool = Query(False),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.patches.patch_provider_doctor(
            allow_network=bool(allow_network)
        )

    @router.get("/system/patch-budget")
    def patch_budget_show(
        project_id: str = Query(...),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        return service.patches.show_patch_budget(project_id)

    # --- real research loops (v2.1.8) ------------------------------------
    @router.get("/real-loops")
    def list_real_loops(
        project_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        items = service.real_loop_list(project_id=project_id, limit=500)
        return page_response(
            items, limit=clamp_limit(limit), offset=clamp_offset(offset)
        )

    @router.get("/real-loops/{session_id}")
    def get_real_loop(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import RealLoopNotFoundError

        try:
            return service.real_loop_show(session_id)
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/real-loops")
    def create_real_loop(
        body: RealLoopCreateBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopValidationError,
        )

        try:
            return service.real_loop_create(
                body.project_id,
                profile_id=body.profile_id,
                protocol_id=body.protocol_id,
                rounds=body.rounds,
                baseline_node_ids=list(body.baseline_node_ids or []),
                tree_id=body.tree_id,
            )
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/check")
    def check_real_loop(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import RealLoopNotFoundError

        try:
            return service.real_loop_check(session_id)
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/plan")
    def plan_real_loop(
        session_id: str,
        body: RealLoopPlanBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopPlanBody()
        try:
            return service.real_loop_plan(
                session_id,
                round_number=payload.round_number,
                allow_network=payload.allow_network,
                provider=payload.provider,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/review")
    def review_real_loop(
        session_id: str,
        body: RealLoopReviewBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopReviewBody()
        try:
            return service.real_loop_review(
                session_id,
                round_number=payload.round_number,
                allow_network=payload.allow_network,
                provider=payload.provider,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/approve")
    def approve_real_loop(
        session_id: str,
        body: RealLoopApproveBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            return service.real_loop_approve(
                session_id,
                candidate_id=body.candidate_id,
                round_number=body.round_number,
                seeds=list(body.seeds or []) or None,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/reject")
    def reject_real_loop(
        session_id: str,
        body: RealLoopRejectBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopRejectBody()
        try:
            return service.real_loop_reject(
                session_id,
                candidate_id=payload.candidate_id,
                round_number=payload.round_number,
                reason=payload.reason,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/execute")
    def execute_real_loop(
        session_id: str,
        body: RealLoopExecuteBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopExecuteBody()
        try:
            return service.real_loop_execute(
                session_id,
                round_number=payload.round_number,
                seeds=list(payload.seeds or []) or None,
                wait=payload.wait,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/record-execution-feedback")
    def record_execution_feedback_real_loop(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            return service.real_loop_record_execution_feedback(session_id)
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/next-round")
    def next_round_real_loop(
        session_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            InvalidRealLoopTransition,
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            return service.real_loop_next_round(session_id)
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (
            RealLoopValidationError,
            RealLoopError,
            InvalidRealLoopTransition,
            ValueError,
        ) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/verify-feedback")
    def verify_feedback_real_loop(
        session_id: str,
        body: RealLoopVerifyBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopVerifyBody()
        try:
            return service.real_loop_verify_feedback(
                session_id,
                round_number=payload.round_number,
                plan_id=payload.plan_id,
                persist=payload.persist,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/real-loops/{session_id}/export")
    def export_real_loop(
        session_id: str,
        body: RealLoopExportBody | None = None,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        payload = body or RealLoopExportBody()
        try:
            return service.real_loop_export(
                session_id,
                output_dir=payload.output_dir,
                allow_incomplete=payload.allow_incomplete,
            )
        except (RealLoopNotFoundError, KeyError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except (RealLoopValidationError, RealLoopError, ValueError) as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    from scientist_lab.api.v1.autonomous_campaigns import build_autonomous_campaigns_router
    from scientist_lab.api.v1.console import build_console_router
    from scientist_lab.api.v1.dataset_workspace import build_dataset_workspace_router
    from scientist_lab.api.v1.local_runs import build_local_runs_router
    from scientist_lab.api.v1.registered_experiments import build_registered_experiments_router
    from scientist_lab.api.v1.trajectories import build_trajectories_router

    router.include_router(build_local_runs_router(get_service))
    router.include_router(build_dataset_workspace_router(get_service))
    router.include_router(build_trajectories_router(get_service))
    router.include_router(build_console_router(get_service))
    router.include_router(build_autonomous_campaigns_router(get_service))
    router.include_router(build_registered_experiments_router(get_service))
    return router
