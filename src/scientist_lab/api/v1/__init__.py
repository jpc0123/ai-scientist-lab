"""REST API v1 routers for Scientist Lab Web Console (v1.7.1)."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query

from scientist_lab.api.errors import http_error
from scientist_lab.api.pagination import clamp_limit, clamp_offset, page_response
from scientist_lab.api.schemas import (
    AuditBuildBody,
    AuditExportBody,
    CandidateRejectBody,
    CompareExecutionsBody,
    CompareNodesBody,
    CompareTriadBody,
    IterationFinalizeBody,
    MergeApproveBody,
    MergeFinalizeBody,
    MergePrepareBody,
    MergeRejectBody,
    MergeRollbackBody,
    MergeTestBody,
    PatchApplySandboxBody,
    ProjectCreateBody,
    ReleaseCandidateCreateBody,
    PatchDecideMergeBody,
    PatchRecordEvidenceBody,
    PatchTestBody,
    ReasonBody,
    ReleaseCreateBody,
    ReleaseDiscardBody,
    ReleaseFreezeBody,
    ReportBuildBody,
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
            "version": "v2.0.3",
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

    return router
