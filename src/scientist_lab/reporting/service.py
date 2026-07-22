"""Reporting service facade (v1.2: context → report → verify → audit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.reporting.audit_bundle import (
    build_audit_bundle,
    export_audit_bundle,
    verify_audit_bundle,
)
from scientist_lab.reporting.context_builder import build_report_context_from_service
from scientist_lab.reporting.models import ReportContext, ResearchReport
from scientist_lab.reporting.report_generator import (
    generate_research_report,
    render_executive_summary,
    render_markdown_report,
)
from scientist_lab.reporting.repository import ReportingRepository
from scientist_lab.reporting.result_tables import build_result_tables
from scientist_lab.reporting.verifier import verify_research_report


class ReportingService:
    def __init__(self, experiment_service: Any) -> None:
        self._experiments = experiment_service
        self._repo = ReportingRepository(experiment_service.settings.outputs_dir)

    def build_context(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
    ) -> ReportContext:
        return build_report_context_from_service(
            self._experiments,
            project_id,
            tree_id=tree_id,
            protocol_id=protocol_id,
        )

    def show_context_dict(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        return self.build_context(
            project_id, tree_id=tree_id, protocol_id=protocol_id
        ).model_dump(mode="json")

    def build_report(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        context = self.build_context(
            project_id, tree_id=tree_id, protocol_id=protocol_id
        )
        tables = build_result_tables(context)
        # Keep refined key_path on context-derived report.
        context = context.model_copy(update={"key_path": tables.key_path})
        report = generate_research_report(context, tables=tables)
        verification = verify_research_report(report)
        report = report.model_copy(
            update={
                "verification": verification,
                "status": "verified" if verification.get("valid") else "draft",
            }
        )
        if persist:
            md = render_markdown_report(report)
            summary = render_executive_summary(report)
            report = self._repo.save_report(report)
            if report.markdown_path:
                Path(report.markdown_path).write_text(md, encoding="utf-8")
            if report.summary_path:
                Path(report.summary_path).write_text(summary, encoding="utf-8")
        return report.model_dump(mode="json")

    def show_report(self, report_id: str) -> dict[str, Any]:
        return self._repo.require_report(report_id).model_dump(mode="json")

    def verify_report(self, report_id: str) -> dict[str, Any]:
        report = self._repo.require_report(report_id)
        result = verify_research_report(report)
        updated = report.model_copy(
            update={
                "verification": result,
                "status": "verified" if result.get("valid") else "failed",
            }
        )
        self._repo.save_report(updated)
        return {
            "report_id": report_id,
            "project_id": report.project_id,
            **result,
        }

    def build_audit(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
        report_id: str | None = None,
    ) -> dict[str, Any]:
        context = self.build_context(
            project_id, tree_id=tree_id, protocol_id=protocol_id
        )
        if report_id:
            report = self._repo.require_report(report_id)
        else:
            tables = build_result_tables(context)
            context = context.model_copy(update={"key_path": tables.key_path})
            report = generate_research_report(context, tables=tables)
            report = self._repo.save_report(report)

        checkpoints: list[dict[str, Any]] = []
        try:
            checkpoints = self._experiments.list_checkpoints(project_id=project_id)
        except Exception:  # noqa: BLE001
            checkpoints = []

        acceptance: dict[str, Any] = {}
        accept_path = (
            Path(self._experiments.settings.project_root)
            / "docs"
            / "acceptance"
            / "v1.1"
            / "v11_acceptance_report.json"
        )
        if accept_path.is_file():
            import json

            try:
                acceptance = json.loads(accept_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                acceptance = {"path": str(accept_path)}

        bundle = build_audit_bundle(
            outputs_root=self._experiments.settings.outputs_dir,
            context=context,
            report=report,
            project_root=Path(self._experiments.settings.project_root),
            checkpoints=checkpoints if isinstance(checkpoints, list) else [],
            acceptance=acceptance,
        )
        self._repo.save_bundle_index(bundle)
        return bundle.model_dump(mode="json")

    def verify_audit(self, bundle_id: str) -> dict[str, Any]:
        root = self._repo.find_bundle_root(bundle_id)
        if root is None:
            raise KeyError(f"audit bundle not found: {bundle_id}")
        return verify_audit_bundle(root)

    def export_audit(self, bundle_id: str, output_dir: str | Path) -> dict[str, Any]:
        root = self._repo.find_bundle_root(bundle_id)
        if root is None:
            raise KeyError(f"audit bundle not found: {bundle_id}")
        return export_audit_bundle(root, output_dir)
