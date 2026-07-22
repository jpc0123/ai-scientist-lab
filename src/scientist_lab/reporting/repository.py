"""Filesystem persistence for research reports and audit bundle indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.reporting.models import AuditBundle, ResearchReport
from scientist_lab.storage.artifact_store import write_json


class ReportingRepository:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = Path(outputs_root)

    def report_dir(self, project_id: str) -> Path:
        return self.outputs_root / project_id / "reports"

    def save_report(self, report: ResearchReport) -> ResearchReport:
        folder = self.report_dir(report.project_id)
        folder.mkdir(parents=True, exist_ok=True)
        json_path = folder / f"{report.report_id}.json"
        md_path = folder / f"{report.report_id}.md"
        summary_path = folder / f"{report.report_id}.summary.md"
        updated = report.model_copy(
            update={
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "summary_path": str(summary_path),
            }
        )
        write_json(json_path, updated.model_dump(mode="json"))
        return updated

    def get_report(self, report_id: str) -> ResearchReport | None:
        if not self.outputs_root.exists():
            return None
        for path in self.outputs_root.glob(f"*/reports/{report_id}.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            return ResearchReport.model_validate(data)
        return None

    def require_report(self, report_id: str) -> ResearchReport:
        report = self.get_report(report_id)
        if report is None:
            raise KeyError(f"report not found: {report_id}")
        return report

    def save_bundle_index(self, bundle: AuditBundle) -> None:
        folder = self.outputs_root / bundle.project_id / "audit"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / f"{bundle.bundle_id}.index.json", bundle.model_dump(mode="json"))

    def find_bundle_root(self, bundle_id: str) -> Path | None:
        if not self.outputs_root.exists():
            return None
        direct = list(self.outputs_root.glob(f"*/audit/{bundle_id}"))
        for path in direct:
            if (path / "audit_bundle.json").is_file():
                return path
        return None

    def load_bundle(self, bundle_id: str) -> dict[str, Any]:
        root = self.find_bundle_root(bundle_id)
        if root is None:
            raise KeyError(f"audit bundle not found: {bundle_id}")
        return json.loads((root / "audit_bundle.json").read_text(encoding="utf-8"))
