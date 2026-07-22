"""Research evidence reporting & audit bundle (v1.2)."""

from scientist_lab.reporting.audit_bundle import (
    build_audit_bundle,
    export_audit_bundle,
    verify_audit_bundle,
)
from scientist_lab.reporting.context_builder import (
    build_report_context,
    build_report_context_from_service,
    report_context_sha256,
)
from scientist_lab.reporting.manifest import (
    build_artifact_manifest,
    build_reproducibility_manifest,
)
from scientist_lab.reporting.models import (
    ArtifactManifest,
    AuditBundle,
    ReportConclusion,
    ReportContext,
    ReproducibilityManifest,
    ResearchReport,
    ResultTables,
)
from scientist_lab.reporting.report_generator import (
    generate_research_report,
    render_executive_summary,
    render_markdown_report,
)
from scientist_lab.reporting.result_tables import build_result_tables, refine_key_path
from scientist_lab.reporting.service import ReportingService
from scientist_lab.reporting.verifier import verify_research_report

__all__ = [
    "ArtifactManifest",
    "AuditBundle",
    "ReportConclusion",
    "ReportContext",
    "ReportingService",
    "ReproducibilityManifest",
    "ResearchReport",
    "ResultTables",
    "build_artifact_manifest",
    "build_audit_bundle",
    "build_report_context",
    "build_report_context_from_service",
    "build_reproducibility_manifest",
    "build_result_tables",
    "export_audit_bundle",
    "generate_research_report",
    "refine_key_path",
    "render_executive_summary",
    "render_markdown_report",
    "report_context_sha256",
    "verify_audit_bundle",
    "verify_research_report",
]

API_VERSION = "v1.2.8"
