"""Assemble and verify Audit Bundle directories (v1.2.7)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.reporting.manifest import (
    build_artifact_manifest,
    build_reproducibility_manifest,
)
from scientist_lab.reporting.models import (
    AuditBundle,
    ReportContext,
    ResearchReport,
)
from scientist_lab.reporting.report_generator import (
    render_executive_summary,
    render_markdown_report,
)
from scientist_lab.reporting.verifier import verify_research_report
from scientist_lab.storage.artifact_store import write_json


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_audit_bundle(
    *,
    outputs_root: Path,
    context: ReportContext,
    report: ResearchReport,
    project_root: Path | None = None,
    checkpoints: list[dict[str, Any]] | None = None,
    acceptance: dict[str, Any] | None = None,
    bundle_id: str | None = None,
) -> AuditBundle:
    bid = bundle_id or new_id("audit")
    root = Path(outputs_root) / context.project_id / "audit" / bid
    if root.exists():
        shutil.rmtree(root)
    dirs = {
        "report": root / "report",
        "protocol": root / "protocol",
        "contracts": root / "contracts",
        "executions": root / "executions",
        "comparisons": root / "comparisons",
        "evidence": root / "evidence",
        "claims": root / "claims",
        "decisions": root / "decisions",
        "tree": root / "tree",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    verification = verify_research_report(report)
    report = report.model_copy(
        update={
            "verification": verification,
            "status": "verified" if verification.get("valid") else "failed",
        }
    )

    report_json = dirs["report"] / "research_report.json"
    report_md = dirs["report"] / "research_report.md"
    summary_md = dirs["report"] / "executive_summary.md"
    write_json(report_json, report.model_dump(mode="json"))
    _write_text(report_md, render_markdown_report(report))
    _write_text(summary_md, render_executive_summary(report))
    report = report.model_copy(
        update={
            "json_path": str(report_json),
            "markdown_path": str(report_md),
            "summary_path": str(summary_md),
        }
    )
    write_json(report_json, report.model_dump(mode="json"))

    if context.protocol:
        write_json(dirs["protocol"] / "protocol.json", context.protocol)
    for node in context.nodes:
        write_json(
            dirs["contracts"] / f"{node.get('node_id')}.json",
            {
                "node_id": node.get("node_id"),
                "parameters": node.get("parameters"),
                "task_config": node.get("task_config"),
                "protocol_id": node.get("protocol_id"),
                "dataset_reference": node.get("dataset_reference"),
                "environment_key": node.get("environment_key"),
                "image_reference": node.get("image_reference"),
            },
        )
    for execution in context.executions:
        eid = execution.get("execution_id") or "unknown"
        write_json(dirs["executions"] / f"{eid}.json", execution)
    for idx, comparison in enumerate(context.comparisons, start=1):
        write_json(dirs["comparisons"] / f"comparison_{idx:03d}.json", comparison)
    for evidence in context.evidence_records:
        eid = evidence.get("evidence_id") or new_id("evidence")
        write_json(dirs["evidence"] / f"{eid}.json", evidence)
    write_json(dirs["claims"] / "claim_support_matrix.json", context.claim_support_matrix)
    for decision in context.decisions:
        did = decision.get("decision_id") or new_id("decision")
        write_json(dirs["decisions"] / f"{did}.json", decision)

    if context.tree:
        write_json(dirs["tree"] / "tree.json", context.tree)
        mermaid = context.tree.get("mermaid")
        if mermaid:
            _write_text(dirs["tree"] / "tree.mmd", str(mermaid) + "\n")

    repro = build_reproducibility_manifest(
        context,
        report,
        project_root=project_root,
        checkpoints=checkpoints,
        acceptance=acceptance,
        bundle_id=bid,
    )
    artifacts = build_artifact_manifest(context, bundle_id=bid)
    write_json(
        dirs["manifests"] / "reproducibility_manifest.json",
        repro.model_dump(mode="json"),
    )
    write_json(
        dirs["manifests"] / "artifact_manifest.json",
        artifacts.model_dump(mode="json"),
    )

    paths = {
        "root": str(root),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "summary_md": str(summary_md),
        "tree_json": str(dirs["tree"] / "tree.json"),
        "reproducibility_manifest": str(
            dirs["manifests"] / "reproducibility_manifest.json"
        ),
        "artifact_manifest": str(dirs["manifests"] / "artifact_manifest.json"),
    }
    bundle = AuditBundle(
        bundle_id=bid,
        project_id=context.project_id,
        report_id=report.report_id,
        tree_id=context.tree_id,
        root_dir=str(root),
        status="built" if verification.get("valid") else "invalid",
        verification=verification,
        paths=paths,
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )
    write_json(root / "audit_bundle.json", bundle.model_dump(mode="json"))
    paths["audit_bundle"] = str(root / "audit_bundle.json")
    bundle = bundle.model_copy(update={"paths": paths})
    write_json(root / "audit_bundle.json", bundle.model_dump(mode="json"))
    return bundle


def verify_audit_bundle(root_dir: Path | str) -> dict[str, Any]:
    root = Path(root_dir)
    meta_path = root / "audit_bundle.json"
    if not meta_path.is_file():
        return {
            "valid": False,
            "blocking_issues": [f"missing audit_bundle.json under {root}"],
            "warnings": [],
        }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = [
        root / "report" / "research_report.json",
        root / "report" / "research_report.md",
        root / "report" / "executive_summary.md",
        root / "manifests" / "reproducibility_manifest.json",
        root / "manifests" / "artifact_manifest.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    report_path = root / "report" / "research_report.json"
    report_issues: list[str] = []
    warnings: list[str] = []
    if report_path.is_file():
        report = ResearchReport.model_validate(
            json.loads(report_path.read_text(encoding="utf-8"))
        )
        checked = verify_research_report(report)
        report_issues = list(checked.get("blocking_issues") or [])
        warnings = list(checked.get("warnings") or [])
    else:
        report_issues = ["research_report.json missing"]

    valid = not missing and not report_issues
    return {
        "valid": valid,
        "bundle_id": meta.get("bundle_id"),
        "project_id": meta.get("project_id"),
        "root_dir": str(root),
        "missing_files": missing,
        "blocking_issues": missing + report_issues,
        "warnings": warnings,
        "meta": meta,
    }


def export_audit_bundle(
    root_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    src = Path(root_dir)
    if not (src / "audit_bundle.json").is_file():
        raise FileNotFoundError(f"audit bundle not found: {src}")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return {
        "bundle_id": src.name,
        "source": str(src),
        "exported_to": str(target),
    }
