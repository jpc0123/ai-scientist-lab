"""Reproducibility and artifact manifests (v1.2.5)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scientist_lab.reporting.models import (
    ArtifactManifest,
    ReportContext,
    ReproducibilityManifest,
    ResearchReport,
)


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_info(cwd: Path | None = None) -> tuple[str | None, list[str]]:
    root = cwd or Path.cwd()
    commit = None
    tags: list[str] = []
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except Exception:  # noqa: BLE001
        commit = None
    try:
        raw = subprocess.check_output(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        tags = [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        tags = []
    return commit, tags


def build_reproducibility_manifest(
    context: ReportContext,
    report: ResearchReport,
    *,
    project_root: Path | None = None,
    checkpoints: list[dict[str, Any]] | None = None,
    acceptance: dict[str, Any] | None = None,
    bundle_id: str | None = None,
) -> ReproducibilityManifest:
    commit, tags = _git_info(project_root)
    api_versions: dict[str, str] = {}
    try:
        from scientist_lab import search as search_pkg
        from scientist_lab import reporting as reporting_pkg

        api_versions["search"] = getattr(search_pkg, "API_VERSION", "unknown")
        api_versions["reporting"] = getattr(reporting_pkg, "API_VERSION", "unknown")
    except Exception:  # noqa: BLE001
        pass

    contracts: list[dict[str, Any]] = []
    for node in context.nodes:
        params = dict(node.get("parameters") or {})
        contracts.append(
            {
                "node_id": node.get("node_id"),
                "protocol_id": node.get("protocol_id"),
                "dataset_reference": node.get("dataset_reference"),
                "environment_key": node.get("environment_key"),
                "image_reference": node.get("image_reference"),
                "parameter_sha256": _sha256_json(params),
                "task_config_sha256": _sha256_json(node.get("task_config") or {}),
            }
        )

    executions = [
        {
            "execution_id": e.get("execution_id"),
            "node_id": e.get("node_id"),
            "seed": e.get("seed"),
            "status": e.get("status"),
            "image_reference": e.get("image_reference"),
            "dataset_version": e.get("dataset_version"),
            "code_version": e.get("code_version"),
        }
        for e in context.executions
    ]
    artifacts = [
        {
            "artifact_id": a.get("artifact_id"),
            "execution_id": a.get("execution_id"),
            "artifact_type": a.get("artifact_type"),
            "sha256": a.get("sha256"),
            "relative_path": a.get("relative_path"),
        }
        for a in context.artifacts
    ]

    return ReproducibilityManifest(
        project_id=context.project_id,
        report_id=report.report_id,
        bundle_id=bundle_id,
        git_commit=commit,
        git_tags=tags,
        api_versions=api_versions,
        protocol_id=context.protocol_id,
        protocol_sha256=_sha256_json(context.protocol),
        dataset_versions=list(context.dataset_references),
        environment_keys=list(context.environment_keys),
        image_references=list(context.image_references),
        contracts=contracts,
        executions=executions,
        artifacts=artifacts,
        checkpoints=list(checkpoints or []),
        acceptance=dict(acceptance or {}),
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


def build_artifact_manifest(
    context: ReportContext,
    *,
    bundle_id: str | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        project_id=context.project_id,
        bundle_id=bundle_id,
        artifacts=[
            {
                "artifact_id": a.get("artifact_id"),
                "execution_id": a.get("execution_id"),
                "artifact_type": a.get("artifact_type"),
                "sha256": a.get("sha256"),
                "relative_path": a.get("relative_path"),
                "size_bytes": a.get("size_bytes"),
            }
            for a in context.artifacts
        ],
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )
