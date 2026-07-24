"""Project export / import bundles (v2.0.10).

Offline, local-only JSON bundles — no network, no shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import ExperimentNode, ResearchProject, utc_now_iso


BUNDLE_SCHEMA = "scientist_lab.project_bundle.v1"


def export_project_bundle(
    experiments: Any,
    project_id: str,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    project = experiments.repo.get_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = experiments.repo.list_nodes(project_id=project_id)
    protocols: list[dict[str, Any]] = []
    for pid in list(project.protocol_ids or []):
        try:
            protocols.append(
                experiments.protocols.require(pid).model_dump(mode="json")
            )
        except Exception:  # noqa: BLE001
            continue

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "exported_at": utc_now_iso(),
        "project": project.model_dump(mode="json"),
        "nodes": [n.model_dump(mode="json") for n in nodes],
        "protocols": protocols,
    }
    path = out / f"{project_id}_bundle.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "project_id": project_id,
        "path": str(path),
        "node_count": len(nodes),
        "protocol_count": len(protocols),
        "schema": BUNDLE_SCHEMA,
    }


def import_project_bundle(
    experiments: Any,
    path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported bundle schema: {payload.get('schema')!r}")

    project_data = dict(payload.get("project") or {})
    project_id = str(project_data.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("bundle missing project_id")

    existing = experiments.repo.get_project(project_id)
    if existing is not None and not force:
        raise ValueError(
            f"project already exists: {project_id}; pass force=true to overwrite metadata"
        )

    for proto in payload.get("protocols") or []:
        try:
            experiments.protocols.create_from_dict(dict(proto))
        except Exception:  # noqa: BLE001
            continue

    project = ResearchProject.model_validate(project_data)
    project.touch()
    experiments.repo.upsert_project(project)

    imported_nodes = 0
    for raw in payload.get("nodes") or []:
        node = ExperimentNode.model_validate(raw)
        experiments.repo.upsert_node(node)
        imported_nodes += 1

    return {
        "project_id": project_id,
        "status": "imported",
        "node_count": imported_nodes,
        "force": force,
        "schema": BUNDLE_SCHEMA,
    }
