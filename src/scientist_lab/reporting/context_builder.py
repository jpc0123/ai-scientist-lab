"""Assemble ReportContext from project / tree state (deterministic, no LLM)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from scientist_lab.reporting.models import ReportContext
from scientist_lab.search.evidence_link import extract_open_gaps


SENSITIVE_KEYS = frozenset(
    {
        "host_path",
        "absolute_path",
        "local_path",
        "auth_token",
        "password",
        "secret",
        "api_key",
        "endpoint_token",
    }
)


def _looks_absolute(value: str) -> bool:
    text = value or ""
    if len(text) > 2 and text[1:3] == ":\\":
        return True
    if text.startswith("/home/") or text.startswith("/Users/") or text.startswith("/var/"):
        return True
    return False


def scrub(value: Any) -> Any:
    """Drop host secrets / absolute paths from report context."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in SENSITIVE_KEYS:
                continue
            if key_l.endswith("_path") and key_l not in {
                "matrix_path",
                "comparison_path",
                "evidence_path",
                "aggregate_path",
                "claim_matrix_path",
                "decision_path",
            }:
                if isinstance(item, str) and _looks_absolute(item):
                    continue
            cleaned[key] = scrub(item)
        return cleaned
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str) and _looks_absolute(value):
        return "<redacted_path>"
    return value


def report_context_sha256(context: ReportContext | dict[str, Any]) -> str:
    if isinstance(context, ReportContext):
        data = context.model_dump(mode="json")
    else:
        data = dict(context)
    data.pop("context_sha256", None)
    data.pop("built_at", None)
    payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _node_summary(node: Any) -> dict[str, Any]:
    contract = dict(getattr(node, "contract_json", None) or {})
    feedback = dict(getattr(node, "feedback_json", None) or {})
    return {
        "node_id": getattr(node, "node_id", None),
        "project_id": getattr(node, "project_id", None),
        "node_type": str(getattr(node, "node_type", "")),
        "stage": str(getattr(node, "stage", "")),
        "status": str(getattr(node, "status", "")),
        "depth": getattr(node, "depth", None),
        "hypothesis": getattr(node, "hypothesis", None),
        "protocol_id": contract.get("protocol_id"),
        "execution_mode": contract.get("execution_mode"),
        "dataset_reference": contract.get("dataset_reference"),
        "environment_key": contract.get("environment_key"),
        "image_reference": contract.get("image_reference")
        or contract.get("code_reference"),
        "parameters": dict(contract.get("parameters") or {}),
        "task_config": {
            key: value
            for key, value in dict(contract.get("task_config") or {}).items()
            if key
            in {
                "claim_level",
                "evaluation_scope",
                "implementation",
                "primary_metric",
                "input_mode",
                "fusion_method",
            }
        },
        "aggregate_metrics": feedback.get("aggregate_metrics"),
    }


def _execution_summary(attempt: Any) -> dict[str, Any]:
    result = dict(getattr(attempt, "result_json", None) or {})
    contract = dict(result.get("contract") or {})
    metrics = dict((result.get("metrics") or {}).get("metrics") or {})
    return {
        "execution_id": getattr(attempt, "execution_id", None),
        "node_id": getattr(attempt, "node_id", None),
        "status": str(getattr(attempt, "status", "")),
        "runner_profile": getattr(attempt, "runner_profile", None),
        "image_reference": getattr(attempt, "image_reference", None),
        "code_version": getattr(attempt, "code_version", None),
        "dataset_version": getattr(attempt, "dataset_version", None),
        "seed": contract.get("seed"),
        "primary_metric": (result.get("metrics") or {}).get("primary_metric"),
        "metric_values": {
            key: metrics.get(key)
            for key in ("mAP50_95", "mAP50", "AP_small", "duration_seconds")
            if key in metrics
        },
    }


def _artifact_summary(artifact: Any) -> dict[str, Any]:
    return {
        "artifact_id": getattr(artifact, "artifact_id", None),
        "execution_id": getattr(artifact, "execution_id", None),
        "artifact_type": getattr(artifact, "artifact_type", None),
        "relative_path": getattr(artifact, "relative_path", None),
        "sha256": getattr(artifact, "sha256", None),
        "size_bytes": getattr(artifact, "size_bytes", None),
    }


def _collect_limitations(
    evidence_records: list[dict[str, Any]],
    claim_matrix: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for record in evidence_records:
        for limitation in record.get("limitations") or []:
            text = str(limitation).strip()
            if text and text not in seen:
                seen.add(text)
                items.append(text)
    for claim in claim_matrix.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        for limitation in claim.get("limitations") or []:
            text = str(limitation).strip()
            if text and text not in seen:
                seen.add(text)
                items.append(text)
        status = str(claim.get("support_status") or "")
        if status in {"blocked", "unsupported"}:
            reason = str(claim.get("reason") or claim.get("claim_text") or status)
            text = f"{status}: {reason}".strip()
            if text and text not in seen:
                seen.add(text)
                items.append(text)
    return items


def _basic_key_path(tree_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """v1.2.1 heuristic: root → highest-score evaluated descendant chain."""
    if not tree_payload:
        return []
    nodes = list(tree_payload.get("nodes") or [])
    if not nodes:
        return []
    by_id = {n["tree_node_id"]: n for n in nodes if n.get("tree_node_id")}
    roots = [n for n in nodes if not n.get("parent_tree_node_id")]
    if not roots:
        return []
    root = roots[0]

    evaluated = [
        n
        for n in nodes
        if n.get("status") in {"evaluated", "selected"} and n.get("score") is not None
    ]
    if not evaluated:
        return [
            {
                "tree_node_id": root.get("tree_node_id"),
                "experiment_node_id": root.get("experiment_node_id"),
                "node_type": root.get("node_type"),
                "status": root.get("status"),
                "depth": root.get("depth"),
                "score": root.get("score"),
            }
        ]

    best = max(evaluated, key=lambda n: float(n.get("score") or 0.0))
    chain: list[dict[str, Any]] = []
    cursor: dict[str, Any] | None = best
    seen: set[str] = set()
    while cursor is not None:
        tid = str(cursor.get("tree_node_id") or "")
        if not tid or tid in seen:
            break
        seen.add(tid)
        chain.append(
            {
                "tree_node_id": cursor.get("tree_node_id"),
                "experiment_node_id": cursor.get("experiment_node_id"),
                "node_type": cursor.get("node_type"),
                "status": cursor.get("status"),
                "depth": cursor.get("depth"),
                "score": cursor.get("score"),
            }
        )
        parent_id = cursor.get("parent_tree_node_id")
        cursor = by_id.get(parent_id) if parent_id else None
    chain.reverse()
    return chain


def build_report_context(
    *,
    project_id: str,
    research_goal: str = "",
    protocol: dict[str, Any] | None = None,
    protocol_id: str | None = None,
    tree_payload: dict[str, Any] | None = None,
    experiment_nodes: list[Any] | None = None,
    executions: list[Any] | None = None,
    artifacts: list[Any] | None = None,
    evidence_records: list[dict[str, Any]] | None = None,
    claim_support_matrix: dict[str, Any] | None = None,
    decisions: list[dict[str, Any]] | None = None,
    comparisons: list[dict[str, Any]] | None = None,
    ablations: list[dict[str, Any]] | None = None,
    remaining_budget: dict[str, Any] | None = None,
    recommended_next_experiments: list[str] | None = None,
) -> ReportContext:
    """Pure assembler: callers inject already-loaded project artifacts."""
    nodes = [_node_summary(n) for n in (experiment_nodes or [])]
    execs = [_execution_summary(e) for e in (executions or [])]
    arts = [_artifact_summary(a) for a in (artifacts or [])]
    evidence = [dict(item) for item in (evidence_records or [])]
    matrix = dict(claim_support_matrix or {})
    tree = dict(tree_payload) if tree_payload else None

    failures: list[dict[str, Any]] = []
    for node in nodes:
        if str(node.get("status") or "").lower() in {"failed", "cancelled"}:
            failures.append({"source": "experiment_node", **node})
    if tree:
        for tnode in tree.get("nodes") or []:
            if str(tnode.get("status") or "") == "failed":
                failures.append({"source": "tree_node", **dict(tnode)})

    datasets = sorted(
        {
            str(n.get("dataset_reference"))
            for n in nodes
            if n.get("dataset_reference")
        }
        | {
            str(e.get("dataset_version"))
            for e in execs
            if e.get("dataset_version")
        }
    )
    envs = sorted(
        {str(n.get("environment_key")) for n in nodes if n.get("environment_key")}
    )
    images = sorted(
        {
            str(n.get("image_reference"))
            for n in nodes
            if n.get("image_reference")
        }
        | {str(e.get("image_reference")) for e in execs if e.get("image_reference")}
    )

    open_gaps = extract_open_gaps(matrix, evidence)
    limitations = _collect_limitations(evidence, matrix)

    context = ReportContext(
        project_id=project_id,
        research_goal=research_goal or "",
        protocol_id=protocol_id or (protocol or {}).get("protocol_id"),
        protocol=dict(protocol or {}),
        tree_id=(tree or {}).get("tree_id") if tree else None,
        tree=tree,
        key_path=_basic_key_path(tree),
        nodes=nodes,
        executions=execs,
        artifacts=arts,
        evidence_records=evidence,
        claim_support_matrix=matrix,
        decisions=[dict(d) for d in (decisions or [])],
        comparisons=[dict(c) for c in (comparisons or [])],
        ablations=[dict(a) for a in (ablations or [])],
        failures=failures,
        open_evidence_gaps=open_gaps,
        limitations=limitations,
        recommended_next_experiments=list(recommended_next_experiments or []),
        dataset_references=datasets,
        environment_keys=envs,
        image_references=images,
        remaining_budget=dict(remaining_budget or {}),
        built_at=datetime.now(timezone.utc).replace(microsecond=0),
        builder_version="v1.2.1",
    )
    scrubbed = ReportContext.model_validate(scrub(context.model_dump(mode="json")))
    return scrubbed.model_copy(
        update={"context_sha256": report_context_sha256(scrubbed)}
    )


def build_report_context_from_service(
    service: Any,
    project_id: str,
    *,
    tree_id: str | None = None,
    protocol_id: str | None = None,
) -> ReportContext:
    """Convenience builder using ExperimentService public APIs."""
    project = service.repo.get_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")

    resolved_protocol_id = protocol_id
    protocol: dict[str, Any] = {}
    if resolved_protocol_id:
        try:
            protocol = service.show_protocol(resolved_protocol_id)
        except Exception:  # noqa: BLE001
            protocol = {}
    else:
        protocols = service.list_protocols(project_id=project_id)
        if protocols:
            protocol = protocols[0]
            resolved_protocol_id = protocol.get("protocol_id")

    tree_payload: dict[str, Any] | None = None
    if tree_id:
        tree_payload = service.tree_export(tree_id, format="json")
    else:
        project_trees = service.trees._repo.list_trees(project_id=project_id)
        if project_trees:
            latest = project_trees[0]
            tree_payload = service.tree_export(latest.tree_id, format="json")
            if not resolved_protocol_id:
                resolved_protocol_id = latest.protocol_id

    nodes = service.repo.list_nodes(project_id=project_id)
    executions = service.repo.list_attempts(project_id=project_id, limit=500)
    artifacts: list[Any] = []
    for attempt in executions:
        artifacts.extend(service.repo.list_artifacts(attempt.execution_id))

    evidence = service.list_evidence(project_id=project_id)
    try:
        matrix = service.show_claim_matrix(project_id)
    except Exception:  # noqa: BLE001
        matrix = {}
    decisions = service.list_decisions(project_id=project_id, limit=100)
    try:
        ablations = service.list_ablations(project_id=project_id)
    except Exception:  # noqa: BLE001
        ablations = []

    budget: dict[str, Any] = {}
    try:
        budget = service.show_budget(project_id)
    except Exception:  # noqa: BLE001
        budget = {}

    return build_report_context(
        project_id=project_id,
        research_goal=str(getattr(project, "research_goal", "") or ""),
        protocol=protocol,
        protocol_id=resolved_protocol_id,
        tree_payload=tree_payload,
        experiment_nodes=nodes,
        executions=executions,
        artifacts=artifacts,
        evidence_records=evidence,
        claim_support_matrix=matrix,
        decisions=decisions,
        ablations=ablations,
        remaining_budget=budget,
    )
