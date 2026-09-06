"""Build result tables and refine key experiment path (v1.2.2)."""

from __future__ import annotations

from typing import Any

from scientist_lab.reporting.models import ReportContext, ResultTables


def _metric_mean(node: dict[str, Any], metric: str) -> float | None:
    agg = node.get("aggregate_metrics") or {}
    # Support both nested and flat shapes used across the codebase.
    metrics = agg.get("aggregate_metrics") or agg
    blob = metrics.get(metric) if isinstance(metrics, dict) else None
    if isinstance(blob, dict) and isinstance(blob.get("mean"), (int, float)):
        return float(blob["mean"])
    if isinstance(blob, (int, float)):
        return float(blob)
    return None


def _metric_std(node: dict[str, Any], metric: str) -> float | None:
    agg = node.get("aggregate_metrics") or {}
    metrics = agg.get("aggregate_metrics") or agg
    blob = metrics.get(metric) if isinstance(metrics, dict) else None
    if isinstance(blob, dict) and isinstance(blob.get("std"), (int, float)):
        return float(blob["std"])
    return None


def _primary_metric(node: dict[str, Any]) -> str:
    agg = node.get("aggregate_metrics") or {}
    primary = agg.get("primary_metric")
    if primary:
        return str(primary)
    task = node.get("task_config") or {}
    if task.get("primary_metric"):
        return str(task["primary_metric"])
    return "mAP50_95"


def refine_key_path(context: ReportContext) -> list[dict[str, Any]]:
    """Prefer decision-selected node, else highest-score evaluated chain to root."""
    tree = context.tree or {}
    nodes = list(tree.get("nodes") or [])
    if not nodes:
        # Fall back to experiment nodes ordered by depth / metric.
        ranked = sorted(
            context.nodes,
            key=lambda n: (
                0 if str(n.get("status")).lower() in {"succeeded", "done"} else 1,
                -(_metric_mean(n, _primary_metric(n)) or -1.0),
            ),
        )
        if not ranked:
            return list(context.key_path)
        best = ranked[0]
        return [
            {
                "experiment_node_id": best.get("node_id"),
                "node_type": best.get("node_type"),
                "status": best.get("status"),
                "depth": best.get("depth"),
                "primary_metric": _primary_metric(best),
                "primary_mean": _metric_mean(best, _primary_metric(best)),
                "source": "experiment_fallback",
            }
        ]

    by_id = {n["tree_node_id"]: n for n in nodes if n.get("tree_node_id")}
    selected_exp = tree.get("selected_node_id")
    tip: dict[str, Any] | None = None
    if selected_exp:
        for node in nodes:
            if node.get("experiment_node_id") == selected_exp and node.get(
                "status"
            ) in {"evaluated", "selected"}:
                tip = node
                break
    if tip is None:
        evaluated = [
            n
            for n in nodes
            if n.get("status") in {"evaluated", "selected"}
            and n.get("score") is not None
        ]
        if evaluated:
            tip = max(evaluated, key=lambda n: float(n.get("score") or 0.0))
        else:
            roots = [n for n in nodes if not n.get("parent_tree_node_id")]
            tip = roots[0] if roots else nodes[0]

    chain: list[dict[str, Any]] = []
    cursor: dict[str, Any] | None = tip
    seen: set[str] = set()
    while cursor is not None:
        tid = str(cursor.get("tree_node_id") or "")
        if not tid or tid in seen:
            break
        seen.add(tid)
        exp_id = cursor.get("experiment_node_id")
        exp = next(
            (n for n in context.nodes if n.get("node_id") == exp_id),
            None,
        )
        primary = _primary_metric(exp) if exp else "mAP50_95"
        chain.append(
            {
                "tree_node_id": cursor.get("tree_node_id"),
                "experiment_node_id": exp_id,
                "node_type": cursor.get("node_type"),
                "status": cursor.get("status"),
                "depth": cursor.get("depth"),
                "score": cursor.get("score"),
                "primary_metric": primary,
                "primary_mean": _metric_mean(exp, primary) if exp else None,
                "primary_std": _metric_std(exp, primary) if exp else None,
            }
        )
        parent_id = cursor.get("parent_tree_node_id")
        cursor = by_id.get(parent_id) if parent_id else None
    chain.reverse()
    return chain


def build_metric_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in context.nodes:
        primary = _primary_metric(node)
        rows.append(
            {
                "node_id": node.get("node_id"),
                "node_type": node.get("node_type"),
                "status": node.get("status"),
                "primary_metric": primary,
                "primary_mean": _metric_mean(node, primary),
                "primary_std": _metric_std(node, primary),
                "mAP50_mean": _metric_mean(node, "mAP50"),
                "AP_small_mean": _metric_mean(node, "AP_small"),
                "claim_level": (node.get("task_config") or {}).get("claim_level"),
                "execution_mode": node.get("execution_mode"),
            }
        )
    rows.sort(
        key=lambda r: (
            -(r["primary_mean"] if isinstance(r.get("primary_mean"), (int, float)) else -1.0),
            str(r.get("node_id") or ""),
        )
    )
    return rows


def build_stability_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in context.nodes:
        primary = _primary_metric(node)
        mean = _metric_mean(node, primary)
        std = _metric_std(node, primary)
        cv = None
        if isinstance(mean, float) and mean != 0 and isinstance(std, float):
            cv = abs(std / mean)
        seed_count = None
        agg = node.get("aggregate_metrics") or {}
        if isinstance(agg.get("seed_count"), int):
            seed_count = agg["seed_count"]
        rows.append(
            {
                "node_id": node.get("node_id"),
                "primary_metric": primary,
                "mean": mean,
                "std": std,
                "cv": cv,
                "seed_count": seed_count,
                "stable": bool(
                    isinstance(cv, float) and cv <= 0.15 and (seed_count or 0) >= 3
                ),
            }
        )
    return rows


def build_resource_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in context.nodes:
        rows.append(
            {
                "node_id": node.get("node_id"),
                "duration_seconds_mean": _metric_mean(node, "duration_seconds"),
                "peak_gpu_memory_mb_mean": _metric_mean(node, "peak_gpu_memory_mb"),
                "parameter_count_mean": _metric_mean(node, "parameter_count"),
                "environment_key": node.get("environment_key"),
                "image_reference": node.get("image_reference"),
            }
        )
    return rows


def build_ablation_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in context.ablations:
        rows.append(
            {
                "ablation_id": plan.get("ablation_id") or plan.get("plan_id"),
                "title": plan.get("title"),
                "project_id": plan.get("project_id"),
                "status": plan.get("status"),
                "variant_count": len(plan.get("variants") or plan.get("contracts") or []),
            }
        )
    # Also surface ablation-typed experiment nodes.
    for node in context.nodes:
        if str(node.get("node_type") or "").lower() == "ablation":
            primary = _primary_metric(node)
            rows.append(
                {
                    "ablation_id": None,
                    "node_id": node.get("node_id"),
                    "title": node.get("hypothesis"),
                    "primary_mean": _metric_mean(node, primary),
                    "status": node.get("status"),
                }
            )
    return rows


def build_failure_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in context.failures:
        rows.append(
            {
                "source": item.get("source"),
                "node_id": item.get("node_id") or item.get("experiment_node_id"),
                "tree_node_id": item.get("tree_node_id"),
                "status": item.get("status"),
                "node_type": item.get("node_type"),
            }
        )
    return rows


def build_claim_table(context: ReportContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim in (context.claim_support_matrix or {}).get("claims") or []:
        if not isinstance(claim, dict):
            continue
        rows.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("claim_text") or claim.get("claim"),
                "support_status": claim.get("support_status"),
                "supporting_evidence_ids": list(
                    claim.get("supporting_evidence_ids") or []
                ),
                "limitations": list(claim.get("limitations") or []),
                "reason": claim.get("reason"),
            }
        )
    return rows


def build_result_tables(context: ReportContext) -> ResultTables:
    key_path = refine_key_path(context)
    return ResultTables(
        key_path=key_path,
        metrics=build_metric_table(context),
        stability=build_stability_table(context),
        resources=build_resource_table(context),
        ablations=build_ablation_table(context),
        failures=build_failure_table(context),
        claims=build_claim_table(context),
        evidence_summary=[
            {
                "evidence_id": e.get("evidence_id"),
                "evidence_type": e.get("evidence_type"),
                "evidence_strength": e.get("evidence_strength"),
                "source_node_ids": list(e.get("source_node_ids") or []),
                "limitations": list(e.get("limitations") or []),
            }
            for e in context.evidence_records
        ],
    )
