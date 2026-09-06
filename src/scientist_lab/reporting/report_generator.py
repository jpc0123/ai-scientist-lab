"""Deterministic ResearchReport generator (JSON + Markdown, no LLM)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.reporting.models import (
    ReportConclusion,
    ReportContext,
    ResearchReport,
    ResultTables,
)
from scientist_lab.reporting.result_tables import build_result_tables


FORBIDDEN_PHRASES = (
    "sota",
    "state-of-the-art",
    "state of the art",
    "proves that",
    "has been proven",
    "已证明",
    "优于现有方法",
    "完整 rgbt-tiny",
    "full rgbt-tiny",
    "surpasses all",
)


def _is_forbidden(text: str) -> bool:
    """Detect strong-claim phrasing; ignore substrings inside identifiers like claim_sota."""
    lower = (text or "").lower()
    for phrase in FORBIDDEN_PHRASES:
        if any("\u4e00" <= ch <= "\u9fff" for ch in phrase):
            if phrase in lower:
                return True
            continue
        pattern = rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])"
        if re.search(pattern, lower):
            return True
    return False


def _scope_prefix(context: ReportContext) -> str:
    modes = {
        str(n.get("execution_mode") or "")
        for n in context.nodes
        if n.get("execution_mode")
    }
    claims = {
        str((n.get("task_config") or {}).get("claim_level") or "")
        for n in context.nodes
    }
    if "fast_eval" in modes or "exploratory_comparison" in claims:
        return "在固定 Fast Eval / 探索性设置下"
    return "在当前 ExperimentProtocol 约束下"


def _build_conclusions(
    context: ReportContext, tables: ResultTables
) -> list[ReportConclusion]:
    conclusions: list[ReportConclusion] = []
    scope = _scope_prefix(context)
    claims = tables.claims

    # One conclusion per claim row, worded conservatively from support_status.
    for idx, claim in enumerate(claims, start=1):
        status = str(claim.get("support_status") or "unsupported")
        claim_id = str(claim.get("claim_id") or f"claim_{idx}")
        claim_text = str(claim.get("claim_text") or claim_id)
        # Never echo raw claim text that already contains forbidden phrasing.
        safe_claim_text = (
            "[redacted strong claim wording]"
            if _is_forbidden(claim_text)
            else claim_text
        )
        evidence_ids = list(claim.get("supporting_evidence_ids") or [])
        limitations = list(claim.get("limitations") or [])
        if not limitations:
            limitations = list(context.limitations[:3])

        if status == "supported":
            text = (
                f"{scope}，claim `{claim_id}`（{safe_claim_text}）目前为 "
                f"supported；该表述仅在所列证据与限制条件下成立。"
            )
            strength = "moderate"
        elif status == "partially_supported":
            text = (
                f"{scope}，claim `{claim_id}`（{safe_claim_text}）仅为 "
                f"partially_supported；尚不足以支持更强泛化结论。"
            )
            strength = "weak"
        elif status == "blocked":
            text = (
                f"claim `{claim_id}`（{safe_claim_text}）被阻断（blocked），"
                f"不得升级为正式科研结论。"
            )
            strength = "blocked"
        else:
            text = (
                f"claim `{claim_id}`（{safe_claim_text}）当前为 unsupported；"
                f"报告不将其表述为已验证发现。"
            )
            strength = "unsupported"

        if _is_forbidden(text):
            text = (
                f"claim `{claim_id}` 的自动表述触发强结论词过滤，"
                f"已降级为中性说明（support_status={status}）。"
            )

        conclusions.append(
            ReportConclusion(
                conclusion_id=f"conclusion_{idx:03d}",
                text=text,
                claim_id=claim_id,
                support_status=status,
                evidence_ids=evidence_ids,
                limitations=limitations,
                strength=strength,
            )
        )

    # Metric observation from key path tip (always scoped, never SOTA).
    if tables.key_path:
        tip = tables.key_path[-1]
        mean = tip.get("primary_mean")
        metric = tip.get("primary_metric") or "primary_metric"
        node_id = tip.get("experiment_node_id")
        if isinstance(mean, (int, float)) and node_id:
            text = (
                f"{scope}，关键路径末端节点 `{node_id}` 的 "
                f"{metric} 均值为 {mean:.4f}。"
            )
            conclusions.append(
                ReportConclusion(
                    conclusion_id=f"conclusion_{len(conclusions) + 1:03d}",
                    text=text,
                    claim_id=None,
                    support_status="partially_supported",
                    evidence_ids=[
                        e.get("evidence_id")
                        for e in context.evidence_records
                        if node_id in (e.get("source_node_ids") or [])
                        and e.get("evidence_id")
                    ],
                    limitations=list(context.limitations[:5])
                    or ["Metric observation is protocol-scoped only."],
                    strength="weak",
                )
            )

    if not conclusions:
        conclusions.append(
            ReportConclusion(
                conclusion_id="conclusion_001",
                text=(
                    f"{scope}，当前尚无足够 Claim / Evidence 支撑正式结论；"
                    "请先补齐比较或消融证据。"
                ),
                claim_id=None,
                support_status="unsupported",
                evidence_ids=[],
                limitations=list(context.limitations)
                or ["Insufficient evidence for formal claims."],
                strength="unsupported",
            )
        )
    return conclusions


def _tree_summary(context: ReportContext) -> dict[str, Any]:
    tree = context.tree or {}
    nodes = list(tree.get("nodes") or [])
    return {
        "tree_id": context.tree_id or tree.get("tree_id"),
        "status": tree.get("status"),
        "node_count": tree.get("node_count") or len(nodes),
        "stop_reason": tree.get("stop_reason"),
        "max_depth": tree.get("max_depth"),
        "max_nodes": tree.get("max_nodes"),
        "ascii_tree": tree.get("ascii_tree"),
    }


def _default_next_steps(context: ReportContext) -> list[str]:
    if context.recommended_next_experiments:
        return list(context.recommended_next_experiments)
    suggestions: list[str] = []
    for gap in context.open_evidence_gaps[:5]:
        suggestions.append(f"设计实验以关闭缺口：{gap}")
    if not suggestions:
        suggestions.append("在协议允许变量内补充消融或 matched-seed 复现。")
    return suggestions


def generate_research_report(
    context: ReportContext,
    *,
    tables: ResultTables | None = None,
    report_id: str | None = None,
) -> ResearchReport:
    tables = tables or build_result_tables(context)
    claims = tables.claims
    supported = [
        c for c in claims if str(c.get("support_status")) in {"supported", "partially_supported"}
    ]
    blocked = [
        c
        for c in claims
        if str(c.get("support_status")) in {"blocked", "unsupported"}
    ]
    conclusions = _build_conclusions(context, tables)
    key_nodes = []
    for step in tables.key_path:
        exp_id = step.get("experiment_node_id")
        match = next((n for n in context.nodes if n.get("node_id") == exp_id), None)
        key_nodes.append({**(match or {}), **step})

    return ResearchReport(
        report_id=report_id or new_id("report"),
        project_id=context.project_id,
        status="draft",
        research_goal=context.research_goal,
        protocol_id=context.protocol_id,
        protocol_summary={
            "protocol_id": context.protocol_id,
            "title": (context.protocol or {}).get("title"),
            "task_type": (context.protocol or {}).get("task_type"),
            "dataset_reference": (context.protocol or {}).get("dataset_reference"),
            "dataset_version": (context.protocol or {}).get("dataset_version"),
        },
        dataset_references=list(context.dataset_references),
        environment_keys=list(context.environment_keys),
        image_references=list(context.image_references),
        tree_id=context.tree_id,
        tree_summary=_tree_summary(context),
        key_path=list(tables.key_path),
        key_nodes=key_nodes,
        metrics_table=list(tables.metrics),
        stability_table=list(tables.stability),
        resource_table=list(tables.resources),
        ablation_table=list(tables.ablations),
        failure_table=list(tables.failures),
        evidence_records=list(tables.evidence_summary),
        supported_claims=supported,
        blocked_claims=blocked,
        conclusions=conclusions,
        limitations=list(context.limitations),
        open_evidence_gaps=list(context.open_evidence_gaps),
        recommended_next_experiments=_default_next_steps(context),
        context_sha256=context.context_sha256,
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
        generator_version="v1.2.4",
    )


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_（无数据）_\n"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| "
        + " | ".join("" if c is None else str(c).replace("|", "/") for c in row)
        + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body]) + "\n"


def render_markdown_report(report: ResearchReport) -> str:
    lines: list[str] = [
        f"# Research Report — {report.project_id}",
        "",
        f"- report_id: `{report.report_id}`",
        f"- status: `{report.status}`",
        f"- protocol_id: `{report.protocol_id or ''}`",
        f"- tree_id: `{report.tree_id or ''}`",
        f"- generator: `{report.generator_version}`",
        "",
        "## 1. 研究目标",
        "",
        report.research_goal or "_（未填写）_",
        "",
        "## 2. 实验协议",
        "",
        "```json",
        json.dumps(report.protocol_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. 数据集与环境",
        "",
        f"- datasets: {', '.join(report.dataset_references) or 'n/a'}",
        f"- environments: {', '.join(report.environment_keys) or 'n/a'}",
        f"- images: {', '.join(report.image_references) or 'n/a'}",
        "",
        "## 4. 实验树摘要",
        "",
        f"- status: `{report.tree_summary.get('status')}`",
        f"- node_count: `{report.tree_summary.get('node_count')}`",
        f"- stop_reason: `{report.tree_summary.get('stop_reason')}`",
        "",
    ]
    ascii_tree = report.tree_summary.get("ascii_tree")
    if ascii_tree:
        lines.extend(["```text", str(ascii_tree), "```", ""])

    lines.extend(
        [
            "## 5. 关键路径",
            "",
            _md_table(
                ["depth", "experiment_node_id", "node_type", "status", "primary_mean"],
                [
                    [
                        s.get("depth"),
                        s.get("experiment_node_id"),
                        s.get("node_type"),
                        s.get("status"),
                        s.get("primary_mean"),
                    ]
                    for s in report.key_path
                ],
            ),
            "",
            "## 6. 主要指标",
            "",
            _md_table(
                ["node_id", "primary_metric", "mean", "std", "claim_level"],
                [
                    [
                        r.get("node_id"),
                        r.get("primary_metric"),
                        r.get("primary_mean"),
                        r.get("primary_std"),
                        r.get("claim_level"),
                    ]
                    for r in report.metrics_table
                ],
            ),
            "",
            "## 7. 稳定性",
            "",
            _md_table(
                ["node_id", "mean", "std", "cv", "seed_count", "stable"],
                [
                    [
                        r.get("node_id"),
                        r.get("mean"),
                        r.get("std"),
                        r.get("cv"),
                        r.get("seed_count"),
                        r.get("stable"),
                    ]
                    for r in report.stability_table
                ],
            ),
            "",
            "## 8. 资源代价",
            "",
            _md_table(
                ["node_id", "duration_s", "peak_gpu_mb", "params"],
                [
                    [
                        r.get("node_id"),
                        r.get("duration_seconds_mean"),
                        r.get("peak_gpu_memory_mb_mean"),
                        r.get("parameter_count_mean"),
                    ]
                    for r in report.resource_table
                ],
            ),
            "",
            "## 9. 消融",
            "",
            _md_table(
                ["ablation_id", "node_id", "title", "status"],
                [
                    [
                        r.get("ablation_id"),
                        r.get("node_id"),
                        r.get("title"),
                        r.get("status"),
                    ]
                    for r in report.ablation_table
                ],
            ),
            "",
            "## 10. 失败实验",
            "",
            _md_table(
                ["source", "node_id", "status", "node_type"],
                [
                    [
                        r.get("source"),
                        r.get("node_id"),
                        r.get("status"),
                        r.get("node_type"),
                    ]
                    for r in report.failure_table
                ],
            ),
            "",
            "## 11. Evidence",
            "",
            _md_table(
                ["evidence_id", "type", "strength", "nodes"],
                [
                    [
                        r.get("evidence_id"),
                        r.get("evidence_type"),
                        r.get("evidence_strength"),
                        ",".join(r.get("source_node_ids") or []),
                    ]
                    for r in report.evidence_records
                ],
            ),
            "",
            "## 12. Claims",
            "",
            "### Supported / Partially supported",
            "",
            _md_table(
                ["claim_id", "status", "text"],
                [
                    [c.get("claim_id"), c.get("support_status"), c.get("claim_text")]
                    for c in report.supported_claims
                ],
            ),
            "",
            "### Blocked / Unsupported",
            "",
            _md_table(
                ["claim_id", "status", "text"],
                [
                    [c.get("claim_id"), c.get("support_status"), c.get("claim_text")]
                    for c in report.blocked_claims
                ],
            ),
            "",
            "## 13. 结论（Claim-gated）",
            "",
        ]
    )
    for item in report.conclusions:
        lines.extend(
            [
                f"### {item.conclusion_id}",
                "",
                item.text,
                "",
                f"- claim_id: `{item.claim_id}`",
                f"- support_status: `{item.support_status}`",
                f"- evidence_ids: {', '.join(item.evidence_ids) or 'n/a'}",
                f"- limitations: {'; '.join(item.limitations) or 'n/a'}",
                "",
            ]
        )

    lines.extend(
        [
            "## 14. 局限性",
            "",
        ]
    )
    for limitation in report.limitations or ["_（无）_"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## 15. 后续实验建议", ""])
    for tip in report.recommended_next_experiments or ["_（无）_"]:
        lines.append(f"- {tip}")
    lines.append("")
    return "\n".join(lines)


def render_executive_summary(report: ResearchReport) -> str:
    tip = report.key_path[-1] if report.key_path else {}
    lines = [
        f"# Executive Summary — {report.project_id}",
        "",
        f"**Goal:** {report.research_goal or 'n/a'}",
        "",
        f"**Protocol:** `{report.protocol_id or 'n/a'}`",
        f"**Tree:** `{report.tree_id or 'n/a'}` status=`{report.tree_summary.get('status')}`",
        "",
        "## Key path tip",
        "",
        f"- node: `{tip.get('experiment_node_id')}`",
        f"- metric: `{tip.get('primary_metric')}` = `{tip.get('primary_mean')}`",
        "",
        "## Claim-gated conclusions",
        "",
    ]
    for item in report.conclusions[:5]:
        lines.append(f"- [{item.support_status}] {item.text}")
    lines.extend(
        [
            "",
            "## Hard limits",
            "",
            "- Weak / blocked evidence is never rewritten as strong claims.",
            "- All conclusions carry claim_id / support_status / evidence_ids / limitations.",
            "",
        ]
    )
    return "\n".join(lines)
