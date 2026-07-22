"""Aggregate case grades into evaluation scorecards (v1.5.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.llm_eval.models import CaseGrade
from scientist_lab.storage.artifact_store import write_json


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


def aggregate_scorecard(
    *,
    evaluation_id: str,
    suite_version: str,
    profile_id: str,
    provider: str,
    grades: list[CaseGrade],
    operations: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planner = [g for g in grades if g.task_type == "planner"]
    critic = [g for g in grades if g.task_type == "critic"]
    safety = [g for g in grades if g.task_type == "safety"]

    def metric_rate(items: list[CaseGrade], name: str) -> float | None:
        vals: list[bool] = []
        for g in items:
            for s in g.scores:
                if s.name == name:
                    vals.append(s.passed)
        if not vals:
            return None
        return _rate(vals)

    safety_violations = sum(1 for g in safety if g.hard_fail and not g.passed)
    # For detection cases, passed=True means expected violation was found.
    safety_pass = all(g.passed for g in safety) if safety else True

    planner_block = {
        "schema_valid_rate": metric_rate(planner, "schema_valid") or 0.0,
        "protocol_compliance_rate": metric_rate(planner, "protocol_compliant") or 0.0,
        "evidence_gap_relevance_rate": metric_rate(planner, "evidence_gap_relevant"),
        "duplicate_candidate_rate": (
            1.0 - (metric_rate(planner, "non_duplicate") or 1.0)
            if planner
            else 0.0
        ),
        "pass_rate": _rate([g.passed for g in planner]),
        "case_count": len(planner),
    }
    critic_block = {
        "valid_acceptance_rate": metric_rate(critic, "accepts_valid_ablation"),
        "invalid_rejection_rate": metric_rate(critic, "rejects_invalid_candidate"),
        "claim_overreach_detection_rate": metric_rate(
            critic, "detects_claim_overreach"
        ),
        "pass_rate": _rate([g.passed for g in critic]),
        "case_count": len(critic),
    }
    safety_block = {
        "violation_count": safety_violations,
        "pass": safety_pass,
        "pass_rate": _rate([g.passed for g in safety]),
        "case_count": len(safety),
    }
    ops = dict(operations or {})
    return {
        "evaluation_id": evaluation_id,
        "suite_version": suite_version,
        "profile_id": profile_id,
        "provider": provider,
        "planner": planner_block,
        "critic": critic_block,
        "safety": safety_block,
        "operations": {
            "average_latency_ms": float(ops.get("average_latency_ms") or 0.0),
            "total_tokens": int(ops.get("total_tokens") or 0),
            "estimated_cost_usd": ops.get("estimated_cost_usd"),
            "case_count": len(grades),
            "passed_count": sum(1 for g in grades if g.passed),
        },
        "metadata": dict(metadata or {}),
    }


def write_evaluation_artifacts(
    out_dir: Path,
    *,
    scorecard: dict[str, Any],
    grades: list[CaseGrade],
    manifest: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "evaluation_report.json"
    report_md = out_dir / "evaluation_report.md"
    cases_path = out_dir / "case_results.jsonl"
    failures_path = out_dir / "failure_cases.jsonl"
    manifest_path = out_dir / "manifest.json"

    write_json(report_json, scorecard)
    write_json(manifest_path, manifest)

    lines = []
    fail_lines = []
    for grade in grades:
        row = grade.model_dump(mode="json")
        lines.append(json.dumps(row, ensure_ascii=False))
        if not grade.passed:
            fail_lines.append(json.dumps(row, ensure_ascii=False))
    cases_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    failures_path.write_text(
        "\n".join(fail_lines) + ("\n" if fail_lines else ""), encoding="utf-8"
    )

    ops = scorecard.get("operations") or {}
    safety = scorecard.get("safety") or {}
    planner = scorecard.get("planner") or {}
    md = "\n".join(
        [
            f"# LLM Evaluation `{scorecard.get('evaluation_id')}`",
            "",
            f"- suite: `{scorecard.get('suite_version')}`",
            f"- profile: `{scorecard.get('profile_id')}`",
            f"- provider: `{scorecard.get('provider')}`",
            f"- planner schema_valid_rate: {planner.get('schema_valid_rate')}",
            f"- planner protocol_compliance_rate: {planner.get('protocol_compliance_rate')}",
            f"- safety pass: {safety.get('pass')}",
            f"- average_latency_ms: {ops.get('average_latency_ms')}",
            f"- total_tokens: {ops.get('total_tokens')}",
            f"- estimated_cost_usd: {ops.get('estimated_cost_usd')}",
            f"- passed: {ops.get('passed_count')}/{ops.get('case_count')}",
            "",
        ]
    )
    report_md.write_text(md, encoding="utf-8")
    return {
        "evaluation_report_json": str(report_json),
        "evaluation_report_md": str(report_md),
        "case_results": str(cases_path),
        "failure_cases": str(failures_path),
        "manifest": str(manifest_path),
    }
