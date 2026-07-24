"""Deterministic FeedbackUseVerifier for multi-round real loops (v2.1.6)."""

from __future__ import annotations

import re
from typing import Any, Mapping

from scientist_lab.planning.candidate_verifier import parameter_fingerprint
from scientist_lab.research_loop.models import FeedbackUseVerification


_EVIDENCE_ID_RE = re.compile(r"evidence_[a-zA-Z0-9_]+")
_METRIC_HINTS = ("accuracy", "f1", "macro_f1", "f1_macro", "loss", "map", "ap")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _candidate_params(candidate: Mapping[str, Any]) -> dict[str, Any]:
    changes = candidate.get("parameter_changes")
    if isinstance(changes, dict) and changes:
        return dict(changes)
    overrides = candidate.get("protocol_overrides")
    if isinstance(overrides, dict) and overrides:
        return dict(overrides)
    params = candidate.get("parameters")
    if isinstance(params, dict) and params:
        return dict(params)
    return {}


def _text_blob(candidate: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "rationale",
        "hypothesis",
        "title",
        "reasoning_summary",
        "expected_effect",
    ):
        val = candidate.get(key)
        if val:
            parts.append(str(val))
    for gap in candidate.get("evidence_gap_addressed") or []:
        parts.append(str(gap))
    for item in candidate.get("expected_outcomes") or []:
        if isinstance(item, dict):
            parts.extend(str(v) for v in item.values() if v is not None)
        else:
            parts.append(str(item))
    for lim in candidate.get("claim_limitations") or []:
        parts.append(str(lim))
    return "\n".join(parts).lower()


def _collect_referenced_evidence_ids(candidates: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for cand in candidates:
        for gap in cand.get("evidence_gap_addressed") or []:
            text = str(gap)
            found.update(_EVIDENCE_ID_RE.findall(text))
            if text.startswith("evidence_"):
                found.add(text.strip())
        blob = _text_blob(cand)
        found.update(_EVIDENCE_ID_RE.findall(blob))
    return found


def _mentions_metric_keys(text: str, metric_keys: set[str]) -> bool:
    lowered = text.lower()
    for key in metric_keys:
        if key and key.lower() in lowered:
            return True
    for hint in _METRIC_HINTS:
        if hint in lowered and (
            not metric_keys or any(hint in mk.lower() for mk in metric_keys)
        ):
            return True
    return False


def verify_feedback_use(
    *,
    feedback_summary: Mapping[str, Any] | None,
    planning_context: Mapping[str, Any] | None,
    candidates: list[Mapping[str, Any]] | None,
    previous_executed_parameters: Mapping[str, Any] | None = None,
    previous_candidate_parameters: list[Mapping[str, Any]] | None = None,
    expected_context_sha256: str | None = None,
) -> FeedbackUseVerification:
    """Run deterministic six-dim feedback-use checks (no LLM / no network)."""
    feedback = _as_dict(feedback_summary)
    context = _as_dict(planning_context)
    cands = [dict(c) for c in (candidates or []) if isinstance(c, Mapping)]
    issues: list[str] = []
    details: dict[str, Any] = {}

    # --- Context contains Round-1 metrics ---
    metric_deltas = dict(feedback.get("metric_deltas") or {})
    recent = dict(context.get("recent_execution_summary") or {})
    ctx_feedback = _as_dict(context.get("round_feedback_summary"))
    ctx_metrics = dict(ctx_feedback.get("metric_deltas") or recent.get("metric_deltas") or {})
    context_contains_round_1_metrics = bool(metric_deltas) and bool(ctx_metrics)
    if metric_deltas and not ctx_metrics:
        issues.append("context missing round-1 metric_deltas")
    if not metric_deltas:
        issues.append("round-1 feedback has empty metric_deltas")
        context_contains_round_1_metrics = False
    details["metric_keys"] = sorted(str(k) for k in metric_deltas.keys())

    # --- Context contains Round-1 evidence ---
    evidence_added = [str(x) for x in (feedback.get("evidence_added") or []) if str(x).strip()]
    ctx_evidence_ids = {
        str(e.get("evidence_id"))
        for e in (context.get("evidence_records") or [])
        if isinstance(e, dict) and e.get("evidence_id")
    }
    ctx_evidence_ids.update(
        str(x) for x in (ctx_feedback.get("evidence_added") or []) if str(x).strip()
    )
    ctx_evidence_ids.update(
        str(x) for x in (recent.get("evidence_added") or []) if str(x).strip()
    )
    if evidence_added:
        overlap = set(evidence_added) & ctx_evidence_ids
        context_contains_round_1_evidence = bool(overlap) or (
            # soft: summary embedded even if full records not listed
            bool(ctx_feedback.get("evidence_added"))
        )
        if not context_contains_round_1_evidence:
            issues.append("context missing round-1 evidence_added ids")
        details["evidence_overlap"] = sorted(overlap)
    else:
        context_contains_round_1_evidence = False
        issues.append("round-1 feedback has empty evidence_added")

    # --- Context contains Round-1 decision ---
    outcome = str(feedback.get("outcome_label") or "").strip()
    executed_id = str(feedback.get("executed_node_id") or "").strip()
    ctx_outcome = str(
        ctx_feedback.get("outcome_label")
        or recent.get("outcome_label")
        or ""
    ).strip()
    ctx_executed = str(
        ctx_feedback.get("executed_node_id")
        or recent.get("executed_node_id")
        or ""
    ).strip()
    context_contains_round_1_decision = bool(outcome or executed_id) and (
        (outcome and outcome == ctx_outcome)
        or (executed_id and executed_id == ctx_executed)
        or bool(ctx_feedback)
    )
    if not context_contains_round_1_decision:
        issues.append("context missing round-1 decision (outcome/executed_node)")

    # context_sha256 consistency (optional)
    ctx_sha = str(context.get("context_sha256") or "").strip()
    if expected_context_sha256 and ctx_sha and ctx_sha != expected_context_sha256:
        issues.append("planning context_sha256 mismatch")
        details["context_sha256_mismatch"] = {
            "expected": expected_context_sha256,
            "actual": ctx_sha,
        }

    # --- Output changes experiment plan + avoids duplicate ---
    prev_params = dict(previous_executed_parameters or feedback.get("executed_parameters") or {})
    prev_fps: set[str] = set()
    if prev_params:
        prev_fps.add(parameter_fingerprint(prev_params))
    for prev_cand in previous_candidate_parameters or []:
        pp = _candidate_params(prev_cand) if isinstance(prev_cand, Mapping) else {}
        # Prefer only variable subset if full params include fixed keys
        if pp:
            # If full contract params, fingerprint on overlapping variable keys only
            # when comparing to candidate changes; else full dict.
            prev_fps.add(parameter_fingerprint(pp))
            if "hidden_units" in pp:
                prev_fps.add(parameter_fingerprint({"hidden_units": pp["hidden_units"]}))

    tested = {
        str(x)
        for x in (context.get("tested_parameter_fingerprints") or [])
        if str(x).strip()
    }
    prev_fps |= tested

    changed = False
    duplicate = False
    cand_fps: list[str] = []
    if not cands:
        issues.append("round-2 plan has no candidates")
        duplicate = True
    for cand in cands:
        params = _candidate_params(cand)
        fp = parameter_fingerprint(params) if params else ""
        if fp:
            cand_fps.append(fp)
        if params and fp and fp in prev_fps:
            duplicate = True
            issues.append(
                f"duplicate candidate fingerprint={fp} params={params}"
            )
        if params and prev_params:
            # Plan change: at least one key differs from executed params.
            differs = False
            for key, val in params.items():
                if prev_params.get(key) != val:
                    differs = True
                    break
            if differs or (fp and fp not in prev_fps):
                changed = True
            elif params == {k: prev_params.get(k) for k in params.keys()}:
                issues.append(f"candidate repeats executed parameters: {params}")
        elif params and not prev_params:
            changed = True
        elif not params:
            issues.append(
                f"candidate {cand.get('candidate_id')} has empty parameter_changes"
            )

    output_changes_experiment_plan = bool(changed) and not duplicate
    output_avoids_duplicate_candidate = bool(cands) and not duplicate
    if cands and not changed and not duplicate:
        # No prev to compare — treat any non-empty params as a change.
        output_changes_experiment_plan = any(_candidate_params(c) for c in cands)
    details["candidate_fingerprints"] = cand_fps
    details["previous_fingerprints"] = sorted(prev_fps)

    # Parent node preference: Round2 parent should be Round1 executed node when present.
    if executed_id and cands:
        parents = {str(c.get("parent_node_id") or "") for c in cands}
        if parents and executed_id not in parents:
            issues.append(
                f"round-2 candidates do not parent from executed node {executed_id}"
            )
            details["parent_node_ids"] = sorted(p for p in parents if p)

    # --- Output references new evidence / metrics ---
    referenced_ids = _collect_referenced_evidence_ids(cands)
    details["referenced_evidence_ids"] = sorted(referenced_ids)
    evidence_ref_ok = False
    if evidence_added:
        evidence_ref_ok = bool(set(evidence_added) & referenced_ids)
    metric_keys = {str(k) for k in metric_deltas.keys()}
    metric_ref_ok = False
    for cand in cands:
        if _mentions_metric_keys(_text_blob(cand), metric_keys):
            metric_ref_ok = True
            break
    # Also accept explicit evidence_gap_addressed non-empty when evidence existed in context
    gap_ok = any(bool(c.get("evidence_gap_addressed")) for c in cands) and (
        context_contains_round_1_evidence or context_contains_round_1_metrics
    )
    output_references_new_evidence = bool(evidence_ref_ok or metric_ref_ok or gap_ok)
    if not output_references_new_evidence:
        issues.append(
            "round-2 candidates do not reference round-1 evidence ids or metric keys"
        )

    # Ghost metric references (fail if inventing unknown metric names as sole signal)
    ghost_hits: list[str] = []
    known = metric_keys | set(_METRIC_HINTS) | {"hidden_units", "learning_rate", "batch_size", "dropout", "epochs"}
    for cand in cands:
        for outcome in cand.get("expected_outcomes") or []:
            if isinstance(outcome, dict):
                m = str(outcome.get("metric") or "").strip().lower()
                if m and m not in known and m not in {k.lower() for k in metric_keys}:
                    # Allow common Digits metrics even if deltas empty.
                    if m not in {"accuracy", "f1_macro", "macro_f1", "log_loss"}:
                        ghost_hits.append(m)
    if ghost_hits:
        issues.append(f"candidates reference unknown metrics: {sorted(set(ghost_hits))}")
        output_references_new_evidence = False

    required = [
        context_contains_round_1_metrics,
        context_contains_round_1_evidence,
        context_contains_round_1_decision,
        output_references_new_evidence,
        output_changes_experiment_plan,
        output_avoids_duplicate_candidate,
    ]
    pass_status = all(required) and not any(
        "duplicate candidate" in i or "unknown metrics" in i for i in issues
    )
    # Deduplicate issues while preserving order.
    seen: set[str] = set()
    uniq_issues: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            uniq_issues.append(item)

    return FeedbackUseVerification(
        context_contains_round_1_metrics=context_contains_round_1_metrics,
        context_contains_round_1_evidence=context_contains_round_1_evidence,
        context_contains_round_1_decision=context_contains_round_1_decision,
        output_references_new_evidence=output_references_new_evidence,
        output_changes_experiment_plan=output_changes_experiment_plan,
        output_avoids_duplicate_candidate=output_avoids_duplicate_candidate,
        pass_status=pass_status,
        issues=uniq_issues,
        details=details,
    )
