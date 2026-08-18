"""LLM Reviewer contract: Evidence + Rubric decision → semantic proposal.

Same Gateway as Planner. Reviewer is not a second physical model and not HOW.
DecisionRubric remains the only KEEP/DISCARD/REPLICATE source.
ClaimGate remains the claim gate. MemoryWriter remains the only Memory writer.
LLM proposes a lesson; it must not overwrite review_decision or invent claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from scientist_lab.core.claim_gate import APS_KEYS, MAP_KEYS
from scientist_lab.llm.config import redact_secrets
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.planner_contract import prompt_hash
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema


class ReviewerContractError(ValueError):
    """LLM output cannot be attached as a semantic proposal."""


_BANNED_OPERATOR = re.compile(
    r"\b(fdpn|new[-_ ]network|write(?:\s+some)?\s+python|implement(?:\s+a)?\s+new"
    r"|custom[-_ ]operator|novel[-_ ]fusion)\b",
    re.IGNORECASE,
)

_BANNED_EFFECTIVENESS = re.compile(
    r"(模块有效|模块无效|\bineffective\b|\beffective\s+module\b"
    r"|module\s+is\s+(?:not\s+)?effective)",
    re.IGNORECASE,
)

_MEMORY_WRITE_KEYS = frozenset(
    {
        "lessons_to_write",
        "strategies_to_write",
        "memory_write",
        "persist_lesson",
        "persist_strategy",
        "write_memory",
    }
)

_REVIEW_OVERRIDE_KEYS = frozenset(
    {
        "review_decision",
        "review_decision_override",
        "keep_discard",
        "rubric_override",
        "claim_gate",
        "claim_gate_override",
        "hypothesis_status_override",
    }
)

ALLOWED_HYPOTHESIS_STATUS = frozenset(
    {
        "not_supported_under_current_protocol",
        "inconclusive_budget",
        "not_a_claim",
        "needs_replication",
        "needs_validation",
    }
)

STATUS_BY_DECISION = {
    "DISCARD": frozenset(
        {"not_supported_under_current_protocol", "inconclusive_budget"}
    ),
    "KEEP": frozenset({"not_a_claim"}),
    "REPLICATE": frozenset({"needs_replication", "inconclusive_budget"}),
    "VALIDATE": frozenset({"needs_validation", "not_a_claim"}),
    "ESCALATE": frozenset({"inconclusive_budget", "not_a_claim"}),
    "STOP": frozenset(
        {"not_supported_under_current_protocol", "inconclusive_budget", "not_a_claim"}
    ),
}

DOCUMENT_HYPOTHESIS_ENUM = frozenset(
    {
        "SUPPORTED",
        "REJECTED",
        "INCONCLUSIVE",
        "NEEDS_REPLICATION",
        "NEEDS_VALIDATION",
    }
)

REVIEWER_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "observation",
        "hypothesis_status",
        "interpretation",
        "alternative_explanations",
        "next_research_priority",
        "evidence_refs",
        "created_from",
        "confidence",
    ],
    "additionalProperties": True,
    "properties": {
        "observation": {"type": "string", "minLength": 1},
        "hypothesis_status": {"type": "string", "minLength": 1},
        "interpretation": {"type": "string", "minLength": 1},
        "alternative_explanations": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "next_research_priority": {"type": "string", "minLength": 1},
        "evidence_refs": {"type": "array", "minItems": 1},
        "created_from": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

REVIEWER_SYSTEM_PROMPT = """You are the Reviewer cognitive backend for Scientist Lab.
You are NOT a fifth Agent. Roles: Manager, Planner, Executor, Reviewer.
Gateway is a backend, not an Agent. Adapter=HOW; you do not emit HOW.

Return JSON only matching the provided schema. This is a PROPOSAL, not a decision.

Hard rules:
- DecisionRubric already locked review_decision (KEEP/DISCARD/REPLICATE/VALIDATE).
  Do not output review_decision, do not override it, do not KEEP/DISCARD yourself.
- ClaimGate remains the claim gate. DISCARD is not a formal claim that a module
  is ineffective. KEEP is not ClaimGate SUPPORTED.
- Do not write Memory. Do not invent lesson/strategy ids. MemoryWriter may persist
  this proposal later only if evidence_refs are valid.
- hypothesis_status must be one of:
  not_supported_under_current_protocol, inconclusive_budget, not_a_claim,
  needs_replication, needs_validation.
  DISCARD → not_supported_under_current_protocol or inconclusive_budget.
  KEEP → not_a_claim (KEEP ≠ claim supported).
- Primary metric is Protocol objective.primary (APS here). Do not impersonate APS with mAP.
- Cite only the provided run_id in evidence_refs and created_from.
- Explain WHY the Rubric decision happened, alternative explanations, and WHAT
  the next research priority should verify. Not HOW (no fusion_method, no Python).
- Do not invent operators or FDPN.
"""


@dataclass
class ReviewerContractInput:
    protocol: Mapping[str, Any]
    plan: Mapping[str, Any]
    result: Mapping[str, Any]
    evidence: Mapping[str, Any]
    rubric: Mapping[str, Any]
    locked_review_decision: str
    document_hypothesis_status: str
    run_id: str
    primary_metric: str
    primary_delta: float | None = None
    budget_class: str = "probe"
    memory: Mapping[str, Any] = field(default_factory=dict)


def _refuse(message: str) -> None:
    raise ReviewerContractError(message)


def _as_text(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts)


def _primary_metric(protocol: Mapping[str, Any]) -> str:
    return str(
        ((protocol.get("objective") or {}).get("primary") or {}).get("metric") or "APS"
    )


def build_contract_input(
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
    rubric: Mapping[str, Any],
    locked_review_decision: str,
    document_hypothesis_status: str,
    memory: Mapping[str, Any] | None = None,
) -> ReviewerContractInput:
    run_id = str(result.get("run_id") or "")
    metric = _primary_metric(protocol)
    check = dict((rubric.get("objective_check") or {}).get(metric) or {})
    delta = rubric.get("primary_delta")
    if delta is None and check.get("delta") is not None:
        delta = check.get("delta")
    return ReviewerContractInput(
        protocol=dict(protocol),
        plan=dict(plan),
        result={
            "run_id": run_id,
            "metrics": dict(result.get("metrics") or {}),
            "execution": dict(result.get("execution") or {}),
        },
        evidence=dict(evidence),
        rubric={
            "objective_check": dict(rubric.get("objective_check") or {}),
            "constraint_check": dict(rubric.get("constraint_check") or {}),
            "primary_delta": rubric.get("primary_delta"),
            "constraints_ok": rubric.get("constraints_ok"),
            "suggest_validate": rubric.get("suggest_validate"),
            "suggest_discard_threshold": rubric.get("suggest_discard_threshold"),
            "keep_threshold_ok": rubric.get("keep_threshold_ok"),
        },
        locked_review_decision=str(locked_review_decision),
        document_hypothesis_status=str(document_hypothesis_status),
        run_id=run_id,
        primary_metric=metric,
        primary_delta=float(delta) if delta is not None else None,
        budget_class=str(plan.get("budget_class") or "probe"),
        memory=dict(memory or {}),
    )


def build_reviewer_request(payload: ReviewerContractInput) -> LLMRequest:
    user = {
        "role": "reviewer",
        "locked_review_decision": payload.locked_review_decision,
        "document_hypothesis_status": payload.document_hypothesis_status,
        "note": (
            "document_hypothesis_status is the freeze review_decision document enum "
            "(REJECTED for DISCARD). Your hypothesis_status is semantic and must NOT "
            "copy SUPPORTED/REJECTED as a claim."
        ),
        "protocol": {
            "protocol_id": payload.protocol.get("protocol_id"),
            "protocol_version": payload.protocol.get("protocol_version"),
            "project_id": payload.protocol.get("project_id"),
            "objective": dict(payload.protocol.get("objective") or {}),
            "decision_policy": dict(payload.protocol.get("decision_policy") or {}),
            "editable_scope": list(payload.protocol.get("editable_scope") or []),
        },
        "plan": {
            "plan_id": payload.plan.get("plan_id"),
            "round_index": payload.plan.get("round_index"),
            "modification_scope": list(payload.plan.get("modification_scope") or []),
            "hypothesis": payload.plan.get("hypothesis"),
            "observation": payload.plan.get("observation"),
            "budget_class": payload.plan.get("budget_class"),
            "expected_effect": dict(payload.plan.get("expected_effect") or {}),
        },
        "result": dict(payload.result),
        "evidence": dict(payload.evidence),
        "rubric": dict(payload.rubric),
        "run_id": payload.run_id,
        "primary_metric": payload.primary_metric,
        "primary_delta": payload.primary_delta,
        "budget_class": payload.budget_class,
        "constraints": {
            "forbid_review_override": True,
            "forbid_claim_gate_override": True,
            "forbid_memory_write": True,
            "forbid_fdpn": True,
            "forbid_how": True,
            "aps_is_not_map": payload.primary_metric in APS_KEYS,
            "discard_is_not_module_ineffective": True,
            "keep_is_not_claim_supported": True,
        },
    }
    return LLMRequest(
        purpose="reviewer",
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=REVIEWER_CONTRACT_SCHEMA,
        temperature=0.0,
        metadata={
            "reviewer_contract": "semantic",
            "project_id": str(payload.protocol.get("project_id") or ""),
            "run_id": payload.run_id,
            "locked_review_decision": payload.locked_review_decision,
        },
    )


def normalize_evidence_refs(refs: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in refs:
        if isinstance(item, str) and item.strip():
            out.append({"run_id": item.strip()})
            continue
        if isinstance(item, Mapping) and item.get("run_id"):
            row: dict[str, Any] = {"run_id": str(item["run_id"])}
            if item.get("metric"):
                row["metric"] = str(item["metric"])
            if "delta" in item:
                row["delta"] = item.get("delta")
            out.append(row)
    return out


def assert_proposal_writable(
    proposal: Mapping[str, Any],
    *,
    run_id: str,
    review_decision: str | None = None,
) -> list[dict[str, Any]]:
    """MemoryWriter gate: missing evidence_refs or banned claims refuse the write."""
    refs = normalize_evidence_refs(list(proposal.get("evidence_refs") or []))
    if not refs:
        _refuse("semantic proposal refused: missing evidence_refs")
    created = [str(x) for x in (proposal.get("created_from") or []) if str(x).strip()]
    if not created:
        _refuse("semantic proposal refused: missing created_from")
    if run_id and not any(row.get("run_id") == run_id for row in refs):
        _refuse(f"semantic proposal refused: evidence_refs must cite run_id={run_id}")
    if run_id and run_id not in created:
        _refuse(f"semantic proposal refused: created_from must include run_id={run_id}")
    haystack = _as_text(
        proposal.get("observation"),
        proposal.get("interpretation"),
        proposal.get("hypothesis_status"),
        proposal.get("next_research_priority"),
        json.dumps(proposal.get("alternative_explanations") or [], ensure_ascii=False),
    )
    if _BANNED_EFFECTIVENESS.search(haystack):
        _refuse(
            "semantic proposal refused: DISCARD/KEEP must not be written as "
            "module effective/ineffective scientific claims"
        )
    status = str(proposal.get("hypothesis_status") or "")
    if status in DOCUMENT_HYPOTHESIS_ENUM or status.lower() in {"supported", "claim_supported"}:
        _refuse(
            "semantic proposal refused: hypothesis_status must not copy ClaimGate/"
            "document SUPPORTED/REJECTED as a scientific claim"
        )
    if str(review_decision or "") == "KEEP" and status != "not_a_claim":
        _refuse("semantic proposal refused: KEEP is not a claim supported")
    return refs


def parse_reviewer_completion(
    response: LLMResponse | str,
    payload: ReviewerContractInput,
) -> dict[str, Any]:
    """Parse LLM JSON into a semantic proposal. Fail closed. Does not KEEP/DISCARD."""
    raw = response.content if isinstance(response, LLMResponse) else str(response)
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        _refuse(f"fail_closed: invalid JSON: {exc}")

    errors = validate_against_schema(data, REVIEWER_CONTRACT_SCHEMA)
    if errors:
        _refuse("fail_closed: contract schema: " + "; ".join(errors))

    for key in _MEMORY_WRITE_KEYS:
        if data.get(key):
            _refuse("fail_closed: LLM must not write Memory")
    for key in _REVIEW_OVERRIDE_KEYS:
        if key in data and data.get(key) not in (None, "", False):
            _refuse("fail_closed: LLM must not override KEEP/DISCARD/ClaimGate")

    locked = str(payload.locked_review_decision)
    if str(data.get("selected_action") or "") in {
        "KEEP",
        "DISCARD",
        "REPLICATE",
        "VALIDATE",
        "ESCALATE",
        "STOP",
    }:
        _refuse("fail_closed: LLM must not emit review_decision/selected_action KEEP/DISCARD")

    status = str(data.get("hypothesis_status") or "").strip()
    if status in DOCUMENT_HYPOTHESIS_ENUM:
        _refuse(
            "fail_closed: hypothesis_status must not copy document enum "
            f"{status}; KEEP is not ClaimGate SUPPORTED and DISCARD is not REJECTED-as-claim"
        )
    if status not in ALLOWED_HYPOTHESIS_STATUS:
        _refuse(f"fail_closed: unknown hypothesis_status={status!r}")
    allowed = STATUS_BY_DECISION.get(locked, ALLOWED_HYPOTHESIS_STATUS)
    if status not in allowed:
        _refuse(
            f"fail_closed: hypothesis_status={status!r} does not align with "
            f"locked review_decision={locked}"
        )

    refs = normalize_evidence_refs(list(data.get("evidence_refs") or []))
    created = [str(x) for x in (data.get("created_from") or []) if str(x).strip()]
    if not refs:
        _refuse("fail_closed: evidence_refs required")
    if not created:
        _refuse("fail_closed: created_from required")
    if payload.run_id and not any(row.get("run_id") == payload.run_id for row in refs):
        _refuse(f"fail_closed: evidence_refs must cite run_id={payload.run_id}")
    if payload.run_id and payload.run_id not in created:
        _refuse(f"fail_closed: created_from must include run_id={payload.run_id}")

    primary = payload.primary_metric
    if primary in APS_KEYS:
        for row in refs:
            metric = str(row.get("metric") or primary)
            if metric in MAP_KEYS:
                _refuse("fail_closed: APS must not be impersonated by mAP")
        haystack_metrics = _as_text(
            data.get("observation"),
            data.get("interpretation"),
            data.get("next_research_priority"),
        )
        if re.search(
            r"(primary|主指标)\s*(metric)?\s*(is|=|:|为)?\s*mAP",
            haystack_metrics,
            re.IGNORECASE,
        ):
            _refuse("fail_closed: APS must not be impersonated by mAP")

    haystack = _as_text(
        data.get("observation"),
        data.get("interpretation"),
        data.get("hypothesis_status"),
        data.get("next_research_priority"),
        json.dumps(data.get("alternative_explanations") or [], ensure_ascii=False),
    )
    if _BANNED_OPERATOR.search(haystack):
        _refuse("fail_closed: invented operator / FDPN / write-Python is forbidden")
    if _BANNED_EFFECTIVENESS.search(haystack):
        _refuse(
            "fail_closed: DISCARD is not a module-ineffective claim; "
            "KEEP is not a module-effective claim"
        )
    if re.search(r"\bearly_concat\b|\bfusion_method\b", haystack, re.IGNORECASE):
        _refuse("fail_closed: Reviewer must not emit HOW")

    proposal = {
        "observation": str(data["observation"]).strip(),
        "hypothesis_status": status,
        "interpretation": str(data["interpretation"]).strip(),
        "alternative_explanations": [
            str(x).strip() for x in data["alternative_explanations"] if str(x).strip()
        ],
        "next_research_priority": str(data["next_research_priority"]).strip(),
        "evidence_refs": refs,
        "created_from": created,
        "confidence": str(data.get("confidence") or "medium"),
        "locked_review_decision": locked,
        "run_id": payload.run_id,
        "primary_metric": primary,
    }
    return proposal


def proposal_to_research_lesson(
    proposal: Mapping[str, Any],
    *,
    run_id: str,
    review_decision: str,
    lesson_type: str,
    module: str,
    task: str,
    metric: str,
    delta: float | None = None,
) -> dict[str, Any]:
    """Map a validated proposal onto the freeze research_lesson schema.

    Does not dump LLM-only keys onto the persisted lesson (additionalProperties false).
    """
    refs = assert_proposal_writable(
        proposal, run_id=run_id, review_decision=review_decision
    )
    evidence: list[dict[str, Any]] = []
    for row in refs:
        item: dict[str, Any] = {"run_id": str(row["run_id"])}
        item["metric"] = str(row.get("metric") or metric)
        if "delta" in row and row.get("delta") is not None:
            item["delta"] = row.get("delta")
        elif delta is not None:
            item["delta"] = delta
        evidence.append(item)
    statement = str(proposal.get("interpretation") or "").strip()
    if proposal.get("next_research_priority"):
        statement = (
            f"{statement} Next research priority: "
            f"{str(proposal['next_research_priority']).strip()}"
        )
    confidence = str(proposal.get("confidence") or "medium")
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "lesson_id": f"LESSON-{run_id}-semantic-001",
        "type": lesson_type,
        "statement": statement,
        "status": "active",
        "evidence": evidence,
        "scope": {"task": task, "module": module},
        "confidence": confidence,
        "created_from": [str(x) for x in proposal.get("created_from") or [run_id]],
        "contradicted_by": [],
        "supersedes": [],
        "expires_when": [],
    }


def redacted_raw(response: LLMResponse | None) -> str:
    if response is None:
        return ""
    return redact_secrets(response.content or "")
