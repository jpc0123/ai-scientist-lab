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
from scientist_lab.core.memory_ids import lesson_id_for_run
from scientist_lab.llm.config import redact_secrets
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.planner_contract import prompt_hash
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema


class ReviewerContractError(ValueError):
    """LLM output cannot be attached as a semantic proposal."""


# Always banned: free invention / write-Python in Reviewer prose.
_BANNED_INVENT = re.compile(
    r"\b(new[-_ ]network|write(?:\s+some)?\s+python|implement(?:\s+a)?\s+new"
    r"|custom[-_ ]operator|novel[-_ ]fusion)\b",
    re.IGNORECASE,
)
# FDPN is catalog HOW (N1/A4), not free invention. Cite only when executed.
_BANNED_FDPN = re.compile(r"\bfdpn\b", re.IGNORECASE)
_EXECUTED_FDPN_HOW = re.compile(r"\b(n1|a4)\b|\bfdpn\b", re.IGNORECASE)

_BANNED_EFFECTIVENESS = re.compile(
    r"(模块有效|模块无效|"
    # Claim-framed English only (not bare "effective module" in KEEP≠Claim hedges).
    r"\b(?:prove[sd]?|show(?:s|ed)?|demonstrate[sd]?|establish(?:es|ed)?|"
    r"confirm(?:s|ed)?)\b.{0,40}\b(?:module\s+)?(?:in)?effective\b|"
    r"\bmodule\s+is\s+(?:in)?effective\b|"
    r"\bmodule\s+is\s+not\s+(?:in)?effective\b)",
    re.IGNORECASE | re.DOTALL,
)
# Meta-disclaimers / KEEP≠Claim hedges (clause-local).
_EFFECTIVENESS_NEGATION = re.compile(
    r"(does\s+not\s+constitute|do(?:es)?\s+not\s+(?:mean|prove|show|imply|support|indicate)|"
    r"not\s+that\b|without\s+(?:claiming|indicating)|"
    r"no(?:t)?\s+(?:making\s+)?(?:an?\s+)?(?:efficacy\s+)?claim|"
    r"not\s+a\s+(?:formal\s+)?(?:scientific\s+)?claim|"
    r"keep\s+(?:here\s+)?means\b|baseline\s+is\s+valid\b|"
    r"不构成|并非.*(?:主张|结论)|不是.*(?:主张|结论)|未主张|不作.*主张)",
    re.IGNORECASE,
)
# Soft strip: remove "not that …" / "do not indicate …" spans before positive match.
_DISCLAIMER_SPAN = re.compile(
    r"(?:not\s+that\b|do(?:es)?\s+not\s+indicate\b|without\s+claiming\b|"
    r"no\s+efficacy\s+claim\b|不构成(?:科学)?主张)"
    r".{0,160}?(?:[.!?\n]|$)",
    re.IGNORECASE | re.DOTALL,
)


def _has_banned_effectiveness_claim(text: str) -> bool:
    """True only for positive effective/ineffective claims, not KEEP≠Claim hedges.

    Prefer: (1) strip disclaimer spans, (2) require claim-ish framing for
    'effective module', (3) still honor local negation windows.
    """
    blob = str(text or "")
    stripped = _DISCLAIMER_SPAN.sub(" ", blob)
    for match in _BANNED_EFFECTIVENESS.finditer(stripped):
        start = max(0, match.start() - 120)
        window = stripped[start : match.end() + 40]
        if _EFFECTIVENESS_NEGATION.search(window):
            continue
        # Bare "effective module" after strip is still banned only with claim verbs
        # or "module is (in)effective"; the regex already encodes that for most
        # forms. Remaining bare hits from Chinese 模块有效/无效 stay hard bans.
        return True
    return False

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

# Structured KEEP≠Claim / DISCARD≠module-ineffective stance (prefer over prose bans).
ALLOWED_CLAIM_STANCE = frozenset(
    {
        "no_module_efficacy_claim",
        "deferred_to_claim_gate",
    }
)
CLAIM_STANCE_BY_DECISION = {
    "DISCARD": "no_module_efficacy_claim",
    "KEEP": "no_module_efficacy_claim",
    "REPLICATE": "deferred_to_claim_gate",
    "VALIDATE": "deferred_to_claim_gate",
    "ESCALATE": "deferred_to_claim_gate",
    "STOP": "no_module_efficacy_claim",
}
_SAFE_INTERPRETATION_BY_DECISION = {
    "KEEP": (
        "DecisionRubric already locked KEEP. KEEP is a next-round action only; "
        "not ClaimGate SUPPORTED and not a module-efficacy claim."
    ),
    "DISCARD": (
        "DecisionRubric already locked DISCARD. DISCARD is a next-action only; "
        "not a ClaimGate verdict on module efficacy."
    ),
    "REPLICATE": (
        "DecisionRubric already locked REPLICATE. Treat this as a replication "
        "priority, not a ClaimGate module-efficacy verdict."
    ),
    "VALIDATE": (
        "DecisionRubric already locked VALIDATE. Treat this as a verification "
        "priority, not a ClaimGate SUPPORTED result."
    ),
}


def default_claim_stance(locked_review_decision: str) -> str:
    return CLAIM_STANCE_BY_DECISION.get(
        str(locked_review_decision or "").strip(),
        "no_module_efficacy_claim",
    )


def _coerce_effectiveness_prose(
    data: dict[str, Any],
    *,
    locked: str,
) -> list[str]:
    """Soft-coerce residual efficacy phrasing instead of fail_closed."""
    warnings: list[str] = []
    haystack = _as_text(
        data.get("observation"),
        data.get("interpretation"),
        data.get("hypothesis_status"),
        data.get("next_research_priority"),
        json.dumps(data.get("alternative_explanations") or [], ensure_ascii=False),
    )
    if not _has_banned_effectiveness_claim(haystack):
        return warnings

    warnings.append("effectiveness_phrasing_coerced")
    data["interpretation"] = _SAFE_INTERPRETATION_BY_DECISION.get(
        locked,
        _SAFE_INTERPRETATION_BY_DECISION["KEEP"],
    )
    cleaned_alts: list[str] = []
    for item in data.get("alternative_explanations") or []:
        text = str(item).strip()
        if text and not _has_banned_effectiveness_claim(text):
            cleaned_alts.append(text)
    if not cleaned_alts:
        cleaned_alts = [
            "Single-seed or probe-budget variance can move the primary metric.",
            "Baseline / control labeling can dominate a one-shot delta.",
        ]
    data["alternative_explanations"] = cleaned_alts

    if _has_banned_effectiveness_claim(str(data.get("observation") or "")):
        data["observation"] = (
            f"VALID evidence reviewed under locked DecisionRubric action {locked}."
        )
        warnings.append("observation_scrubbed")
    if _has_banned_effectiveness_claim(str(data.get("next_research_priority") or "")):
        data["next_research_priority"] = (
            "Continue under protocol with the locked DecisionRubric action; "
            "do not promote it into a ClaimGate module-efficacy verdict."
        )
        warnings.append("next_research_priority_scrubbed")
    return warnings


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

# Live M1: LLMs often emit budget/replication labels under KEEP. Rubric already
# locked KEEP (= next-round, not a claim). Coerce instead of fail_closed.
KEEP_STATUS_COERCE = frozenset(
    {
        "inconclusive_budget",
        "needs_replication",
        "needs_validation",
    }
)

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
        "claim_stance": {
            "type": "string",
            "enum": sorted(ALLOWED_CLAIM_STANCE),
            "description": (
                "Structured KEEP≠Claim / DISCARD≠module-ineffective stance. "
                "KEEP/DISCARD → no_module_efficacy_claim; "
                "REPLICATE/VALIDATE → deferred_to_claim_gate."
            ),
        },
    },
}

REVIEWER_SYSTEM_PROMPT = """You are the Reviewer cognitive backend for Scientist Lab.
You are NOT a fifth Agent. Roles: Manager, Planner, Executor, Reviewer.
Gateway is a backend, not an Agent. Adapter=HOW; you do not emit HOW.

Return JSON only matching the provided schema. This is a PROPOSAL, not a decision.
The root MUST contain: observation, hypothesis_status, interpretation,
alternative_explanations, next_research_priority, evidence_refs, created_from,
confidence. Also emit claim_stance (preferred over efficacy prose).
Do NOT emit a Planner `selected` object. Do NOT emit review_decision.

Example (KEEP; shape only — copy the keys, not invented metrics):
{"observation":"VALID evidence for run_x: primary metric judged KEEP by DecisionRubric.","hypothesis_status":"not_a_claim","claim_stance":"no_module_efficacy_claim","interpretation":"DecisionRubric already locked KEEP. KEEP is a next-round action, not ClaimGate SUPPORTED, and a single-seed delta is not a G2 success claim.","alternative_explanations":["A tiny val slice can move the primary metric without a second seed.","Training noise can dominate one formal run."],"next_research_priority":"Replicate the same protocol on another seed before any G2 claim.","evidence_refs":[{"run_id":"run_x","metric":"APS_lowlight"}],"created_from":["run_x"],"confidence":"medium"}

Example (DISCARD; shape only):
{"observation":"VALID evidence for run_x: primary metric judged DISCARD by DecisionRubric.","hypothesis_status":"not_supported_under_current_protocol","claim_stance":"no_module_efficacy_claim","interpretation":"DecisionRubric already locked DISCARD. This is a next-action DISCARD, not a ClaimGate verdict that a module is ineffective.","alternative_explanations":["Probe budget can yield a large delta without a formal matched pair.","The labeled control may dominate the delta."],"next_research_priority":"Verify an allowed Adapter HOW that is not the discarded scope, still under the current budget.","evidence_refs":[{"run_id":"run_x","metric":"APS"}],"created_from":["run_x"],"confidence":"medium"}

Hard rules:
- DecisionRubric already locked review_decision (KEEP/DISCARD/REPLICATE/VALIDATE).
  Do not output review_decision, do not override it, do not KEEP/DISCARD yourself.
- ClaimGate remains the claim gate. DISCARD is not a formal claim that a module
  is ineffective. KEEP is not ClaimGate SUPPORTED.
- claim_stance is required intent: KEEP/DISCARD → no_module_efficacy_claim;
  REPLICATE/VALIDATE → deferred_to_claim_gate. Prefer this structured field
  over efficacy prose. Residual "effective module" wording is soft-coerced,
  not a campaign fail.
- Wording: avoid the phrases "effective module" / "module is effective" even inside
  hedges. Prefer: "KEEP is next-round only; not ClaimGate SUPPORTED" /
  "DISCARD is next-action only; not a ClaimGate module verdict".
- Do not write Memory. Do not invent lesson/strategy ids. MemoryWriter may persist
  this proposal later only if evidence_refs are valid.
- hypothesis_status must be one of:
  not_supported_under_current_protocol, inconclusive_budget, not_a_claim,
  needs_replication, needs_validation.
  DISCARD → not_supported_under_current_protocol or inconclusive_budget.
  KEEP → not_a_claim (KEEP ≠ claim supported). Do not copy SUPPORTED/REJECTED.
  Under KEEP, do not emit inconclusive_budget / needs_replication /
  needs_validation; those will be coerced to not_a_claim if you slip.
- Primary metric is Protocol objective.primary. Do not impersonate APS with mAP.
- Cite only the provided run_id in evidence_refs and created_from.
  Copy that run_id EXACTLY. Do not truncate it. Do not shorten the
  trailing hex like a git SHA (the full suffix is part of the contract).
- Explain WHY the Rubric decision happened, alternative explanations, and WHAT
  the next research priority should verify. You may cite HOW names already in
  the executed plan as observation (including N1/A4/FDPN when that plan ran
  them). Do not propose a new fusion_method, Python, or invented operator.
- Do not invent operators. Do not introduce FDPN unless the executed plan
  already used N1, A4, or neck/fusion type fdpn.
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
        "copy_this_run_id_exactly": payload.run_id,
        "primary_metric": payload.primary_metric,
        "primary_delta": payload.primary_delta,
        "budget_class": payload.budget_class,
        "constraints": {
            "forbid_review_override": True,
            "forbid_claim_gate_override": True,
            "forbid_memory_write": True,
            "forbid_fdpn_invention": True,
            "allow_cite_executed_fdpn": True,
            "forbid_how": True,
            "aps_is_not_map": payload.primary_metric in APS_KEYS,
            "discard_is_not_module_ineffective": True,
            "keep_is_not_claim_supported": True,
            "output_root_must_not_contain": ["selected", "review_decision", "claim_gate"],
        },
        "output_schema": REVIEWER_CONTRACT_SCHEMA,
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


# Nested campaign ids grow as run_plan_roundN_from_<parent>. Live models
# often clip the trailing exec hex (git-short-SHA habit). 32 chars is past
# "run_plan_round3_from" and still a unique prefix of the contract run_id.
_RUN_ID_PREFIX_MIN = 32


def canonicalize_cited_run_id(cited: str, expected: str) -> str:
    """Map an unambiguous truncation of the contract run_id back onto it.

    Does not invent a different run. A cited token that is not a long
    prefix of ``expected`` is left unchanged so the later equality check
    still fail-closes.
    """
    token = str(cited or "").strip()
    want = str(expected or "").strip()
    if not token or not want or token == want:
        return token
    if len(token) >= _RUN_ID_PREFIX_MIN and want.startswith(token):
        return want
    return token


def normalize_evidence_refs(
    refs: Sequence[Any],
    *,
    expected_run_id: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    expected = str(expected_run_id or "").strip()
    for item in refs:
        if isinstance(item, str) and item.strip():
            out.append(
                {"run_id": canonicalize_cited_run_id(item.strip(), expected)}
            )
            continue
        if isinstance(item, Mapping) and item.get("run_id"):
            row: dict[str, Any] = {
                "run_id": canonicalize_cited_run_id(str(item["run_id"]), expected)
            }
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
    refs = normalize_evidence_refs(
        list(proposal.get("evidence_refs") or []),
        expected_run_id=run_id,
    )
    if not refs:
        _refuse("semantic proposal refused: missing evidence_refs")
    created = [
        canonicalize_cited_run_id(str(x), run_id)
        for x in (proposal.get("created_from") or [])
        if str(x).strip()
    ]
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
    if _has_banned_effectiveness_claim(haystack):
        stance = str(proposal.get("claim_stance") or "").strip()
        if stance not in ALLOWED_CLAIM_STANCE:
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
    if isinstance(data.get("selected"), dict):
        _refuse("fail_closed: Reviewer must not emit Planner selected")

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
    # KEEP is next-round, not a claim. Soft-coerce common LLM slips.
    if locked == "KEEP" and status in KEEP_STATUS_COERCE:
        status = "not_a_claim"
        data["hypothesis_status"] = status
    allowed = STATUS_BY_DECISION.get(locked, ALLOWED_HYPOTHESIS_STATUS)
    if status not in allowed:
        _refuse(
            f"fail_closed: hypothesis_status={status!r} does not align with "
            f"locked review_decision={locked}"
        )

    refs = normalize_evidence_refs(
        list(data.get("evidence_refs") or []),
        expected_run_id=payload.run_id,
    )
    created = [
        canonicalize_cited_run_id(str(x), payload.run_id)
        for x in (data.get("created_from") or [])
        if str(x).strip()
    ]
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
    executed_how = _as_text(
        json.dumps(payload.plan, ensure_ascii=False),
        json.dumps(payload.result, ensure_ascii=False),
        json.dumps(payload.protocol, ensure_ascii=False),
    ).lower()
    if _BANNED_INVENT.search(haystack):
        _refuse("fail_closed: invented operator / FDPN / write-Python is forbidden")
    # Live M1: citing executed N1/A4/FDPN is observation, not invention.
    if _BANNED_FDPN.search(haystack) and not _EXECUTED_FDPN_HOW.search(executed_how):
        _refuse("fail_closed: invented operator / FDPN / write-Python is forbidden")
    contract_warnings = _coerce_effectiveness_prose(data, locked=locked)
    # Recompute haystack after soft-coerce so HOW checks see scrubbed prose.
    haystack = _as_text(
        data.get("observation"),
        data.get("interpretation"),
        data.get("hypothesis_status"),
        data.get("next_research_priority"),
        json.dumps(data.get("alternative_explanations") or [], ensure_ascii=False),
    )
    claim_stance = str(data.get("claim_stance") or "").strip()
    if claim_stance not in ALLOWED_CLAIM_STANCE:
        if claim_stance:
            contract_warnings.append("claim_stance_coerced_to_default")
        claim_stance = default_claim_stance(locked)
    if locked in {"KEEP", "DISCARD"} and claim_stance != "no_module_efficacy_claim":
        claim_stance = "no_module_efficacy_claim"
        contract_warnings.append("claim_stance_forced_no_module_efficacy_claim")
    proposes_new_how = bool(
        re.search(
            r"(set|use|switch(?:\s+to)?|change(?:\s+to)?|emit)\s+fusion_method\b",
            haystack,
            re.IGNORECASE,
        )
    )
    # Mentioning fusion_method=<executed token> in next_research_priority is OK
    # (replicate same HOW). Only refuse unknown / novel fusion_method= values.
    for match in re.finditer(
        r"\bfusion_method\s*=\s*([A-Za-z0-9_:.\-]+)",
        haystack,
        re.IGNORECASE,
    ):
        token = str(match.group(1) or "").strip().lower()
        if not token:
            continue
        if token in executed_how:
            continue
        if token.startswith("plugin:") and token in executed_how:
            continue
        bare = token.split(":", 1)[-1]
        if bare and (bare in executed_how or f"plugin:{bare}" in executed_how):
            continue
        proposes_new_how = True
        break
    cites_unknown_how = bool(
        re.search(r"\bearly_concat\b|\bfusion_method\b", haystack, re.IGNORECASE)
        and "early_concat" not in executed_how
        and "fusion_method" not in executed_how
        and "fusion_type" not in executed_how
    )
    if proposes_new_how or cites_unknown_how:
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
        "claim_stance": claim_stance,
        "contract_warnings": contract_warnings,
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
        "lesson_id": lesson_id_for_run(run_id, semantic=True),
        "type": lesson_type,
        "statement": statement,
        "status": "active",
        "evidence": evidence,
        "scope": {"task": task, "module": module},
        "confidence": confidence,
        "created_from": [
            canonicalize_cited_run_id(str(x), run_id)
            for x in (proposal.get("created_from") or [run_id])
            if str(x).strip()
        ]
        or [run_id],
        "contradicted_by": [],
        "supersedes": [],
        "expires_when": [],
    }


def redacted_raw(response: LLMResponse | None) -> str:
    if response is None:
        return ""
    return redact_secrets(response.content or "")
