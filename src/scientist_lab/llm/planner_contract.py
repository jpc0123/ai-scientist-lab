"""LLM Planner contract: Evidence → structured WHAT/WHY → experiment_plan fields.

Gateway is not a fifth Agent. Planner=WHAT/WHY; Adapter=HOW.
Fail closed on bad JSON, scope violations, invented operators (FDPN),
formal promotion, Memory writes, or KEEP/DISCARD overrides.

v2.5-B: historical REPLAY exam (this module + plan_replay --ab).
v2.5-C: same Gateway with a Reviewer role prompt (`reviewer_contract`).
DecisionRubric still emits KEEP/DISCARD/REPLICATE; LLM must not overwrite it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from scientist_lab.adapters.dfine.how import HOW_MODULES, list_adapter_capabilities
from scientist_lab.llm.config import redact_secrets
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema


class PlannerContractError(ValueError):
    """LLM output cannot be mapped into a Gate-legal ExperimentPlan."""


# Budget rank: LLM may keep or lower, never raise to skip Human Gate.
_BUDGET_RANK = {"probe": 0, "validation": 1, "formal": 2}

_BANNED_OPERATOR = re.compile(
    r"\b(fdpn|new[-_ ]network|write(?:\s+some)?\s+python|implement(?:\s+a)?\s+new"
    r"|custom[-_ ]operator|novel[-_ ]fusion)\b",
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
    }
)

PLANNER_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["selected"],
    "additionalProperties": True,
    "properties": {
        "selected": {
            "type": "object",
            "required": [
                "requested_module",
                "hypothesis",
                "proposed_changes",
            ],
            "properties": {
                "candidate_id": {"type": "string"},
                "requested_module": {"type": "string"},
                "observation": {"type": "string"},
                "hypothesis": {"type": "string"},
                "proposed_changes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["target", "summary"],
                        "properties": {
                            "target": {"type": "string"},
                            "summary": {"type": "string"},
                            "detail": {"type": "object"},
                        },
                    },
                },
                "expected_effect": {"type": "object"},
                "budget_class": {"type": "string"},
                "selected_action": {"type": "string"},
                "rationale": {"type": "string"},
            },
        },
        "candidates": {"type": "array"},
        "memory_refs": {"type": "object"},
        "invented_operators": {"type": "array"},
    },
}

PLANNER_SYSTEM_PROMPT = """You are the Planner cognitive backend for Scientist Lab.
You are NOT a fifth Agent. Roles: Planner=WHAT/WHY, Adapter=HOW, Gate=approve/reject.
Return JSON only matching the provided schema.

Hard rules:
- requested_module must be in protocol.editable_scope AND have existing Adapter HOW
  (neck → rgb + fusion_method=none; fusion → rgbt + early_concat).
- Do not invent operators, FDPN, new networks, or ask to write Python.
- Do not raise budget_class to formal (Human Gate owns formal).
- Do not write Memory, invent lesson/strategy ids, or change KEEP/DISCARD/REPLICATE.
- Hypothesis is a mechanism guess about an already-allowed module, not a new architecture.
- You may propose multiple candidates; only `selected` will be sent to Gate.
- After DISCARD / negative_evidence, do NOT select the discarded modification_scope
  again. You MAY list it in candidates with reason_not_selected.
- After DISCARD, return at least two candidates besides selected.
- Cite only memory_refs provided in the user payload.
"""


def infer_discarded_modules(
    *,
    previous_plan: Mapping[str, Any],
    last_review_decision: str | None,
    memory_catalog: Mapping[str, Any] | None = None,
) -> list[str]:
    """Modules with DISCARD / negative_evidence / deprioritize. Selected must not repeat them."""
    found: list[str] = []
    if str(last_review_decision or "") == "DISCARD":
        found.extend(str(tok) for tok in (previous_plan.get("modification_scope") or []))
    catalog = dict(memory_catalog or {})
    lessons = catalog.get("lessons") or {}
    if isinstance(lessons, Mapping):
        for row in lessons.values():
            if not isinstance(row, Mapping):
                continue
            if str(row.get("type") or "") not in {"negative_evidence", "constraint_violation"}:
                continue
            module = str((row.get("scope") or {}).get("module") or "").strip()
            if module:
                found.append(module)
    strategies = catalog.get("strategies") or {}
    if isinstance(strategies, Mapping):
        for row in strategies.values():
            if not isinstance(row, Mapping):
                continue
            if str(row.get("action") or "") != "deprioritize":
                continue
            target = str(row.get("target") or "").strip()
            if target:
                found.append(target)
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in found:
        if tok and tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered


@dataclass
class PlannerContractInput:
    goal: Mapping[str, Any]
    protocol: Mapping[str, Any]
    previous_plan: Mapping[str, Any]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    rubric: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    memory_refs: Mapping[str, Sequence[str]] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    adapter_capabilities: Sequence[Mapping[str, Any]] = field(default_factory=list)
    last_review_decision: str | None = None
    parent_run_id: str = ""
    observation: str | None = None
    discarded_modules: Sequence[str] = field(default_factory=list)


def prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    blob = json.dumps(list(messages), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_contract_input(
    *,
    protocol: Mapping[str, Any],
    previous_plan: Mapping[str, Any],
    memory_refs: Mapping[str, Sequence[str]],
    memory_catalog: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    last_review_decision: str | None = None,
    parent_run_id: str = "",
    observation: str | None = None,
) -> PlannerContractInput:
    budget_class = str(previous_plan.get("budget_class") or "probe")
    catalog = dict(memory_catalog or {})
    discarded = infer_discarded_modules(
        previous_plan=previous_plan,
        last_review_decision=last_review_decision,
        memory_catalog=catalog,
    )
    return PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=dict(protocol),
        previous_plan=dict(previous_plan),
        evidence=dict(evidence or {}),
        rubric=dict(protocol.get("decision_policy") or {}),
        memory=catalog,
        memory_refs={
            "lesson_ids": [str(i) for i in (memory_refs.get("lesson_ids") or [])],
            "strategy_ids": [str(i) for i in (memory_refs.get("strategy_ids") or [])],
        },
        budget={
            "budget_class": budget_class,
            "experiment_budget": dict(protocol.get("experiment_budget") or {}),
        },
        adapter_capabilities=list_adapter_capabilities(),
        last_review_decision=last_review_decision,
        parent_run_id=str(parent_run_id),
        observation=observation,
        discarded_modules=discarded,
    )


def build_planner_request(payload: PlannerContractInput) -> LLMRequest:
    user = {
        "goal": dict(payload.goal),
        "protocol": {
            "protocol_id": payload.protocol.get("protocol_id"),
            "protocol_version": payload.protocol.get("protocol_version"),
            "project_id": payload.protocol.get("project_id"),
            "editable_scope": list(payload.protocol.get("editable_scope") or []),
            "frozen_scope": list(payload.protocol.get("frozen_scope") or []),
            "objective": dict(payload.protocol.get("objective") or {}),
            "fingerprint_id": payload.protocol.get("fingerprint_id"),
        },
        "previous_plan": {
            "plan_id": payload.previous_plan.get("plan_id"),
            "round_index": payload.previous_plan.get("round_index"),
            "modification_scope": list(payload.previous_plan.get("modification_scope") or []),
            "proposed_changes": list(payload.previous_plan.get("proposed_changes") or []),
            "hypothesis": payload.previous_plan.get("hypothesis"),
            "budget_class": payload.previous_plan.get("budget_class"),
            "risk_level": payload.previous_plan.get("risk_level"),
            "evaluation": dict(payload.previous_plan.get("evaluation") or {}),
        },
        "evidence": dict(payload.evidence),
        "rubric": dict(payload.rubric),
        "memory": dict(payload.memory),
        "memory_refs": {
            "lesson_ids": list(payload.memory_refs.get("lesson_ids") or []),
            "strategy_ids": list(payload.memory_refs.get("strategy_ids") or []),
        },
        "budget": dict(payload.budget),
        "adapter_capabilities": [dict(row) for row in payload.adapter_capabilities],
        "last_review_decision": payload.last_review_decision,
        "parent_run_id": payload.parent_run_id,
        "observation": payload.observation,
        "discarded_modules": [str(x) for x in payload.discarded_modules],
        "constraints": {
            "how_modules": sorted(HOW_MODULES),
            "forbid_fdpn": True,
            "forbid_formal_promotion": True,
            "forbid_memory_write": True,
            "forbid_review_override": True,
            "forbid_repeat_discarded_selected": True,
        },
    }
    return LLMRequest(
        purpose="planner",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=PLANNER_CONTRACT_SCHEMA,
        temperature=0.0,
        metadata={
            "planner_contract": "experiment_plan",
            "project_id": str(payload.protocol.get("project_id") or ""),
            "parent_run_id": payload.parent_run_id,
        },
    )


def _as_text(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts)


def _refuse(message: str) -> None:
    raise PlannerContractError(message)


def parse_planner_completion(
    response: LLMResponse | str,
    payload: PlannerContractInput,
    *,
    known_lesson_ids: Sequence[str] | None = None,
    known_strategy_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse LLM JSON and map to experiment_plan overlays. Fail closed."""
    raw = response.content if isinstance(response, LLMResponse) else str(response)
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        _refuse(f"fail_closed: invalid JSON: {exc}")

    errors = validate_against_schema(data, PLANNER_CONTRACT_SCHEMA)
    if errors:
        _refuse("fail_closed: contract schema: " + "; ".join(errors))

    for key in _MEMORY_WRITE_KEYS:
        if data.get(key):
            _refuse("fail_closed: LLM must not write Memory")
    for key in _REVIEW_OVERRIDE_KEYS:
        if key in data and data.get(key) not in (None, "", False):
            _refuse("fail_closed: LLM must not override KEEP/DISCARD/ClaimGate")

    invented = [str(x) for x in (data.get("invented_operators") or []) if str(x).strip()]
    if invented:
        _refuse(f"fail_closed: invented operators not allowed: {invented}")

    selected = dict(data.get("selected") or {})
    module = str(selected.get("requested_module") or "").strip()
    editable = {str(tok) for tok in (payload.protocol.get("editable_scope") or [])}
    frozen = {str(tok) for tok in (payload.protocol.get("frozen_scope") or [])}
    if not module:
        _refuse("fail_closed: selected.requested_module missing")
    if module in frozen or module not in editable:
        _refuse(
            f"fail_closed: requested_module={module!r} outside editable_scope "
            f"(editable={sorted(editable)})"
        )
    if module not in HOW_MODULES:
        _refuse(
            f"fail_closed: Adapter has no HOW for module={module!r}; "
            f"allowed HOW modules={sorted(HOW_MODULES)}"
        )
    discarded = [str(x) for x in payload.discarded_modules]
    if not discarded:
        discarded = infer_discarded_modules(
            previous_plan=payload.previous_plan,
            last_review_decision=payload.last_review_decision,
            memory_catalog=payload.memory,
        )
    if module in discarded:
        _refuse(
            f"fail_closed: selected.requested_module={module!r} repeats DISCARD/"
            f"negative_evidence module {discarded}"
        )

    changes = [dict(row) for row in (selected.get("proposed_changes") or [])]
    if not changes or not all(row.get("target") and row.get("summary") for row in changes):
        _refuse("fail_closed: proposed_changes need target+summary")
    for row in changes:
        target = str(row.get("target") or "")
        if target != module:
            _refuse(
                f"fail_closed: proposed_changes.target={target!r} != requested_module={module!r}"
            )
        if target not in editable or target in frozen:
            _refuse(f"fail_closed: proposed_changes.target={target!r} not editable")

    haystack = _as_text(
        module,
        selected.get("hypothesis"),
        selected.get("observation"),
        selected.get("rationale"),
        json.dumps(changes, ensure_ascii=False),
        json.dumps(data.get("candidates") or [], ensure_ascii=False),
    )
    if _BANNED_OPERATOR.search(haystack):
        _refuse("fail_closed: invented operator / FDPN / write-Python is forbidden")

    prev_budget = str(
        (payload.budget or {}).get("budget_class")
        or payload.previous_plan.get("budget_class")
        or "probe"
    )
    llm_budget = str(selected.get("budget_class") or prev_budget).strip().lower()
    if llm_budget not in _BUDGET_RANK:
        _refuse(f"fail_closed: unknown budget_class={llm_budget!r}")
    if _BUDGET_RANK[llm_budget] > _BUDGET_RANK.get(prev_budget, 0):
        _refuse(
            f"fail_closed: LLM cannot promote budget_class {prev_budget!r} → {llm_budget!r}"
        )

    known_lessons = set(known_lesson_ids or [])
    known_strategies = set(known_strategy_ids or [])
    cited = data.get("memory_refs") or selected.get("memory_refs") or {}
    cited_lessons = [str(i) for i in (cited.get("lesson_ids") or [])]
    cited_strategies = [str(i) for i in (cited.get("strategy_ids") or [])]
    if cited_lessons or cited_strategies:
        missing_l = [i for i in cited_lessons if i not in known_lessons]
        missing_s = [i for i in cited_strategies if i not in known_strategies]
        if missing_l or missing_s:
            _refuse(
                "fail_closed: LLM invented unresolved memory_refs: "
                f"lessons={missing_l} strategies={missing_s}"
            )
        refs = {"lesson_ids": cited_lessons, "strategy_ids": cited_strategies}
    else:
        refs = {
            "lesson_ids": [str(i) for i in (payload.memory_refs.get("lesson_ids") or [])],
            "strategy_ids": [str(i) for i in (payload.memory_refs.get("strategy_ids") or [])],
        }

    metric = str(
        ((payload.protocol.get("objective") or {}).get("primary") or {}).get("metric")
        or "APS"
    )
    direction = "increase"
    if str(((payload.protocol.get("objective") or {}).get("primary") or {}).get("direction") or "") == "minimize":
        direction = "decrease"
    expected = dict(selected.get("expected_effect") or {})
    expected.setdefault("primary_metric", metric)
    expected.setdefault("direction", direction)
    expected.setdefault(
        "rationale",
        f"LLM Planner mechanism guess on allowed module {module}; Adapter remains HOW.",
    )

    observation = str(
        selected.get("observation")
        or payload.observation
        or payload.previous_plan.get("observation")
        or "LLM Planner citing written memory."
    ).strip()
    hypothesis = str(selected.get("hypothesis") or "").strip()
    if not hypothesis:
        _refuse("fail_closed: hypothesis missing")

    candidates = [dict(row) for row in (data.get("candidates") or []) if isinstance(row, Mapping)]
    selected_id = str(selected.get("candidate_id") or "selected")
    attachments = [row for row in candidates if str(row.get("candidate_id") or "") != selected_id]
    if not attachments and candidates:
        attachments = [row for row in candidates if str(row.get("requested_module") or "") != module]
    if str(payload.last_review_decision or "") == "DISCARD" and len(attachments) < 2:
        _refuse(
            "fail_closed: after DISCARD, candidate_experiments must have >= 2 "
            "non-selected candidates (with reason_not_selected for discarded modules)"
        )

    mapped = {
        "modification_scope": [module],
        "proposed_changes": changes,
        "hypothesis": hypothesis,
        "observation": observation,
        "expected_effect": expected,
        "budget_class": llm_budget,
        "selected_action": str(selected.get("selected_action") or f"probe_{module}"),
        "memory_refs": refs,
        "candidate_experiments": attachments,
        "selected_candidate_id": selected_id,
        "parsed": data,
        "raw_output": redact_secrets(raw),
    }
    return mapped
