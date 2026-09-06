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
_ALLOWED_EFFECT_DIRECTIONS = frozenset({"increase", "decrease", "stabilize", "unclear"})


def coerce_expected_direction(value: Any, default: str = "increase") -> str:
    """Clamp LLM direction tokens onto experiment_plan.enum. Does not invent HOW."""
    fallback = default if default in _ALLOWED_EFFECT_DIRECTIONS else "unclear"
    token = str(value or "").strip().lower()
    if token in _ALLOWED_EFFECT_DIRECTIONS:
        return token
    if token.startswith("decrease") or any(part in token for part in ("worsen", "drop", "lower")):
        return "decrease"
    if token.startswith("increase") or any(part in token for part in ("improv", "raise", "higher")):
        return "increase"
    if token.startswith("stabil") or token in {"maintain", "unchanged", "same"}:
        return "stabilize"
    return "unclear" if fallback != "unclear" else fallback


# experiment_plan.schema.json expected_effect.additionalProperties=false
_EXPECTED_EFFECT_KEYS = frozenset({"primary_metric", "direction", "rationale"})


def sanitize_expected_effect(
    value: Mapping[str, Any] | None,
    *,
    default_metric: str = "APS",
    default_direction: str = "increase",
) -> dict[str, Any]:
    """Keep only schema-legal expected_effect keys. Strip LLM extras (e.g. reference_last)."""
    raw = dict(value or {})
    direction = coerce_expected_direction(
        raw.get("direction") or default_direction, default_direction
    )
    out: dict[str, Any] = {
        "primary_metric": str(raw.get("primary_metric") or default_metric),
        "direction": direction,
    }
    rationale = raw.get("rationale")
    if rationale is not None and str(rationale).strip():
        out["rationale"] = str(rationale)
    # Drop unknown keys (reference_last, reference_last_delta, …) — schema refuse.
    for key, val in raw.items():
        if key in _EXPECTED_EFFECT_KEYS and key not in out and val is not None:
            out[key] = val
    return {k: out[k] for k in out if k in _EXPECTED_EFFECT_KEYS}


def _coerce_string_list(value: Any) -> list[str] | None:
    """Normalize LLM list-or-string slips into a string list. None = leave unset."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return [str(value)]


def _merge_coverage_lists(existing: Any, extra: Any) -> list[str]:
    base = _coerce_string_list(existing) or []
    add = _coerce_string_list(extra) or []
    out = list(base)
    for item in add:
        if item not in out:
            out.append(item)
    return out


def _normalize_verification_plan(vp: Mapping[str, Any] | None) -> dict[str, Any]:
    """Type-normalize verification_plan fields before schema validate."""
    out = dict(vp or {})
    # LLM alias: addresses_question → addresses_coverage (schema has no extra keys).
    for alias in ("addresses_question", "address_question", "open_question"):
        if alias not in out:
            continue
        alias_val = out.pop(alias)
        out["addresses_coverage"] = _merge_coverage_lists(
            out.get("addresses_coverage"), alias_val
        )
    for key in (
        "scientific_question",
        "what_to_run",
        "how_to_verify",
        "success_criterion",
        "falsified_if",
    ):
        if key in out and out[key] is not None and not isinstance(out[key], str):
            out[key] = str(out[key])
    for key in ("controlled_variables", "addresses_coverage"):
        if key not in out:
            continue
        coerced = _coerce_string_list(out.get(key))
        if coerced is not None:
            out[key] = coerced
    return out


def coerce_planner_json_object(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Best-effort alias normalize before schema validate. Does not invent science."""
    out: dict[str, Any] = dict(data or {})
    # Repair meta-wrapper sometimes comes back as the model content.
    if not isinstance(out.get("selected"), dict) and isinstance(out.get("invalid_output"), str):
        try:
            nested = extract_json_object(str(out.get("invalid_output") or ""))
        except (ValueError, json.JSONDecodeError):
            nested = None
        if isinstance(nested, dict) and (
            isinstance(nested.get("selected"), dict)
            or any(isinstance(nested.get(k), dict) for k in ("plan", "chosen", "decision"))
        ):
            out = dict(nested)
    if not isinstance(out.get("selected"), dict):
        for alt in ("plan", "chosen", "decision", "selected_plan", "choice"):
            blob = out.get(alt)
            if isinstance(blob, dict):
                out["selected"] = dict(blob)
                break
    selected = dict(out.get("selected") or {})
    if not str(selected.get("requested_module") or "").strip():
        for alt in ("module", "target_module", "editable_module"):
            if str(selected.get(alt) or "").strip():
                selected["requested_module"] = str(selected.get(alt)).strip()
                break
        scope = selected.get("modification_scope") or selected.get("editable_scope")
        if not selected.get("requested_module"):
            if isinstance(scope, list) and scope:
                selected["requested_module"] = str(scope[0]).strip()
            elif isinstance(scope, str) and scope.strip():
                selected["requested_module"] = scope.strip()
    if not isinstance(selected.get("proposed_changes"), list):
        for alt in ("changes", "modifications", "edits", "actions", "patches"):
            if isinstance(selected.get(alt), list):
                selected["proposed_changes"] = list(selected.get(alt) or [])
                break
    changes: list[dict[str, Any]] = []
    for row in selected.get("proposed_changes") or []:
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        if not str(item.get("target") or "").strip():
            for alt in ("module", "scope", "requested_module"):
                if str(item.get(alt) or "").strip():
                    item["target"] = str(item.get(alt)).strip()
                    break
        if not str(item.get("target") or "").strip() and selected.get("requested_module"):
            item["target"] = str(selected["requested_module"]).strip()
        if not str(item.get("summary") or "").strip():
            for alt in ("description", "change", "action", "text", "rationale", "detail_summary"):
                if str(item.get(alt) or "").strip():
                    item["summary"] = str(item.get(alt)).strip()
                    break
        if not str(item.get("summary") or "").strip():
            how = str(item.get("how_id") or selected.get("how_id") or "").strip()
            if how:
                item["summary"] = f"Run registered HOW {how}."
        changes.append(item)
    if changes:
        selected["proposed_changes"] = changes
    if not str(selected.get("hypothesis") or "").strip():
        for alt in ("hypothesis_text", "claim", "idea", "rationale"):
            if str(selected.get(alt) or "").strip():
                selected["hypothesis"] = str(selected.get(alt)).strip()
                break
    if not str(selected.get("hypothesis") or "").strip():
        how = str(selected.get("how_id") or "").strip()
        module = str(selected.get("requested_module") or "module").strip()
        selected["hypothesis"] = (
            f"Evaluate {how or module} under the current protocol without raising ClaimGate."
        )
    if "seed" in selected and selected.get("seed") is not None and not isinstance(
        selected.get("seed"), bool
    ):
        seed_val = selected.get("seed")
        if isinstance(seed_val, str) and seed_val.strip().lstrip("-").isdigit():
            selected["seed"] = int(seed_val.strip())
        elif isinstance(seed_val, float) and seed_val.is_integer():
            selected["seed"] = int(seed_val)
    if isinstance(selected.get("verification_plan"), Mapping):
        selected["verification_plan"] = _normalize_verification_plan(
            selected.get("verification_plan")
        )
    out["selected"] = selected
    if isinstance(out.get("verification_plan"), Mapping):
        out["verification_plan"] = _normalize_verification_plan(out.get("verification_plan"))
    if "candidates" not in out and isinstance(out.get("alternatives"), list):
        out["candidates"] = list(out.get("alternatives") or [])
    if "invented_operators" not in out:
        out["invented_operators"] = list(out.get("invented_ops") or [])
    if "memory_refs" not in out and isinstance(out.get("memory_references"), dict):
        out["memory_refs"] = dict(out.get("memory_references") or {})
    return out


_BANNED_OPERATOR = re.compile(
    r"\b(new[-_ ]network|write(?:\s+some)?\s+python|implement(?:\s+a)?\s+new"
    r"|custom[-_ ]operator|novel[-_ ]fusion)\b",
    re.IGNORECASE,
)

# Avoid matching ordinary English "from …" / "class of".
_HOW_CANDIDATE_CODE = re.compile(
    r"(```|write python|\bdef\s+\w|\bclass\s+\w+\s*[:\(]|\bimport\s+\w|"
    r"\bfrom\s+\w+\s+import\b|new network|\bfdpn\b)",
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
        "verification_plan": {
            "type": "object",
            "required": ["what_to_run", "how_to_verify", "success_criterion"],
            "properties": {
                "scientific_question": {"type": "string"},
                "what_to_run": {"type": "string"},
                "how_to_verify": {"type": "string"},
                "controlled_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "success_criterion": {"type": "string"},
                "falsified_if": {"type": "string"},
                "addresses_coverage": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
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
                "how_id": {"type": "string"},
                "seed": {"type": "integer"},
                "expected_effect": {"type": "object"},
                "budget_class": {"type": "string"},
                "selected_action": {"type": "string"},
                "rationale": {"type": "string"},
            },
        },
        "candidates": {"type": "array"},
        "memory_refs": {"type": "object"},
        "invented_operators": {"type": "array"},
        "how_candidates": {"type": "array"},
    },
}

def _planner_visible_ids(
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    from scientist_lab.adapters.dfine.how_catalog import planner_visible_how_ids

    return sorted(planner_visible_how_ids(overlay))


def _not_registered_ids(
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    from scientist_lab.adapters.dfine.how_catalog import NOT_REGISTERED

    extra = {str(k).strip().upper() for k in dict(overlay or {})}
    return sorted(token for token in NOT_REGISTERED if token not in extra)


def _family_for_how(how_id: str, module: str) -> str:
    token = str(how_id or "").strip().upper()
    if token.startswith("T"):
        return "training"
    if token.startswith("N"):
        return "neck"
    if str(module or "") in {"fusion", "neck", "training"}:
        return str(module)
    return "fusion"


def planner_system_prompt(
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    visible = ", ".join(_planner_visible_ids(overlay))
    blocked = ", ".join(_not_registered_ids(overlay))
    return f"""You are the Planner cognitive backend for Scientist Lab.
You are NOT a fifth Agent. Roles: Planner=WHAT/WHY, Adapter=HOW, Gate=approve/reject.
Return JSON only matching the provided schema. The root MUST contain `selected`.

Example (shape only):
{{"selected":{{"requested_module":"fusion","how_id":"F3","hypothesis":"...","proposed_changes":[{{"target":"fusion","summary":"Use registered HOW F3.","detail":{{"how_id":"F3"}}}}],"expected_effect":{{"primary_metric":"APS_lowlight","direction":"increase"}},"budget_class":"formal"}},"candidates":[{{"candidate_id":"cand_not_selected","requested_module":"fusion","how_id":"F0","reason_not_selected":"RGB-only control not selected this round."}}],"invented_operators":[],"memory_refs":{{"lesson_ids":[],"strategy_ids":[]}}}}

Hard rules:
- requested_module must be in protocol.editable_scope AND have existing Adapter HOW.
- Set selected.how_id to a materializable catalog id when you intend the Adapter to run:
  {visible}.
  F0 = RGB-only (none), F1 = early_concat, F3 = gated_multiscale,
  N0 = standard neck, N1 = existing fdpn neck, A4 = existing early_concat + fdpn.
- Do NOT rewrite selected.how_id to a different catalog id. If you pick N1, it stays N1.
- {blocked} are not materializable unless this campaign already registered them
  as overlay plugins (then they appear in the materializable list above).
  You MAY put unregistered ids in how_candidates[] as drafts.
  If you put an unregistered id in selected.how_id, it becomes a pending draft;
  the campaign LLM HOW lifecycle may author/accept overlay code. GPU still
  waits until the HOW is materializable. You do not write Python in this JSON.
- Do not invent operators, new networks, or ask to write Python.
- Do not raise budget_class above the previous plan. If previous is already formal, keep formal.
- Do not write Memory, invent lesson/strategy ids, or change KEEP/DISCARD/REPLICATE.
- Hypothesis is a mechanism guess about an already-allowed module, not a new architecture.
- You may propose multiple candidates; only `selected` will be sent to Gate.
- After DISCARD / negative_evidence, do NOT select the discarded modification_scope
  again. You MAY list it in candidates with reason_not_selected.
- After DISCARD, return at least two candidates besides selected.
- Cite only memory_refs provided in the user payload. LiteratureEvidence is not ExperimentEvidence.
- literature in the user payload is a ranked table (rank/title/year/url/why_relevant). Use only rows with url. Do not invent a paper that is not in the table. It is NOT Claim evidence and MUST NOT enter ClaimGate. Do not treat a paper as proving a hypothesis. KEEP is not a Claim.
- Do not invent an unregistered HOW because a paper used it. If you want to follow a paper, either change scout (out of band) or emit how_candidates[] drafts with paper_refs copied from the payload, or select a materializable registered HOW. Unregistered selected.how_id becomes a pending draft and MUST NOT run GPU.
- Use evidence.last_metrics / last_primary_delta as the previous-shot numbers. Do not invent Δ.
- After REPLICATE of the same HOW, set selected.seed to last_seed+1 so evaluation.seeds actually changes. Repeating seed 42 is not a replication.
- You choose the next materializable HOW. No human will pick it for you. Same HOW + same seed is forbidden.
- If evidence.last_primary_delta is negative, do not repeat that how_id unless you are explicitly replicating on a new seed.
- You MAY emit how_candidates[] as drafts of new methods found in literature. Those are NOT executable HOW. Copy literature.provenance.literature_query_id and paper_ids from the user payload. invented_operators must stay []. Do not write Python in this JSON. Optional implementation_intent is natural-language only; Python for a fusion plugin is authored later via the how_plugins Diff path.
- experiment_brief (when present) lists coverage gaps and open_scientific_questions. These are ADVISORY for a professional write-up — NOT a fixed run script. You MUST design the next experiment from evidence: state what you test, how you verify it, and what would falsify your hypothesis in verification_plan.
- human_steer (when present and planner_may_use=true) is a mid-campaign human direction for THIS next round only. Honor it in hypothesis / selected HOW choice when legal. It is NOT Protocol Amendment: never change frozen dataset, slice, primary metric, evaluator, or claim policy. If needs_protocol_amendment=true, refuse to rewrite frozen fields and stay on Adapter-legal HOW.
- idea_snapshot (when present) is the Idea Interview brief frozen at register. Use as provenance for the research question; do not treat it as ExperimentEvidence or a Claim.
- verification_plan is required every round: what_to_run, how_to_verify (metric, controls, seeds), success_criterion, optional falsified_if and addresses_coverage (which gap/question this round targets).
- Do NOT pick the next catalog HOW only because it appears in remaining_catalog_how_ids. Justify from hypothesis + last_metrics + review_decision.
- If experiment_brief.unused_smoked_plugins is non-empty, you SHOULD select one of those how_id values for selected.how_id (they are already materializable overlay plugins). Prefer them over repeating catalog F0/F1/F3/N0/N1/A4 when the catalog axis is already variance-dominated or Stage A is validating the plugin path. State in verification_plan that you are testing the overlay plugin.
"""


PLANNER_SYSTEM_PROMPT = planner_system_prompt()


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
    literature: Mapping[str, Any] = field(default_factory=dict)
    how_overlay: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    experiment_brief: Mapping[str, Any] = field(default_factory=dict)
    human_steer: Mapping[str, Any] = field(default_factory=dict)
    idea_snapshot: Mapping[str, Any] = field(default_factory=dict)


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
    literature: Mapping[str, Any] | None = None,
    how_overlay: Mapping[str, Mapping[str, Any]] | None = None,
    experiment_brief: Mapping[str, Any] | None = None,
    human_steer: Mapping[str, Any] | None = None,
    idea_snapshot: Mapping[str, Any] | None = None,
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
        literature=dict(literature or {}),
        how_overlay={
            str(k).upper(): dict(v)
            for k, v in dict(how_overlay or {}).items()
            if isinstance(v, Mapping)
        },
        experiment_brief=dict(experiment_brief or {}),
        human_steer=dict(human_steer or {}),
        idea_snapshot=dict(idea_snapshot or {}),
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
            "forbid_fdpn_invention": True,
            "forbid_formal_promotion": True,
            "forbid_memory_write": True,
            "forbid_review_override": True,
            "forbid_repeat_discarded_selected": True,
            "registered_how_ids": sorted(_planner_visible_ids(payload.how_overlay)),
            "hidden_how_ids": [],
            "not_registered_how_ids": sorted(_not_registered_ids(payload.how_overlay)),
            "keep_formal_if_previous_formal": True,
            "output_root_must_contain": ["selected"],
            "how_candidates_are_drafts_only": True,
            "selected_how_must_not_be_rewritten": True,
            "unmaterializable_selected_goes_pending": True,
        },
        "literature": dict(payload.literature or {}),
        "experiment_brief": dict(payload.experiment_brief or {}),
        "human_steer": dict(payload.human_steer or {}),
        "idea_snapshot": dict(payload.idea_snapshot or {}),
    }
    return LLMRequest(
        purpose="planner",
        messages=[
            {"role": "system", "content": planner_system_prompt(payload.how_overlay)},
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


# Nested campaign run_ids make LESSON-/STRATEGY- bodies huge; live models clip
# mid-token. Require enough shared prefix that "LESSON-" alone never matches.
_MEMORY_REF_PREFIX_MIN = 24


def _prefer_memory_ref_candidates(
    matches: Sequence[str],
    preferred: Sequence[str] | None,
) -> str | None:
    """Disambiguate non-unique prefix hits without inventing foreign IDs.

    Order: preferred catalog (payload memory_refs) → primary ``-001`` (not
    ``-semantic-001``) → shortest → lexicographic.
    """
    cands = [str(m) for m in matches if str(m).strip()]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    pref = [str(p) for p in (preferred or []) if str(p).strip()]
    if pref:
        ranked = [p for p in pref if p in set(cands)]
        if ranked:
            return ranked[0]
    primary = [
        c
        for c in cands
        if c.endswith("-001") and "-semantic-001" not in c
    ]
    if len(primary) == 1:
        return primary[0]
    if primary:
        return sorted(primary, key=lambda x: (len(x), x))[0]
    return sorted(cands, key=lambda x: (len(x), x))[0]


def _resolve_memory_ref_token(
    token: str,
    known: set[str],
    *,
    preferred: Sequence[str] | None = None,
) -> str | None:
    """Exact match, else unique/preferred prefix (LLM often clips long IDs)."""
    tid = str(token or "").strip()
    if not tid:
        return None
    if tid in known:
        return tid
    if len(tid) < _MEMORY_REF_PREFIX_MIN:
        return None
    prefix_matches = [kid for kid in known if kid.startswith(tid)]
    if not prefix_matches:
        return None
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return _prefer_memory_ref_candidates(prefix_matches, preferred)


def _resolve_cited_memory_refs(
    cited: Mapping[str, Any],
    *,
    known_lessons: set[str],
    known_strategies: set[str],
    fallback: Mapping[str, Sequence[str]],
) -> dict[str, list[str]] | None:
    cited_lessons = [str(i) for i in (cited.get("lesson_ids") or []) if str(i).strip()]
    cited_strategies = [str(i) for i in (cited.get("strategy_ids") or []) if str(i).strip()]
    fallback_lessons = [str(i) for i in (fallback.get("lesson_ids") or [])]
    fallback_strategies = [str(i) for i in (fallback.get("strategy_ids") or [])]
    if not cited_lessons and not cited_strategies:
        return {
            "lesson_ids": fallback_lessons,
            "strategy_ids": fallback_strategies,
        }
    missing_l: list[str] = []
    missing_s: list[str] = []
    resolved_lessons: list[str] = []
    resolved_strategies: list[str] = []
    for lid in cited_lessons:
        resolved = _resolve_memory_ref_token(
            lid, known_lessons, preferred=fallback_lessons
        )
        if resolved:
            resolved_lessons.append(resolved)
        else:
            missing_l.append(lid)
    for sid in cited_strategies:
        resolved = _resolve_memory_ref_token(
            sid, known_strategies, preferred=fallback_strategies
        )
        if resolved:
            resolved_strategies.append(resolved)
        else:
            missing_s.append(sid)
    if missing_l or missing_s:
        return None
    return {"lesson_ids": resolved_lessons, "strategy_ids": resolved_strategies}


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

    # Alias + verification_plan type slips (string lists, seed strings, repair wrappers).
    data = coerce_planner_json_object(data)

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
    from scientist_lab.adapters.dfine.how_catalog import (
        ALLOWED_HOW,
        NOT_REGISTERED,
        plan_how_id,
        planner_visible_how_ids,
    )
    from scientist_lab.datasets.low_light_subset import (
        SliceAmendmentRequired,
        assert_slice_not_rewritten,
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
    try:
        assert_slice_not_rewritten({"modification_scope": [module], "proposed_changes": changes})
    except SliceAmendmentRequired as exc:
        _refuse(f"fail_closed: {exc}")
    how_id = plan_how_id({"how_id": selected.get("how_id"), "proposed_changes": changes})
    how_executable = True
    pending_unmaterializable: dict[str, Any] | None = None
    if how_id:
        token = str(how_id).strip().upper()
        visible = planner_visible_how_ids(payload.how_overlay)
        if token in visible:
            spec = ALLOWED_HOW.get(token) or dict(payload.how_overlay.get(token) or {})
            primary = str(spec.get("primary_module") or "")
            if primary and primary != module:
                _refuse(
                    f"fail_closed: how_id={token} maps to module {primary!r}, "
                    f"not selected.requested_module={module!r}"
                )
            how_id = token
        else:
            # Keep the LLM's id. Do not rewrite F2/T* (or unknown ids) to F1.
            how_executable = False
            how_id = token
            pending_unmaterializable = {
                "how_id": token,
                "family": _family_for_how(token, module),
                "mechanism": str(selected.get("hypothesis") or "").strip()
                or f"LLM selected unmaterializable HOW {token}",
                "requested_module": module,
                "not_registered": token in NOT_REGISTERED,
            }

    haystack = _as_text(
        module,
        selected.get("hypothesis"),
        selected.get("observation"),
        selected.get("rationale"),
        json.dumps(changes, ensure_ascii=False),
        json.dumps(data.get("candidates") or [], ensure_ascii=False),
        json.dumps(data.get("how_candidates") or [], ensure_ascii=False),
    )
    if _BANNED_OPERATOR.search(haystack):
        _refuse("fail_closed: invented operator / FDPN / write-Python is forbidden")
    # FDPN invent check uses selected prose only. candidates[] may name N1/A4
    # (FDPN neck) as not-selected coverage without inventing a HOW.
    selected_hay = _as_text(
        module,
        selected.get("hypothesis"),
        selected.get("observation"),
        selected.get("rationale"),
        json.dumps(changes, ensure_ascii=False),
    )
    if re.search(r"\bfdpn\b", selected_hay, re.IGNORECASE):
        token = str(how_id or "").strip().upper()
        # Live M1: selecting N1/A4 may name FDPN; other catalog HOWs may cite a
        # prior executed N1/A4 as observation.
        history = _as_text(json.dumps(payload.previous_plan or {}, ensure_ascii=False))
        prior_fdpn = bool(
            re.search(r"\b(n1|a4)\b|\bfdpn\b", history, re.IGNORECASE)
        )
        if token not in {"N1", "A4"} and not prior_fdpn:
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
    if prev_budget == "formal" and llm_budget != "formal":
        _refuse(
            "fail_closed: previous plan is formal (v2.6 G2 family); "
            "LLM cannot drop budget_class to probe"
        )

    known_lessons = set(known_lesson_ids or [])
    known_strategies = set(known_strategy_ids or [])
    cited = data.get("memory_refs") or selected.get("memory_refs") or {}
    resolved_refs = _resolve_cited_memory_refs(
        cited,
        known_lessons=known_lessons,
        known_strategies=known_strategies,
        fallback=payload.memory_refs,
    )
    if resolved_refs is None:
        cited_lessons = [str(i) for i in (cited.get("lesson_ids") or [])]
        cited_strategies = [str(i) for i in (cited.get("strategy_ids") or [])]
        missing_l = [i for i in cited_lessons if _resolve_memory_ref_token(i, known_lessons) is None]
        missing_s = [
            i for i in cited_strategies if _resolve_memory_ref_token(i, known_strategies) is None
        ]
        _refuse(
            "fail_closed: LLM invented unresolved memory_refs: "
            f"lessons={missing_l} strategies={missing_s}"
        )
    refs = resolved_refs

    metric = str(
        ((payload.protocol.get("objective") or {}).get("primary") or {}).get("metric")
        or "APS"
    )
    direction = "increase"
    if str(((payload.protocol.get("objective") or {}).get("primary") or {}).get("direction") or "") == "minimize":
        direction = "decrease"
    expected = sanitize_expected_effect(
        selected.get("expected_effect") or {},
        default_metric=metric,
        default_direction=direction,
    )
    expected.setdefault(
        "rationale",
        f"LLM Planner mechanism guess on allowed module {module}; Adapter remains HOW.",
    )
    expected = sanitize_expected_effect(
        expected, default_metric=metric, default_direction=direction
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

    verification = dict(data.get("verification_plan") or selected.get("verification_plan") or {})
    if payload.experiment_brief.get("live_m1"):
        how_token = str(how_id or "").strip().upper() or module
        seed_hint = selected.get("seed")
        if seed_hint is None:
            seeds = list((selected.get("evaluation") or {}).get("seeds") or []) if isinstance(selected.get("evaluation"), Mapping) else []
            seed_hint = seeds[0] if seeds else None
        hypo = str(selected.get("hypothesis") or "").strip()
        metric_name = str(
            ((payload.protocol.get("objective") or {}).get("primary") or {}).get("metric")
            or expected.get("primary_metric")
            or "APS"
        )
        direction = str(expected.get("direction") or "increase")
        if not str(verification.get("what_to_run") or "").strip():
            verification["what_to_run"] = (
                f"Run registered HOW {how_token}"
                + (f" on seed {seed_hint}" if seed_hint is not None else "")
                + (f": {hypo}" if hypo else ".")
            ).strip()
            verification.setdefault("synthesized", True)
        if not str(verification.get("how_to_verify") or "").strip():
            verification["how_to_verify"] = (
                f"Compare primary metric {metric_name} against the previous round "
                f"under the same frozen protocol; report delta and keep/discard rubric."
            )
            verification.setdefault("synthesized", True)
        if not str(verification.get("success_criterion") or "").strip():
            verification["success_criterion"] = (
                f"{metric_name} moves in the expected direction ({direction}) "
                "relative to the prior evidence run, within formal budget constraints."
            )
            verification.setdefault("synthesized", True)
        for key in ("what_to_run", "how_to_verify", "success_criterion"):
            if not str(verification.get(key) or "").strip():
                _refuse(f"fail_closed: verification_plan.{key} missing (live M1)")
    # LLM often returns a prose string; schema requires string[].
    coverage = verification.get("addresses_coverage")
    if isinstance(coverage, str):
        text = coverage.strip()
        verification["addresses_coverage"] = [text] if text else []
    elif coverage is not None and not isinstance(coverage, list):
        verification["addresses_coverage"] = [str(coverage)]
    elif isinstance(coverage, list):
        verification["addresses_coverage"] = [
            str(x).strip() for x in coverage if str(x).strip()
        ]
    for alias in ("addresses_question", "address_question", "open_question"):
        if alias not in verification:
            continue
        verification["addresses_coverage"] = _merge_coverage_lists(
            verification.get("addresses_coverage"), verification.pop(alias)
        )

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

    if how_id:
        first = dict(changes[0])
        detail = dict(first.get("detail") or {})
        detail.setdefault("how_id", str(how_id).strip().upper())
        first["detail"] = detail
        changes[0] = first

    requested_seed = selected.get("seed")
    if requested_seed is None:
        selected_eval = selected.get("evaluation")
        if isinstance(selected_eval, Mapping):
            seeds = list(selected_eval.get("seeds") or [])
            if seeds:
                requested_seed = seeds[0]

    how_candidates = _extract_how_candidates(data, payload)

    mapped = {
        "modification_scope": [module],
        "proposed_changes": changes,
        "hypothesis": hypothesis,
        "observation": observation,
        "expected_effect": expected,
        "budget_class": llm_budget,
        "how_id": str(how_id).strip().upper() if how_id else None,
        "seed": requested_seed,
        "selected_action": str(selected.get("selected_action") or f"probe_{module}"),
        "memory_refs": refs,
        "candidate_experiments": attachments,
        "selected_candidate_id": selected_id,
        "how_candidates": how_candidates,
        "how_executable": how_executable,
        "pending_unmaterializable": pending_unmaterializable,
        "verification_plan": verification,
        "parsed": data,
        "raw_output": redact_secrets(raw),
    }
    return mapped


def _literature_paper_ids(literature: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    blob = dict(literature or {})
    for key in ("planner_admissible", "planner_admissible"):
        for row in blob.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
            token = str((paper or {}).get("paper_id") or (paper or {}).get("paper_id") or "").strip()
            if token:
                ids.add(token)
    provenance = blob.get("provenance") if isinstance(blob.get("provenance"), Mapping) else {}
    for token in list(blob.get("paper_refs") or []) + list((provenance or {}).get("paper_refs") or []):
        if str(token).strip():
            ids.add(str(token).strip())
    return ids


def _extract_how_candidates(
    data: Mapping[str, Any],
    payload: PlannerContractInput,
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in (data.get("how_candidates") or []) if isinstance(item, Mapping)]
    literature = dict(getattr(payload, "literature", None) or {})
    if not rows:
        return []
    if literature.get("fail_closed") or literature.get("fail_closed"):
        return []
    known = _literature_paper_ids(literature)
    qid = str(
        literature.get("literature_query_id")
        or literature.get("literature_query_id")
        or (literature.get("provenance") or {}).get("literature_query_id")
        or (literature.get("provenance") or {}).get("literature_query_id")
        or ""
    ).strip()
    if not qid or not known:
        # No successful scout: drop drafts, do not fail the whole Plan.
        return []
    cleaned: list[dict[str, Any]] = []
    for item in rows:
        invented = [str(x).strip() for x in (item.get("invented_operators") or []) if str(x).strip()]
        if invented:
            _refuse(f"fail_closed: invented operators not allowed on HOW candidates: {invented}")
        intent = str(item.get("implementation_intent") or "").strip()
        hay = " ".join(
            [
                str(item.get("mechanism") or ""),
                str(item.get("hypothesis") or ""),
                intent,
            ]
        )
        if _BANNED_OPERATOR.search(hay) or _HOW_CANDIDATE_CODE.search(hay):
            _refuse(
                "fail_closed: HOW candidate must not include Python or invented operators"
            )
        papers = [str(x).strip() for x in (item.get("paper_refs") or []) if str(x).strip()]
        query_id = str(item.get("literature_query_id") or qid).strip()
        if query_id != qid:
            _refuse("fail_closed: HOW candidate literature_query_id does not match the scout packet")
        if not papers or any(pid not in known for pid in papers):
            _refuse("fail_closed: HOW candidate paper_refs must be scout-admissible paper ids")
        row = dict(item)
        row["literature_query_id"] = qid
        row["paper_refs"] = papers
        row["invented_operators"] = []
        cleaned.append(row)
    return cleaned
