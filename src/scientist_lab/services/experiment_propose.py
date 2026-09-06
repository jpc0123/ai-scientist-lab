"""LLM drafts a new object-detection experiment. Human confirms. No GPU."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import (
    CATALOG_ID,
    ALLOWED_HOW,
    NOT_REGISTERED,
    plan_how_id,
)
from scientist_lab.core.schema_registry import load_json
from scientist_lab.llm.gateway import complete_chat
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.planner_contract import prompt_hash
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema
from scientist_lab.services.registered_experiments import (
    BUILTIN_EXPERIMENT_ID,
    BUILTIN_PLAN,
    BUILTIN_PROTOCOL,
    RegisteredExperimentError,
    adapter_token,
    builtin_science_identity,
    parse_dataset_id,
    same_science_as_builtin,
    sanitize_experiment_id,
    science_identity,
)

BUILTIN_PROTOCOL_ID = "research_protocol_rgbt_dfine_v26"
_ALLOWED_ADAPTERS = frozenset({"dfine", "rtdetr", "rt_detr"})
_ALLOWED_METRICS = frozenset(
    {
        "APS",
        "AP_small",
        "APS_lowlight",
        "mAP50",
        "mAP50_95",
        "mAP",
        "AP",
        "AP50_lowlight",
        "mAP50_95_lowlight",
    }
)
_BANNED_DETECTOR = re.compile(
    r"\b(yolo|yolov\d+|new[-_ ]backbone|new[-_ ]detector)\b",
    re.IGNORECASE,
)
_IDLE_QUESTION = re.compile(
    r"(再比一次|三选一|compare\s+again|again\s+compare).{0,40}(f0|f1|f3)"
    r"|(f0).{0,12}(f1).{0,12}(f3).{0,20}(再比|again|same\s+comparison)",
    re.IGNORECASE,
)

EXPERIMENT_PROPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "research_question",
        "dataset_id",
        "primary_metric",
        "adapter_id",
        "seed_how_id",
    ],
    "additionalProperties": True,
    "properties": {
        "research_question": {"type": "string", "minLength": 8},
        "dataset_id": {"type": "string"},
        "slice_id": {"type": ["string", "null"]},
        "primary_metric": {"type": "string"},
        "adapter_id": {"type": "string"},
        "seed_how_id": {"type": "string"},
        "catalog_id": {"type": "string"},
        "difference_from_builtin": {"type": "string"},
        "title": {"type": "string"},
        "protocol_id": {"type": "string"},
        "experiment_id": {"type": "string"},
    },
}

PROPOSE_SYSTEM_PROMPT = """You draft ONE new object_detection experiment for Scientist Lab.
You are not a fifth Agent. You are not the campaign Planner.
Protocol / Adapter / Gate / ClaimGate stay in charge.
Return JSON only matching the schema.

If user_intent / idea_brief / dialogue are provided, the draft MUST reflect that human
direction. Do not ignore the interview. Prefer their metric/slice/adapter/HOW hints when
legal; if illegal, pick the closest Adapter-runnable alternative and explain in
difference_from_builtin.

The built-in experiment exp_rgbt_dfine_v26_lowlight already covers:
dataset rgbt_tiny_v1 + slice low_light_subset_v1 + metric APS_lowlight + seed HOW F1.
Do NOT propose that same science fingerprint. At least one of dataset_id, slice_id,
primary_metric, seed_how_id MUST differ. Do not write "compare F0/F1/F3 again".

Stay on adapters dfine or rtdetr, dataset rgbt_tiny_v1, and materializable HOW ids
the Adapter already has (F0, F1, F3, N0, N1, A4 on D-FINE; RT-DETR cannot run N1/A4).
Do not propose YOLO, a new backbone, F2/weighted_fusion, or T* training operators.
Literature is not a Claim.

The framework must prove the LLM can pose a different scientific question the current
Adapter can actually run. Example directions (only when the human gave no intent):
- error analysis by brightness buckets on the existing RGBT-Tiny data
- seed-stability of one already-materializable HOW
- RGB-only vs thermal on a NON-low-light / full-val subset

Fields:
- research_question: one sentence reflecting the human idea when present
- dataset_id, slice_id (null means full val, not the frozen low-light slice),
  primary_metric, adapter_id, seed_how_id, catalog_id
- difference_from_builtin: which fingerprint field(s) changed and why
"""


class ExperimentProposeError(ValueError):
    """LLM draft cannot become a startable experiment."""


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip()).strip("_").lower()
    return (slug or fallback)[:48]


def looks_like_idle_f013_question(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if _IDLE_QUESTION.search(raw):
        return True
    compact = re.sub(r"[\s,/+]+", "", raw.lower())
    return all(tok in compact for tok in ("f0", "f1", "f3")) and (
        "再比" in raw or "三选一" in raw or "again" in raw.lower()
    )


def build_propose_request(
    *,
    user_intent: str | None = None,
    idea_brief: Mapping[str, Any] | None = None,
    dialogue: list[Mapping[str, Any]] | None = None,
    interview_id: str | None = None,
) -> LLMRequest:
    builtin = load_json(BUILTIN_PROTOCOL)
    plan = load_json(BUILTIN_PLAN)
    intent = str(user_intent or "").strip()
    brief = dict(idea_brief or {}) if idea_brief else {}
    history = [
        {
            "role": str(row.get("role") or "human"),
            "text": str(row.get("text") or "")[:1200],
        }
        for row in (dialogue or [])[-16:]
        if str(row.get("text") or "").strip()
    ]
    user = {
        "builtin_experiment_id": BUILTIN_EXPERIMENT_ID,
        "builtin_science_identity": builtin_science_identity(),
        "builtin_protocol_id": builtin.get("protocol_id"),
        "builtin_title": builtin.get("title"),
        "allowed_adapters": sorted(_ALLOWED_ADAPTERS),
        "allowed_metrics": sorted(_ALLOWED_METRICS),
        "materializable_how_ids": sorted(ALLOWED_HOW),
        "not_registered_how_ids": sorted(NOT_REGISTERED),
        "catalog_id": CATALOG_ID,
        "seed_how_of_builtin": plan_how_id(plan),
        "interview_id": interview_id,
        "user_intent": intent or None,
        "idea_brief": brief or None,
        "dialogue": history or None,
        "constraints": {
            "task_type": "object_detection",
            "forbid_yolo": True,
            "forbid_new_backbone": True,
            "forbid_same_science_fingerprint": True,
            "forbid_rewrite_to_builtin_protocol": True,
            "no_gpu": True,
            "not_a_claim": True,
            "not_campaign_planner": True,
            "honor_user_intent_when_present": True,
        },
    }
    return LLMRequest(
        purpose="other",
        messages=[
            {"role": "system", "content": PROPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=EXPERIMENT_PROPOSE_SCHEMA,
        temperature=0.2,
        metadata={"planner_contract": "experiment_protocol"},
    )


def _parse_draft(raw: str) -> dict[str, Any]:
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ExperimentProposeError(f"fail_closed: invalid JSON: {exc}") from exc
    errors = validate_against_schema(data, EXPERIMENT_PROPOSE_SCHEMA)
    if errors:
        raise ExperimentProposeError("fail_closed: propose schema: " + "; ".join(errors))
    return dict(data)


def _apply_slice(protocol: dict[str, Any], slice_id: str | None) -> None:
    token = str(slice_id or "").strip()
    if not token or token.lower() in {"none", "null", "full", "full_val", "all"}:
        protocol.pop("condition_slice", None)
        frozen = [str(x) for x in (protocol.get("frozen_scope") or [])]
        protocol["frozen_scope"] = [x for x in frozen if x != "condition_slice"]
        return
    previous = dict(protocol.get("condition_slice") or {})
    previous["id"] = token
    protocol["condition_slice"] = previous


def materialize_protocol_from_draft(draft: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay LLM science fields onto the V26 schema skeleton. Never copy builtin ids."""
    template = load_json(BUILTIN_PROTOCOL)
    seed_template = load_json(BUILTIN_PLAN)
    question = str(draft.get("research_question") or "").strip()
    adapter = str(draft.get("adapter_id") or "dfine").strip().lower().replace("-", "_")
    if adapter == "rt_detr":
        adapter = "rtdetr"
    how_id = str(draft.get("seed_how_id") or "").strip().upper()
    dataset = parse_dataset_id(str(draft.get("dataset_id") or "rgbt_tiny_v1"))
    metric = str(draft.get("primary_metric") or "").strip()
    catalog_id = str(draft.get("catalog_id") or CATALOG_ID).strip() or CATALOG_ID
    title = str(draft.get("title") or question or "LLM drafted detection experiment").strip()
    slug = _slug(str(draft.get("protocol_id") or title or how_id), "llm_detection")
    protocol_id = str(draft.get("protocol_id") or f"research_protocol_{slug}").strip()
    if protocol_id == BUILTIN_PROTOCOL_ID:
        protocol_id = f"research_protocol_{slug}"
        if protocol_id == BUILTIN_PROTOCOL_ID:
            protocol_id = f"research_protocol_{slug}_alt"

    protocol = dict(template)
    protocol["protocol_id"] = protocol_id
    protocol["title"] = title
    protocol["fingerprint_id"] = f"FP-{slug.upper().replace('_', '-')}"[:80]
    if protocol["fingerprint_id"] == str(template.get("fingerprint_id") or ""):
        digest = hashlib.sha256(protocol_id.encode("utf-8")).hexdigest()[:12]
        protocol["fingerprint_id"] = f"FP-LLM-{digest}"
    protocol["goal"] = {
        "improve": str((template.get("goal") or {}).get("improve") or "object_detection"),
        "task_type": "object_detection",
        "notes": question,
    }
    baseline = dict(protocol.get("baseline") or {})
    baseline["adapter"] = adapter
    baseline["dataset"] = f"dataset:{dataset}" if not str(dataset).startswith("dataset:") else dataset
    if adapter == "rtdetr":
        baseline["model"] = str(draft.get("model") or "RT-DETR-S")
    else:
        baseline["model"] = str(draft.get("model") or "DFINE-S")
    protocol["baseline"] = baseline
    objective = dict(protocol.get("objective") or {})
    primary = dict(objective.get("primary") or {})
    primary["metric"] = metric or primary.get("metric") or "APS"
    primary.setdefault("direction", "maximize")
    objective["primary"] = primary
    protocol["objective"] = objective
    if "slice_id" in draft:
        _apply_slice(protocol, draft.get("slice_id"))
    notes = [
        str(protocol.get("notes") or "").strip(),
        f"LLM research question: {question}",
        str(draft.get("difference_from_builtin") or "").strip(),
        "KEEP ≠ Claim. This document is not a Claim.",
    ]
    protocol["notes"] = " ".join(part for part in notes if part)

    spec = dict(ALLOWED_HOW.get(how_id) or {})
    module = str(spec.get("primary_module") or "fusion")
    plan = dict(seed_template)
    plan["protocol_id"] = protocol_id
    plan["plan_id"] = f"plan_{slug}_seed"
    plan["how_id"] = how_id or None
    plan["observation"] = question or "LLM drafted a new registered experiment."
    plan["hypothesis"] = question or "Adapter-runnable probe of an existing HOW."
    plan["modification_scope"] = [module]
    plan["proposed_changes"] = [
        {
            "target": module,
            "summary": f"Seed HOW {how_id or 'unset'} for a new experiment (not a V26 copy).",
            "detail": {"how_id": how_id} if how_id else {},
        }
    ]
    expected = dict(plan.get("expected_effect") or {})
    expected["primary_metric"] = primary["metric"]
    expected.setdefault("direction", "unclear")
    expected["rationale"] = question
    plan["expected_effect"] = expected
    plan["rationale"] = (
        "LLM-drafted seed plan. Human must confirm register. Not a Claim. No GPU yet."
    )
    plan["bootstrap"] = True
    return protocol, plan


def _reasons_for_draft(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    question: str,
) -> list[str]:
    reasons: list[str] = []
    hay = " ".join(
        [
            question,
            str(protocol.get("title") or ""),
            str(((protocol.get("baseline") or {}).get("adapter") or "")),
            str(((protocol.get("baseline") or {}).get("model") or "")),
        ]
    )
    if looks_like_idle_f013_question(question):
        reasons.append("research_question looks like another F0/F1/F3 idle comparison")
    if _BANNED_DETECTOR.search(hay):
        reasons.append("YOLO / new backbone is forbidden; stay on dfine or rtdetr")
    adapter = adapter_token(protocol)
    if adapter not in _ALLOWED_ADAPTERS:
        reasons.append(f"adapter {adapter!r} is not in {sorted(_ALLOWED_ADAPTERS)}")
    how_id = plan_how_id(plan)
    if how_id and str(how_id).strip().upper() in NOT_REGISTERED:
        reasons.append(
            f"seed HOW {how_id} is not materializable; pending Adapter work, cannot start"
        )
    if same_science_as_builtin(protocol, plan):
        reasons.append(
            "science fingerprint matches builtin V26 "
            f"({science_identity(protocol, plan)}); refusing idle loop"
        )
    if str(protocol.get("protocol_id") or "") == BUILTIN_PROTOCOL_ID:
        reasons.append("refusing to reuse builtin protocol_id")
    metric = str(((protocol.get("objective") or {}).get("primary") or {}).get("metric") or "")
    if metric and metric not in _ALLOWED_METRICS:
        reasons.append(f"primary_metric {metric!r} is not an Adapter-known detection metric")
    return reasons


def propose_experiment_draft(
    *,
    provider: Any | None = None,
    live: bool = False,
    user_intent: str | None = None,
    idea_brief: Mapping[str, Any] | None = None,
    dialogue: list[Mapping[str, Any]] | None = None,
    interview_id: str | None = None,
) -> dict[str, Any]:
    """Call LLM (or FakeProvider) and return a preview. Does not register. No GPU."""
    intent = str(user_intent or "").strip() or None
    request = build_propose_request(
        user_intent=intent,
        idea_brief=idea_brief,
        dialogue=dialogue,
        interview_id=interview_id,
    )
    response = complete_chat(request, provider=provider, live=live)
    raw = str(response.content or "")
    try:
        draft = _parse_draft(raw)
    except ExperimentProposeError as exc:
        return {
            "ok": False,
            "registered": False,
            "gpu": False,
            "is_claim": False,
            "status_if_registered": "draft",
            "unsupported_reason": str(exc),
            "raw_output": raw,
            "interview_id": interview_id,
            "user_intent": intent,
            "idea_brief": dict(idea_brief or {}) or None,
            "llm": {
                "provider": getattr(response, "provider", None),
                "model": getattr(response, "model", None),
                "prompt_hash": prompt_hash(request.messages),
            },
        }
    protocol, plan = materialize_protocol_from_draft(draft)
    if str(protocol.get("protocol_id") or "") == BUILTIN_PROTOCOL_ID:
        protocol["protocol_id"] = f"research_protocol_{_slug(draft.get('title') or 'llm', 'llm')}_alt"
        plan["protocol_id"] = protocol["protocol_id"]
    reasons = _reasons_for_draft(
        protocol, plan, str(draft.get("research_question") or "")
    )
    identity = science_identity(protocol, plan)
    preview_status = "ready"
    unsupported: str | None = None
    if reasons:
        preview_status = "draft"
        unsupported = "; ".join(reasons)
    else:
        from scientist_lab.services.registered_experiments import assert_seed_how_materializable

        try:
            assert_seed_how_materializable(protocol, plan)
        except RegisteredExperimentError as exc:
            preview_status = "draft"
            unsupported = str(exc)
    experiment_id = sanitize_experiment_id(
        str(draft.get("experiment_id") or f"exp_{_slug(protocol['protocol_id'], 'llm')}")
    )
    if experiment_id == BUILTIN_EXPERIMENT_ID:
        experiment_id = f"{experiment_id}_alt"
        preview_status = "draft"
        unsupported = (unsupported or "refusing builtin experiment_id").strip()
    return {
        "ok": True,
        "registered": False,
        "gpu": False,
        "is_claim": False,
        "research_question": draft.get("research_question"),
        "difference_from_builtin": draft.get("difference_from_builtin"),
        "experiment_id": experiment_id,
        "title": protocol.get("title"),
        "protocol": protocol,
        "seed_plan": plan,
        "catalog_id": str(draft.get("catalog_id") or CATALOG_ID),
        "science_identity": identity,
        "builtin_science_identity": builtin_science_identity(),
        "differs_from_builtin": identity != builtin_science_identity(),
        "status_if_registered": preview_status,
        "unsupported_reason": unsupported,
        "interview_id": interview_id,
        "user_intent": intent,
        "idea_brief": dict(idea_brief or {}) or None,
        "llm": {
            "provider": getattr(response, "provider", None),
            "model": getattr(response, "model", None),
            "prompt_hash": prompt_hash(request.messages),
        },
        "note": (
            "Preview only. Human must confirm register. "
            "Confirmation does not start GPU. KEEP ≠ Claim. "
            "Idea Interview is not the campaign Planner."
        ),
    }
