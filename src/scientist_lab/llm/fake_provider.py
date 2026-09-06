"""Deterministic FakeProvider — no network, fixed structured outputs."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.llm.audit import build_usage_from_messages
from scientist_lab.llm.models import LLMRequest, LLMResponse
from scientist_lab.llm.provider import BaseLLMProvider, request_fingerprint
from scientist_lab.llm.schema_parser import (
    CRITIC_REVIEW_SCHEMA,
    PLANNER_OUTPUT_SCHEMA,
    parse_and_validate,
)


def _uses_reviewer_contract(request: LLMRequest) -> bool:
    meta = dict(request.metadata or {})
    if str(meta.get("reviewer_contract") or "") == "semantic":
        return True
    if request.purpose == "reviewer":
        return True
    required = list((request.response_schema or {}).get("required") or [])
    return "interpretation" in required and "hypothesis_status" in required


def _uses_experiment_plan_contract(request: LLMRequest) -> bool:
    meta = dict(request.metadata or {})
    if str(meta.get("planner_contract") or "") == "experiment_plan":
        return True
    required = list((request.response_schema or {}).get("required") or [])
    return "selected" in required


def _project_id_from_request(request: LLMRequest) -> str:
    meta = dict(request.metadata or {})
    if meta.get("project_id"):
        return str(meta["project_id"])
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, dict) and content.get("project_id"):
            return str(content["project_id"])
        if isinstance(content, str) and "project_id" in content:
            try:
                blob = json.loads(content)
                if isinstance(blob, dict) and blob.get("project_id"):
                    return str(blob["project_id"])
            except Exception:  # noqa: BLE001
                pass
    return "project_unknown"


def _parent_from_request(request: LLMRequest) -> str:
    meta = dict(request.metadata or {})
    if meta.get("current_best_node_id"):
        return str(meta["current_best_node_id"])
    for message in request.messages:
        content = message.get("content")
        blob: dict[str, Any] | None = None
        if isinstance(content, dict):
            blob = content
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    blob = parsed
            except Exception:  # noqa: BLE001
                blob = None
        if not blob:
            continue
        if blob.get("current_best_node_id"):
            return str(blob["current_best_node_id"])
        nodes = blob.get("nodes") or []
        if nodes and isinstance(nodes[0], dict) and nodes[0].get("node_id"):
            return str(nodes[0]["node_id"])
    return "node_unknown"


def _user_blob(request: LLMRequest) -> dict[str, Any]:
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(parsed, dict) and (
                "protocol" in parsed
                or "adapter_capabilities" in parsed
                or "memory_refs" in parsed
                or "locked_review_decision" in parsed
            ):
                return parsed
    return {}


def build_fake_experiment_plan_payload(request: LLMRequest) -> dict[str, Any]:
    """v2.5-A freeze Planner JSON. Legal fusion HOW; no FDPN; no formal promotion."""
    blob = _user_blob(request)
    protocol = dict(blob.get("protocol") or {})
    previous = dict(blob.get("previous_plan") or {})
    refs = dict(blob.get("memory_refs") or {})
    review = str(blob.get("last_review_decision") or "")
    prev_scope = [str(x) for x in (previous.get("modification_scope") or [])]
    discarded = prev_scope[0] if prev_scope else "neck"
    editable = [str(x) for x in (protocol.get("editable_scope") or ["fusion", "neck"])]
    module = "fusion"
    if review == "DISCARD" and discarded == "fusion" and "neck" in editable:
        module = "neck"
    elif "fusion" not in editable and "neck" in editable:
        module = "neck"
    discarded_list = [str(x) for x in (blob.get("discarded_modules") or [])]
    if not discarded_list and review == "DISCARD":
        discarded_list = [discarded]
    if module in discarded_list:
        alt = [tok for tok in ("fusion", "neck") if tok in editable and tok not in discarded_list]
        if alt:
            module = alt[0]
    budget = str((blob.get("budget") or {}).get("budget_class") or previous.get("budget_class") or "probe")
    if budget == "formal":
        budget = "probe"
    metric = str(((protocol.get("objective") or {}).get("primary") or {}).get("metric") or "APS")
    lesson_ids = list(refs.get("lesson_ids") or [])
    strategy_ids = list(refs.get("strategy_ids") or [])
    cited_lesson = lesson_ids[0] if lesson_ids else "written_memory"
    hypothesis = (
        f"After DISCARD/negative_evidence on {discarded} ({cited_lesson}), "
        f"a {module} change is a better probe of {metric} than repeating "
        f"the discarded direction."
    )
    summary = f"Probe {module} using existing Adapter HOW; do not invent operators."
    not_selected = [tok for tok in discarded_list if tok != module] or [discarded]
    alt_how = [tok for tok in ("fusion", "neck") if tok != module]
    candidates = [
        {
            "candidate_id": "cand_discarded_not_selected",
            "requested_module": not_selected[0],
            "hypothesis": (
                f"{not_selected[0]} already has DISCARD/negative_evidence; "
                "listed only to explain why it is not selected."
            ),
            "summary": f"Not selected: repeats discarded module {not_selected[0]}.",
            "reason_not_selected": (
                f"Rubric=DISCARD on {not_selected[0]}; do not repeat that modification_scope."
            ),
        },
        {
            "candidate_id": "cand_alt_not_selected",
            "requested_module": alt_how[0] if alt_how else "neck",
            "hypothesis": "Alternate allowed-module probe kept as attachment only.",
            "summary": "Not sent to Gate.",
            "reason_not_selected": (
                "Selected fusion/neck HOW already covers the DISCARD switch; "
                "this attachment is for multi-candidate exam completeness."
                if (alt_how and alt_how[0] == module)
                else (
                    f"Not selected: prefer {module} HOW after DISCARD on {discarded}."
                    if alt_how and alt_how[0] in discarded_list
                    else f"Not selected: {module} is the DISCARD-aware selected HOW."
                )
            ),
        },
    ]
    if alt_how and alt_how[0] == not_selected[0]:
        candidates[1]["requested_module"] = "hyperparameter"
        candidates[1]["reason_not_selected"] = (
            "hyperparameter is editable but Adapter has no distinct HOW; "
            "not selected, not sent to Gate."
        )
        candidates[1]["summary"] = "Attachment only; no Adapter HOW."
    return {
        "selected": {
            "candidate_id": "cand_selected_fusion" if module == "fusion" else "cand_selected_neck",
            "requested_module": module,
            "observation": (
                f"DISCARD on {discarded}; citing {cited_lesson}. "
                "Probe an allowed Adapter HOW module that is not the discarded scope."
            ),
            "hypothesis": hypothesis,
            "proposed_changes": [{"target": module, "summary": summary}],
            "expected_effect": {
                "primary_metric": metric,
                "direction": "increase",
                "rationale": "Protocol objective; FakeProvider freeze-plan contract.",
            },
            "budget_class": budget,
            "selected_action": f"switch_to_{module}" if review == "DISCARD" else f"probe_{module}",
        },
        "candidates": candidates,
        "memory_refs": {
            "lesson_ids": lesson_ids,
            "strategy_ids": strategy_ids,
        },
        "invented_operators": [],
        "stop_recommended": False,
    }


def build_fake_planner_payload(request: LLMRequest) -> dict[str, Any]:
    project_id = _project_id_from_request(request)
    parent = _parent_from_request(request)
    return {
        "project_id": project_id,
        "reasoning_summary": (
            "FakeProvider fixed planner output for offline replay testing."
        ),
        "candidates": [
            {
                "candidate_id": "candidate_fake_ablation_001",
                "parent_node_id": parent,
                "title": "Fake ablation: RGB-only control",
                "hypothesis": (
                    "Removing fusion under a matched protocol should reduce AP_small "
                    "if fusion contributes."
                ),
                "experiment_type": "ablation",
                "parameter_changes": {
                    "input_mode": "rgb",
                    "fusion_method": "none",
                },
                "expected_outcomes": [
                    {
                        "metric": "AP_small",
                        "direction": "decrease",
                        "rationale": "Ablating fusion should hurt small-object AP.",
                    }
                ],
                "evidence_gap_addressed": [
                    "Missing controlled ablation for fusion contribution."
                ],
                "priority": 0.84,
                "rationale": "Deterministic fake candidate for provider-layer tests.",
                "claim_limitations": [
                    "FakeProvider output is not scientific evidence."
                ],
            }
        ],
        "stop_recommended": False,
        "stop_reason": None,
    }


def build_fake_reviewer_payload(request: LLMRequest) -> dict[str, Any]:
    """v2.5-C semantic Reviewer JSON. Explains Rubric DISCARD; does not override it."""
    blob = _user_blob(request)
    meta = dict(request.metadata or {})
    decision = str(
        blob.get("locked_review_decision")
        or meta.get("locked_review_decision")
        or "DISCARD"
    )
    result = dict(blob.get("result") or {})
    run_id = str(
        result.get("run_id") or blob.get("run_id") or meta.get("run_id") or "run_unknown"
    )
    protocol = dict(blob.get("protocol") or {})
    metric = str(
        blob.get("primary_metric")
        or ((protocol.get("objective") or {}).get("primary") or {}).get("metric")
        or "APS"
    )
    plan = dict(blob.get("plan") or {})
    scope = list(plan.get("modification_scope") or [])
    target = str(scope[0] if scope else "unknown")
    budget = str(blob.get("budget_class") or plan.get("budget_class") or "probe")
    if decision == "DISCARD":
        status = "not_supported_under_current_protocol"
        interpretation = (
            f"DecisionRubric already locked {decision}. Probe {metric} declined past "
            f"discard_if under the current protocol for {target}. This is a next-action "
            f"DISCARD, not a ClaimGate verdict and not a formal effectiveness conclusion."
        )
        next_pri = (
            "Verify an allowed Adapter HOW module that is not the discarded scope, "
            f"still {budget}-class; do not promote DISCARD into a scientific conclusion."
        )
    elif decision == "KEEP":
        status = "not_a_claim"
        interpretation = (
            f"DecisionRubric already locked {decision} on {metric}. KEEP answers the "
            "next-round action only; it is not ClaimGate SUPPORTED."
        )
        next_pri = (
            "If the hypothesis still matters, schedule a higher-budget verification "
            "under the same protocol; do not treat KEEP as a supported claim."
        )
    elif decision == "REPLICATE":
        status = "needs_replication"
        interpretation = (
            f"DecisionRubric already locked {decision}. The {metric} delta did not "
            "cross validate/discard thresholds; replication is still required."
        )
        next_pri = "Replicate the same protocol probe before changing the scientific story."
    else:
        status = "needs_validation"
        interpretation = (
            f"DecisionRubric already locked {decision}. Treat this as a verification "
            "priority, not a ClaimGate SUPPORTED result."
        )
        next_pri = "Run the protocol validation budget before any formal claim."
    claim_stance = (
        "no_module_efficacy_claim"
        if decision in {"KEEP", "DISCARD", "STOP"}
        else "deferred_to_claim_gate"
    )
    return {
        "observation": (
            f"VALID evidence for {run_id}: primary {metric} judged by DecisionRubric "
            f"as {decision}."
        ),
        "hypothesis_status": status,
        "claim_stance": claim_stance,
        "interpretation": interpretation,
        "alternative_explanations": [
            "Probe budget or a tiny subset can yield a large APS delta without a formal pair.",
            "The labeled control may be synthetic_control rather than a matched fingerprint baseline.",
            "Under-trained weights can dominate the observed metric movement.",
        ],
        "next_research_priority": next_pri,
        "evidence_refs": [{"run_id": run_id, "metric": metric}],
        "created_from": [run_id],
        "confidence": "medium",
    }


def build_fake_critic_payload(request: LLMRequest) -> dict[str, Any]:
    candidate_id = str(
        (request.metadata or {}).get("candidate_id") or "candidate_fake_ablation_001"
    )
    return {
        "candidate_id": candidate_id,
        "scientific_validity": "valid",
        "novelty_status": "new",
        "expected_information_gain": 0.7,
        "cost_effectiveness": 0.65,
        "risk_level": "low",
        "strengths": ["Single-variable ablation under protocol constraints."],
        "weaknesses": ["FakeProvider review only; not a real scientific critique."],
        "required_revisions": [],
        "recommendation": "accept",
    }


def build_fake_experiment_protocol_payload(request: LLMRequest) -> dict[str, Any]:
    """Offline draft of a non-V26 object_detection experiment. Not a Claim. No GPU."""
    intent = ""
    try:
        user = json.loads(str((request.messages or [{}])[-1].get("content") or "{}"))
        if isinstance(user, dict):
            intent = str(user.get("user_intent") or "").strip()
            brief = user.get("idea_brief") if isinstance(user.get("idea_brief"), dict) else {}
            if not intent and brief:
                intent = str(brief.get("core_intent") or brief.get("research_direction") or "").strip()
    except (TypeError, ValueError, json.JSONDecodeError, IndexError, AttributeError):
        intent = ""
    question = (
        intent[:220]
        if intent
        else (
            "On the full RGBT-Tiny val split (not the frozen low-light slice), "
            "does RGB-only F0 underperform early-concat fusion for APS?"
        )
    )
    if intent and "aps" not in intent.lower() and "指标" not in intent:
        question = (
            f"{intent.rstrip('。.')} — evaluated on full-val APS "
            "(not the frozen low-light APS_lowlight slice)."
        )
    return {
        "research_question": question[:400],
        "dataset_id": "rgbt_tiny_v1",
        "slice_id": None,
        "primary_metric": "APS",
        "adapter_id": "dfine",
        "seed_how_id": "F0",
        "catalog_id": "how_catalog_v26_p2",
        "difference_from_builtin": (
            "Honors Idea Interview / user_intent when present; "
            "slice_id is unset (full val, not low_light_subset_v1); "
            "primary_metric is APS not APS_lowlight; seed_how_id is F0 not F1."
        ),
        "title": "Full-val RGB-only APS vs thermal fusion",
        "protocol_id": "research_protocol_rgbt_dfine_fullval_aps_f0",
        "experiment_id": "exp_rgbt_dfine_fullval_aps_f0",
    }


def build_fake_idea_interview_payload(request: LLMRequest) -> dict[str, Any]:
    """Offline Idea Interview turn. Not Planner. No GPU."""
    message = ""
    try:
        user = json.loads(str((request.messages or [{}])[-1].get("content") or "{}"))
        dialogue = user.get("dialogue") if isinstance(user, dict) else None
        if isinstance(dialogue, list) and dialogue:
            message = str((dialogue[-1] or {}).get("text") or "").strip()
    except (TypeError, ValueError, json.JSONDecodeError, IndexError, AttributeError):
        message = ""
    other = bool(
        re.search(
            r"分类|分割|segmentation|classification|nlp|diffusion",
            message,
            re.IGNORECASE,
        )
    )
    if other:
        return {
            "reply": (
                "已记下。分类/分割现在不能开战；可改成目标检测方向后再起草。"
                "这不是战役 Planner。"
            ),
            "brief": {
                "core_intent": message[:400],
                "research_direction": "",
                "hypothesis": "",
                "constraints": ["unsupported_task_for_ready_campaign"],
                "preferred_adapter": None,
                "preferred_metric": None,
                "preferred_slice": None,
                "open_questions": ["是否改成 object_detection 方向？"],
                "ready_to_draft": False,
                "unsupported_task_note": "non object_detection task",
            },
        }
    ready = len(message) >= 12
    return {
        "reply": (
            "已提取你的方向。可继续补充主指标/对照，或基于想法起草协议。"
            "我不是战役 Planner，也不会点火 GPU。"
            if ready
            else "请再写具体一点：想验证什么假设、盯哪个指标？"
        ),
        "brief": {
            "core_intent": message[:400] or "unspecified detection direction",
            "research_direction": message[:240],
            "hypothesis": message[:240] if ready else "",
            "constraints": ["object_detection", "adapter dfine|rtdetr"],
            "preferred_adapter": "dfine",
            "preferred_metric": "APS" if ("aps" in message.lower() or "full" in message.lower()) else None,
            "preferred_slice": None,
            "open_questions": [] if ready else ["主指标？", "对照设置？"],
            "ready_to_draft": ready,
            "unsupported_task_note": None,
        },
    }


class FakeProvider(BaseLLMProvider):
    """Returns deterministic structured JSON. Never opens network sockets."""

    def __init__(self, *, model: str = "fake-llm-v1") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if request.purpose == "planner":
            if _uses_experiment_plan_contract(request):
                payload = build_fake_experiment_plan_payload(request)
            else:
                payload = build_fake_planner_payload(request)
            schema = request.response_schema or PLANNER_OUTPUT_SCHEMA
        elif _uses_reviewer_contract(request):
            from scientist_lab.llm.reviewer_contract import REVIEWER_CONTRACT_SCHEMA

            payload = build_fake_reviewer_payload(request)
            schema = request.response_schema or REVIEWER_CONTRACT_SCHEMA
        elif request.purpose == "critic":
            payload = build_fake_critic_payload(request)
            schema = request.response_schema or CRITIC_REVIEW_SCHEMA
        elif str((request.metadata or {}).get("planner_contract") or "") == "idea_interview":
            payload = build_fake_idea_interview_payload(request)
            schema = request.response_schema
        elif str((request.metadata or {}).get("planner_contract") or "") == "experiment_protocol":
            payload = build_fake_experiment_protocol_payload(request)
            schema = request.response_schema
        else:
            payload = {
                "ok": True,
                "echo_purpose": request.purpose,
                "message": "FakeProvider generic response.",
            }
            schema = request.response_schema

        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        parsed, errors = parse_and_validate(content, schema)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            request_id=new_id("llmreq"),
            content=content,
            parsed_json=parsed,
            usage=build_usage_from_messages(request.messages, content),
            latency_ms=elapsed_ms,
            provider=self.name,
            model=self.model,
            request_fingerprint=request_fingerprint(request),
            schema_valid=not errors,
            schema_errors=errors,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )
