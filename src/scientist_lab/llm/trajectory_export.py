"""Read-only ATDP six-tuple export from a Manager run directory.

Does not write research_events.jsonl. Does not drive Planner / Gate / GPU.
Detection scores never become contrast labels or SFT/DPO/RL rewards.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scientist_lab.adapters.dfine.how_catalog import ALLOWED_HOW, plan_how_id
from scientist_lab.core.schema_registry import SchemaValidationError, validate_named
from scientist_lab.llm.contrast_holdout import (
    bucket_from_label,
    label_contrast,
    training_leaks_holdout,
)

EXPORTER_VERSION = "atdp6_v1"
REWARD_EXPORT_VERSION = "contrast_v1"
SCHEMA_VERSION = "1.0.0"

STEP_KINDS = (
    "plan_proposal",
    "gate_decision",
    "execution",
    "review_decision",
)

HOW_ZH = {
    "F0": "只用可见光，不用热成像",
    "F1": "可见光和热成像一开始就拼成一张图",
    "F3": "可见光、热成像各走一路，再用门控融合",
    "N0": "脖子（特征金字塔）用仓库里的标准版",
    "N1": "另一种脖子（FDPN），对 Planner 隐藏",
    "A4": "F1 拼接融合再加 N1 脖子，不是 Planner 发现答案",
}

REWARD_BY_LABEL = {
    "how_contrast": 1.0,
    "seed_contrast": 1.0,
    "stop_ok": 1.0,
    "idle": -2.0,
    "fake_contrast": -1.0,
    "confound": -1.0,
    "illegal": -1.0,
}


def how_zh(how_id: str | None) -> str:
    token = str(how_id or "").strip().upper()
    if not token:
        return "未写明做法"
    return HOW_ZH.get(token, f"目录外做法 {token}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _as_seeds(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    return [int(x) for x in value]


def _parse_dataset_id(raw: Any) -> str:
    text = str(raw or "").strip()
    if text.startswith("dataset:"):
        return text.split(":", 1)[1].strip()
    return text


def last_event(
    events: Sequence[Mapping[str, Any]], event_type: str
) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for row in events:
        if str(row.get("event_type") or "") == event_type:
            found = dict(row)
    return found


def events_of_type(
    events: Sequence[Mapping[str, Any]], event_type: str
) -> list[dict[str, Any]]:
    return [dict(row) for row in events if str(row.get("event_type") or "") == event_type]


def extract_how_id(*blobs: Mapping[str, Any] | None) -> str:
    for blob in blobs:
        if not blob:
            continue
        token = plan_how_id(blob)
        if token:
            return str(token).strip().upper()
        how = blob.get("how") if isinstance(blob.get("how"), Mapping) else None
        if how and how.get("how_id"):
            return str(how.get("how_id")).strip().upper()
        mat = blob.get("materialization") if isinstance(blob.get("materialization"), Mapping) else None
        nested = (mat or {}).get("how") if isinstance((mat or {}).get("how"), Mapping) else {}
        if nested.get("how_id"):
            return str(nested.get("how_id")).strip().upper()
        training = blob.get("training") if isinstance(blob.get("training"), Mapping) else None
        if training and training.get("how_id"):
            return str(training.get("how_id")).strip().upper()
    return ""


def extract_seeds(*blobs: Mapping[str, Any] | None) -> list[int]:
    for blob in blobs:
        if not blob:
            continue
        evaluation = blob.get("evaluation") if isinstance(blob.get("evaluation"), Mapping) else None
        if evaluation and evaluation.get("seeds"):
            return _as_seeds(evaluation.get("seeds"))
        if blob.get("seed") is not None:
            return _as_seeds(blob.get("seed"))
        if blob.get("seeds"):
            return _as_seeds(blob.get("seeds"))
        training = blob.get("training") if isinstance(blob.get("training"), Mapping) else None
        if training and training.get("seed") is not None:
            return _as_seeds(training.get("seed"))
    return []


def extract_frozen_fields(
    protocol: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    protocol = protocol or {}
    plan = plan or {}
    contract = contract or {}
    dataset = contract.get("dataset") if isinstance(contract.get("dataset"), Mapping) else {}
    dataset_id = _parse_dataset_id(
        dataset.get("reference")
        or (protocol.get("baseline") or {}).get("dataset")
        or plan.get("dataset_id")
    )
    slice_id = str(
        ((protocol.get("condition_slice") or {}).get("id"))
        or plan.get("slice_id")
        or ""
    ).strip()
    budget = str(
        contract.get("budget_class") or plan.get("budget_class") or ""
    ).strip()
    return {
        "dataset_id": dataset_id or None,
        "slice_id": slice_id or None,
        "budget_class": budget or None,
    }


def machine_contract(
    *,
    how_id: str,
    seeds: Sequence[int],
    frozen: Mapping[str, Any],
    stop: bool = False,
) -> dict[str, Any]:
    return {
        "how_id": how_id or None,
        "how_zh": how_zh(how_id),
        "seeds": list(seeds),
        "dataset_id": frozen.get("dataset_id"),
        "slice_id": frozen.get("slice_id"),
        "budget_class": frozen.get("budget_class"),
        "stop": bool(stop),
    }


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "research_events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _first_existing(run_dir: Path, *names: str) -> dict[str, Any]:
    for name in names:
        payload = _read_json(Path(run_dir) / name)
        if payload:
            return payload
        nested = _read_json(Path(run_dir) / "run" / name)
        if nested:
            return nested
    return {}


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir)
    events = load_events(root)
    plan = _first_existing(root, "plan.json")
    previous_plan = _first_existing(root, "previous_plan.json")
    protocol = _first_existing(root, "protocol.json")
    contract = _first_existing(root, "contract.json")
    result = _first_existing(root, "result.json")
    review = _first_existing(root, "review.json")
    claim_gate = _first_existing(root, "claim_gate.json")
    run_doc = _first_existing(root, "experiment_run.json")
    metrics = _first_existing(root, "metrics.json")
    handle = _first_existing(root, "handle.json")
    baseline_metrics = _first_existing(root, "baseline_metrics.json", "previous_round_metrics.json")
    run_id = str(
        run_doc.get("run_id")
        or contract.get("run_id")
        or (result.get("run_id") if result else "")
        or root.name
    )
    return {
        "run_dir": str(root),
        "run_id": run_id,
        "events": events,
        "plan": plan,
        "previous_plan": previous_plan,
        "protocol": protocol,
        "contract": contract,
        "result": result,
        "review": review,
        "claim_gate": claim_gate,
        "run": run_doc,
        "metrics": metrics,
        "handle": handle,
        "baseline_metrics": baseline_metrics,
        "reconstructed": not bool(events),
    }


def _primary_metric(bundle: Mapping[str, Any]) -> str:
    protocol = dict(bundle.get("protocol") or {})
    plan = dict(bundle.get("plan") or {})
    result = dict(bundle.get("result") or {})
    expected = ((plan.get("expected_effect") or {}).get("primary_metric"))
    if expected:
        return str(expected)
    primary = ((protocol.get("objective") or {}).get("primary") or {}).get("metric")
    if primary:
        return str(primary)
    metrics = dict(result.get("metrics") or bundle.get("metrics") or {})
    if "APS_lowlight" in metrics:
        return "APS_lowlight"
    return "APS"


def _metric_value(blob: Mapping[str, Any] | None, key: str) -> float | None:
    if not blob:
        return None
    metrics = blob.get("metrics") if isinstance(blob.get("metrics"), Mapping) else blob
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _literature(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in events:
        payload = dict(row.get("payload") or {})
        tool = str(payload.get("tool") or payload.get("name") or payload.get("action") or "")
        blob = json.dumps(payload, ensure_ascii=False).lower()
        if "literature" not in tool.lower() and "literature" not in blob and "semantic_scholar" not in blob:
            continue
        fake = bool(
            payload.get("fake")
            or str(payload.get("provider") or "").lower() in {"fake", "mock"}
            or "fake" in blob
        )
        return {
            "tool": tool or "literature-search",
            "fake": fake,
            "query_id": payload.get("query_id") or payload.get("literature_query_id"),
            "provider": payload.get("provider"),
        }
    return None


def _plan_legal(plan: Mapping[str, Any]) -> dict[str, Any]:
    how = extract_how_id(plan)
    schema_ok = True
    schema_error = None
    if plan:
        try:
            validate_named("experiment_plan", dict(plan))
        except SchemaValidationError as exc:
            schema_ok = False
            schema_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — export must not crash a round
            schema_ok = False
            schema_error = str(exc)
    in_catalog = how in ALLOWED_HOW
    return {
        "plan_schema_valid": schema_ok,
        "plan_schema_error": schema_error,
        "how_in_catalog": in_catalog,
        "how_visible_to_planner": how in {"F0", "F1", "F3", "N0"},
    }


def _clip_summary(text: str, limit: int = 400) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1] + "…"


def _decision_summary(
    source: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    for blob in (source, fallback):
        if not blob:
            continue
        summary = blob.get("decision_summary")
        if isinstance(summary, Mapping) and summary:
            allowed = {
                "problem_observed",
                "hypothesis",
                "candidate_actions",
                "selected_action",
                "decision_basis",
                "expected_effect",
                "risk",
            }
            return {key: summary[key] for key in allowed if key in summary}
    return None


def _meta(
    bundle: Mapping[str, Any],
    *,
    step_kind: str,
    event: Mapping[str, Any] | None,
    reconstructed: bool,
    source_event_types: Sequence[str],
    actor_role: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    run_doc = dict(bundle.get("run") or {})
    plan = dict(bundle.get("plan") or {})
    protocol = dict(bundle.get("protocol") or {})
    event = dict(event or {})
    round_index = plan.get("round_index")
    if round_index is None:
        round_index = run_doc.get("round_index")
    return {
        "exporter_version": EXPORTER_VERSION,
        "step_kind": step_kind,
        "project_id": (
            event.get("project_id")
            or run_doc.get("project_id")
            or plan.get("project_id")
            or protocol.get("project_id")
        ),
        "round_id": None if round_index is None else str(round_index),
        "run_id": str(bundle.get("run_id") or ""),
        "event_id": event.get("event_id"),
        "actor_role": actor_role or event.get("actor_role"),
        "phase": phase or event.get("phase"),
        "plan_id": event.get("plan_id") or plan.get("plan_id") or run_doc.get("plan_id"),
        "fingerprint_id": (
            event.get("fingerprint_id")
            or protocol.get("fingerprint_id")
            or run_doc.get("fingerprint_id")
        ),
        "git_sha": event.get("git_sha") or run_doc.get("experiment_sha"),
        "reconstructed": bool(reconstructed),
        "source_event_types": list(source_event_types),
    }


def resolve_anchor(
    bundle: Mapping[str, Any],
    parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = parent or {}
    previous = dict(bundle.get("previous_plan") or {})
    parent_plan = dict(parent.get("plan") or previous)
    parent_contract = dict(parent.get("contract") or {})
    parent_protocol = dict(parent.get("protocol") or bundle.get("protocol") or {})
    parent_metrics = dict(parent.get("metrics") or parent.get("result") or {})
    how_id = extract_how_id(parent_plan, parent_contract, parent_metrics)
    seeds = extract_seeds(parent_plan, parent_contract, parent_metrics)
    frozen = extract_frozen_fields(parent_protocol, parent_plan, parent_contract)
    status = None
    parent_result = dict(parent.get("result") or {})
    if parent_result:
        status = str((parent_result.get("execution") or {}).get("status") or "") or None
    missing = not (how_id or seeds or frozen.get("dataset_id"))
    contract = machine_contract(how_id=how_id, seeds=seeds, frozen=frozen)
    contract["status"] = status
    contract["missing"] = missing
    contract["parent_run_id"] = (
        (bundle.get("plan") or {}).get("parent_run_id")
        or (bundle.get("run") or {}).get("parent_run_id")
        or parent.get("run_id")
    )
    return contract


def _protocol_view(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "editable_scope": list(protocol.get("editable_scope") or []),
        "frozen_scope": list(protocol.get("frozen_scope") or []),
        "may_change_dataset": False,
        "fingerprint_id": protocol.get("fingerprint_id"),
    }


def _gate_action(bundle: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    events = list(bundle.get("events") or [])
    gate = last_event(events, "gate_decision")
    humans = events_of_type(events, "human_governance")
    payload = dict((gate or {}).get("payload") or {})
    status = str(payload.get("status") or "").upper()
    training_eligible = True
    action = "approve"
    reason_type = None
    if status in {"REJECTED", "BLOCKED", "NEED_HUMAN", "HUMAN_REQUIRED"}:
        action = "reject"
        training_eligible = False
    confirmed = any(
        str((row.get("payload") or {}).get("action") or "")
        == "campaign_human_gate_confirmed"
        for row in humans
    )
    if confirmed and status in {"HUMAN_REQUIRED", "APPROVED", "NEED_HUMAN", ""}:
        action = "approve"
        status = status or "APPROVED"
        training_eligible = True
    need_human = next(
        (
            row
            for row in reversed(humans)
            if str((row.get("payload") or {}).get("action") or "")
            in {"NEED_HUMAN", "human_reject"}
        ),
        None,
    )
    if need_human and not confirmed:
        action = "reject"
        reason = str((need_human.get("payload") or {}).get("reason") or "")
        reason_type = "budget" if "budget" in reason.lower() else "governance"
        if reason_type == "budget":
            training_eligible = False
        status = status or "NEED_HUMAN"
    return action, {
        "action": action,
        "gate_status": status or None,
        "reason_type": reason_type,
    }, {
        "status": status or None,
        "training_eligible": training_eligible,
        "human_gate_confirmed": confirmed,
        "reasons": list(payload.get("reasons") or []),
    }


def _executed_contract(bundle: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(bundle.get("contract") or {})
    metrics = dict(bundle.get("metrics") or {})
    plan = dict(bundle.get("plan") or {})
    how_id = extract_how_id(contract, metrics, plan)
    seeds = extract_seeds(metrics, contract, plan)
    source = "contract"
    if extract_how_id(contract):
        source = "contract"
    elif extract_how_id(metrics):
        source = "metrics"
    elif extract_how_id(plan):
        source = "plan_fallback"
    out = machine_contract(how_id=how_id, seeds=seeds, frozen=frozen)
    out["source"] = source
    return out


def build_steps(
    bundle: Mapping[str, Any],
    parent: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = list(bundle.get("events") or [])
    reconstructed = bool(bundle.get("reconstructed"))
    plan = dict(bundle.get("plan") or {})
    protocol = dict(bundle.get("protocol") or {})
    contract = dict(bundle.get("contract") or {})
    result = dict(bundle.get("result") or {})
    review = dict(bundle.get("review") or {})
    claim_gate = dict(bundle.get("claim_gate") or {})
    run_doc = dict(bundle.get("run") or {})
    frozen = extract_frozen_fields(protocol, plan, contract)
    anchor = resolve_anchor(bundle, parent)
    planned = machine_contract(
        how_id=extract_how_id(plan, contract),
        seeds=extract_seeds(plan, contract),
        frozen=frozen,
    )
    executed = _executed_contract(bundle, frozen)
    literature = _literature(events)
    plan_event = last_event(events, "plan_proposal")
    gate_event = last_event(events, "gate_decision")
    exec_event = last_event(events, "execution")
    metrics_event = last_event(events, "metrics_parsed")
    review_event = last_event(events, "review_decision")
    memory_event = last_event(events, "memory_write")
    claim_event = last_event(events, "claim_gate")
    materialize_event = last_event(events, "materialize")

    if not plan_event:
        reconstructed = True
    plan_h = _decision_summary(plan, fallback=plan_event) or {
        "problem_observed": _clip_summary(str(plan.get("observation") or "")),
        "hypothesis": _clip_summary(str(plan.get("hypothesis") or "")),
        "selected_action": str(plan.get("how_id") or planned.get("how_id") or ""),
        "decision_basis": list(plan.get("controlled_variables") or [])[:6],
        "expected_effect": _clip_summary(
            str(((plan.get("expected_effect") or {}).get("rationale") or ""))
        ),
        "risk": str(plan.get("risk_level") or ""),
    }
    step1 = {
        "schema_version": SCHEMA_VERSION,
        "step_index": 1,
        "step_kind": "plan_proposal",
        "o": {
            "anchor": anchor,
            "protocol": _protocol_view(protocol),
            "literature": literature,
        },
        "h": plan_h,
        "a": {
            "tool": "propose_plan",
            "how_id": planned.get("how_id"),
            "how_zh": planned.get("how_zh"),
            "seeds": planned.get("seeds"),
            "stop": False,
        },
        "y": _plan_legal(plan),
        "r": None,
        "m": _meta(
            bundle,
            step_kind="plan_proposal",
            event=plan_event,
            reconstructed=bool(not plan_event),
            source_event_types=["plan_proposal"] if plan_event else [],
            actor_role="planner",
            phase="planning",
        ),
    }

    gate_action, gate_a, gate_y = _gate_action(bundle)
    human_event = last_event(events, "human_governance")
    gate_reasons = list(gate_y.get("reasons") or [])
    gate_h = _decision_summary(gate_event) or {
        "problem_observed": "计划已写出，等待协议闸门 / 人闸",
        "selected_action": gate_action,
        "decision_basis": [_clip_summary(str(x)) for x in gate_reasons][:6]
        or [str(gate_y.get("status") or "unobserved")],
        "risk": str(gate_y.get("reason_type") or ""),
    }
    step2 = {
        "schema_version": SCHEMA_VERSION,
        "step_index": 2,
        "step_kind": "gate_decision",
        "o": {
            "plan_contract": planned,
            "allowed_changes": list(contract.get("allowed_changes") or plan.get("modification_scope") or []),
            "frozen_variables": list(
                contract.get("frozen_variables") or protocol.get("frozen_scope") or []
            ),
        },
        "h": gate_h,
        "a": gate_a,
        "y": gate_y,
        "r": None,
        "m": _meta(
            bundle,
            step_kind="gate_decision",
            event=gate_event or human_event,
            reconstructed=bool(not gate_event and not human_event),
            source_event_types=[
                name
                for name, ev in (("gate_decision", gate_event), ("human_governance", human_event))
                if ev
            ],
            actor_role="gate",
            phase="planning",
        ),
    }

    primary = _primary_metric(bundle)
    primary_value = _metric_value(result, primary) or _metric_value(
        bundle.get("metrics"), primary
    )
    exec_payload = dict((exec_event or {}).get("payload") or {})
    metrics_payload = dict((metrics_event or {}).get("payload") or {})
    run_state = str(run_doc.get("run_state") or "") or None
    forged = bool(
        exec_payload.get("metrics_forged")
        or metrics_payload.get("metrics_forged")
        or False
    )
    adapter_how = ((contract.get("materialization") or {}).get("how") or {})
    adapter_h = None
    if adapter_how:
        adapter_h = {
            "problem_observed": "已批准合同，Adapter 翻译 HOW",
            "selected_action": str(adapter_how.get("how_id") or executed.get("how_id") or ""),
            "decision_basis": [
                f"fusion_method={adapter_how.get('fusion_method')}",
                f"neck_type={adapter_how.get('neck_type')}",
            ],
            "expected_effect": "按合同执行，不发明算子",
        }
    step3 = {
        "schema_version": SCHEMA_VERSION,
        "step_index": 3,
        "step_kind": "execution",
        "o": {
            "approved_contract": {
                **executed,
                "how_id": executed.get("how_id") or planned.get("how_id"),
                "seeds": executed.get("seeds") or planned.get("seeds"),
            }
        },
        "h": adapter_h,
        "a": {
            "tool": "run_experiment",
            "executed_how_id": executed.get("how_id"),
            "executed_how_zh": executed.get("how_zh"),
            "executed_seeds": executed.get("seeds"),
        },
        "y": {
            "observed": bool(exec_event or result or bundle.get("metrics")),
            "run_state": run_state,
            "execution_status": str(
                exec_payload.get("status")
                or ((result.get("execution") or {}).get("status") or "")
                or ""
            )
            or None,
            "primary_metric": primary,
            "primary_value": primary_value,
            "metrics_forged": forged,
            "APS_lowlight": _metric_value(result, "APS_lowlight")
            or _metric_value(bundle.get("metrics"), "APS_lowlight"),
        },
        "r": None,
        "m": _meta(
            bundle,
            step_kind="execution",
            event=exec_event or metrics_event,
            reconstructed=bool(not exec_event and not metrics_event and not result),
            source_event_types=[
                name
                for name, ev in (
                    ("execution", exec_event),
                    ("metrics_parsed", metrics_event),
                    ("materialize", materialize_event),
                )
                if ev
            ],
            actor_role="executor",
            phase="experiment",
        ),
    }

    before = _metric_value(bundle.get("baseline_metrics"), primary)
    if before is None and parent:
        before = _metric_value(parent.get("result"), primary) or _metric_value(
            parent.get("metrics"), primary
        )
    delta = None
    judgment = dict(review.get("primary_metric_judgment") or {})
    if judgment.get("delta") is not None:
        delta = judgment.get("delta")
        if before is None:
            before = judgment.get("before")
        if primary_value is None:
            primary_value = judgment.get("after")
    elif primary_value is not None and before is not None:
        delta = float(primary_value) - float(before)
    review_decision = str(
        review.get("review_decision")
        or ((review_event or {}).get("payload") or {}).get("review_decision")
        or run_doc.get("review_decision")
        or ""
    ).upper() or None
    lessons = [
        str(row.get("lesson_id"))
        for row in (review.get("research_lessons") or [])
        if isinstance(row, Mapping) and row.get("lesson_id")
    ]
    strategies = [
        str(row.get("strategy_id"))
        for row in (review.get("strategies") or [])
        if isinstance(row, Mapping) and row.get("strategy_id")
    ]
    review_h = _decision_summary(review_event, fallback=review) or {
        "problem_observed": "读到本轮结果，对照上一轮",
        "selected_action": review_decision or "",
        "decision_basis": [_clip_summary(str(review.get("reasoning_summary") or ""))],
        "risk": "KEEP ≠ Claim",
    }
    step4 = {
        "schema_version": SCHEMA_VERSION,
        "step_index": 4,
        "step_kind": "review_decision",
        "o": {
            "primary_metric": primary,
            "after": primary_value,
            "before": before,
            "delta": delta,
            "execution_matched_plan": (
                str(executed.get("how_id") or "") == str(planned.get("how_id") or "")
                and list(executed.get("seeds") or []) == list(planned.get("seeds") or [])
            ),
        },
        "h": review_h,
        "a": {"action": review_decision},
        "y": {
            "review_decision": review_decision,
            "hypothesis_status": review.get("hypothesis_status"),
            "lesson_ids": lessons,
            "strategy_ids": strategies,
            "claim_gate_status": claim_gate.get("status"),
            "keep_is_not_claim": True,
            "memory_written": bool(memory_event),
        },
        "r": None,
        "m": _meta(
            bundle,
            step_kind="review_decision",
            event=review_event or memory_event or claim_event,
            reconstructed=bool(not review_event and not review),
            source_event_types=[
                name
                for name, ev in (
                    ("review_decision", review_event),
                    ("memory_write", memory_event),
                    ("claim_gate", claim_event),
                )
                if ev
            ],
            actor_role="reviewer",
            phase="review",
        ),
    }
    steps = [step1, step2, step3, step4]
    if reconstructed:
        for step in steps:
            step["m"]["reconstructed"] = True
    for step in steps:
        validate_named("trajectory_step", step)
        if step["r"] is not None:
            raise RuntimeError("live trajectory export must leave r=null")
    scores_in_early = json.dumps({"o": step1["o"], "a": step1["a"]}, ensure_ascii=False)
    if "APS_lowlight" in scores_in_early or '"APS"' in scores_in_early:
        raise RuntimeError("detection scores leaked into plan_proposal o/a")
    return steps


def _raw_round(
    bundle: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    step1 = steps[0]
    step2 = steps[1]
    step3 = steps[2]
    frozen = extract_frozen_fields(
        bundle.get("protocol") or {},
        bundle.get("plan") or {},
        bundle.get("contract") or {},
    )
    anchor = dict(step1["o"].get("anchor") or {})
    plan = machine_contract(
        how_id=str(step1["a"].get("how_id") or ""),
        seeds=list(step1["a"].get("seeds") or []),
        frozen=frozen,
        stop=bool(step1["a"].get("stop")),
    )
    executed = machine_contract(
        how_id=str(step3["a"].get("executed_how_id") or ""),
        seeds=list(step3["a"].get("executed_seeds") or []),
        frozen=frozen,
    )
    gate_y = dict(step2.get("y") or {})
    human_gate = None
    if gate_y.get("status") in {"NEED_HUMAN", "REJECTED", "BLOCKED"} and not gate_y.get(
        "human_gate_confirmed"
    ):
        human_gate = {
            "decision": "reject",
            "reason_type": (step2.get("a") or {}).get("reason_type") or "governance",
            "training_eligible": bool(gate_y.get("training_eligible")),
        }
    rationale = str(((bundle.get("plan") or {}).get("hypothesis") or ""))
    plan["rationale"] = rationale
    return {
        "reconstructed": any(bool((row.get("m") or {}).get("reconstructed")) for row in steps),
        "anchor": None if anchor.get("missing") else anchor,
        "plan": plan,
        "executed": executed if step3["y"].get("observed") else {},
        "human_gate": human_gate,
        "tested_fingerprints": [],
        "parent_run_id": (parent or {}).get("run_id") or anchor.get("parent_run_id"),
    }


def build_story(
    bundle: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = _raw_round(bundle, steps, parent)
    label = label_contrast(raw)
    bucket = bucket_from_label(label)
    reconstructed = bool(raw.get("reconstructed"))
    gate_y = dict(steps[1].get("y") or {})
    training_eligible = True
    if reconstructed:
        training_eligible = False
    if raw.get("human_gate") and raw["human_gate"].get("decision") == "reject":
        training_eligible = False
        bucket = "drop"
    if not steps[2]["y"].get("observed"):
        training_eligible = False
    executed = dict(raw.get("executed") or {})
    plan = dict(raw.get("plan") or {})
    if executed and (
        executed.get("how_id") != plan.get("how_id")
        or list(executed.get("seeds") or []) != list(plan.get("seeds") or [])
    ):
        training_eligible = False
    if bucket != "sft_positive":
        high_trust_sft = False
    else:
        high_trust_sft = training_eligible
    anchor = dict(raw.get("anchor") or {})
    see = (
        f"上一轮已经用「{how_zh(anchor.get('how_id'))}」（{anchor.get('how_id') or '未知'}）"
        f"跑过了，随机种子 {anchor.get('seeds') or '未记录'}。"
        f"数据 {anchor.get('dataset_id') or '未写'}，切片 {anchor.get('slice_id') or '未写'}，"
        f"预算档 {anchor.get('budget_class') or '未写'}。"
        "协议规定不能改数据集/切片。只能从已接好的做法里做单变量对照。"
    )
    if not raw.get("anchor"):
        see = (
            "没有上一轮已执行合同（缺比较基准）。"
            "协议仍禁止改数据集。编号必须带人话，不能只背 F0/F1。"
        )
    decide = (
        f"决定改成「{how_zh(plan.get('how_id'))}」（{plan.get('how_id') or '未知'}），"
        f"种子 {plan.get('seeds') or '未写'}。"
    )
    if label == "idle":
        decide = (
            f"决定仍用「{how_zh(plan.get('how_id'))}」（{plan.get('how_id')}），"
            f"种子 {plan.get('seeds')}。这是空转，不是对照。"
        )
    elif label == "seed_contrast":
        decide = (
            f"决定保持「{how_zh(plan.get('how_id'))}」（{plan.get('how_id')}），"
            f"只换种子为 {plan.get('seeds')}。"
        )
    review_action = (steps[3].get("a") or {}).get("action") or "未评审"
    matched = bool((steps[3].get("o") or {}).get("execution_matched_plan"))
    happen = (
        f"实际执行 {executed.get('how_id') or '未跑'}/{executed.get('seeds') or []}，"
        f"与计划{'一致' if matched else '不一致'}。"
        f"评审 {review_action}。"
        f"声称闸门 {(steps[3].get('y') or {}).get('claim_gate_status') or '未写'}"
        "（KEEP ≠ 声称成立）。"
    )
    reward = REWARD_BY_LABEL.get(label)
    return {
        "exporter_version": EXPORTER_VERSION,
        "reward_export_version": REWARD_EXPORT_VERSION,
        "run_id": bundle.get("run_id"),
        "reconstructed": reconstructed,
        "contrast_label": label,
        "bucket": bucket,
        "training_eligible": training_eligible,
        "high_trust_sft": high_trust_sft,
        "machine": {
            "anchor": raw.get("anchor"),
            "plan": {k: v for k, v in plan.items() if k != "rationale"},
            "executed": executed or None,
        },
        "story": {"see": see, "decide": decide, "happen": happen},
        "reward_export": {
            "version": REWARD_EXPORT_VERSION,
            "r": reward,
            "note": "contrast rule score; not APS / mAP",
        },
        "gate": {
            "training_eligible": gate_y.get("training_eligible"),
            "human_gate_confirmed": gate_y.get("human_gate_confirmed"),
            "status": gate_y.get("status"),
        },
    }


def _sft_record(story: Mapping[str, Any]) -> dict[str, Any]:
    machine = dict(story.get("machine") or {})
    return {
        "id": f"sft_{story.get('run_id')}",
        "source_run_id": story.get("run_id"),
        "contrast_label": story.get("contrast_label"),
        "reconstructed": bool(story.get("reconstructed")),
        "input": (story.get("story") or {}).get("see"),
        "output": {
            "text": (story.get("story") or {}).get("decide"),
            "contract": machine.get("plan"),
        },
    }


def _dpo_record(story: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"dpo_{story.get('run_id')}",
        "source_run_id": story.get("run_id"),
        "prompt": (story.get("story") or {}).get("see"),
        "rejected": {
            "text": (story.get("story") or {}).get("decide"),
            "contract": ((story.get("machine") or {}).get("plan")),
            "contrast_label": story.get("contrast_label"),
        },
        "chosen": None,
        "pair_complete": False,
        "note": "same-anchor positive pair filled later in the batch if available",
    }


def _rl_record(story: Mapping[str, Any]) -> dict[str, Any]:
    reward = dict(story.get("reward_export") or {})
    return {
        "id": f"rl_{story.get('run_id')}",
        "source_run_id": story.get("run_id"),
        "prompt": (story.get("story") or {}).get("see"),
        "constraint_reward": reward.get("r"),
        "reward_version": reward.get("version"),
        "contrast_label": story.get("contrast_label"),
        "note": "do not use detection scores as reward",
    }


def _pair_dpo(stories: Sequence[Mapping[str, Any]], dpo_rows: list[dict[str, Any]]) -> None:
    positives: dict[str, Mapping[str, Any]] = {}
    for story in stories:
        if not story.get("high_trust_sft"):
            continue
        anchor = dict((story.get("machine") or {}).get("anchor") or {})
        key = json.dumps(
            {
                "how_id": anchor.get("how_id"),
                "seeds": anchor.get("seeds"),
                "dataset_id": anchor.get("dataset_id"),
                "slice_id": anchor.get("slice_id"),
                "budget_class": anchor.get("budget_class"),
            },
            sort_keys=True,
        )
        positives[key] = story
    for row in dpo_rows:
        story = next(
            (item for item in stories if item.get("run_id") == row.get("source_run_id")),
            None,
        )
        if not story:
            continue
        anchor = dict((story.get("machine") or {}).get("anchor") or {})
        key = json.dumps(
            {
                "how_id": anchor.get("how_id"),
                "seeds": anchor.get("seeds"),
                "dataset_id": anchor.get("dataset_id"),
                "slice_id": anchor.get("slice_id"),
                "budget_class": anchor.get("budget_class"),
            },
            sort_keys=True,
        )
        chosen = positives.get(key)
        if not chosen:
            continue
        row["chosen"] = {
            "text": (chosen.get("story") or {}).get("decide"),
            "contract": ((chosen.get("machine") or {}).get("plan")),
            "contrast_label": chosen.get("contrast_label"),
            "source_run_id": chosen.get("run_id"),
        }
        row["pair_complete"] = True


def export_run(
    run_dir: Path,
    export_root: Path,
    *,
    parent: Mapping[str, Any] | None = None,
    parent_dir: Path | None = None,
    include_training: bool = True,
) -> dict[str, Any]:
    bundle = load_run_bundle(run_dir)
    parent_bundle = parent
    if parent_bundle is None and parent_dir is not None:
        parent_bundle = load_run_bundle(parent_dir)
    steps = build_steps(bundle, parent_bundle)
    story = build_story(bundle, steps, parent_bundle)
    run_id = str(bundle.get("run_id") or Path(run_dir).name)
    traces_path = Path(export_root) / "traces" / f"{run_id}.jsonl"
    story_path = Path(export_root) / "stories" / f"{run_id}.json"
    reward_path = Path(export_root) / "reward_export" / f"{run_id}.json"
    _write_jsonl(traces_path, steps)
    _write_json(story_path, story)
    _write_json(
        reward_path,
        {
            "run_id": run_id,
            "version": REWARD_EXPORT_VERSION,
            "r": (story.get("reward_export") or {}).get("r"),
            "contrast_label": story.get("contrast_label"),
            "source_trace": str(traces_path),
            "note": "sidecar only; research_events.jsonl is unchanged",
        },
    )
    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    rl: list[dict[str, Any]] = []
    if include_training:
        if story.get("high_trust_sft"):
            sft.append(_sft_record(story))
        if story.get("bucket") == "dpo_rejected" and story.get("training_eligible") is not False:
            if not story.get("reconstructed"):
                dpo.append(_dpo_record(story))
        if story.get("reward_export", {}).get("r") is not None and not story.get("reconstructed"):
            rl.append(_rl_record(story))
        leaks = training_leaks_holdout(sft + dpo + rl)
        if leaks:
            raise ValueError(f"holdout exam leaked into training export: {leaks}")
        if sft:
            _write_jsonl(Path(export_root) / "sft" / f"{run_id}.jsonl", sft)
        if dpo:
            _write_jsonl(Path(export_root) / "dpo" / f"{run_id}.jsonl", dpo)
        if rl:
            _write_jsonl(Path(export_root) / "rl" / f"{run_id}.jsonl", rl)
    return {
        "run_id": run_id,
        "run_dir": str(Path(run_dir)),
        "reconstructed": bool(story.get("reconstructed")),
        "contrast_label": story.get("contrast_label"),
        "bucket": story.get("bucket"),
        "high_trust_sft": bool(story.get("high_trust_sft")),
        "traces": str(traces_path),
        "story": str(story_path),
        "n_steps": len(steps),
        "sft": sft,
        "dpo": dpo,
        "rl": rl,
        "events_source": "research_events.jsonl" if not bundle.get("reconstructed") else "reconstructed",
    }


def export_runs(
    run_dirs: Sequence[Path],
    export_root: Path,
    *,
    parent_dir: Path | None = None,
    include_training: bool = True,
) -> dict[str, Any]:
    export_root = Path(export_root)
    bundles = [load_run_bundle(path) for path in run_dirs]
    by_id = {str(row.get("run_id")): row for row in bundles}
    reports = []
    sft_all: list[dict[str, Any]] = []
    dpo_all: list[dict[str, Any]] = []
    rl_all: list[dict[str, Any]] = []
    stories: list[dict[str, Any]] = []
    for path, bundle in zip(run_dirs, bundles):
        parent = None
        parent_run_id = str(
            (bundle.get("plan") or {}).get("parent_run_id")
            or (bundle.get("run") or {}).get("parent_run_id")
            or ""
        )
        if parent_run_id and parent_run_id in by_id:
            parent = by_id[parent_run_id]
        report = export_run(
            path,
            export_root,
            parent=parent,
            parent_dir=parent_dir if parent is None else None,
            include_training=include_training,
        )
        story = _read_json(Path(report["story"])) or {}
        stories.append(story)
        reports.append({k: v for k, v in report.items() if k not in {"sft", "dpo", "rl"}})
        sft_all.extend(report.get("sft") or [])
        dpo_all.extend(report.get("dpo") or [])
        rl_all.extend(report.get("rl") or [])
    _pair_dpo(stories, dpo_all)
    leaks = training_leaks_holdout(sft_all + dpo_all + rl_all)
    if leaks:
        raise ValueError(f"holdout exam leaked into training export: {leaks}")
    if include_training:
        _write_jsonl(export_root / "sft" / "batch.jsonl", sft_all)
        _write_jsonl(export_root / "dpo" / "batch.jsonl", dpo_all)
        _write_jsonl(export_root / "rl" / "prompts.jsonl", rl_all)
    manifest = {
        "exporter_version": EXPORTER_VERSION,
        "reward_export_version": REWARD_EXPORT_VERSION,
        "exported_at": _now(),
        "runs": reports,
        "counts": {
            "runs": len(reports),
            "sft": len(sft_all),
            "dpo": len(dpo_all),
            "rl": len(rl_all),
        },
        "holdout_isolated": True,
        "note": (
            "SFT input is step-1 observation only; do not concatenate four steps. "
            "r in traces is null. Contrast reward is sidecar-only."
        ),
    }
    _write_json(export_root / "manifests" / "batch.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export ATDP six-tuple traces from Manager run directories (read-only)"
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        dest="run_dirs",
        help="Manager run directory (repeatable)",
    )
    parser.add_argument("--parent-dir", type=Path, default=None)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Export root. Default: <run-dir>/export when a single run is given.",
    )
    parser.add_argument(
        "--traces-only",
        action="store_true",
        help="Write traces/stories/reward_export only; skip SFT/DPO/RL cuts",
    )
    args = parser.parse_args(argv)
    run_dirs = list(args.run_dirs or [])
    if not run_dirs:
        parser.error("at least one --run-dir is required")
    export_dir = args.export_dir
    if export_dir is None:
        if len(run_dirs) != 1:
            parser.error("--export-dir is required when exporting multiple run directories")
        export_dir = run_dirs[0] / "export"
    manifest = export_runs(
        run_dirs,
        export_dir,
        parent_dir=args.parent_dir,
        include_training=not args.traces_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
