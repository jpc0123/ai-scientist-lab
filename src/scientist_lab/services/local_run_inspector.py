"""Read-only local run-dir inspector for the Demo UI.

Opens existing Manager / LLM-loop packs under ``.run/`` and
``tests/fixtures/``. Never forges GPU metrics. Never writes into git.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ALLOWED_REL_PREFIXES = (".run", "tests/fixtures")
SKIP_DIR_NAMES = {
    "_web_actions",
    "node_modules",
    "__pycache__",
    "run",
    "memory",
}
JSON_ALLOWLIST = (
    "protocol.json",
    "plan.json",
    "previous_plan.json",
    "result.json",
    "review.json",
    "claim_gate.json",
    "claim_gate_c1.json",
    "loop_report.json",
    "experiment_run.json",
    "contract.json",
    "SOURCE.json",
    "baseline_metrics.json",
    "formal_01_baseline_metrics.json",
    "manager_status.json",
    "PAUSE.json",
    "handle.json",
    "memory/research_memory.json",
    "memory/strategy_memory.json",
    "memory/research_trace.json",
)

CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "formal_c1_aps_early_concat",
        "relpath": ".run/formal_c1_aps_early_concat",
        "inspect_subdir": "candidate_resume",
        "kind": "formal_c1",
        "role": "formal_c1_pair",
        "title": "Formal C1 · early_concat vs rgb+none",
        "blurb": "唯一允许展示 C1 对照数字的 pack。KEEP ≠ Claim。",
        "gitignored": True,
    },
    {
        "id": "v25d_llm_real_loop",
        "relpath": ".run/v25d_llm_real_loop",
        "inspect_subdir": "round",
        "kind": "llm_loop",
        "role": "probe_loop",
        "title": "v2.5-D LLM 工程闭环（probe）",
        "blurb": "LLM Planner → Gate → Adapter HOW → 可选 GPU。probe APS≠科学声称。",
        "gitignored": True,
    },
    {
        "id": "fixture_m4_rounds3_discard",
        "relpath": "tests/fixtures/llm_plan_replay/m4_rounds3_discard",
        "inspect_subdir": "",
        "kind": "fixture",
        "role": "fixture_discard",
        "title": "Fixture · M4 Round1 DISCARD",
        "blurb": "历史 DISCARD 证据包。baseline APS=0.6 是 synthetic_control，不是本轮 GPU。",
        "gitignored": False,
    },
    {
        "id": "fixture_discard_stub",
        "relpath": "tests/fixtures/llm_plan_replay/discard_stub",
        "inspect_subdir": "",
        "kind": "fixture",
        "role": "fixture_stub",
        "title": "Fixture · DISCARD stub",
        "blurb": "离线 replay 用的精简 stub，不含 GPU 产物。",
        "gitignored": False,
    },
)

LOOP_STAGE_IDS = (
    "protocol",
    "gate",
    "run",
    "evidence",
    "rubric",
    "memory",
    "next_plan",
)


def _posix(rel: str) -> str:
    return str(rel).replace("\\", "/").strip("/")


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _round_aps(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def resolve_project_root(root: Path | str) -> Path:
    return Path(root).resolve()


def is_allowed_relpath(relpath: str) -> bool:
    rel = _posix(relpath)
    if not rel or rel.startswith("/") or ":" in rel.split("/")[0]:
        return False
    parts = Path(rel).parts
    if ".." in parts:
        return False
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in ALLOWED_REL_PREFIXES)


def resolve_allowed_path(root: Path, relpath: str) -> Path:
    if not is_allowed_relpath(relpath):
        raise PermissionError(f"path outside allowlist: {relpath}")
    full = (root / Path(_posix(relpath))).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise PermissionError(f"path escapes project root: {relpath}") from exc
    return full


def _inspect_root_for(pack: Path, inspect_subdir: str = "") -> Path:
    if inspect_subdir:
        nested = pack / inspect_subdir
        if nested.is_dir():
            return nested
    if (pack / "protocol.json").is_file() or (pack / "plan.json").is_file():
        return pack
    for name in ("round", "candidate_resume", "candidate"):
        nested = pack / name
        if nested.is_dir() and (
            (nested / "protocol.json").is_file() or (nested / "plan.json").is_file()
        ):
            return nested
    return pack


def _present_files(pack: Path, inspect_root: Path) -> list[str]:
    found: list[str] = []
    for name in JSON_ALLOWLIST:
        if (inspect_root / name).is_file() or (pack / name).is_file():
            found.append(name)
    return found


def _how_from(contract: Mapping[str, Any], loop_report: Mapping[str, Any]) -> dict[str, Any]:
    how = _as_dict((_as_dict(contract.get("materialization")).get("how")))
    if not how:
        how = _as_dict(loop_report.get("how"))
    if not how:
        return {}
    return {
        "primary_module": how.get("primary_module"),
        "input_mode": how.get("input_mode"),
        "fusion_method": how.get("fusion_method"),
        "neck_type": how.get("neck_type"),
        "existing_capability": how.get("existing_capability"),
        "invented_operators": _as_list(how.get("invented_operators")),
        "gap": how.get("gap"),
        "budget_class": how.get("budget_class"),
        "execution_mode": how.get("execution_mode"),
        "claim_level": how.get("claim_level"),
        "label": _how_label(how),
    }


def _how_label(how: Mapping[str, Any]) -> str:
    module = str(how.get("primary_module") or "")
    fusion = str(how.get("fusion_method") or "")
    neck = str(how.get("neck_type") or "")
    if module == "fusion" and fusion:
        return f"fusion / {fusion}"
    if module == "neck" and neck:
        return f"neck / {neck}"
    if module:
        return module
    if fusion:
        return f"fusion / {fusion}"
    return "unknown"


def _candidates(plan: Mapping[str, Any]) -> list[Any]:
    direct = _as_list(plan.get("candidate_experiments"))
    if direct:
        return direct
    parsed = _as_dict(_as_dict(plan.get("llm_trace")).get("parsed"))
    return _as_list(parsed.get("candidates"))


def _lessons(memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    lessons = memory.get("lessons")
    if isinstance(lessons, Mapping):
        rows = list(lessons.values())
    elif isinstance(lessons, list):
        rows = lessons
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for row in rows[:12]:
        item = _as_dict(row)
        out.append(
            {
                "lesson_id": item.get("lesson_id"),
                "type": item.get("type"),
                "statement": item.get("statement"),
                "status": item.get("status"),
            }
        )
    return out


def _gate_from(
    loop_report: Mapping[str, Any],
    experiment_run: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _as_dict(loop_report.get("gate"))
    if gate.get("status"):
        return {
            "status": str(gate.get("status")),
            "reasons": _as_list(gate.get("reasons")),
            "source": "loop_report",
        }
    run_state = str(experiment_run.get("run_state") or "")
    approval = str(contract.get("approval_status") or "")
    if run_state == "BLOCKED":
        return {"status": "REJECTED", "reasons": ["run_state=BLOCKED"], "source": "experiment_run"}
    if run_state and run_state not in {"CREATED", "PLANNED"}:
        return {
            "status": "APPROVED",
            "reasons": [f"run_state={run_state}"],
            "source": "experiment_run",
        }
    if approval in {"approved", "candidate"}:
        return {
            "status": "APPROVED" if approval == "approved" else "CANDIDATE",
            "reasons": [f"approval_status={approval}"],
            "source": "contract",
        }
    return {"status": "UNKNOWN", "reasons": ["no gate artifact"], "source": None}


def _stage(
    stage_id: str,
    *,
    status: str,
    summary: str,
    ready: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": stage_id,
        "status": status,
        "summary": summary,
        "ready": ready,
    }
    if extra:
        row.update(dict(extra))
    return row


def _build_stages(
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    gate: Mapping[str, Any],
    result: Mapping[str, Any],
    experiment_run: Mapping[str, Any],
    review: Mapping[str, Any],
    memory_pack: Mapping[str, Any],
    claim: Mapping[str, Any],
    loop_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protocol_id = protocol.get("protocol_id") or protocol.get("title")
    budget = plan.get("budget_class") or protocol.get("experiment_budget")
    exec_status = str(_as_dict(result.get("execution")).get("status") or "")
    evidence_status = str(experiment_run.get("evidence_status") or "")
    review_decision = str(review.get("review_decision") or "")
    lessons = _lessons(memory_pack)
    next_summary = _as_dict(plan.get("decision_summary"))
    selected = next_summary.get("selected_action") or plan.get("modification_scope")
    memory_refs = _as_dict(plan.get("memory_refs") or loop_report.get("memory_refs"))

    return [
        _stage(
            "protocol",
            status="ready" if protocol else "missing",
            summary=str(protocol_id or "protocol.json 缺失"),
            ready=bool(protocol),
            extra={"budget_class": budget, "protocol_id": protocol.get("protocol_id")},
        ),
        _stage(
            "gate",
            status=str(gate.get("status") or "UNKNOWN"),
            summary="; ".join(str(x) for x in _as_list(gate.get("reasons"))[:3])
            or str(gate.get("status") or "未知"),
            ready=str(gate.get("status") or "") not in {"", "UNKNOWN", "REJECTED"},
            extra={"source": gate.get("source")},
        ),
        _stage(
            "run",
            status=exec_status or str(experiment_run.get("run_state") or "missing"),
            summary=str(
                experiment_run.get("run_id")
                or result.get("run_id")
                or "尚无 result.json"
            ),
            ready=bool(result) or bool(experiment_run),
            extra={"run_state": experiment_run.get("run_state")},
        ),
        _stage(
            "evidence",
            status=evidence_status or ("present" if result else "missing"),
            summary=_evidence_summary(result, experiment_run),
            ready=bool(result),
            extra={
                "raw_metric_refs": _as_list(result.get("raw_metric_refs")),
                "artifact_paths": _as_list(_as_dict(result.get("artifacts")).get("paths")),
            },
        ),
        _stage(
            "rubric",
            status=review_decision or "missing",
            summary=str(review.get("reasoning_summary") or review_decision or "尚无 review.json"),
            ready=bool(review_decision),
            extra={
                "keep_is_not_claim": True,
                "hypothesis_status": review.get("hypothesis_status"),
            },
        ),
        _stage(
            "memory",
            status="written" if lessons else ("present" if memory_pack else "missing"),
            summary=(
                f"{len(lessons)} lessons"
                if lessons
                else ("Memory 目录存在" if memory_pack else "尚无 Memory")
            ),
            ready=bool(lessons) or bool(memory_pack),
            extra={"lesson_ids": [row.get("lesson_id") for row in lessons]},
        ),
        _stage(
            "next_plan",
            status="ready" if plan else "missing",
            summary=str(selected or plan.get("plan_id") or "尚无下一轮 Plan"),
            ready=bool(plan),
            extra={
                "memory_refs": memory_refs,
                "has_decision_summary": bool(next_summary),
            },
        ),
    ]


def _evidence_summary(result: Mapping[str, Any], experiment_run: Mapping[str, Any]) -> str:
    metrics = _as_dict(result.get("metrics"))
    aps = metrics.get("APS")
    status = experiment_run.get("evidence_status") or _as_dict(result.get("execution")).get(
        "status"
    )
    if aps is None:
        return str(status or "无 metrics")
    return f"APS={aps} · evidence={status or 'n/a'}"


def _c1_display(
    *,
    kind: str,
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
    review: Mapping[str, Any],
    baseline: Mapping[str, Any],
    pack_claim: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    run_level = str(
        pack_claim.get("run_level") or claim.get("run_level") or ""
    ).lower()
    budget_hint = str(claim.get("budget_class") or "")
    is_probe = kind in {"llm_loop", "fixture"} or run_level == "probe" or budget_hint == "probe"
    is_formal = kind == "formal_c1" or run_level == "formal"

    candidate_aps = _as_dict(result.get("metrics")).get("APS")
    if candidate_aps is None:
        candidate_aps = _as_dict(_as_dict(review.get("objective_check")).get("APS")).get(
            "current"
        )
    baseline_aps = baseline.get("APS")
    if baseline_aps is None:
        baseline_aps = _as_dict(_as_dict(review.get("objective_check")).get("APS")).get(
            "baseline"
        )

    verdict = pack_claim if pack_claim.get("status") else claim
    status = str(verdict.get("status") or "")

    if is_probe:
        aps_zero = False
        try:
            aps_zero = float(candidate_aps) == 0.0
        except (TypeError, ValueError):
            aps_zero = candidate_aps in {0, 0.0, "0", "0.0"}
        note = source.get("note") or (
            "probe APS 只证明工程闭环可走通，不是 C1 科学声称。"
        )
        if aps_zero:
            note = "probe APS=0：这是工程闭环，不是科学声称。ClaimGate 不得标 SUPPORTED。"
        return {
            "allowed": False,
            "engineering_not_claim": True,
            "aps_is_zero": aps_zero,
            "probe_aps": candidate_aps,
            "probe_aps_display": _round_aps(candidate_aps),
            "claim_status": status or "BLOCKED",
            "show_supported": False,
            "warning": note,
        }

    if not is_formal:
        return {
            "allowed": False,
            "engineering_not_claim": True,
            "show_supported": False,
            "warning": "非 Formal C1 对照，不展示声称数字。",
        }

    show_supported = status == "SUPPORTED" and run_level == "formal"
    return {
        "allowed": True,
        "engineering_not_claim": False,
        "baseline_aps": baseline_aps,
        "candidate_aps": candidate_aps,
        "baseline_aps_display": _round_aps(baseline_aps),
        "candidate_aps_display": _round_aps(candidate_aps),
        "claim_status": status,
        "claim_reason": verdict.get("reason"),
        "keep_is_not_claim": bool(verdict.get("keep_is_not_claim", True)),
        "review_decision": verdict.get("review_decision") or review.get("review_decision"),
        "show_supported": show_supported,
        "note": "Formal 对照：baseline rgb+none vs early_concat。KEEP 不是 Claim。",
    }


def _memory_pack(inspect_root: Path) -> dict[str, Any]:
    mem_dir = inspect_root / "memory"
    payload = {
        "research_memory": _read_json(mem_dir / "research_memory.json"),
        "strategy_memory": _read_json(mem_dir / "strategy_memory.json"),
        "research_trace": _read_json(mem_dir / "research_trace.json"),
    }
    research = _as_dict(payload["research_memory"])
    lessons = research.get("lessons") or research
    strategies = _as_dict(payload["strategy_memory"]).get("strategies")
    return {
        "lessons": lessons if isinstance(lessons, (Mapping, list)) else research,
        "strategies": strategies,
        "raw_present": {
            key: value is not None for key, value in payload.items()
        },
    }


def _entry_exists(root: Path, entry: Mapping[str, Any]) -> bool:
    pack = resolve_allowed_path(root, str(entry["relpath"]))
    return pack.exists()


def list_local_runs(root: Path | str) -> dict[str, Any]:
    project = resolve_project_root(root)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in CATALOG:
        pack_exists = False
        try:
            pack_exists = _entry_exists(project, entry)
        except PermissionError:
            pack_exists = False
        item = {
            **entry,
            "available": pack_exists,
            "relpath": _posix(str(entry["relpath"])),
        }
        items.append(item)
        seen.add(str(entry["id"]))

    run_root = project / ".run"
    if run_root.is_dir():
        for child in sorted(run_root.iterdir()):
            if not child.is_dir() or child.name in SKIP_DIR_NAMES or child.name.startswith("."):
                continue
            rid = child.name
            if rid in seen:
                continue
            inspect = _inspect_root_for(child)
            if not (inspect / "protocol.json").is_file() and not (inspect / "loop_report.json").is_file():
                if not (child / "loop_report.json").is_file() and not (
                    child / "claim_gate_c1.json"
                ).is_file():
                    continue
            items.append(
                {
                    "id": rid,
                    "relpath": f".run/{rid}",
                    "inspect_subdir": inspect.name if inspect != child else "",
                    "kind": "discovered",
                    "role": "discovered",
                    "title": rid,
                    "blurb": "本机 .run/ 发现的其它 pack（只读）。",
                    "gitignored": True,
                    "available": True,
                }
            )

    return {
        "items": items,
        "narrative": {
            "product": "Scientist Lab = 规则/状态驱动的可控自主实验系统 + v2.5 LLM 科研认知层",
            "not": ["更好的检测器", "完整自动发论文系统"],
            "loop": list(LOOP_STAGE_IDS),
            "claim": "KEEP ≠ Claim",
            "v25": "LLM Planner（WHAT/WHY）→ Plan schema → Gate → Adapter HOW →（可选）GPU；LLM Reviewer 只解释，不覆盖 Rubric",
        },
        "notes": [
            ".run/ 已被 gitignore，不会进入版本库。",
            "C1 数字只在 Formal 对照 pack 展示。",
            "probe APS=0 标注为工程闭环，不是科学声称。",
        ],
    }


def inspect_local_run(root: Path | str, run_id: str) -> dict[str, Any]:
    project = resolve_project_root(root)
    catalog = list_local_runs(project)
    entry = next((row for row in catalog["items"] if row.get("id") == run_id), None)
    if entry is None:
        raise FileNotFoundError(f"unknown local run: {run_id}")
    if not entry.get("available"):
        raise FileNotFoundError(f"local run not on disk: {entry.get('relpath')}")

    pack = resolve_allowed_path(project, str(entry["relpath"]))
    inspect_root = _inspect_root_for(pack, str(entry.get("inspect_subdir") or ""))
    protocol = _as_dict(_read_json(inspect_root / "protocol.json"))
    plan = _as_dict(
        _read_json(inspect_root / "plan.json") or _read_json(inspect_root / "previous_plan.json")
    )
    result = _as_dict(_read_json(inspect_root / "result.json"))
    review = _as_dict(_read_json(inspect_root / "review.json"))
    claim = _as_dict(_read_json(inspect_root / "claim_gate.json"))
    pack_claim = _as_dict(_read_json(pack / "claim_gate_c1.json"))
    loop_report = _as_dict(
        _read_json(pack / "loop_report.json") or _read_json(inspect_root / "loop_report.json")
    )
    experiment_run = _as_dict(_read_json(inspect_root / "experiment_run.json"))
    contract = _as_dict(_read_json(inspect_root / "contract.json"))
    source = _as_dict(_read_json(inspect_root / "SOURCE.json"))
    baseline = _as_dict(
        _read_json(pack / "formal_01_baseline_metrics.json")
        or _read_json(inspect_root / "baseline_metrics.json")
    )
    memory_pack = _memory_pack(inspect_root)
    how = _how_from(contract, loop_report)
    gate = _gate_from(loop_report, experiment_run, contract)
    decision_summary = _as_dict(plan.get("decision_summary") or loop_report.get("plan"))
    if not decision_summary.get("selected_action") and loop_report:
        decision_summary = _as_dict(plan.get("decision_summary"))
    memory_refs = _as_dict(plan.get("memory_refs") or loop_report.get("memory_refs"))
    candidates = _candidates(plan)
    c1 = _c1_display(
        kind=str(entry.get("kind") or ""),
        claim=claim,
        result=result,
        review=review,
        baseline=baseline,
        pack_claim=pack_claim,
        source=source,
    )
    llm_trace = _as_dict(plan.get("llm_trace"))
    if llm_trace.get("raw_output") and isinstance(llm_trace["raw_output"], str):
        raw = llm_trace["raw_output"]
        if len(raw) > 4000:
            llm_trace = {**llm_trace, "raw_output": raw[:4000] + "…"}

    claim_for_ui = pack_claim or claim
    if c1.get("engineering_not_claim") and str(claim_for_ui.get("status")) == "SUPPORTED":
        # Fail-closed display: never paint SUPPORTED on a probe pack.
        claim_for_ui = {**claim_for_ui, "status": "BLOCKED", "display_override": "probe_not_c1"}

    return {
        "id": run_id,
        "available": True,
        "catalog": entry,
        "paths": {
            "pack": str(pack),
            "inspect_root": str(inspect_root),
            "relpath": entry.get("relpath"),
        },
        "files": _present_files(pack, inspect_root),
        "loop": _build_stages(
            protocol=protocol,
            plan=plan,
            gate=gate,
            result=result,
            experiment_run=experiment_run,
            review=review,
            memory_pack=memory_pack,
            claim=claim_for_ui,
            loop_report=loop_report,
        ),
        "protocol": {
            "protocol_id": protocol.get("protocol_id"),
            "title": protocol.get("title"),
            "goal": protocol.get("goal"),
            "objective": protocol.get("objective"),
            "editable_scope": protocol.get("editable_scope"),
            "frozen_scope": protocol.get("frozen_scope"),
            "experiment_budget": protocol.get("experiment_budget"),
        },
        "plan": {
            "plan_id": plan.get("plan_id"),
            "hypothesis": plan.get("hypothesis"),
            "observation": plan.get("observation"),
            "modification_scope": plan.get("modification_scope"),
            "budget_class": plan.get("budget_class"),
            "rationale": plan.get("rationale"),
            "parent_run_id": plan.get("parent_run_id"),
            "source": plan.get("source") or llm_trace.get("backend"),
        },
        "gate": gate,
        "how": how,
        "run": {
            "run_id": experiment_run.get("run_id") or result.get("run_id"),
            "run_state": experiment_run.get("run_state"),
            "evidence_status": experiment_run.get("evidence_status"),
            "execution": result.get("execution"),
            "scientific_outcome": experiment_run.get("scientific_outcome"),
        },
        "evidence": {
            "metrics": result.get("metrics"),
            "raw_metric_refs": result.get("raw_metric_refs"),
            "artifacts": result.get("artifacts"),
            "source_note": source.get("note"),
            "gpu_metrics_aps": source.get("gpu_metrics_aps"),
            "control_baseline_aps": source.get("control_baseline_aps"),
        },
        "rubric": {
            "review_decision": review.get("review_decision"),
            "hypothesis_status": review.get("hypothesis_status"),
            "reasoning_summary": review.get("reasoning_summary"),
            "objective_check": review.get("objective_check"),
            "keep_is_not_claim": True,
        },
        "claim_gate": {
            "status": claim_for_ui.get("status"),
            "reason": claim_for_ui.get("reason"),
            "claim_text": claim_for_ui.get("claim_text"),
            "claim_strength": claim_for_ui.get("claim_strength"),
            "run_level": claim_for_ui.get("run_level"),
            "review_decision": claim_for_ui.get("review_decision"),
            "keep_is_not_claim": bool(claim_for_ui.get("keep_is_not_claim", True)),
            "scientific_outcome": claim_for_ui.get("scientific_outcome"),
            "in_run_status": claim.get("status"),
            "pack_status": pack_claim.get("status"),
        },
        "memory": {
            "refs": memory_refs,
            "lessons": _lessons(memory_pack),
        },
        "llm": {
            "backend": llm_trace.get("backend") or loop_report.get("planner_backend"),
            "provider": llm_trace.get("provider") or loop_report.get("reviewer_backend"),
            "decision_summary": _as_dict(plan.get("decision_summary")),
            "memory_refs": memory_refs,
            "candidate_experiments": candidates,
            "llm_live": loop_report.get("llm_live"),
            "reviewer_explains_only": True,
        },
        "c1": c1,
        "loop_report": {
            "present": bool(loop_report),
            "exam": loop_report.get("exam"),
            "evidence_class": loop_report.get("evidence_class"),
            "metrics_forged": loop_report.get("metrics_forged"),
            "gpu": loop_report.get("gpu"),
            "execute": loop_report.get("execute"),
            "llm_live": loop_report.get("llm_live"),
            "aps": loop_report.get("aps"),
            "claim_gate_not_c1_supported": loop_report.get("claim_gate_not_c1_supported"),
        } if loop_report else None,
        "actions_enabled": {
            "llm_plan_replay": (inspect_root / "protocol.json").is_file()
            and (
                (inspect_root / "plan.json").is_file()
                or (inspect_root / "previous_plan.json").is_file()
            ),
            "llm_review_replay": (inspect_root / "protocol.json").is_file()
            and (inspect_root / "review.json").is_file(),
            "manager_run": (inspect_root / "protocol.json").is_file()
            and (
                (inspect_root / "plan.json").is_file()
                or (inspect_root / "previous_plan.json").is_file()
            ),
        },
    }


def read_local_run_file(root: Path | str, run_id: str, name: str) -> dict[str, Any]:
    rel = _posix(name)
    if rel not in JSON_ALLOWLIST:
        raise PermissionError(f"file not on allowlist: {name}")
    payload = inspect_local_run(root, run_id)
    inspect_root = Path(payload["paths"]["inspect_root"])
    pack = Path(payload["paths"]["pack"])
    path = inspect_root / rel
    if not path.is_file():
        path = pack / rel
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(f"file not found: {name}")
    return {"name": rel, "path": str(path), "data": data}


def _copy_replay_source(inspect_root: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    names = (
        "protocol.json",
        "plan.json",
        "previous_plan.json",
        "result.json",
        "review.json",
        "baseline_metrics.json",
        "SOURCE.json",
        "experiment_run.json",
        "claim_gate.json",
        "contract.json",
    )
    for name in names:
        src = inspect_root / name
        if src.is_file():
            shutil.copyfile(src, dest / name)
    mem_src = inspect_root / "memory"
    if mem_src.is_dir():
        mem_dest = dest / "memory"
        mem_dest.mkdir(parents=True, exist_ok=True)
        for name in ("research_memory.json", "strategy_memory.json", "research_trace.json"):
            src = mem_src / name
            if src.is_file():
                shutil.copyfile(src, mem_dest / name)
    return dest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_local_action(
    root: Path | str,
    run_id: str,
    *,
    action: str,
    live: bool = False,
    execute: bool = False,
    confirm_live: bool = False,
    confirm_execute: bool = False,
    live_ready: bool | None = None,
    llm_ready: bool | None = None,
    provider: str = "mock",
) -> dict[str, Any]:
    """Replay / dry-run against an isolated copy. Default is mock + no GPU."""
    project = resolve_project_root(root)
    inspected = inspect_local_run(project, run_id)
    inspect_root = Path(inspected["paths"]["inspect_root"])
    enabled = inspected.get("actions_enabled") or {}
    if action not in {"llm_plan_replay", "llm_review_replay", "manager_run"}:
        raise ValueError(f"unknown action: {action}")
    if not enabled.get(action):
        raise ValueError(f"action not available for this pack: {action}")

    if live and not confirm_live:
        raise PermissionError("真实 --live 需要二次确认（confirm_live）")
    if execute and not confirm_execute:
        raise PermissionError("真实 --execute GPU 需要二次确认（confirm_execute）")
    if execute and live_ready is False:
        return {
            "ok": False,
            "fail_closed": True,
            "action": action,
            "execute": True,
            "live": bool(live),
            "gpu": False,
            "metrics_forged": False,
            "error": "cuda doctor live_ready=false; refusing GPU and forged metrics.json",
        }
    if live and llm_ready is False:
        return {
            "ok": False,
            "fail_closed": True,
            "action": action,
            "execute": False,
            "live": True,
            "gpu": False,
            "metrics_forged": False,
            "error": "LLM 未就绪（缺 Key / 未允许联网）；拒绝伪造成功",
        }

    work = project / ".run" / "_web_actions" / run_id / f"{action}_{_stamp()}"
    replica = _copy_replay_source(inspect_root, work / "source")
    report: dict[str, Any]

    if action == "llm_plan_replay":
        if execute:
            raise PermissionError("llm-plan-replay 永不 --execute / GPU")
        from scientist_lab.llm.plan_replay import run_llm_plan_replay

        report = run_llm_plan_replay(
            replica,
            live=bool(live),
            provider=provider if not live else "openai-compatible",
            output=work / "replay_report.json",
        )
    elif action == "llm_review_replay":
        if execute:
            raise PermissionError("llm-review-replay 永不 --execute / GPU")
        from scientist_lab.llm.review_replay import run_llm_review_replay

        report = run_llm_review_replay(
            replica,
            live=bool(live),
            provider=provider if not live else "openai-compatible",
            output=work / "replay_report.json",
        )
    else:
        from scientist_lab.core.manager_cli import run_manager_from_files

        protocol_path = replica / "protocol.json"
        plan_path = replica / "plan.json"
        if not plan_path.is_file():
            plan_path = replica / "previous_plan.json"
        baseline_path = replica / "baseline_metrics.json"
        report = run_manager_from_files(
            protocol_path,
            plan_path,
            output_dir=work / "manager",
            execute=bool(execute),
            require_live_ready=bool(execute),
            llm_live=bool(live),
            planner_backend="rules" if not live else "llm",
            reviewer_backend="rules" if not live else "llm",
            baseline_metrics_path=baseline_path if baseline_path.is_file() else None,
            max_steps=24,
        )
        report["metrics_forged"] = False

    report.setdefault("metrics_forged", False)
    report["work_dir"] = str(work)
    report["source_run_id"] = run_id
    report["action"] = action
    report["cli"] = {
        "llm_plan_replay": "scientist-lab llm-plan-replay --run-dir <dir>",
        "llm_review_replay": "scientist-lab llm-review-replay --run-dir <dir>",
        "manager_run": "scientist-lab manager-run --protocol <p> --plan <plan> --output-dir <out>",
    }.get(action)
    return report
