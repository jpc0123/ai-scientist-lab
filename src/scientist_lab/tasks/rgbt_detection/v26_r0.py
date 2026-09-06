"""v2.6 V26.4 R0 baseline anchor. Not an Agent. Does not invent APS_lowlight."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint
from scientist_lab.adapters.dfine.how import resolve_adapter_how
from scientist_lab.adapters.dfine.how_catalog import resolve_how_id
from scientist_lab.adapters.dfine.metrics_parser import parse_metrics
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.datasets.low_light_subset import SLICE_ID, rule_hash
from scientist_lab.metrics.aps_lowlight import PRIMARY_METRIC, SECONDARY_ALL, SECONDARY_SLICE, evaluate_aps_lowlight

ANCHOR_ID = "v26_r0_dfine_f1_lowlight"
R0_HOW_ID = "F1"
R0_PLAN_ID = "plan_v26_r0_dfine_f1"
R0_RUN_KEY = "r0_dfine_f1_lowlight_v1"
DATASET_ID = "rgbt_tiny_v1"
SCHEMA_VERSION = "1.0.0"
G2_BUDGET_CLASS = "formal"
FORBIDDEN_SUBSTITUTES = ("APS", "APS_all", "mAP50_95", "mAP", "mAP50")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SKIP_RUN_DIRS = frozenset({"_dfine_run", "_dfine_stage", "__pycache__"})

DEFAULT_FREEZE_PATH = _REPO_ROOT / "docs" / "research" / "v26" / "R0_BASELINE_FREEZE.json"
DEFAULT_SLICE_FREEZE_PATH = _REPO_ROOT / "docs" / "research" / "v26" / "LOW_LIGHT_SUBSET_V1_FREEZE.json"
DEFAULT_VAL_GT_PATH = (
    _REPO_ROOT / "datasets" / "registered" / "rgbt_tiny_v1" / "annotations" / "instances_val.json"
)
DEFAULT_METRIC_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4] / "docs" / "research" / "v26" / "R0_METRIC_CONTRACT.json"
)
DEFAULT_PROTOCOL_PATH = SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json"
DEFAULT_PLAN_PATH = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json"


class R0BaselineError(ValueError):
    """R0 identity or metrics cannot be accepted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def r0_how() -> dict[str, Any]:
    spec = resolve_how_id(R0_HOW_ID)
    return {
        "how_id": spec["id"],
        "family": spec["family"],
        "primary_module": spec["primary_module"],
        "input_mode": spec["input_mode"],
        "fusion_method": spec["fusion_method"],
        "neck_type": spec["neck_type"],
        "existing_capability": spec["existing_capability"],
        "hidden_control": "Not A4 (F1+N1). N1 is not the R0 neck. R0 seed stays F1.",
    }


def metric_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": "v26_r0_aps_lowlight",
        "task": "rgbt_object_detection",
        "dataset_id": DATASET_ID,
        "slice_id": SLICE_ID,
        "split": "val",
        "primary": {"metric": PRIMARY_METRIC, "direction": "maximize"},
        "secondary_slice": [{"metric": name, "direction": "maximize"} for name in SECONDARY_SLICE],
        "secondary_all": [{"metric": name, "direction": "maximize"} for name in SECONDARY_ALL],
        "evaluator": "coco_ap_small_on_low_light_subset_v1",
        "required_keys": [PRIMARY_METRIC],
        "forbidden_substitutes": list(FORBIDDEN_SUBSTITUTES),
        "official_labels": False,
        "llm_may_rewrite": False,
        "note": (
            "G2 compares APS_lowlight on frozen low_light_subset_v1 val. "
            "Full-set APS / mAP cannot fill this key. v2.5 A4 and v2.5-D probe APS=0.0 are not R0."
        ),
    }


def default_r0_plan() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": R0_PLAN_ID,
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "protocol_version": 1,
        "parent_run_id": None,
        "round_index": 0,
        "observation": (
            "v2.6 needs a D-FINE R0 on the frozen low-light slice before Live LLM rounds. "
            "This is the trusted RGB-T starting point, not an LLM discovery."
        ),
        "hypothesis": (
            "early_concat (F1) with the standard neck is the v2.6 R0 comparator for APS_lowlight."
        ),
        "modification_scope": ["fusion"],
        "proposed_changes": [
            {
                "target": "fusion",
                "summary": "R0 trusted HOW F1 early_concat + standard neck. Not A4, not F3, not F2.",
                "detail": {"how_id": R0_HOW_ID, "neck_type": "standard"},
            }
        ],
        "controlled_variables": [
            "dataset_split",
            "evaluator",
            "metric_definition",
            "condition_slice",
            "seed",
        ],
        "expected_effect": {
            "primary_metric": PRIMARY_METRIC,
            "direction": "stabilize",
            "rationale": "R0 is the comparator, not a claimed improvement.",
        },
        "evaluation": {
            "method": "full_train",
            "seeds": [42],
            "notes": (
                "budget_class=formal so G2 is not 16/8 smoke. Same Frozen Fingerprint family as later rounds. "
                "Not Formal E and not a C1 upgrade of v2.5-D."
            ),
        },
        "budget_class": G2_BUDGET_CLASS,
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": [], "strategy_ids": []},
        "evidence_runs": [],
        "bootstrap": True,
        "how_id": R0_HOW_ID,
        "rationale": "Campaign constitution R0. LLM did not select this HOW.",
    }


def execute_cli(*, output_dir: str = "outputs/v26_r0") -> str:
    protocol = DEFAULT_PROTOCOL_PATH.as_posix()
    plan = DEFAULT_PLAN_PATH.as_posix()
    return (
        "scientist-lab v26-r0-run "
        f"--output-dir {output_dir} "
        "--execute --confirm-human-gate --require-live-ready"
        f"  # protocol={protocol} plan={plan}"
    )


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def refuse_forged_metrics(raw: Mapping[str, Any] | None) -> None:
    if not raw:
        return
    if raw.get("source") in {"v25_a4", "v25d_probe", "hand_edit", "invented"}:
        raise R0BaselineError(f"refusing forged R0 metrics from {raw.get('source')}")
    aps_ll = raw.get(PRIMARY_METRIC)
    if aps_ll is None:
        return
    if not _finite(aps_ll):
        raise R0BaselineError("APS_lowlight is not a finite number")
    for key in FORBIDDEN_SUBSTITUTES:
        if key in raw and raw.get(key) is not None and raw.get(PRIMARY_METRIC) == raw.get(key):
            if raw.get("evaluator_backend") != "pycocotools":
                raise R0BaselineError(
                    f"refusing APS_lowlight copied from {key}; use coco AP_small on the frozen slice"
                )


def _find_run_file(run_dir: Path, name: str) -> Path | None:
    direct = run_dir / name
    if direct.is_file():
        return direct
    if not run_dir.is_dir():
        return None
    for child in run_dir.iterdir():
        if child.is_dir() and child.name not in _SKIP_RUN_DIRS:
            nested = child / name
            if nested.is_file():
                return nested
    return None


def compute_aps_lowlight_from_run(run_dir: Path) -> dict[str, Any]:
    """COCO AP_small on frozen low_light_subset_v1 val. Never copies full-set APS/mAP."""
    det_path = _find_run_file(run_dir, "val_detections_coco.json")
    if det_path is None:
        raise R0BaselineError(
            "R0 cannot bind: APS_lowlight missing (no val_detections_coco.json). "
            "Do not copy APS_all / mAP / v2.5-D probe APS=0.0."
        )
    payload = json.loads(det_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        detections = payload
    elif isinstance(payload, Mapping):
        detections = payload.get("detections") or payload.get("annotations") or []
    else:
        detections = []
    if not isinstance(detections, list):
        raise R0BaselineError("val_detections_coco.json detections must be a list")
    freeze_path = DEFAULT_SLICE_FREEZE_PATH
    if not freeze_path.is_file():
        raise R0BaselineError(f"low_light_subset_v1 freeze missing: {freeze_path}")
    gt_path = DEFAULT_VAL_GT_PATH
    if not gt_path.is_file():
        raise R0BaselineError(f"R0 cannot bind: APS_lowlight missing (val GT not found: {gt_path})")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    gt = json.loads(gt_path.read_text(encoding="utf-8-sig"))
    result = evaluate_aps_lowlight(gt, detections, freeze, split="val")
    result["detections_path"] = str(det_path)
    result["gt_path"] = str(gt_path)
    write_json(run_dir / "aps_lowlight.json", result)
    return result


def _merge_slice_metrics(metrics_path: Path, slice_eval: Mapping[str, Any]) -> None:
    if not metrics_path.is_file():
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return
    keys = (PRIMARY_METRIC, "mAP50_95_lowlight", "AP50_lowlight", "evaluator_backend")
    for key in keys:
        if key in slice_eval:
            payload[key] = slice_eval[key]
    nested = payload.get("metrics")
    if isinstance(nested, dict):
        for key in keys:
            if key in slice_eval:
                nested[key] = slice_eval[key]
    write_json(metrics_path, payload)


def _live_exec_output_dirs(*roots: Path | str | None) -> list[Path]:
    """Resolve scientist-exec bind-mount dirs from live_execution.json pointers."""
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        base = Path(root)
        for pointer in (base / "live_execution.json", base / "run" / "live_execution.json"):
            if not pointer.is_file():
                continue
            try:
                data = json.loads(pointer.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, Mapping):
                continue
            raw = data.get("output_directory") or data.get("output_dir")
            if not raw:
                continue
            dest = Path(str(raw))
            key = str(dest.resolve()) if dest.exists() else str(dest)
            if key in seen:
                continue
            seen.add(key)
            found.append(dest)
    return found


def sync_aps_lowlight_to_dirs(
    slice_eval: Mapping[str, Any],
    *dest_dirs: Path | str | None,
) -> list[str]:
    """Write aps_lowlight.json + merge metrics into GPU exec dirs (and any mirrors).

    Host bind happens on the campaign run/; without this sync, outputs/exec_*/metrics.json
    still lack APS_lowlight and mislead ablation reads. Does not invent values.
    """
    written: list[str] = []
    seen: set[str] = set()
    for raw in dest_dirs:
        if raw is None:
            continue
        dest = Path(raw)
        if not dest.exists():
            continue
        key = str(dest.resolve())
        if key in seen:
            continue
        seen.add(key)
        write_json(dest / "aps_lowlight.json", dict(slice_eval))
        metrics_path = dest / "metrics.json"
        if metrics_path.is_file():
            _merge_slice_metrics(metrics_path, slice_eval)
        written.append(str(dest))
    return written


def bind_metrics_from_run(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        raise R0BaselineError(f"R0 run directory missing: {root}")
    parsed = parse_metrics(root)
    metrics = dict(parsed.get("metrics") or {})
    refuse_forged_metrics(metrics)
    refuse_forged_metrics(parsed.get("raw") or {})
    sources = list(parsed.get("sources") or [])
    if not _finite(metrics.get(PRIMARY_METRIC)):
        slice_eval = compute_aps_lowlight_from_run(root)
        if slice_eval.get("evaluator_backend") != "pycocotools" or not _finite(
            slice_eval.get(PRIMARY_METRIC)
        ):
            note = slice_eval.get("note") or "Do not copy APS_all / mAP / v2.5-D probe APS=0.0."
            raise R0BaselineError(f"R0 cannot bind: APS_lowlight missing. {note}")
        metrics[PRIMARY_METRIC] = float(slice_eval[PRIMARY_METRIC])
        metrics["mAP50_95_lowlight"] = slice_eval.get("mAP50_95_lowlight")
        metrics["AP50_lowlight"] = slice_eval.get("AP50_lowlight")
        metrics["evaluator_backend"] = "pycocotools"
        sources.append(str(root / "aps_lowlight.json"))
        _merge_slice_metrics(root / "metrics.json", slice_eval)
        refuse_forged_metrics(metrics)
    if not _finite(metrics.get(PRIMARY_METRIC)):
        raise R0BaselineError(
            "R0 cannot bind: APS_lowlight missing. Do not copy APS_all / mAP / v2.5-D probe APS=0.0."
        )
    return {
        PRIMARY_METRIC: float(metrics[PRIMARY_METRIC]),
        "mAP50_95_lowlight": metrics.get("mAP50_95_lowlight"),
        "AP50_lowlight": metrics.get("AP50_lowlight"),
        "APS": metrics.get("APS"),
        "mAP50_95": metrics.get("mAP50_95"),
        "evaluator_backend": metrics.get("evaluator_backend"),
        "sources": sources,
        "bound_at": utc_now(),
    }


def build_r0_freeze(
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    doctor: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    how = r0_how()
    resolved_how = resolve_adapter_how(plan, protocol)
    if str(resolved_how.get("how_id") or "").upper() != R0_HOW_ID:
        raise R0BaselineError(f"R0 HOW must be {R0_HOW_ID}, got {resolved_how.get('how_id')}")
    if resolved_how.get("neck_type") != "standard":
        raise R0BaselineError("R0 neck must be standard; A4/FDPN is hidden from this anchor")
    if str(plan.get("budget_class") or "") != G2_BUDGET_CLASS:
        raise R0BaselineError(f"R0 G2 run_level must be budget_class={G2_BUDGET_CLASS}")
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(plan, protocol)
    fingerprint = compute_fingerprint(protocol, contract)
    doctor = dict(doctor or {})
    live_ready = bool(doctor.get("live_ready"))
    missing = []
    if not live_ready:
        missing.append("cuda_doctor.live_ready")
        runtime = doctor.get("runtime") or {}
        for key in ("docker", "cuda_image", "nvidia_smi"):
            if not bool((runtime.get(key) or {}).get("ok")):
                missing.append(f"runtime.{key}")
    if not dataset_contract.get("probe", {}).get("processed_present"):
        missing.append("processed_root")
    if metrics:
        refuse_forged_metrics(metrics)
        if not _finite(metrics.get(PRIMARY_METRIC)):
            raise R0BaselineError("refusing incomplete R0 metrics payload")
        status = "metrics_bound"
    else:
        status = "protocol_frozen_metrics_pending"
        metrics = {
            PRIMARY_METRIC: None,
            "mAP50_95_lowlight": None,
            "AP50_lowlight": None,
            "source": None,
            "execution_id": None,
            "note": "Metrics pending a live_ready GPU run. Not filled from v2.5 or probe APS=0.0.",
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "anchor_id": ANCHOR_ID,
        "step": "V26.4",
        "campaign": "v2.6",
        "status": status,
        "frozen_at": utc_now(),
        "not_an_agent": True,
        "llm_may_rewrite": False,
        "llm_may_select_slice": True,
        "how": how,
        "run_key": R0_RUN_KEY,
        "plan_id": plan.get("plan_id"),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "dataset": {
            "dataset_id": dataset_contract.get("dataset_id") or DATASET_ID,
            "dataset_ref": dataset_contract.get("dataset_ref"),
            "slice_id": dataset_contract.get("slice_id") or SLICE_ID,
            "split_reference": dataset_contract.get("split_reference"),
            "fingerprint": dataset_contract.get("fingerprint"),
            "version": dataset_contract.get("version"),
            "rule_hash": rule_hash(),
        },
        "metric_contract": metric_contract(),
        "run_level": {
            "budget_class": G2_BUDGET_CLASS,
            "execution_mode": "full_train",
            "evaluation_scope": "low_light_subset_v1_val",
            "seed": 42,
            "note": (
                "probe/fast_eval 16/8 cannot fill G2. This formal run_level is the v2.6 "
                "comparator family, not Formal E / C1 of v2.5-D."
            ),
        },
        "frozen_fingerprint": fingerprint,
        "experiment_contract_run_id": contract.get("run_id"),
        "metrics": dict(metrics),
        "gates": {
            "live_ready": live_ready,
            "doctor_ok": doctor.get("ok"),
            "missing": missing,
            "human_gate": "full_training remains approval_required; --execute --confirm-human-gate is the one-time campaign permission",
        },
        "execute": {
            "cli": execute_cli(),
            "protocol": str(DEFAULT_PROTOCOL_PATH),
            "plan": str(DEFAULT_PLAN_PATH),
        },
        "do_not": [
            "copy v2.5 A4 mAP50_95 / last@20 as APS_lowlight",
            "copy v2.5-D probe APS=0.0",
            "let LLM rewrite low_light_subset_v1",
            "fill APS_lowlight from APS_all or mAP",
            "use F2/T1/T2 or A4 as R0",
        ],
    }
    validate_named("r0_baseline", payload)
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def freeze_r0_anchor(
    *,
    dataset_contract: Mapping[str, Any],
    doctor: Mapping[str, Any] | None = None,
    protocol_path: Path | str | None = None,
    plan_path: Path | str | None = None,
    output: Path | str | None = None,
    metric_contract_output: Path | str | None = None,
    metrics_dir: Path | str | None = None,
) -> dict[str, Any]:
    protocol = load_json(protocol_path or DEFAULT_PROTOCOL_PATH)
    plan_file = Path(plan_path) if plan_path else DEFAULT_PLAN_PATH
    plan = load_json(plan_file) if plan_file.is_file() else default_r0_plan()
    validate_named("research_protocol", protocol)
    validate_named("experiment_plan", plan)
    metrics = bind_metrics_from_run(metrics_dir) if metrics_dir else None
    freeze = build_r0_freeze(
        protocol=protocol,
        plan=plan,
        dataset_contract=dataset_contract,
        doctor=doctor,
        metrics=metrics,
    )
    dest = Path(output) if output is not None else DEFAULT_FREEZE_PATH
    write_json(dest, freeze)
    contract_dest = Path(metric_contract_output) if metric_contract_output is not None else DEFAULT_METRIC_CONTRACT_PATH
    write_json(contract_dest, freeze["metric_contract"])
    freeze["output"] = str(dest)
    freeze["metric_contract_output"] = str(contract_dest)
    return freeze


def run_r0(
    *,
    dataset_contract: Mapping[str, Any],
    doctor: Mapping[str, Any],
    output_dir: Path | str,
    execute: bool = False,
    confirm_human_gate: bool = False,
    require_live_ready: bool = False,
    experiments: Any | None = None,
    protocol_path: Path | str | None = None,
    plan_path: Path | str | None = None,
) -> dict[str, Any]:
    """Dry-run by default. GPU only with human confirm + live_ready. Never forges metrics."""
    from scientist_lab.adapters.dfine.cuda_runner import make_cuda_live_runner

    freeze = freeze_r0_anchor(
        dataset_contract=dataset_contract,
        doctor=doctor,
        protocol_path=protocol_path,
        plan_path=plan_path,
    )
    report: dict[str, Any] = {
        "ok": True,
        "executed": False,
        "dry_run": not execute,
        "freeze": {
            "status": freeze["status"],
            "anchor_id": freeze["anchor_id"],
            "dataset_fingerprint": freeze["dataset"]["fingerprint"],
            "how_id": freeze["how"]["how_id"],
            "output": freeze.get("output"),
        },
        "live_ready": bool(doctor.get("live_ready")),
        "missing": list(freeze.get("gates", {}).get("missing") or []),
    }
    if not execute:
        report["next"] = execute_cli(output_dir=str(output_dir))
        return report
    if not confirm_human_gate:
        raise R0BaselineError(
            "R0 GPU requires --confirm-human-gate (one-time campaign Human Gate). "
            "Protocol still forbids silent formal training."
        )
    if not require_live_ready:
        raise R0BaselineError("R0 GPU requires --require-live-ready; refusing forged APS_lowlight")
    if not doctor.get("live_ready"):
        raise R0BaselineError(
            "cuda doctor live_ready=false; refusing GPU and forged APS_lowlight"
        )
    protocol = load_json(protocol_path or DEFAULT_PROTOCOL_PATH)
    plan_file = Path(plan_path) if plan_path else DEFAULT_PLAN_PATH
    plan = load_json(plan_file) if plan_file.is_file() else default_r0_plan()
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(plan, protocol)
    if experiments is None:
        from scientist_lab.services.experiment_service import ExperimentService

        experiments = ExperimentService()
    handle = adapter.execute(
        contract,
        protocol,
        output_dir=output_dir,
        dry_run=False,
        live_runner=make_cuda_live_runner(
            experiments,
            execute=True,
            require_live_ready=True,
        ),
        expected_fingerprint=freeze["frozen_fingerprint"],
    )
    report["executed"] = True
    report["dry_run"] = False
    report["human_gate_confirmed"] = True
    report["handle"] = {
        "status": handle.get("status"),
        "dry_run": handle.get("dry_run"),
    }
    return report


R0_LESSON_ID = "LESSON-V26-R0-001"
R0_STRATEGY_ID = "STRATEGY-V26-R0-001"
R0_PARENT_RUN_ID = "exec_31eff20c4e0c"


def seed_r0_campaign_memory(
    memory_dir: Path | str,
    *,
    run_id: str | None = None,
    aps_lowlight: float | None = None,
) -> dict[str, Any]:
    """Write a citable R0 comparator lesson. Not an LLM discovery. Not a fifth Agent."""
    from scientist_lab.core.memory_writer import MemoryWriter

    parent = str(run_id or R0_PARENT_RUN_ID)
    aps = float(aps_lowlight) if aps_lowlight is not None else None
    if aps is None:
        freeze_path = DEFAULT_FREEZE_PATH
        if freeze_path.is_file():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
            metrics = freeze.get("metrics") or {}
            if _finite(metrics.get(PRIMARY_METRIC)):
                aps = float(metrics[PRIMARY_METRIC])
        aps_path = _REPO_ROOT / "outputs" / "v26_r0" / "aps_lowlight.json"
        if aps is None and aps_path.is_file():
            payload = json.loads(aps_path.read_text(encoding="utf-8-sig"))
            if _finite(payload.get(PRIMARY_METRIC)):
                aps = float(payload[PRIMARY_METRIC])
    if aps is None:
        raise R0BaselineError("cannot seed R0 memory: APS_lowlight missing")
    writer = MemoryWriter(memory_dir)
    lesson = {
        "lesson_id": R0_LESSON_ID,
        "type": "process",
        "statement": (
            "V26.4 R0 comparator is F1 early_concat + standard neck on "
            f"low_light_subset_v1 val: {PRIMARY_METRIC}={aps}. "
            "Not A4. Later rounds must keep dataset_id/slice_id/fingerprint family. "
            "LiteratureEvidence is not ExperimentEvidence."
        ),
        "status": "active",
        "evidence": [{"run_id": parent, "metric": PRIMARY_METRIC, "delta": None}],
        "scope": {"task": "rgbt_detection", "module": "fusion"},
        "confidence": "high",
        "created_from": [parent],
        "contradicted_by": [],
        "supersedes": [],
        "expires_when": [],
    }
    strategy = {
        "strategy_id": R0_STRATEGY_ID,
        "action": "keep",
        "target": "fusion",
        "reason_lesson_ids": [R0_LESSON_ID],
        "status": "active",
    }
    writer.persist_lesson(lesson)
    writer.persist_strategy(strategy)
    return {
        "lesson_id": R0_LESSON_ID,
        "strategy_id": R0_STRATEGY_ID,
        "run_id": parent,
        PRIMARY_METRIC: aps,
    }
