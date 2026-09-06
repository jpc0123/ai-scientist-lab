"""Annotate Vendor / stand-in DFINE Fast Eval evidence (v2.3.3).

Vendor runs are labeled non-stand-in so claim_formal_dfine is no longer
stand-in-blocked; Fast Eval still keeps scientific strength weak and
formal_success=false until later gate upgrades (v2.3.5+).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scientist_lab.domain.models import new_id, utc_now_iso
from scientist_lab.evidence.claim_gate import assess_evidence_strength
from scientist_lab.evidence.claim_matrix import evaluate_formal_dfine_claim
from scientist_lab.evidence.models import EvidenceRecord
from scientist_lab.tasks.rgbt_detection.vendor_audit import (
    BASELINE_IMPLEMENTATION,
    PINNED_COMMIT,
)

BackendKind = Literal["vendor", "standin", "unknown"]


def classify_dfine_backend(
    *,
    execution_metadata: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    model_summary: dict[str, Any] | None = None,
) -> BackendKind:
    meta = dict(execution_metadata or {})
    tagged = str(meta.get("standin_or_vendor") or "").strip().lower()
    if tagged in {"vendor", "standin"}:
        return tagged  # type: ignore[return-value]

    requested = str(
        meta.get("dfine_backend_requested")
        or (contract or {}).get("parameters", {}).get("dfine_backend")
        or ""
    ).strip().lower()
    if requested in {"standin", "torch_mini", "mini"}:
        return "standin"
    if requested in {"dfine", "dfine_s", "vendor"}:
        # Prefer model_summary when present.
        impl = str(
            (model_summary or {}).get("baseline_implementation")
            or meta.get("baseline_implementation")
            or ""
        ).lower()
        if "standin" in impl or "stand_in" in impl:
            return "standin"
        return "vendor"

    impl = str(
        (model_summary or {}).get("baseline_implementation")
        or meta.get("baseline_implementation")
        or ((contract or {}).get("task_config") or {}).get("implementation")
        or ""
    ).lower().replace("-", "_")
    if not impl:
        return "unknown"
    if "standin" in impl or "stand_in" in impl or "torch_mini" in impl:
        return "standin"
    if "vendored" in impl or "dfine" in impl:
        return "vendor"
    return "unknown"


def _implementation_label(kind: BackendKind, meta: dict[str, Any]) -> str:
    if kind == "standin":
        return "torch_mini_standin_v0_8_1"
    if kind == "vendor":
        return str(
            meta.get("baseline_implementation") or BASELINE_IMPLEMENTATION
        )
    return str(meta.get("baseline_implementation") or "unknown")


def build_dfine_fast_eval_evidence(
    *,
    project_id: str,
    execution_id: str,
    node_id: str | None = None,
    metrics: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
    model_summary: dict[str, Any] | None = None,
    evidence_id: str | None = None,
) -> EvidenceRecord:
    """Build a single_execution EvidenceRecord for one DFINE Fast Eval run."""
    meta = dict(execution_metadata or {})
    contract_payload = dict(contract or {})
    task_config = dict(contract_payload.get("task_config") or {})
    kind = classify_dfine_backend(
        execution_metadata=meta,
        contract=contract_payload,
        model_summary=model_summary,
    )
    impl = _implementation_label(kind, meta)
    claim_level = str(
        meta.get("claim_level")
        or task_config.get("claim_level")
        or "exploratory_comparison"
    )
    evaluation_scope = str(task_config.get("evaluation_scope") or "fast_eval_subset")
    protocol_id = contract_payload.get("protocol_id")

    assessment = assess_evidence_strength(
        seed_count=1,
        execution_mode=str(
            contract_payload.get("execution_mode") or "fast_eval"
        ),
        evaluation_scope=evaluation_scope,
        implementation=impl,
        claim_level=claim_level,
        has_protocol=bool(protocol_id),
        has_ablation=False,
        formal_implementation=(kind == "vendor"),
        stable_direction=False,
        requested_type="single_execution",
    )

    limitations = list(assessment.get("limitations") or [])
    if kind == "vendor":
        limitations.append(
            "Vendor DFINE Fast Eval only; not a formal benchmark acceptance."
        )
        limitations.append(
            "Vendor annotation removes standin-backend block only; "
            "formal DFINE superiority still requires stronger acceptance."
        )
    elif kind == "standin":
        # assess_evidence_strength already adds stand-in lines; keep explicit tag
        if not any("Stand-in" in item or "stand-in" in item for item in limitations):
            limitations.append("Stand-in implementation was used.")
    else:
        limitations.append("DFINE backend classification is unknown.")

    metrics_payload = dict(metrics or {})
    primary = task_config.get("primary_metric") or "mAP50_95"
    metric_summary: dict[str, Any] = {
        "primary_metric": primary,
        "primary_value": metrics_payload.get(primary),
        "metrics": metrics_payload,
        "standin_or_vendor": kind,
        "dfine_backend_requested": meta.get("dfine_backend_requested"),
        "baseline_implementation": impl,
        "vendor_commit_expected": meta.get("vendor_commit_expected") or PINNED_COMMIT,
        "environment_key": meta.get("environment_key")
        or contract_payload.get("environment_key"),
        "runner_profile": meta.get("runner_profile")
        or contract_payload.get("runner_profile"),
        "device": meta.get("device"),
        "cuda_available": meta.get("cuda_available"),
        "orchestrator": meta.get("orchestrator"),
        "triad_role": meta.get("triad_role") or meta.get("role"),
        "input_mode": (contract_payload.get("parameters") or {}).get("input_mode")
        or meta.get("input_mode"),
        "fusion_method": (contract_payload.get("parameters") or {}).get("fusion_method")
        or meta.get("fusion_method"),
        "evaluation_scope": evaluation_scope,
        "exploratory_only": True,
        "formal_success": False,
        "recorded_at": utc_now_iso(),
    }

    return EvidenceRecord(
        evidence_id=evidence_id or new_id("ev"),
        project_id=project_id,
        evidence_type="single_execution",
        source_node_ids=[node_id] if node_id else [],
        source_execution_ids=[execution_id],
        protocol_id=str(protocol_id) if protocol_id else None,
        metric_summary=metric_summary,
        evidence_strength=assessment["evidence_strength"],
        engineering_evidence_level=assessment.get("engineering_evidence_level"),
        scientific_evidence_level=assessment.get("scientific_evidence_level"),
        limitations=limitations,
        valid=True,
        claim_level=claim_level,
    )


def record_dfine_fast_eval_evidence(
    experiments: Any,
    *,
    execution_id: str,
    execution_metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    refresh_claim_matrix: bool = True,
) -> dict[str, Any]:
    """Persist Fast Eval evidence and optionally refresh claim matrix."""
    meta = dict(execution_metadata or {})
    project_id = str(
        meta.get("project_id")
        or (contract or {}).get("project_id")
        or ""
    )
    if not project_id:
        # Try metadata path layout / attempt
        attempt = experiments.repo.get_attempt(execution_id)
        if attempt is not None:
            node = experiments.repo.get_node(attempt.node_id)
            if node is not None:
                project_id = node.project_id
                contract = contract or dict(node.contract_json or {})
            if metrics is None and attempt.result_json:
                metrics = dict(
                    (attempt.result_json or {}).get("metrics")
                    or attempt.result_json
                    or {}
                )
        if not project_id:
            raise ValueError(
                "project_id required to record DFINE Fast Eval evidence"
            )

    node_id = None
    attempt = experiments.repo.get_attempt(execution_id)
    if attempt is not None:
        node_id = attempt.node_id
        if contract is None:
            node = experiments.repo.get_node(attempt.node_id)
            if node is not None:
                contract = dict(node.contract_json or {})
        if metrics is None and attempt.result_json:
            metrics = dict((attempt.result_json or {}).get("metrics") or {})

    # Load model_summary if output_directory known
    model_summary = None
    output_dir = meta.get("output_directory")
    if output_dir:
        ms_path = Path(str(output_dir)) / "model_summary.json"
        if ms_path.is_file():
            model_summary = json.loads(ms_path.read_text(encoding="utf-8"))

    if not meta.get("standin_or_vendor"):
        # Reuse orchestrator metadata file when present
        meta_path = (
            Path(experiments.settings.outputs_dir)
            / project_id
            / "_dfine_cuda_runs"
            / f"{execution_id}.json"
        )
        if meta_path.is_file():
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = {**loaded, **meta}

    record = build_dfine_fast_eval_evidence(
        project_id=project_id,
        execution_id=execution_id,
        node_id=node_id,
        metrics=metrics,
        contract=contract,
        execution_metadata=meta,
        model_summary=model_summary,
    )
    stored = experiments.evidence.persist(record)

    claim_preview = evaluate_formal_dfine_claim(
        project_id=project_id, records=[stored]
    )
    from scientist_lab.evidence.formal_dfine_gate import (
        assess_formal_dfine_path_gate,
        evaluate_formal_dfine_path_claim,
    )

    # Include prior project vendor evidence when assessing path openness.
    prior = [
        item
        for item in experiments.evidence.list_evidence(project_id=project_id)
        if item.evidence_id != stored.evidence_id
    ]
    gate_records = [*prior, stored]
    path_claim = evaluate_formal_dfine_path_claim(
        project_id=project_id, records=gate_records
    )
    gate = assess_formal_dfine_path_gate(gate_records)
    matrix = None
    if refresh_claim_matrix:
        matrix = experiments.evidence.build_claim_matrix(project_id)

    kind = str((stored.metric_summary or {}).get("standin_or_vendor") or "unknown")
    return {
        "evidence": stored.model_dump(mode="json"),
        "evidence_id": stored.evidence_id,
        "standin_or_vendor": kind,
        "non_standin": kind == "vendor",
        "exploratory_only": True,
        "formal_success": False,
        "formal_dfine_claim": {
            "support_status": claim_preview.support_status,
            "reason": claim_preview.reason,
        },
        "formal_dfine_path": {
            "support_status": path_claim.support_status,
            "formal_path_open": gate.get("formal_path_open"),
            "formal_superiority_eligible": gate.get("formal_superiority_eligible"),
            "thresholds": gate.get("thresholds"),
            "blocking_reasons": gate.get("blocking_reasons"),
        },
        "claim_matrix": matrix,
        "evidence_path": str(
            experiments.evidence.evidence_path(stored.evidence_id)
        ),
    }
