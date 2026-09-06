"""v0.9 CLI acceptance: seed formal triad data, then exercise §11 commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentArtifact,
    ExperimentNode,
    ResearchProject,
)
from scientist_lab.services.experiment_service import ExperimentService


def _seed_attempt(
    *,
    execution_id: str,
    node_id: str,
    seed: int,
    map50_95: float,
    duration: float,
    attempt_index: int,
    input_mode: str,
    fusion_method: str,
) -> ExecutionAttempt:
    return ExecutionAttempt(
        execution_id=execution_id,
        node_id=node_id,
        attempt_index=attempt_index,
        runner_profile="local",
        status=JobStatus.COMPLETED,
        image_reference="scientist-rgbt-detection:v2",
        code_version="image:rgbt-detection-v2",
        dataset_version="dataset:rgbt_fast_eval_v1",
        result_json={
            "metrics": {
                "primary_metric": "mAP50_95",
                "metrics": {
                    "mAP50_95": map50_95,
                    "mAP50": map50_95 + 0.1,
                    "AP_small": map50_95 / 2,
                    "precision": 0.4,
                    "recall": 0.3,
                    "duration_seconds": duration,
                    "peak_gpu_memory_mb": 100.0 if input_mode == "rgb" else 200.0,
                    "parameter_count": 1000.0 if input_mode == "rgb" else 1500.0,
                },
            },
            "contract": {
                "seed": seed,
                "project_id": "project_rgbt_003",
                "protocol_id": "protocol_rgbt_001",
                "dataset_reference": "dataset:rgbt_fast_eval_v1",
                "code_reference": "image:rgbt-detection-v2",
                "environment_key": "rgbt-detection-v2",
                "entrypoint": "run_detection_experiment.py",
                "execution_mode": "fast_eval",
                "task_type": "rgbt_detection",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "evaluation_scope": "fast_eval_subset",
                    "implementation": "stand_in",
                    "primary_metric": "mAP50_95",
                },
                "parameters": {
                    "baseline": "dfine_s",
                    "input_mode": input_mode,
                    "fusion_method": fusion_method,
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.0001,
                    "image_width": 640,
                    "image_height": 512,
                },
            },
        },
        created_at=f"2026-07-21T00:00:{attempt_index:02d}+00:00",
        completed_at=f"2026-07-21T00:01:{attempt_index:02d}+00:00",
    )


def seed_formal_triad(service: ExperimentService) -> None:
    now = "2026-07-21T00:00:00+00:00"
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T formal acceptance",
            research_goal="v0.9 CLI acceptance",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    try:
        service.create_protocol(ROOT / "examples" / "rgbt_protocol.json")
    except Exception:
        pass

    specs = {
        "rgbt_formal_node_001": {
            "input_mode": "rgb",
            "fusion_method": "none",
            "maps": [0.10, 0.11, 0.09],
            "duration": [10.0, 11.0, 9.0],
        },
        "rgbt_formal_node_002": {
            "input_mode": "thermal",
            "fusion_method": "none",
            "maps": [0.08, 0.09, 0.07],
            "duration": [11.0, 12.0, 10.0],
        },
        "rgbt_formal_node_003": {
            "input_mode": "rgbt",
            "fusion_method": "early_concat",
            "maps": [0.12, 0.13, 0.11],
            "duration": [20.0, 21.0, 19.0],
        },
    }
    for node_id, spec in specs.items():
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_rgbt_003",
                node_type=NodeType.BASELINE,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=0,
                contract_json={
                    "project_id": "project_rgbt_003",
                    "node_id": node_id,
                    "protocol_id": "protocol_rgbt_001",
                    "task_type": "rgbt_detection",
                    "execution_mode": "fast_eval",
                    "dataset_reference": "dataset:rgbt_fast_eval_v1",
                    "code_reference": "image:rgbt-detection-v2",
                    "environment_key": "rgbt-detection-v2",
                    "task_config": {
                        "claim_level": "exploratory_comparison",
                        "evaluation_scope": "fast_eval_subset",
                        "implementation": "stand_in",
                        "primary_metric": "mAP50_95",
                    },
                    "parameters": {
                        "baseline": "dfine_s",
                        "input_mode": spec["input_mode"],
                        "fusion_method": spec["fusion_method"],
                        "epochs": 5,
                        "batch_size": 4,
                        "learning_rate": 0.0001,
                        "image_width": 640,
                        "image_height": 512,
                    },
                },
                created_at=now,
                updated_at=now,
            )
        )
        for index, seed in enumerate([42, 43, 44], start=1):
            attempt = _seed_attempt(
                execution_id=f"exec_{node_id}_{seed}",
                node_id=node_id,
                seed=seed,
                map50_95=spec["maps"][index - 1],
                duration=spec["duration"][index - 1],
                attempt_index=index,
                input_mode=spec["input_mode"],
                fusion_method=spec["fusion_method"],
            )
            service.repo.upsert_attempt(attempt)
            service.repo.add_artifact(
                ExperimentArtifact(
                    artifact_id=f"art_{node_id}_{seed}",
                    execution_id=attempt.execution_id,
                    artifact_type="metrics",
                    relative_path="metrics.json",
                    size_bytes=12,
                    sha256="abc",
                    created_at=now,
                )
            )


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"step": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    service = ExperimentService()
    seed_formal_triad(service)
    results: list[dict] = []

    # 11.1 protocol
    protocol = service.show_protocol("protocol_rgbt_001")
    results.append(_check("show-protocol", protocol.get("protocol_id") == "protocol_rgbt_001"))

    # 11.2 formal triad
    triad = service.validate_formal_triad_contracts(
        [
            str(ROOT / "examples" / "rgbt_formal_rgb_contract.json"),
            str(ROOT / "examples" / "rgbt_formal_thermal_contract.json"),
            str(ROOT / "examples" / "rgbt_formal_fusion_contract.json"),
        ],
        protocol_id="protocol_rgbt_001",
    )
    results.append(
        _check(
            "validate-formal-triad",
            bool(triad.get("ok") or triad.get("valid")),
            json.dumps(
                {
                    k: triad.get(k)
                    for k in ("ok", "valid", "blocking_issues", "warnings")
                    if k in triad or True
                },
                ensure_ascii=False,
                default=str,
            )[:500],
        )
    )

    # 11.3 multi-metric comparison
    comparison = service.compare_node_groups(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    need = [
        "metric_relations",
        "resource_relations",
        "paired_win_count",
        "paired_loss_count",
        "claim_level",
    ]
    missing = [k for k in need if k not in comparison]
    results.append(
        _check(
            "compare-node-groups",
            not missing,
            f"missing={missing}; claim_level={comparison.get('claim_level')}",
        )
    )

    # 11.4 ablation
    try:
        service.create_ablation(ROOT / "examples" / "rgbt_ablation_modality_plan.json")
    except Exception:
        pass
    ablation_report = service.validate_ablation("ablation_rgbt_modality_001")
    results.append(
        _check(
            "validate-ablation",
            bool(ablation_report.get("valid")),
            json.dumps(ablation_report.get("blocking_issues") or [], ensure_ascii=False),
        )
    )
    materialized = service.materialize_ablation(
        "ablation_rgbt_modality_001",
        reference_contract=ROOT / "examples" / "rgbt_formal_fusion_contract.json",
    )
    results.append(
        _check(
            "materialize-ablation",
            bool(
                materialized.get("contracts")
                or materialized.get("paths")
                or materialized.get("written")
                or materialized.get("variants")
            ),
            str(sorted((materialized or {}).keys())),
        )
    )

    # 11.5 evidence
    evidence = service.build_evidence(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    results.append(
        _check(
            "build-evidence",
            len(evidence.get("evidence_ids") or []) >= 1,
            f"ids={evidence.get('evidence_ids')}",
        )
    )
    listed = service.list_evidence(project_id="project_rgbt_003")
    results.append(_check("list-evidence", len(listed) >= 1, f"count={len(listed)}"))
    shown_e = service.show_evidence(evidence["evidence_ids"][0])
    results.append(
        _check(
            "show-evidence",
            shown_e.get("evidence_strength") == "weak",
            f"strength={shown_e.get('evidence_strength')}; path={shown_e.get('evidence_path')}",
        )
    )

    # 11.6 claim matrix
    matrix = service.build_claim_matrix("project_rgbt_003")
    by_id = {c["claim_id"]: c for c in matrix.get("claims") or []}
    results.append(
        _check(
            "build-claim-matrix",
            by_id.get("claim_sota", {}).get("support_status") == "blocked"
            and by_id.get("claim_full_rgbt_tiny", {}).get("support_status") == "blocked",
            f"sota={by_id.get('claim_sota', {}).get('support_status')}; "
            f"full={by_id.get('claim_full_rgbt_tiny', {}).get('support_status')}; "
            f"ap_small={by_id.get('claim_fast_eval_ap_small', {}).get('support_status')}",
        )
    )
    shown_m = service.show_claim_matrix("project_rgbt_003")
    results.append(
        _check(
            "show-claim-matrix",
            Path(str(shown_m.get("matrix_path") or "")).is_file()
            or bool(shown_m.get("claims")),
            f"path={shown_m.get('matrix_path')}",
        )
    )

    # 11.7 decision + feedback fields
    decision = service.record_decision(
        selected_node_id="rgbt_formal_node_003",
        alternatives=["rgbt_formal_node_001", "rgbt_formal_node_002"],
        decision_type="exploratory_improvement",
        reason="v0.9 acceptance: fusion wins under fixed Fast Eval protocol",
        evidence_strength="strong",
        baseline_node_id="rgbt_formal_node_001",
        candidate_node_id="rgbt_formal_node_003",
    )
    results.append(
        _check(
            "record-decision",
            bool(decision.get("supporting_evidence_ids"))
            and bool(decision.get("claim_matrix_path"))
            and decision.get("protocol_id") == "protocol_rgbt_001"
            and decision.get("evidence_strength") == "weak",
            json.dumps(
                {
                    "evidence_strength": decision.get("evidence_strength"),
                    "protocol_id": decision.get("protocol_id"),
                    "supporting_evidence_ids": decision.get("supporting_evidence_ids"),
                    "claim_matrix_path": decision.get("claim_matrix_path"),
                },
                ensure_ascii=False,
            ),
        )
    )
    shown_d = service.show_decision(decision["decision_id"])
    results.append(
        _check(
            "show-decision",
            shown_d.get("decision_id") == decision["decision_id"],
            decision["decision_id"],
        )
    )

    feedback = service.analyze_feedback(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    v09_keys = [
        "scientific_interpretation",
        "engineering_findings",
        "performance_findings",
        "resource_tradeoffs",
        "evidence_gaps",
        "claim_restrictions",
        "recommended_next_experiments",
    ]
    missing_fb = [k for k in v09_keys if not feedback.get(k)]
    restrictions = " ".join(feedback.get("claim_restrictions") or []).lower()
    results.append(
        _check(
            "analyze-feedback-v09-fields",
            not missing_fb
            and ("sota" in restrictions or "state-of-the-art" in restrictions),
            f"missing={missing_fb}; evidence_strength={feedback.get('evidence_strength')}",
        )
    )

    ok_all = all(item["ok"] for item in results)
    report = {
        "version": "v0.9",
        "ok": ok_all,
        "project_id": "project_rgbt_003",
        "protocol_id": "protocol_rgbt_001",
        "decision_id": decision.get("decision_id"),
        "evidence_ids": evidence.get("evidence_ids"),
        "claim_matrix_path": matrix.get("matrix_path"),
        "steps": results,
    }
    out_dir = service.settings.outputs_dir / "project_rgbt_003" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "v09_acceptance_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {path}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
