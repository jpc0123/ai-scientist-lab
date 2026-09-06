"""GateEngine + CUDA live_runner wiring (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.cuda_runner import (
    freeze_to_legacy_contract,
    make_cuda_live_runner,
)
from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.domain.contracts import ExperimentContract

EXAMPLES = SCHEMA_DIR / "examples"


def _plan(**overrides):
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_round1_neck_hr",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v1",
        "protocol_version": 1,
        "parent_run_id": "EXP-007",
        "round_index": 1,
        "observation": "localization still weak",
        "hypothesis": "high-res neck path helps APS",
        "modification_scope": ["neck"],
        "proposed_changes": [{"target": "neck", "summary": "Adjust high-res path"}],
        "controlled_variables": ["evaluator"],
        "expected_effect": {"primary_metric": "APS", "direction": "increase"},
        "evaluation": {"method": "fast_eval", "seeds": [42]},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": ["LESSON-017"], "strategy_ids": ["STRATEGY-009"]},
        "evidence_runs": ["EXP-007"],
    }
    plan.update(overrides)
    return plan


def test_gate_approves_editable_probe() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    verdict = GateEngine().evaluate(protocol, plan, contract)
    assert verdict.status == GateStatus.APPROVED
    assert verdict.fingerprint_comparable is True


def test_gate_blocks_fingerprint_change() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    contract = dict(contract)
    contract["dataset"] = {**dict(contract["dataset"]), "split_reference": "split:leaked_train"}
    verdict = GateEngine().evaluate(protocol, plan, contract)
    assert verdict.status == GateStatus.BLOCKED
    assert any("Fingerprint CHANGED" in r for r in verdict.reasons)


def test_gate_rejects_frozen_evaluator_change() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    plan = _plan(modification_scope=["evaluator"])
    contract = dict(contract)
    contract["allowed_changes"] = ["evaluator"]
    verdict = GateEngine().evaluate(protocol, plan, contract)
    assert verdict.status == GateStatus.REJECTED


def test_gate_human_required_for_formal() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan(budget_class="formal", risk_level="approval_required")
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    verdict = GateEngine().evaluate(protocol, plan, contract)
    assert verdict.status == GateStatus.HUMAN_REQUIRED


def test_frozen_overlap_plus_fingerprint_is_blocked() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    plan = _plan(modification_scope=["evaluator"])
    contract = dict(contract)
    contract["allowed_changes"] = ["evaluator"]
    contract["dataset"] = {**dict(contract["dataset"]), "split_reference": "split:x"}
    verdict = GateEngine().evaluate(protocol, plan, contract)
    assert verdict.status == GateStatus.BLOCKED


def test_legacy_mapping_validates() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    contract = DFINEAdapter().materialize_contract(_plan(), protocol)
    legacy = ExperimentContract.model_validate(freeze_to_legacy_contract(contract))
    assert legacy.task_type == "rgbt_detection"
    assert legacy.execution_mode == "fast_eval"
    assert legacy.task_config["claim_level"] == "exploratory_comparison"
    dumped = freeze_to_legacy_contract(contract)
    assert dumped.get("protocol_id") in {None, ""}
    assert dumped["runner_profile"] == "local"
    assert dumped["parameters"]["epochs"] == 1
    assert dumped["parameters"]["max_train_images"] == 16
    assert dumped["parameters"]["input_mode"] == "rgb"
    assert dumped["parameters"]["fusion_method"] == "none"
    assert dumped["parameters"]["neck"]["type"] == "standard"
    assert dumped["execution_mode"] == "fast_eval"


def test_blocked_gate_skips_live_runner(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("orchestrator must not run")

    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(modification_scope=["evaluator"], proposed_changes=[
            {"target": "evaluator", "summary": "illegal"}
        ]),
        output_dir=tmp_path,
        execute=True,
        live_runner=lambda *_: boom(),
    )
    assert report["gate"]["status"] == GateStatus.REJECTED
    assert report["executed"] is False
    assert called["n"] == 0


def test_execute_uses_live_runner_when_approved(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    calls = []

    def fake_eval(experiments, **kwargs):
        calls.append(kwargs)
        dest = Path(str(kwargs["contract_path"])).parent
        (dest / "metrics.json").write_text(
            json.dumps({"APS": 0.1, "mAP50_95": 0.09}), encoding="utf-8"
        )
        (dest / "checkpoint_selection.json").write_text("{}", encoding="utf-8")
        return {
            "status": "completed",
            "orchestrator": "stub",
            "run": {"status": "completed", "output_directory": str(dest)},
        }

    runner = make_cuda_live_runner(
        experiments=object(),
        execute=True,
        call_eval=fake_eval,
        require_live_ready=False,
        probe_runtime=False,
    )
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=True,
        live_runner=runner,
    )
    assert report["gate"]["status"] == GateStatus.APPROVED
    assert report["executed"] is True
    assert report["handle"]["status"] == "completed"
    assert report["result"]["metrics"]["APS"] == 0.1
    assert report["evidence"]["evidence_status"] == "VALID"
    assert report["evidence"]["review_allowed"] is True
    # No baseline → Reviewer must not invent KEEP; REPLICATE, not PENDING.
    assert report["review_decision"] == "REPLICATE"
    assert report["rubric"]["constraints_ok"] is True
    assert report["scientific_outcome"] == "INCONCLUSIVE"
    assert report["memory"]["lessons_written"]
    assert calls and calls[0]["dry_run"] is False
