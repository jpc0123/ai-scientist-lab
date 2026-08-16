"""Manager REAL execute wiring. No GPU. Stub live_runner / mocked doctor only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scientist_lab.cli import build_parser, main
from scientist_lab.core.manager import Manager
from scientist_lab.core.manager_cli import run_manager_from_files
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    OrchestrationAction,
    RunState,
)

EXAMPLES = SCHEMA_DIR / "examples"

LESSON_OTHER = {
    "lesson_id": "LESSON-OTHER",
    "type": "negative_evidence",
    "statement": "Unrelated stored lesson so Gate can REJECT unresolved refs.",
    "status": "active",
    "evidence": [{"run_id": "EXP-007", "metric": "APS", "delta": -0.1}],
    "scope": {"task": "rgbt_tiny_detection", "module": "neck"},
    "confidence": "medium",
    "created_from": ["EXP-007"],
    "contradicted_by": [],
    "supersedes": [],
    "expires_when": [],
}


def _seed_plan(**overrides):
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _install_seed(root: Path, *, with_metrics: bool = False) -> dict:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "plan.json", _seed_plan())
    _write_json(root / "baseline_metrics.json", {"APS": best["APS"], "mAP50_95": best["mAP50_95"]})
    if with_metrics:
        shutil.copyfile(EXAMPLES / "recovered_k2c44_last_metrics.json", root / "metrics.json")
        shutil.copyfile(
            EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
            root / "checkpoint_selection.json",
        )
    return protocol


def _stub_runner(calls: list) -> object:
    def runner(contract, output_dir):
        calls.append({"run_id": contract.get("run_id"), "output_dir": str(output_dir)})
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXAMPLES / "recovered_k2c44_last_metrics.json", dest / "metrics.json")
        shutil.copyfile(
            EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
            dest / "checkpoint_selection.json",
        )
        return {"status": "completed"}

    return runner


def test_execute_true_stub_runner_recovers_valid(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path, with_metrics=False)
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_runner(calls),
        max_extra_rounds=0,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert calls, "NEED_EXECUTION must call stub live_runner when execute=True"
    assert steps[-1].state.evidence_status == EvidenceStatus.VALID
    assert (tmp_path / "run" / "metrics.json").is_file()
    exec_steps = [s for s in steps if s.action == OrchestrationAction.NEED_EXECUTION.value]
    assert exec_steps
    assert exec_steps[0].report.get("runner_called") is True
    assert exec_steps[0].report.get("dry_run") is False


def test_require_live_ready_false_doctor_does_not_call_runner_or_forge_metrics(
    tmp_path: Path,
) -> None:
    protocol = _install_seed(tmp_path, with_metrics=False)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("live_runner must not run when live_ready is false")

    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        require_live_ready=True,
        live_runner=boom,
        doctor_fn=lambda: {"live_ready": False, "overall": "error"},
        max_extra_rounds=0,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert calls == []
    assert not (tmp_path / "metrics.json").is_file()
    assert steps[-1].state.run_state == RunState.BLOCKED
    assert steps[-1].report.get("runner_called") is False
    assert steps[-1].report.get("metrics_forged") is False
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert '"tool": "cuda_doctor"' in events
    assert '"live_ready": false' in events
    dumped = json.dumps(steps[-1].to_dict())
    assert "0.41" not in dumped
    assert "APS" not in dumped or steps[-1].report.get("metrics_forged") is False


def test_execute_false_does_not_call_live_runner(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path, with_metrics=True)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("execute=False must not call live_runner")

    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=False,
        live_runner=boom,
        max_extra_rounds=0,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert calls == []
    exec_steps = [s for s in steps if s.action == OrchestrationAction.NEED_EXECUTION.value]
    assert exec_steps
    assert exec_steps[0].report.get("runner_called") is False
    assert exec_steps[0].report.get("dry_run") is True


def test_gate_reject_does_not_call_live_runner(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path, with_metrics=False)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("Gate REJECT must not ignite live_runner")

    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=boom,
        max_extra_rounds=0,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    first = mgr.step()
    assert first.action == OrchestrationAction.NEED_PLAN.value
    MemoryWriter(tmp_path / "memory").persist_lesson(LESSON_OTHER)
    steps = mgr.run_until(max_steps=10)
    assert calls == []
    assert not (tmp_path / "metrics.json").is_file()
    gate_steps = [s for s in steps if s.action == OrchestrationAction.NEED_GATE.value]
    assert gate_steps
    assert gate_steps[-1].report.get("status") == "REJECTED"
    assert steps[-1].state.run_state == RunState.BLOCKED
    assert OrchestrationAction.NEED_EXECUTION.value not in [s.action for s in steps]


def test_cli_has_manager_run_and_does_not_bind_experiment_service() -> None:
    import scientist_lab.cli as cli

    assert not hasattr(cli, "ExperimentService")
    parser = build_parser()
    assert "manager-run" in parser.format_help()
    assert "dfine-adapter-run" in parser.format_help()


def test_cli_manager_run_dispatches_to_manager(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake(*_a, **kwargs):
        seen.update(kwargs)
        return {"exit_code": 0, "live_ready": False}

    monkeypatch.setattr(
        "scientist_lab.core.manager_cli.run_manager_from_files", fake
    )
    protocol = tmp_path / "protocol.json"
    plan = tmp_path / "plan.json"
    protocol.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    code = main(
        [
            "manager-run",
            "--protocol",
            str(protocol),
            "--plan",
            str(plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--execute",
            "--require-live-ready",
            "--max-steps",
            "8",
        ]
    )
    assert code == 0
    assert seen.get("execute") is True
    assert seen.get("require_live_ready") is True
    assert seen.get("max_steps") == 8
    assert seen.get("max_extra_rounds") == 0


def test_cli_parses_max_extra_rounds(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake(*_a, **kwargs):
        seen.update(kwargs)
        return {"exit_code": 0, "live_ready": True}

    monkeypatch.setattr(
        "scientist_lab.core.manager_cli.run_manager_from_files", fake
    )
    protocol = tmp_path / "protocol.json"
    plan = tmp_path / "plan.json"
    protocol.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    code = main(
        [
            "manager-run",
            "--protocol",
            str(protocol),
            "--plan",
            str(plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-extra-rounds",
            "1",
            "--max-steps",
            "24",
        ]
    )
    assert code == 0
    assert seen.get("max_extra_rounds") == 1
    assert seen.get("max_steps") == 24
    assert seen.get("execute") is False


def test_manager_cli_require_live_ready_exits_nonzero_without_metrics(
    tmp_path: Path,
) -> None:
    protocol = _install_seed(tmp_path, with_metrics=False)
    _write_json(tmp_path / "protocol.json", protocol)
    _write_json(tmp_path / "plan.json", _seed_plan())
    out = tmp_path / "run_out"
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("must not run")

    result = run_manager_from_files(
        tmp_path / "protocol.json",
        tmp_path / "plan.json",
        output_dir=out,
        execute=True,
        require_live_ready=True,
        live_runner=boom,
        doctor_fn=lambda: {"live_ready": False, "overall": "error"},
        max_steps=20,
    )
    assert result["exit_code"] == 1
    assert result["live_ready"] is False
    assert calls == []
    assert not (out / "metrics.json").is_file()
    doctor_dump = json.loads((out / "cuda_doctor.json").read_text(encoding="utf-8"))
    assert doctor_dump.get("live_ready") is False
    events = (out / "research_events.jsonl").read_text(encoding="utf-8")
    assert '"tool": "cuda_doctor"' in events


def test_timed_out_does_not_overwrite_freeze_contract_or_keep(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path, with_metrics=False)
    calls: list = []

    def runner(contract, output_dir):
        calls.append(str(output_dir))
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "contract.json").write_text(
            json.dumps({"schema_version": "1.2", "title": "legacy overwrite"}),
            encoding="utf-8",
        )
        return {
            "status": "timed_out",
            "run": {
                "status": "timed_out",
                "error": {
                    "error_type": "timeout",
                    "message": "实验超过最大运行时间：1200 秒。",
                },
            },
        }

    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=runner,
        max_extra_rounds=0,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert calls
    assert Path(calls[0]).name == "run"
    freeze_contract = load_json(tmp_path / "contract.json")
    assert freeze_contract.get("run_id")
    assert freeze_contract.get("schema_version") == "1.0.0"
    assert steps[-1].state.run_state == RunState.FAILED
    assert steps[-1].idle is True
    assert steps[-1].state.review_decision.value == "PENDING"
    assert not (tmp_path / "metrics.json").is_file()
    assert not (tmp_path / "run" / "metrics.json").is_file()
    dumped = json.dumps(steps[-1].to_dict())
    assert "KEEP" not in dumped
    assert "DISCARD" not in dumped
