"""v2.5-D Human-gated probe REAL loop. No GPU in these tests; stub/dry-run only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scientist_lab.cli import build_parser, main
from scientist_lab.core.gate_engine import GateStatus
from scientist_lab.core.invariants import assert_keep_is_not_claim
from scientist_lab.core.manager import Manager
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.gateway import ScriptedProvider, resolve_gateway_provider
from scientist_lab.llm.real_loop import run_v25d_real_loop

EXAMPLES = SCHEMA_DIR / "examples"
FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llm_plan_replay" / "m4_rounds3_discard"
)
REAL_LESSON = "LESSON-run_plan_round1_neck_hr-001"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "m4_rounds3_discard"
    shutil.copytree(FIXTURE, dest)
    return dest


def _stub_runner(calls: list):
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


def test_manager_explicit_llm_backends(tmp_path: Path) -> None:
    protocol = load_json(FIXTURE / "protocol.json")
    mgr = Manager(
        tmp_path / "mgr",
        protocol=protocol,
        planner_backend="llm",
        reviewer_backend="llm",
        llm_provider=FakeProvider(),
        max_extra_rounds=0,
    )
    assert mgr.planner.backend == "llm"
    assert mgr.reviewer.backend == "llm"
    default = Manager(tmp_path / "mgr_default", protocol=protocol, max_extra_rounds=0)
    assert default.planner.backend == "rules"
    assert default.reviewer.backend == "rules"


def test_dry_run_loop_plan_enters_gate_without_live_runner(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    calls: list = []
    report = run_v25d_real_loop(
        source_run_dir=source,
        output_dir=tmp_path / "loop",
        execute=False,
        live_runner=_stub_runner(calls),
        planner_backend="llm",
        reviewer_backend="llm",
        llm_provider=FakeProvider(),
        human_probe_permission=False,
    )
    assert report["ok"] is True
    assert report["gate"]["status"] == GateStatus.APPROVED
    assert report["runner_called"] is False
    assert report["ignited"] is False
    assert report["gpu"] is False
    assert calls == []
    assert report["plan"]["modification_scope"] == ["fusion"]
    cited = (report["memory_refs"] or {}).get("lesson_ids") or []
    assert REAL_LESSON in cited or any("semantic" in str(x) for x in cited)
    assert report["historical"]["review_decision"] == "DISCARD"
    assert report["historical"]["rubric_locked"] is True
    assert report["plan"]["budget_class"] == "probe"
    assert (tmp_path / "loop" / "loop_report.json").is_file()


def test_unapproved_gate_does_not_call_live_runner(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("live_runner must not run when Gate is not APPROVED")

    illegal = json.dumps(
        {
            "selected": {
                "candidate_id": "cand_bad",
                "requested_module": "fusion",
                "observation": "after DISCARD",
                "hypothesis": "fusion probe",
                "proposed_changes": [{"target": "fusion", "summary": "probe fusion"}],
                "budget_class": "probe",
            },
            "memory_refs": {"lesson_ids": ["LESSON-BOGUS-NOT-WRITTEN"], "strategy_ids": []},
            "invented_operators": [],
        }
    )

    def scripted(request):
        if getattr(request, "purpose", "") == "reviewer":
            return FakeProvider().complete(request).content
        return illegal

    report = run_v25d_real_loop(
        source_run_dir=source,
        output_dir=tmp_path / "loop_reject",
        execute=True,
        live_runner=boom,
        planner_backend="llm",
        reviewer_backend="llm",
        llm_provider=ScriptedProvider(scripted),
        human_probe_permission=True,
    )
    assert calls == []
    assert report["ignited"] is False
    assert report["metrics_forged"] is False
    assert report["ok"] is False


def test_probe_claim_gate_is_not_supported_c1(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    calls: list = []
    report = run_v25d_real_loop(
        source_run_dir=source,
        output_dir=tmp_path / "loop_stub",
        execute=True,
        live_runner=_stub_runner(calls),
        planner_backend="llm",
        reviewer_backend="llm",
        llm_provider=FakeProvider(),
        human_probe_permission=True,
    )
    assert calls, "Gate APPROVED + execute must call stub live_runner"
    assert report["ok"] is True
    assert report["ignited"] is True
    assert report["gpu"] is False
    claim = report["claim_gate"] or {}
    assert claim.get("status") != "SUPPORTED" or claim.get("claim_strength") != "C1"
    assert report.get("claim_gate_not_c1_supported") is True
    if claim:
        assert claim.get("keep_is_not_claim") is True
        assert_keep_is_not_claim(claim)
    assert report["how"]
    assert str(report["how"].get("fusion_method") or report["how"].get("primary_module") or "")
    assert report["rubric_review_decision"] in {"KEEP", "DISCARD", "REPLICATE", "VALIDATE"}
    assert report["evidence_class"] == "engineering_probe_not_c1"


def test_doctor_false_refuses_gpu_and_does_not_forge(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    calls: list = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("doctor false must not call live_runner")

    report = run_v25d_real_loop(
        source_run_dir=source,
        output_dir=tmp_path / "loop_doctor",
        execute=True,
        require_live_ready=True,
        live_runner=boom,
        doctor_fn=lambda: {"live_ready": False, "overall": "error"},
        planner_backend="llm",
        reviewer_backend="llm",
        llm_provider=FakeProvider(),
        human_probe_permission=True,
    )
    assert calls == []
    assert report["ok"] is False
    assert report["ignited"] is False
    assert report["metrics_forged"] is False
    round_dir = tmp_path / "loop_doctor" / "round"
    assert not (round_dir / "metrics.json").is_file()
    assert not (round_dir / "run" / "metrics.json").is_file()


def test_execute_without_human_permission_is_fail_closed(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    report = run_v25d_real_loop(
        source_run_dir=source,
        output_dir=tmp_path / "loop_nohuman",
        execute=True,
        live_runner=_stub_runner([]),
        llm_provider=FakeProvider(),
        human_probe_permission=False,
    )
    assert report["ok"] is False
    assert report["fail_closed"] is True
    assert report["ignited"] is False


def test_cli_manager_run_backends_default_rules() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "manager-run",
            "--protocol",
            "p.json",
            "--plan",
            "plan.json",
            "--output-dir",
            "out",
        ]
    )
    assert args.planner_backend == "rules"
    assert args.reviewer_backend == "rules"
    help_text = parser.format_help()
    assert "llm-real-loop" in help_text
    llm_args = parser.parse_args(
        ["manager-run", "--protocol", "p.json", "--plan", "plan.json", "--output-dir", "out", "--planner-backend", "llm"]
    )
    assert llm_args.planner_backend == "llm"


def test_cli_llm_real_loop_dry_run(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    out = tmp_path / "cli_loop"
    code = main(
        [
            "llm-real-loop",
            "--run-dir",
            str(source),
            "--output-dir",
            str(out),
            "--planner-backend",
            "llm",
            "--reviewer-backend",
            "llm",
        ]
    )
    assert code == 0
    report = load_json(out / "loop_report.json")
    assert report["ok"] is True
    assert report["execute"] is False
    assert report["ignited"] is False
    assert report["gate"]["status"] == "APPROVED"


def test_cli_live_without_key_does_not_fake_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _copy_fixture(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    out = tmp_path / "cli_live"
    code = main(
        [
            "llm-real-loop",
            "--run-dir",
            str(source),
            "--output-dir",
            str(out),
            "--live",
        ]
    )
    assert code == 1
    report = load_json(out / "loop_report.json")
    assert report["ok"] is False
    assert report["fail_closed"] is True
    assert report["ignited"] is False


def test_live_provider_without_key_is_fail_closed() -> None:
    with pytest.raises((MissingAPIKeyError, RealProviderNotEnabledError)):
        resolve_gateway_provider(live=True, environ={"LLM_PROVIDER": "openai-compatible"})
