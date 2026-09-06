"""v1.4.4 llm-eval suite gates and offline suite runner."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.llm.eval_suite import (
    real_eval_gates,
    run_llm_eval_suite,
    write_suite_report,
)


def test_real_gates_default_block():
    ok, reason = real_eval_gates(
        provider="real",
        allow_network=False,
        environ={},
    )
    assert ok is False
    assert "allow-network" in reason


def test_real_gates_require_env_flags():
    ok, reason = real_eval_gates(
        provider="real",
        allow_network=True,
        environ={"LLM_ALLOW_NETWORK": "true"},
    )
    assert ok is False
    assert "RUN_REAL_LLM_TESTS" in reason


def test_real_suite_skipped_without_gates(tmp_path: Path):
    report = run_llm_eval_suite(
        suite="planner-basic",
        provider="real",
        allow_network=False,
        audit_root=tmp_path / "llm",
        evals_root=Path(__file__).resolve().parents[2] / "evals" / "llm",
    )
    assert report.skipped is True
    assert report.real_network_called is False
    assert "skipped" in (report.skip_reason or "").lower()


def test_mock_suite_runs_offline(tmp_path: Path):
    report = run_llm_eval_suite(
        suite="planner-basic",
        provider="mock",
        project_id="project_rgbt_003",
        audit_root=tmp_path / "llm",
        evals_root=Path(__file__).resolve().parents[2] / "evals" / "llm",
    )
    assert report.skipped is False
    assert report.case_count == 3
    assert report.executed_count == 3
    assert report.schema_valid_rate == 1.0
    assert report.real_network_called is False
    out = write_suite_report(report, tmp_path / "suite.json")
    assert Path(out.path or "").is_file()


def test_fake_and_replay_suite(tmp_path: Path):
    root = Path(__file__).resolve().parents[2] / "evals" / "llm"
    fake = run_llm_eval_suite(
        suite="planner-basic",
        provider="fake",
        audit_root=tmp_path / "llm",
        evals_root=root,
        max_cases=1,
    )
    assert fake.executed_count == 1
    replay = run_llm_eval_suite(
        suite="planner-basic",
        provider="replay",
        audit_root=tmp_path / "llm",
        evals_root=root,
        max_cases=1,
    )
    assert replay.executed_count == 1
    assert replay.provider == "replay"
