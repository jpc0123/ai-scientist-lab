"""v1.5.3–v1.5.5: eval runner, profiles, scorecard artifacts."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.llm_eval.runner import run_evaluation_suite
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def test_run_evaluation_suite_mock_writes_artifacts(tmp_path: Path):
    out = tmp_path / "outputs"
    result = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=out,
    )
    assert result["status"] == "completed"
    assert result["provider"] == "mock"
    assert result["suite_version"] == "eval_suite_v1"
    assert (result.get("safety") or {}).get("pass") is True
    assert (result.get("safety") or {}).get("case_count") == 5
    paths = result["paths"]
    for key in (
        "evaluation_report_json",
        "evaluation_report_md",
        "case_results",
        "failure_cases",
        "manifest",
    ):
        assert Path(paths[key]).is_file()
    assert result["operations"]["case_count"] == 25


def test_run_evaluation_suite_real_skipped_without_gates(tmp_path: Path):
    result = run_evaluation_suite(
        "eval_suite_v1",
        provider="real",
        allow_network=False,
        project_id="project_rgbt_003",
        output_root=tmp_path / "outputs",
    )
    assert result["status"] == "skipped"
    assert result["real_network_called"] is False
    assert "allow-network" in str(result.get("skip_reason") or "").lower() or result.get(
        "skip_reason"
    )


def test_run_evaluation_suite_replay(tmp_path: Path):
    out = tmp_path / "outputs"
    result = run_evaluation_suite(
        "eval_suite_v1",
        provider="replay",
        project_id="project_rgbt_003",
        output_root=out,
        seed_fake_for_replay=True,
        task_types=["planner", "safety"],
    )
    assert result["status"] == "completed"
    assert result["provider"] == "replay"
    assert (result.get("safety") or {}).get("pass") is True


def test_profile_register_list_show(tmp_path: Path):
    service = _service(tmp_path)
    root = Path(__file__).resolve().parents[2]
    registered = service.register_llm_profile(root / "examples" / "llm_profile.json")
    assert registered["profile_id"] == "mock_default"
    listed = service.list_llm_profiles()
    assert any(p["profile_id"] == "mock_default" for p in listed)
    shown = service.show_llm_profile("mock_default")
    assert shown["model"] == "mock-planner-v1"

    custom = LLMModelProfile(
        profile_id="openai_model_a",
        provider="openai-compatible",
        model="gpt-test",
        api_mode="chat_completions",
        planner_prompt_version="planner_v2",
        critic_prompt_version="critic_v2",
        temperature=0.0,
        max_output_tokens=1024,
    )
    service.register_llm_profile(profile=custom)
    assert service.show_llm_profile("openai_model_a")["planner_prompt_version"] == "planner_v2"


def test_service_run_llm_eval_suite(tmp_path: Path):
    service = _service(tmp_path)
    service.register_llm_profile(
        Path(__file__).resolve().parents[2] / "examples" / "llm_profile.json"
    )
    result = service.run_llm_eval_suite(
        "project_rgbt_003",
        suite="eval_suite_v1",
        provider="mock",
        profile_id="mock_default",
    )
    assert result["status"] == "completed"
    assert result["profile_id"] == "mock_default"
    stored = service.get_llm_evaluation(result["evaluation_id"])
    assert stored["evaluation_id"] == result["evaluation_id"]
    assert stored["suite_version"] == "eval_suite_v1"
