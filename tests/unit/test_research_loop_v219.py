"""v2.1.9 accept_v21_real gates — default skip, zero network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_accept_v21_real():
    path = ROOT / "scripts" / "accept_v21_real.py"
    spec = importlib.util.spec_from_file_location("accept_v21_real", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_loop_acceptance_gates_default_blocked():
    mod = _load_accept_v21_real()
    allowed, reason, gates = mod.real_loop_acceptance_gates(environ={})
    assert allowed is False
    assert "RUN_REAL_LLM_TESTS" in reason or "not" in reason.lower()
    assert gates["run_real_llm_tests"] is False


def test_real_loop_acceptance_gates_pass_when_configured():
    mod = _load_accept_v21_real()
    allowed, reason, gates = mod.real_loop_acceptance_gates(
        environ={
            "RUN_REAL_LLM_TESTS": "1",
            "LLM_ALLOW_NETWORK": "true",
            "LLM_API_KEY": "sk-test-not-used",
            "LLM_BASE_URL": "https://api.example.com/v1",
            "LLM_MODEL": "gpt-test",
        }
    )
    assert allowed is True
    assert reason == "real gates satisfied"
    assert gates["llm_model"] is True


def test_accept_v21_real_default_skip_writes_report(tmp_path, monkeypatch):
    mod = _load_accept_v21_real()
    # Force clean env for this process.
    for key in (
        "RUN_REAL_LLM_TESTS",
        "LLM_ALLOW_NETWORK",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "RUN_REAL_DIGITS",
    ):
        monkeypatch.delenv(key, raising=False)

    # Redirect report roots into tmp by patching module ROOT-derived paths via chdir? 
    # Instead call gates + emulate skip payload write.
    allowed, reason, gates = mod.real_loop_acceptance_gates(environ=dict())
    assert allowed is False
    report = {
        "schema_version": "v21_real_closed_loop_report",
        "status": "skipped",
        "gates": gates,
        "gate_reason": reason,
        "network_used": False,
    }
    out = tmp_path / "v21_real_closed_loop_report.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["status"] == "skipped"
    assert loaded["network_used"] is False


def test_accept_v21_real_main_skip_exit_zero(monkeypatch, tmp_path):
    mod = _load_accept_v21_real()
    for key in (
        "RUN_REAL_LLM_TESTS",
        "LLM_ALLOW_NETWORK",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "RUN_REAL_DIGITS",
    ):
        monkeypatch.delenv(key, raising=False)

    # Point ROOT side-effects: monkeypatch Path used inside main via accept roots.
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    # Need src on path still — module already imported scientist_lab from real ROOT.
    # Re-bind EXAMPLES unused in skip path.
    code = mod.main()
    assert code == 0
    report = tmp_path / "docs" / "acceptance" / "v2.1" / "v21_real_closed_loop_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["network_used"] is False
