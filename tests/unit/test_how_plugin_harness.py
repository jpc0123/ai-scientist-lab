"""Harness Docker HOW plugin worker. Injected runtime; no GPU; no image pull."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how_catalog import resolve_how_id
from scientist_lab.core.how_plugin_author import (
    HowPluginAuthorError,
    author_how_patch,
    example_weighted_plugin_source,
)
from scientist_lab.core.how_plugin_harness import (
    DEFAULT_DSH_IMAGE,
    DockerHarnessRunner,
    HarnessContainerRequest,
    HarnessPluginWorker,
    InjectedHarnessRunner,
    PLUGIN_STUB,
    harness_plugin_task_prompt,
    plugin_relpath_for,
    redact_harness_env,
    stage_harness_workspace,
)
from scientist_lab.core.how_pending import load_store
from scientist_lab.core.how_plugin_worker import PluginAuthorWorkerError

from tests.unit.test_how_plugin_bridge import ROOT, _seed_f2


def test_cordis_template_uses_openai_compatible_not_deepseek_official() -> None:
    tmpl = (
        ROOT / "docker" / "dsh-plugin-author" / "cordis.yml.tmpl"
    ).read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-llm-pi-ai" in tmpl
    assert "openai-completions" in tmpl
    assert "supportsDeveloperRole: false" in tmpl
    assert "maxTokensField: max_tokens" in tmpl
    assert "dsh-llm-deepseek" not in tmpl
    rendered = tmpl.replace("__LAB_BASE_URL__", "https://example.invalid/v1").replace(
        "__LAB_MODEL__", "qwen3.7-max"
    )
    assert "https://example.invalid/v1" in rendered
    assert "qwen3.7-max" in rendered
    assert "sk-" not in rendered


def test_harness_env_defaults_to_lab_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from scientist_lab.core.how_plugin_harness import _harness_env

    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-max")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.delenv("DSH_PROVIDER", raising=False)
    env = _harness_env("F2", plugin_relpath_for("F2"))
    assert env["DSH_PROVIDER"] == "lab"
    assert env["DSH_MODEL"] == "qwen3.7-max"
    assert env["DSH_SESSION_ID"].startswith("F2-")
    assert env["DEEPSEEK_API_KEY"] == "sk-test"
    redacted = redact_harness_env(env)
    assert redacted["DEEPSEEK_API_KEY"] == "***"


def test_harness_task_prompt_asks_to_write_file_not_json_diff() -> None:
    prompt = harness_plugin_task_prompt(
        how_id="F2",
        rel=plugin_relpath_for("F2"),
        system_prompt="Return JSON with a Unified Diff only.",
        goal="Author HOW plugin F2",
    )
    assert "Write a complete Python module" in prompt
    assert "Do not return JSON" in prompt
    assert "Return JSON with a Unified Diff only" not in prompt
    assert "Author HOW plugin F2" in prompt


def _example_runner() -> InjectedHarnessRunner:
    source = example_weighted_plugin_source(ROOT)

    def write(_dest: Path, _request: HarnessContainerRequest) -> str:
        return source

    return InjectedHarnessRunner(write)


def test_stage_workspace_excludes_trainer_and_vendor(tmp_path: Path) -> None:
    workspace = stage_harness_workspace(
        project_root=ROOT,
        work_root=tmp_path,
        how_id="F2",
        prompt="write plugin.py",
    )
    rels = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
    assert "PLUGIN_TASK.md" in rels
    assert "experiment_apps/rgbt_detection_real/models/feature_fusion.py" in rels
    assert plugin_relpath_for("F2") in rels
    plugin_body = (workspace / plugin_relpath_for("F2")).read_text(encoding="utf-8")
    assert PLUGIN_STUB.strip() in plugin_body
    assert "def build_fusion" not in plugin_body
    assert not any("train_dfine.py" in row for row in rels)
    assert not any("third_party" in row for row in rels)
    assert not any("fusion_factory.py" in row for row in rels)


def test_injected_harness_authors_plugin_no_gpu(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    worker = HarnessPluginWorker(
        project_root=ROOT,
        runner=_example_runner(),
        work_root=tmp_path / "harness",
    )
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        worker=worker,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["gpu"] is False
    assert authored["registered"] is False
    assert authored["plugin_authored_by"] == "harness"
    assert authored["plugin_worker_id"] == "harness"
    saved = load_store(path)
    assert saved["candidates"][0]["smoke_ok"] is True
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_injected_harness_empty_plugin_fail_closed(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)

    def write(_dest: Path, _request: HarnessContainerRequest) -> str:
        return ""

    worker = HarnessPluginWorker(
        project_root=ROOT,
        runner=InjectedHarnessRunner(write),
        work_root=tmp_path / "harness",
    )
    with pytest.raises(HowPluginAuthorError, match="empty plugin"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            worker=worker,
            sandbox_root=tmp_path / "sandboxes",
        )


def test_injected_harness_subprocess_plugin_fail_closed(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)

    def write(_dest: Path, _request: HarnessContainerRequest) -> str:
        return (
            "import subprocess\n\n"
            "def build_fusion(**k):\n"
            "    subprocess.call(['echo', 'x'])\n"
        )

    worker = HarnessPluginWorker(
        project_root=ROOT,
        runner=InjectedHarnessRunner(write),
        work_root=tmp_path / "harness",
    )
    with pytest.raises(HowPluginAuthorError):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            worker=worker,
            sandbox_root=tmp_path / "sandboxes",
        )
    assert load_store(path)["candidates"][0].get("smoke_ok") is not True


def test_docker_runner_missing_image_fail_closed(tmp_path: Path) -> None:
    class _Images:
        def get(self, _name: str) -> None:
            raise RuntimeError("missing")

    class _Client:
        images = _Images()

    runner = DockerHarnessRunner(client=_Client(), image=DEFAULT_DSH_IMAGE)
    request = HarnessContainerRequest(
        workspace=tmp_path,
        plugin_relpath=plugin_relpath_for("F2"),
        how_id="F2",
        prompt="x",
        image=DEFAULT_DSH_IMAGE,
        environment={},
    )
    with pytest.raises(PluginAuthorWorkerError, match="image_not_found"):
        runner.run(request)


def test_redact_api_key() -> None:
    redacted = redact_harness_env({"DEEPSEEK_API_KEY": "sk-secret", "HOW_ID": "F2"})
    assert redacted["DEEPSEEK_API_KEY"] == "***"
    assert redacted["HOW_ID"] == "F2"
