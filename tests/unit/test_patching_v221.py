"""v2.2.1 CodeContextBundle — offline, zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.context_bundle import (
    ContextBundleError,
    build_code_context_bundle,
    code_context_sha256,
    digits_improvement_patch_request,
)
from scientist_lab.patching.context_models import (
    AllowedSourceFile,
    ContextSizeBudget,
    PatchRequest,
)
from scientist_lab.patching.path_policy import PathPolicy


ROOT = Path(__file__).resolve().parents[2]


def test_digits_context_bundle_reads_only_allowed_entrypoint():
    req = digits_improvement_patch_request(
        request_id="preq_digits_stable",
        source_commit="fixed_commit_for_test",
    )
    bundle = build_code_context_bundle(
        req,
        project_root=ROOT,
        bundle_id="ctx_digits_stable",
    )
    assert bundle.source_commit == "fixed_commit_for_test"
    assert len(bundle.snapshots) == 1
    snap = bundle.snapshots[0]
    assert snap.path == "experiment_app/run_experiment.py"
    assert snap.content_sha256
    assert "MLPClassifier" in snap.content
    assert bundle.context_sha256
    assert bundle.total_bytes > 0
    assert bundle.metadata.get("provider_call") is False


def test_context_sha256_stable_across_rebuild():
    req = digits_improvement_patch_request(
        request_id="preq_digits_stable",
        source_commit="fixed_commit_for_test",
    )
    a = build_code_context_bundle(req, project_root=ROOT, bundle_id="ctx_a")
    b = build_code_context_bundle(req, project_root=ROOT, bundle_id="ctx_b")
    assert a.context_sha256 == b.context_sha256
    assert a.bundle_id != b.bundle_id
    assert code_context_sha256(a) == a.context_sha256


def test_context_excludes_secrets_and_env():
    req = PatchRequest(
        request_id="preq_deny",
        project_id="p1",
        goal="try forbidden files",
        allowed_files=[
            AllowedSourceFile(path="experiment_app/run_experiment.py", role="entrypoint"),
            AllowedSourceFile(path=".env", reason="should be denied"),
            AllowedSourceFile(path="src/scientist_lab/llm/config.py", reason="denied"),
            AllowedSourceFile(path="../outside.py", reason="escape"),
        ],
        source_commit="c0",
    )
    bundle = build_code_context_bundle(req, project_root=ROOT)
    paths = {s.path for s in bundle.snapshots}
    assert paths == {"experiment_app/run_experiment.py"}
    reasons = {e["path"]: e["reason"] for e in bundle.excluded_paths}
    assert ".env" in reasons
    assert "src/scientist_lab/llm/config.py" in reasons
    assert "../outside.py" in reasons or any("escape" in r for r in reasons.values())


def test_default_path_policy_still_blocks_digits_for_legacy_patch():
    """Legacy PatchProposal policy must not silently expand to Digits."""
    policy = PathPolicy()
    ok, reason = policy.is_allowed("experiment_app/run_experiment.py")
    assert ok is False
    assert "allowed" in reason


def test_context_policy_allows_digits():
    policy = PathPolicy.for_code_context()
    ok, _ = policy.is_allowed("experiment_app/run_experiment.py")
    assert ok is True


def test_budget_max_files():
    req = digits_improvement_patch_request(source_commit="c0")
    with pytest.raises(ContextBundleError, match="too many allowed files"):
        build_code_context_bundle(
            req,
            project_root=ROOT,
            budget=ContextSizeBudget(max_files=0),
        )


def test_empty_allowed_files_rejected():
    req = PatchRequest(
        request_id="preq_empty",
        project_id="p1",
        goal="x",
        allowed_files=[],
    )
    with pytest.raises(ContextBundleError, match="empty"):
        build_code_context_bundle(req, project_root=ROOT)


def test_persist_export_and_reload(tmp_path: Path):
    from scientist_lab.patching.context_store import (
        CodeContextRepository,
        export_code_context_bundle,
        load_exported_code_context_bundle,
    )
    from scientist_lab.storage.database import init_db

    db_path = str(tmp_path / "ctx.db")
    Session = init_db(db_path)
    repo = CodeContextRepository(Session)

    req = digits_improvement_patch_request(
        request_id="preq_persist",
        source_commit="fixed_commit_for_test",
    )
    bundle = build_code_context_bundle(
        req, project_root=ROOT, bundle_id="ctx_persist_1"
    )
    repo.upsert(bundle)

    loaded = repo.require("ctx_persist_1")
    assert loaded.context_sha256 == bundle.context_sha256
    assert loaded.snapshots[0].path == "experiment_app/run_experiment.py"

    export_path = tmp_path / "bundle.json"
    meta = export_code_context_bundle(loaded, output_path=export_path)
    assert export_path.is_file()
    assert meta["context_sha256"] == bundle.context_sha256

    from_disk = load_exported_code_context_bundle(export_path)
    assert from_disk.context_sha256 == bundle.context_sha256
    assert from_disk.snapshots[0].content_sha256 == bundle.snapshots[0].content_sha256


def test_denies_git_db_and_home_style_paths():
    policy = PathPolicy.for_code_context()
    for path, needle in [
        (".git/config", "denied"),
        ("scientist_lab.db", "denied"),
        ("~/secrets.txt", "absolute"),
        ("C:/Users/me/.env", "absolute"),
    ]:
        ok, reason = policy.is_allowed(path)
        assert ok is False
        assert needle in reason
