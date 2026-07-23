"""v1.9.8 Release Candidate + manifest tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.release.manifest import build_release_manifest, verify_release_manifest
from scientist_lab.release.models import MergeCandidate, ReleaseCandidate
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "rc.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_manifest_rejects_remote_published_flag():
    candidate = ReleaseCandidate(
        version="1.9.0",
        project_id="p1",
        base_tag="v1.8.0",
        commit_sha="abc123",
        included_patch_ids=["patch_1"],
        included_merge_candidate_ids=["mc_1"],
        status="draft",
    )
    manifest = build_release_manifest(candidate)
    candidate.manifest = {**manifest, "remote_published": True}
    result = verify_release_manifest(candidate)
    assert result["valid"] is False
    assert any("remote_published" in i for i in result["blocking_issues"])


def test_release_candidate_create_requires_merged(tmp_path: Path):
    service = _service(tmp_path)
    mc = MergeCandidate(
        project_id="project_rc_demo",
        patch_id="patch_fake",
        patch_evidence_id="pe_fake",
        source_commit="deadbeef",
        target_branch="main",
        workspace_path=str(tmp_path / "wt"),
        patch_sha256="a" * 64,
        diff_sha256="b" * 64,
        status="created",
    )
    service.merges._repo.upsert(mc)  # noqa: SLF001
    with pytest.raises(ValueError, match="must be merged"):
        service.create_release_candidate(
            version="1.9.0-rc1",
            project_id="project_rc_demo",
            merge_candidate_ids=[mc.merge_candidate_id],
        )


def test_release_candidate_create_and_verify_without_merges(tmp_path: Path):
    service = _service(tmp_path)
    created = service.create_release_candidate(
        version="1.9.0-rc1",
        project_id="project_rc_demo",
        base_tag="v1.8.0",
        merge_candidate_ids=[],
        notes="snapshot without merges",
    )
    assert created["release_candidate_id"]
    assert created["status"] == "draft"
    assert created["can_publish_remote"] is False
    assert created["manifest_path"]
    assert Path(created["manifest_path"]).is_file()
    assert created["manifest"]["schema_version"] == "1.9.8"
    assert created["manifest"]["remote_published"] is False

    verified = service.verify_release_candidate(created["release_candidate_id"])
    assert verified["verification"]["valid"] is True
    assert verified["status"] == "verified"

    shown = service.show_release_candidate(created["release_candidate_id"])
    assert shown["version"] == "1.9.0-rc1"
    listed = service.list_release_candidates(project_id="project_rc_demo")
    assert len(listed) == 1
