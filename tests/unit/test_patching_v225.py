"""v2.2.5 approval content seals — tamper voids approval (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.approval_seal import (
    ApprovalSealError,
    compare_approval_seal,
    proposal_content_sha256,
)
from scientist_lab.patching.service import PatchingService, build_mock_unified_diff
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_approve_binds_content_seal(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/seal_note.md"
            )
        ),
    )
    # Attach context fingerprints as real proposals would.
    item = service.patches._repo.require(proposed["patch_id"])
    item.metadata["context_sha256"] = "ctx_deadbeef"
    item.metadata["source_commit"] = "abc123"
    service.patches._repo.upsert(item)

    approved = service.patches.approve(proposed["patch_id"], reason="seal me")
    assert approved["status"] == "approved"
    seal = approved["approval"]["content_seal"]
    assert seal["context_sha256"] == "ctx_deadbeef"
    assert seal["source_commit"] == "abc123"
    assert seal["patch_sha256"]
    assert seal["proposal_sha256"]
    assert seal["seal_version"] == "v2.2.5"

    checked = service.patches.check_approval_seal(proposed["patch_id"])
    assert checked["seal_report"]["ok"] is True
    assert checked["status"] == "approved"


def test_tamper_invalidates_approval_and_blocks_sandbox(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/tamper_note.md"
            )
        ),
    )
    service.patches.approve(proposed["patch_id"])

    item = service.patches._repo.require(proposed["patch_id"])
    item.unified_diff = item.unified_diff + "\n+# tampered\n"
    item.fingerprint_sha256 = "not_the_original_fingerprint"
    service.patches._repo.upsert(item)

    checked = service.patches.check_approval_seal(proposed["patch_id"])
    assert checked["seal_report"]["ok"] is False
    assert checked["status"] == "verified"
    assert checked["approval"]["invalidated"] is True

    # Re-approve current content, then tamper title → apply must fail.
    service.patches.approve(proposed["patch_id"])
    item = service.patches._repo.require(proposed["patch_id"])
    item.title = item.title + " TAMPERED"
    service.patches._repo.upsert(item)
    with pytest.raises(ApprovalSealError, match="re-approve"):
        service.patches.apply_sandbox(proposed["patch_id"])
    after = service.patches.show(proposed["patch_id"])
    assert after["status"] == "verified"
    assert after["can_apply_sandbox"] is False


def test_proposal_sha_stable_for_same_content(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    a = service.patches._repo.require(proposed["patch_id"])
    b = service.patches._repo.require(proposed["patch_id"])
    assert proposal_content_sha256(a) == proposal_content_sha256(b)


def test_compare_reports_mismatch_fields(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/cmp_note.md"
            )
        ),
    )
    service.patches.approve(proposed["patch_id"])
    item = service.patches._repo.require(proposed["patch_id"])
    item.rationale = "changed after approval"
    report = compare_approval_seal(item)
    assert report["ok"] is False
    assert "proposal_sha256" in report["mismatches"]
