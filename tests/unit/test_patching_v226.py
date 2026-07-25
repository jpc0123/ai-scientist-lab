"""v2.2.6 sandbox apply + registered test profiles (Digits path; offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.sandbox_registry import (
    list_sandbox_test_profiles,
    require_sandbox_test_profile,
)
from scientist_lab.patching.service import PatchingService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.database import init_db


ROOT = Path(__file__).resolve().parents[2]

DIGITS_DIFF = """\
diff --git a/experiment_app/run_experiment.py b/experiment_app/run_experiment.py
--- a/experiment_app/run_experiment.py
+++ b/experiment_app/run_experiment.py
@@ -88,7 +88,7 @@ def main() -> None:
     random.seed(seed)
     np.random.seed(seed)
 
-    print("Starting real Digits MLP experiment", flush=True)
+    print("Starting real Digits MLP experiment (sandbox-v226)", flush=True)
     print(f"seed={seed}", flush=True)
     print(f"learning_rate={learning_rate}", flush=True)
     print(f"epochs={epochs}", flush=True)
"""


def _service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def test_registry_lists_unit_and_rejects_unknown():
    ids = {p["id"] for p in list_sandbox_test_profiles()}
    assert ids == {"smoke", "syntax", "unit", "mock_experiment"}
    assert require_sandbox_test_profile("unit") == "unit"
    with pytest.raises(ValueError, match="unknown sandbox test profile"):
        require_sandbox_test_profile("pytest")
    with pytest.raises(ValueError, match="unknown sandbox test profile"):
        require_sandbox_test_profile("rm -rf /")


def test_digits_sandbox_apply_and_unit_profile(tmp_path: Path):
    service = _service(tmp_path)
    # Propose via PatchingService with context verifier path: use propose_mock
    # with digits diff would fail default verifier — go through repo + verify
    # using context policy via apply after manual insert... easier: use
    # PatchingService directly and verify with context policy.
    Session = init_db(str(tmp_path / "v226.db"))
    patches = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=tmp_path / "outputs",
        sandbox_root=tmp_path / "outputs" / "_patch_sandboxes",
    )
    from scientist_lab.domain.models import new_id
    from scientist_lab.patching.fingerprint import fingerprint_diff
    from scientist_lab.patching.models import PatchProposal
    from scientist_lab.patching.verifier import PatchVerifier

    diff = DIGITS_DIFF
    verification = PatchVerifier(patches.context_policy).verify(diff)
    assert verification.ok, verification.issues
    proposal = PatchProposal(
        patch_id=new_id("patch"),
        project_id="digits_sandbox_v226",
        status="verified",
        title="Digits sandbox v2.2.6",
        rationale="cosmetic log",
        unified_diff=diff,
        files_touched=list(verification.files_touched),
        fingerprint_sha256=fingerprint_diff(diff),
        provider="openai-compatible",
        verification=verification,
        metadata={
            "context_sha256": "ctx_test",
            "source_commit": "c0",
            "bundle_id": "ctx_test",
        },
    )
    patches._repo.upsert(proposal)
    approved = patches.approve(proposal.patch_id)
    assert approved["status"] == "approved"
    assert approved["approval"]["content_seal"]["context_sha256"] == "ctx_test"

    applied = patches.apply_sandbox(proposal.patch_id)
    assert applied["status"] == "applied_sandbox"
    assert applied["sandbox"]["ok"] is True
    assert applied["sandbox"]["main_workspace_modified"] is False
    assert "experiment_app/run_experiment.py" in applied["sandbox"]["files_written"]
    sandbox_file = (
        Path(applied["sandbox"]["sandbox_dir"]) / "experiment_app/run_experiment.py"
    )
    assert "sandbox-v226" in sandbox_file.read_text(encoding="utf-8")
    # Main tree unchanged.
    main = ROOT / "experiment_app" / "run_experiment.py"
    assert "sandbox-v226" not in main.read_text(encoding="utf-8")

    unit = patches.test_sandbox(proposal.patch_id, profile="unit")
    assert unit["sandbox_tests"]["ok"] is True
    names = {c["name"] for c in unit["sandbox_tests"]["checks"]}
    assert "unit_registry_only" in names
    assert "unit_digits_entrypoint" in names
    assert "python_syntax" in names


def test_unknown_profile_rejected_on_service(tmp_path: Path):
    service = _service(tmp_path)
    profiles = service.patches.list_sandbox_test_profiles()
    assert profiles["arbitrary_commands_forbidden"] is True
    assert profiles["total"] == 4
