"""v2.3.1 unit tests — Vendor DFINE offline audit (zero GPU / zero network)."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.evidence.claim_matrix import evaluate_formal_dfine_claim
from scientist_lab.evidence.models import EvidenceRecord
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter
from scientist_lab.tasks.rgbt_detection.vendor_audit import (
    BASELINE_IMPLEMENTATION,
    PINNED_COMMIT,
    audit_dfine_vendor_pin,
    cuda_dockerfile_present,
    resolve_dfine_vendor_root,
    stage_dfine_vendor,
)

ROOT = Path(__file__).resolve().parents[2]


def test_audit_real_vendor_pin():
    audit = audit_dfine_vendor_pin(ROOT)
    assert audit["ok"] is True
    assert audit["pinned_commit"] == PINNED_COMMIT
    assert audit["gpu_used"] is False
    assert audit["network_used"] is False
    assert resolve_dfine_vendor_root(ROOT) is not None


def test_stage_dfine_vendor_copies_train_py(tmp_path: Path):
    fake = tmp_path / "proj"
    dfine = fake / "third_party" / "DFINE"
    dfine.mkdir(parents=True)
    (dfine / "train.py").write_text("print('dfine')\n", encoding="utf-8")
    (dfine / "LICENSE").write_text("MIT\n", encoding="utf-8")

    ws = tmp_path / "workspace"
    result = stage_dfine_vendor(ws, project_root=fake)
    assert result["ok"] is True
    assert result["copied"] is True
    dest = Path(result["dest"])
    assert (dest / "train.py").is_file()
    assert (dest / "LICENSE").is_file()

    again = stage_dfine_vendor(ws, project_root=fake)
    assert again["ok"] is True
    assert again["copied"] is False
    assert again["already_present"] is True


def test_cuda_dockerfile_and_registry():
    assert cuda_dockerfile_present(ROOT)
    settings = Settings(project_root=ROOT).resolve()
    assert settings.image_registry["rgbt-detection-v2-cuda"] == (
        "scientist-rgbt-detection:v2-cuda"
    )


def test_adapter_matches_vendor_pin():
    adapter = DFineSBaselineAdapter()
    contract = ExperimentContract.model_validate_json(
        (ROOT / "examples" / "rgbt_remote_cuda_dfine_rgb_contract.json").read_text(
            encoding="utf-8"
        )
    )
    native = adapter.build_native_config(contract)
    assert native["baseline_implementation"] == BASELINE_IMPLEMENTATION
    assert native["vendor_commit"] == PINNED_COMMIT
    assert contract.environment_key == "rgbt-detection-v2-cuda"
    assert contract.parameters.get("dfine_backend") == "dfine"


def test_formal_claim_blocked_for_standin():
    record = EvidenceRecord(
        evidence_id="ev_standin",
        project_id="p1",
        evidence_type="paired_comparison",
        limitations=["stand-in evidence only"],
    )
    claim = evaluate_formal_dfine_claim(project_id="p1", records=[record])
    assert claim.support_status == "blocked"


def test_cuda_doctor_offline_no_probe():
    from scientist_lab.tasks.rgbt_detection.cuda_doctor import build_dfine_cuda_doctor

    settings = Settings(project_root=ROOT).resolve()
    report = build_dfine_cuda_doctor(
        ROOT,
        image_registry=dict(settings.image_registry or {}),
        probe_runtime=False,
    )
    assert report["ok"] is True
    assert report["runtime"]["probed"] is False
    assert report["gpu_required_for_live"] is True
    assert report.get("doctor_version") == "v2.3.7"
    ids = {c["id"] for c in report["checks"]}
    assert "vendor_pin" in ids
    assert "cuda_dockerfile" in ids
    assert "example_contract" in ids
    assert "cuda_protocol" in ids
