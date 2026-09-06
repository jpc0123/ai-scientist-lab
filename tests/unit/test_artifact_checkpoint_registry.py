from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.artifacts.bundle import validate_bundle_archive
from scientist_lab.artifacts.policy import ArtifactPolicy
from scientist_lab.checkpoints.registry import CheckpointRegistry
from scientist_lab.runners.remote_errors import RemoteArtifactCorrupt
from scientist_lab.storage.database import init_db
from scientist_worker.artifact_packager import package_artifacts, should_include_path


def test_artifact_policy_excludes_intermediate_checkpoints():
    policy = ArtifactPolicy(include_intermediate_checkpoints=False)
    assert policy.should_include("metrics.json")
    assert policy.should_include("checkpoint/last.pt")
    assert policy.should_include("checkpoint/best.pt")
    assert not policy.should_include("checkpoint/epoch_3.pt")
    assert not policy.should_include("checkpoint/intermediate/foo.pt")


def test_package_artifacts_respects_policy(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (output_dir / "execution.json").write_text("{}", encoding="utf-8")
    ckpt = output_dir / "checkpoint"
    ckpt.mkdir()
    (ckpt / "last.pt").write_bytes(b"last-bytes")
    (ckpt / "epoch_1.pt").write_bytes(b"epoch-bytes")

    packaged = package_artifacts(
        output_dir,
        job_id="job_1",
        execution_id="exec_1",
        policy={"include_intermediate_checkpoints": False},
    )
    paths = {item["path"] for item in packaged["manifest"]["files"]}
    assert "checkpoint/last.pt" in paths
    assert "checkpoint/epoch_1.pt" not in paths
    assert "metrics.json" in paths


def test_validate_bundle_detects_hash_mismatch(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text('{"ok":true}', encoding="utf-8")
    (output_dir / "execution.json").write_text("{}", encoding="utf-8")
    packaged = package_artifacts(
        output_dir, job_id="job_x", execution_id="exec_x"
    )
    bundle = Path(packaged["bundle_path"])
    manifest = dict(packaged["manifest"])
    manifest["bundle_sha256"] = "0" * 64
    with pytest.raises(RemoteArtifactCorrupt, match="bundle hash mismatch"):
        validate_bundle_archive(
            bundle,
            manifest=manifest,
            policy=ArtifactPolicy(),
            extract_to=tmp_path / "extract",
        )


def test_validate_bundle_extracts_and_checks_file_hashes(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text('{"m":1}', encoding="utf-8")
    (output_dir / "execution.json").write_text("{}", encoding="utf-8")
    packaged = package_artifacts(
        output_dir, job_id="job_y", execution_id="exec_y"
    )
    extract_to = tmp_path / "extract"
    report = validate_bundle_archive(
        Path(packaged["bundle_path"]),
        manifest=packaged["manifest"],
        policy=ArtifactPolicy(),
        extract_to=extract_to,
    )
    assert (extract_to / "metrics.json").is_file()
    assert "metrics.json" in report["extracted_files"]


def test_checkpoint_registry_register_list_verify(tmp_path: Path):
    outputs = tmp_path / "outputs"
    project_id = "proj_a"
    execution_id = "exec_a"
    ckpt_dir = outputs / project_id / execution_id / "checkpoint"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "last.pt"
    ckpt_path.write_bytes(b"fake-weights-001")

    session_factory = init_db(str(tmp_path / "lab.db"))
    registry = CheckpointRegistry(session_factory, outputs_root=outputs)
    record = registry.register(
        project_id=project_id,
        execution_id=execution_id,
        relative_path="checkpoint/last.pt",
        baseline_key="dfine_s",
    )
    assert record.role == "last"
    assert record.format == "pt"
    assert record.verified is True

    listed = registry.list_checkpoints(project_id=project_id)
    assert len(listed) == 1
    assert listed[0].checkpoint_id == record.checkpoint_id

    verified = registry.verify(record.checkpoint_id)
    assert verified.verified is True

    ckpt_path.write_bytes(b"tampered")
    failed = registry.verify(record.checkpoint_id)
    assert failed.verified is False
    assert "hash mismatch" in (failed.verification_message or "")


def test_checkpoint_registry_discover_from_execution(tmp_path: Path):
    outputs = tmp_path / "outputs"
    project_id = "proj_b"
    execution_id = "exec_b"
    root = outputs / project_id / execution_id / "checkpoint"
    root.mkdir(parents=True)
    (root / "best.pt").write_bytes(b"best")
    (root / "last.pt").write_bytes(b"last")
    (root / "epoch_2.pt").write_bytes(b"epoch")

    session_factory = init_db(str(tmp_path / "lab2.db"))
    registry = CheckpointRegistry(session_factory, outputs_root=outputs)
    registered = registry.register_from_execution(
        project_id=project_id,
        execution_id=execution_id,
        preferred_only=True,
    )
    roles = {item.role for item in registered}
    assert roles == {"best", "last"}


def test_should_include_path_worker_helper():
    assert should_include_path("combined.log", None)
    assert not should_include_path(
        "checkpoint/epoch_9.pt",
        {"include_intermediate_checkpoints": False},
    )
