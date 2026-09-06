from __future__ import annotations

from pathlib import Path

from scientist_lab.checkpoints.models import CheckpointFormat, CheckpointRecord, CheckpointRole
from scientist_lab.storage.artifact_store import sha256_file


_FORMAT_BY_SUFFIX = {
    ".pt": "pt",
    ".pth": "pt",
    ".npz": "npz",
    ".bin": "bin",
    ".safetensors": "safetensors",
    ".txt": "txt",
}


def infer_role(relative_path: str) -> CheckpointRole:
    name = Path(relative_path).name.lower()
    if "best" in name:
        return "best"
    if "last" in name:
        return "last"
    if name.startswith(("epoch_", "step_", "ckpt_")) or "/intermediate/" in (
        "/" + relative_path.replace("\\", "/")
    ):
        return "intermediate"
    return "other"


def infer_format(relative_path: str) -> CheckpointFormat:
    suffix = Path(relative_path).suffix.lower()
    return _FORMAT_BY_SUFFIX.get(suffix, "other")  # type: ignore[return-value]


def verify_checkpoint_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    min_bytes: int = 1,
) -> tuple[bool, str, int, str]:
    """Return (ok, message, size_bytes, sha256)."""
    if not path.is_file():
        return False, f"checkpoint_missing: {path}", 0, ""
    if path.is_symlink():
        return False, f"checkpoint_corrupt: symlink rejected ({path})", 0, ""
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"checkpoint_corrupt: empty or too small ({size} bytes)", size, ""
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        return (
            False,
            f"checkpoint_corrupt: hash mismatch expected={expected_sha256} actual={digest}",
            size,
            digest,
        )
    # Light format sniff for known extensions.
    suffix = path.suffix.lower()
    try:
        with path.open("rb") as file:
            head = file.read(16)
    except OSError as exc:
        return False, f"checkpoint_corrupt: unreadable ({exc})", size, digest
    if suffix in {".pt", ".pth", ".npz"} and not head:
        return False, "checkpoint_corrupt: empty payload", size, digest
    return True, "ok", size, digest


def verify_record(
    record: CheckpointRecord,
    *,
    outputs_root: Path,
) -> tuple[bool, str, CheckpointRecord]:
    path = Path(outputs_root) / record.project_id / record.execution_id / record.relative_path
    ok, message, size, digest = verify_checkpoint_file(
        path, expected_sha256=record.sha256
    )
    updated = record.model_copy(
        update={
            "verified": ok,
            "verification_message": message,
            "size_bytes": size or record.size_bytes,
            "sha256": digest or record.sha256,
        }
    )
    return ok, message, updated
