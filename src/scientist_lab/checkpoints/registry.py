from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import sessionmaker

from scientist_lab.checkpoints.models import CheckpointRecord
from scientist_lab.checkpoints.repository import CheckpointRepository
from scientist_lab.checkpoints.verifier import (
    infer_format,
    infer_role,
    verify_checkpoint_file,
    verify_record,
)
from scientist_lab.domain.models import new_id, utc_now_iso


class CheckpointRegistry:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        outputs_root: Path,
    ) -> None:
        self._repo = CheckpointRepository(session_factory)
        self.outputs_root = Path(outputs_root).resolve()

    def register(
        self,
        *,
        project_id: str,
        execution_id: str,
        relative_path: str,
        node_id: str | None = None,
        role: str | None = None,
        baseline_key: str | None = None,
        metadata: dict | None = None,
        verify: bool = True,
    ) -> CheckpointRecord:
        rel = relative_path.replace("\\", "/").lstrip("/")
        path = self.outputs_root / project_id / execution_id / rel
        ok, message, size, digest = verify_checkpoint_file(path)
        if not ok and verify:
            raise FileNotFoundError(message) if "missing" in message else ValueError(message)

        existing = self._repo.get_by_execution_path(execution_id, rel)
        now = utc_now_iso()
        record = CheckpointRecord(
            checkpoint_id=existing.checkpoint_id if existing else new_id("ckpt"),
            project_id=project_id,
            execution_id=execution_id,
            node_id=node_id,
            relative_path=rel,
            role=(role or infer_role(rel)),  # type: ignore[arg-type]
            format=infer_format(rel),
            size_bytes=size,
            sha256=digest,
            baseline_key=baseline_key,
            verified=ok,
            verification_message=message if verify else None,
            metadata=dict(metadata or {}),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        return self._repo.upsert(record)

    def register_from_execution(
        self,
        *,
        project_id: str,
        execution_id: str,
        node_id: str | None = None,
        baseline_key: str | None = None,
        preferred_only: bool = True,
    ) -> list[CheckpointRecord]:
        """Discover and register checkpoint files under an execution output dir."""
        output_dir = self.outputs_root / project_id / execution_id
        if not output_dir.is_dir():
            return []

        candidates: list[Path] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir).as_posix()
            if not rel.startswith("checkpoint/"):
                continue
            if path.suffix.lower() not in {".pt", ".pth", ".npz", ".bin", ".safetensors", ".txt"}:
                continue
            role = infer_role(rel)
            if preferred_only and role not in {"best", "last"}:
                continue
            candidates.append(path)

        # Prefer best over last when both exist for the same stem family.
        registered: list[CheckpointRecord] = []
        for path in candidates:
            rel = path.relative_to(output_dir).as_posix()
            try:
                registered.append(
                    self.register(
                        project_id=project_id,
                        execution_id=execution_id,
                        relative_path=rel,
                        node_id=node_id,
                        baseline_key=baseline_key,
                        verify=True,
                    )
                )
            except (FileNotFoundError, ValueError):
                continue
        return registered

    def get(self, checkpoint_id: str) -> CheckpointRecord | None:
        return self._repo.get(checkpoint_id)

    def require(self, checkpoint_id: str) -> CheckpointRecord:
        record = self.get(checkpoint_id)
        if record is None:
            raise KeyError(f"checkpoint not found: {checkpoint_id}")
        return record

    def list_checkpoints(
        self,
        *,
        project_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[CheckpointRecord]:
        return self._repo.list_checkpoints(
            project_id=project_id, execution_id=execution_id
        )

    def verify(self, checkpoint_id: str) -> CheckpointRecord:
        record = self.require(checkpoint_id)
        ok, _message, updated = verify_record(record, outputs_root=self.outputs_root)
        updated = updated.model_copy(update={"updated_at": utc_now_iso(), "verified": ok})
        return self._repo.upsert(updated)

    def resolve_path(self, checkpoint_id: str) -> Path:
        record = self.require(checkpoint_id)
        return (
            self.outputs_root
            / record.project_id
            / record.execution_id
            / record.relative_path
        )
