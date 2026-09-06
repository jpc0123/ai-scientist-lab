from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from sqlalchemy.orm import sessionmaker

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.datasets.repository import DatasetRepository
from scientist_lab.domain.models import utc_now_iso


DEFAULT_RGBT_METADATA = {
    "modalities": ["rgb", "thermal"],
    "annotation_format": "coco",
    "dataset_scope": "debug_subset",
    "claim_level": "pipeline_validation_only",
}


def parse_dataset_reference(reference: str) -> str | None:
    """Return dataset_key if reference looks like dataset:<key>, else None."""
    text = (reference or "").strip()
    if text.startswith("dataset:"):
        key = text.split(":", 1)[1].strip()
        return key or None
    return None


def is_linux_absolute_path(path: str) -> bool:
    text = (path or "").strip()
    if not text.startswith("/"):
        return False
    try:
        pure = PurePosixPath(text)
    except Exception:  # noqa: BLE001
        return False
    return pure.is_absolute() and "\\" not in text


def is_dangerous_host_path(host_path: Path, *, project_root: Path | None = None) -> str | None:
    """Return reason if path is forbidden for dataset registration."""
    path = host_path.resolve()
    parts = path.parts
    # Drive root: D:\ or /
    if len(parts) <= 1 or (os.name == "nt" and len(parts) == 1):
        return f"dangerous mount root forbidden: {path}"
    if os.name == "nt" and len(parts) == 2 and parts[1] in {"", "."}:
        return f"dangerous mount root forbidden: {path}"
    # Unix root children like / only
    if str(path) in {"/", "\\"}:
        return f"dangerous mount root forbidden: {path}"

    home = Path.home().resolve()
    if path == home or home in path.parents and path.parent == home.parent:
        # Exact home directory forbidden (not subdirs under home necessarily —
        # plan forbids 用户主目录 itself).
        if path == home:
            return f"user home directory forbidden: {path}"

    if project_root is not None:
        root = project_root.resolve()
        # Forbid registering the project tree root or any ancestor of it.
        if path == root or path in root.parents:
            return f"project root or ancestor path forbidden: {path}"
        workspace = root.parent if root.name else root
        if path == workspace:
            return f"workspace root forbidden: {path}"

    return None


class DatasetRegistry:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        project_root: Path | None = None,
    ) -> None:
        self._repo = DatasetRepository(session_factory)
        self._project_root = project_root

    def register(
        self,
        *,
        dataset_key: str,
        task_type: str,
        host_path: str | Path,
        container_path: str | None = None,
        read_only: bool = True,
        metadata: dict | None = None,
        enabled: bool = True,
    ) -> DatasetRegistration:
        key = dataset_key.strip()
        if not key:
            raise ValueError("dataset_key 不能为空")
        if self._repo.exists(key):
            raise ValueError(f"dataset_key already registered: {key}")

        if not read_only:
            raise ValueError("first version only allows read-only dataset registration")

        path = Path(host_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"数据集路径不存在：{path}")
        if not path.is_dir():
            raise ValueError(f"数据集路径必须是目录：{path}")

        danger = is_dangerous_host_path(path, project_root=self._project_root)
        if danger:
            raise ValueError(danger)

        container = (container_path or f"/datasets/{key}").strip()
        if not is_linux_absolute_path(container):
            raise ValueError(
                f"container_path must be a Linux absolute path, got: {container}"
            )

        now = utc_now_iso()
        meta = dict(metadata or {})
        if task_type == "rgbt_detection":
            for field, value in DEFAULT_RGBT_METADATA.items():
                meta.setdefault(field, value)

        registration = DatasetRegistration(
            dataset_key=key,
            task_type=task_type,
            host_path=str(path),
            container_path=container,
            read_only=True,
            enabled=enabled,
            metadata=meta,
            created_at=now,
            updated_at=now,
        )
        return self._repo.insert(registration)

    def get(self, dataset_key: str) -> DatasetRegistration | None:
        return self._repo.get(dataset_key)

    def require(self, dataset_key: str, *, require_enabled: bool = True) -> DatasetRegistration:
        item = self.get(dataset_key)
        if item is None:
            raise KeyError(f"数据集未注册：{dataset_key}")
        if require_enabled and not item.enabled:
            raise ValueError(f"数据集已禁用：{dataset_key}")
        return item

    def resolve_reference(self, dataset_reference: str) -> DatasetRegistration | None:
        key = parse_dataset_reference(dataset_reference)
        if key is None:
            return None
        return self.require(key, require_enabled=True)

    def list_datasets(self) -> list[DatasetRegistration]:
        return self._repo.list_all()

    def disable(self, dataset_key: str) -> DatasetRegistration:
        return self._repo.update_flags(
            dataset_key, enabled=False, updated_at=utc_now_iso()
        )

    def enable(self, dataset_key: str) -> DatasetRegistration:
        return self._repo.update_flags(
            dataset_key, enabled=True, updated_at=utc_now_iso()
        )

    def update_host_path(
        self,
        dataset_key: str,
        host_path: str | Path,
    ) -> DatasetRegistration:
        """Rebind mount path for an existing dataset_key. Does not change identity."""
        if self.get(dataset_key) is None:
            raise KeyError(f"数据集未注册：{dataset_key}")

        path = Path(host_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"数据集路径不存在：{path}")
        if not path.is_dir():
            raise ValueError(f"数据集路径必须是目录：{path}")

        danger = is_dangerous_host_path(path, project_root=self._project_root)
        if danger:
            raise ValueError(danger)

        return self._repo.update_host_path(
            dataset_key,
            host_path=str(path),
            updated_at=utc_now_iso(),
        )
