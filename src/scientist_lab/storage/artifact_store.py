from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_relative_safe(output_dir: Path, relative_path: str) -> Path:
    candidate = (output_dir / relative_path).resolve()
    root = output_dir.resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"Artifact path escapes output dir: {relative_path}")
    return candidate


class ArtifactStore:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def execution_output_dir(self, project_id: str, execution_id: str) -> Path:
        path = self.outputs_root / project_id / execution_id
        path.mkdir(parents=True, exist_ok=True)
        return path
