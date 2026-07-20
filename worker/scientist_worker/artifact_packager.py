from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _default_policy() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "include_metrics": True,
        "include_logs": True,
        "include_previews": True,
        "include_best_checkpoint": True,
        "include_last_checkpoint": True,
        "include_intermediate_checkpoints": False,
        "max_bundle_size_gb": 2.0,
        "required_paths": ["metrics.json", "execution.json"],
    }


def _is_intermediate_checkpoint(rel: str, name: str) -> bool:
    lowered = rel.lower()
    if "/intermediate/" in f"/{lowered}":
        return True
    if name.startswith(("epoch_", "step_", "ckpt_", "checkpoint_")):
        return True
    if name.startswith("epoch") and len(name) > 5 and name[5].isdigit():
        return True
    return False


def should_include_path(relative_path: str, policy: dict[str, Any] | None) -> bool:
    cfg = {**_default_policy(), **(policy or {})}
    rel = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    if ".." in rel.split("/"):
        return False
    name = Path(rel).name.lower()

    if rel.startswith("checkpoint/") or "/checkpoint/" in f"/{rel}":
        if _is_intermediate_checkpoint(rel, name):
            return bool(cfg.get("include_intermediate_checkpoints"))
        if "best" in name:
            return bool(cfg.get("include_best_checkpoint", True))
        if "last" in name:
            return bool(cfg.get("include_last_checkpoint", True))
        return bool(cfg.get("include_intermediate_checkpoints"))

    if rel.startswith("previews/") or rel.startswith("preview/"):
        return bool(cfg.get("include_previews", True))

    if rel.endswith(".log") or name in {
        "stdout.log",
        "stderr.log",
        "combined.log",
        "worker.log",
    }:
        return bool(cfg.get("include_logs", True))

    if name in {
        "metrics.json",
        "training_history.csv",
        "model_summary.json",
        "resource_usage.json",
        "detection_metrics.json",
    }:
        return bool(cfg.get("include_metrics", True))

    return True


def load_job_artifact_policy(record: JobRecord) -> dict[str, Any]:
    """Read artifact_policy from the persisted submit payload if present."""
    payload_path = Path(record.output_path).parent / "request" / "payload.json"
    if not payload_path.is_file():
        return {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    policy = payload.get("artifact_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def package_artifacts(
    output_dir: Path,
    *,
    job_id: str,
    execution_id: str,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    result_dir = output_dir.parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**_default_policy(), **(policy or {})}
    max_bytes = int(float(cfg.get("max_bundle_size_gb", 2.0)) * (1024**3))

    selected: list[Path] = []
    total_size = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"symlink_rejected: {path}")
        rel = path.relative_to(output_dir).as_posix()
        if ".." in rel.split("/"):
            raise ValueError(f"path_escape: {rel}")
        if not should_include_path(rel, cfg):
            continue
        total_size += path.stat().st_size
        if total_size > max_bytes:
            raise ValueError(
                f"bundle too large before pack: {total_size} > {max_bytes} bytes"
            )
        selected.append(path)

    required = list(cfg.get("required_paths") or [])
    present = {p.relative_to(output_dir).as_posix() for p in selected}
    missing = [item for item in required if item not in present]
    if missing:
        raise ValueError(f"missing_required_artifact: {', '.join(missing)}")

    bundle_path = result_dir / "artifact_bundle.tar.gz"
    files: list[dict[str, Any]] = []
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in selected:
            rel = path.relative_to(output_dir).as_posix()
            tar.add(path, arcname=rel)
            files.append(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    bundle_size = bundle_path.stat().st_size
    if bundle_size > max_bytes:
        raise ValueError(
            f"bundle too large after pack: {bundle_size} > {max_bytes} bytes"
        )

    manifest = {
        "schema_version": "1.0",
        "job_id": job_id,
        "execution_id": execution_id,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_size_bytes": bundle_size,
        "policy": cfg,
        "files": files,
    }
    manifest_path = result_dir / "artifact_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
