from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from scientist_lab.artifacts.policy import ArtifactPolicy, default_artifact_policy
from scientist_lab.runners.remote_errors import RemoteArtifactCorrupt
from scientist_lab.storage.artifact_store import sha256_file


def validate_bundle_archive(
    bundle_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    policy: ArtifactPolicy | None = None,
    extract_to: Path | None = None,
) -> dict[str, Any]:
    """Validate remote artifact bundle; optionally extract into extract_to."""
    policy = policy or default_artifact_policy()
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise RemoteArtifactCorrupt(f"bundle missing: {bundle_path}")

    size = bundle_path.stat().st_size
    max_bytes = policy.max_bundle_size_bytes()
    if size > max_bytes:
        raise RemoteArtifactCorrupt(
            f"bundle too large: {size} bytes > max {max_bytes} "
            f"({policy.max_bundle_size_gb} GiB)"
        )

    manifest = dict(manifest or {})
    expected_bundle_hash = manifest.get("bundle_sha256")
    if expected_bundle_hash:
        actual = sha256_file(bundle_path)
        if actual != str(expected_bundle_hash):
            raise RemoteArtifactCorrupt(
                f"bundle hash mismatch expected={expected_bundle_hash} actual={actual}"
            )

    file_index = {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in (manifest.get("files") or [])
        if isinstance(item, dict) and item.get("path")
    }

    extracted: list[str] = []
    with tarfile.open(bundle_path, "r:gz") as tar:
        members = list(tar.getmembers())
        for member in members:
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise RemoteArtifactCorrupt(f"path_escape: {name}")
            if member.issym() or member.islnk():
                raise RemoteArtifactCorrupt(f"symlink_rejected: {name}")
            if member.isfile() and not policy.should_include(name):
                # Extra files outside policy are rejected after download to keep
                # control-plane inventory honest.
                raise RemoteArtifactCorrupt(f"policy_rejected_path: {name}")

        if extract_to is not None:
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_to, filter="data")
            for member in members:
                if not member.isfile():
                    continue
                name = member.name.replace("\\", "/")
                target = extract_to / name
                if not target.is_file():
                    raise RemoteArtifactCorrupt(f"extract_missing: {name}")
                meta = file_index.get(name)
                if meta and meta.get("sha256"):
                    actual = sha256_file(target)
                    if actual != str(meta["sha256"]):
                        raise RemoteArtifactCorrupt(
                            f"file hash mismatch path={name} "
                            f"expected={meta['sha256']} actual={actual}"
                        )
                if meta and meta.get("size_bytes") is not None:
                    actual_size = target.stat().st_size
                    if int(meta["size_bytes"]) != actual_size:
                        raise RemoteArtifactCorrupt(
                            f"file size mismatch path={name} "
                            f"expected={meta['size_bytes']} actual={actual_size}"
                        )
                extracted.append(name)

    missing_required = [
        path
        for path in policy.required_paths
        if extract_to is not None and not (Path(extract_to) / path).is_file()
    ]
    if missing_required:
        raise RemoteArtifactCorrupt(
            f"missing_required_artifact: {', '.join(missing_required)}"
        )

    return {
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": size,
        "bundle_sha256": str(expected_bundle_hash or sha256_file(bundle_path)),
        "extracted_files": extracted,
        "policy": policy.model_dump(),
    }
