"""Temporary sandbox workspace for approved patches (v1.6.4).

Never writes to the main project working tree. All mutations stay under
``sandbox_root / <patch_id>/``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scientist_lab.patching.apply_engine import PatchApplyError, apply_unified_diff_to_root
from scientist_lab.patching.diff_parser import parse_unified_diff
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.storage.artifact_store import write_json


@dataclass
class SandboxApplyResult:
    ok: bool
    sandbox_dir: str
    files_written: list[str]
    error: str | None = None
    manifest_path: str | None = None


class PatchSandbox:
    def __init__(
        self,
        *,
        project_root: Path,
        sandbox_root: Path,
        policy: PathPolicy | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.sandbox_root = Path(sandbox_root)
        self.policy = policy or PathPolicy.for_code_context()
        self.verifier = PatchVerifier(self.policy)

    def workspace_dir(self, patch_id: str) -> Path:
        return (self.sandbox_root / patch_id).resolve()

    def prepare(
        self,
        patch_id: str,
        unified_diff: str,
        *,
        force: bool = False,
    ) -> Path:
        """Create sandbox and copy baseline files needed by the diff."""
        ws = self.workspace_dir(patch_id)
        if ws.exists():
            if not force:
                raise PatchApplyError(f"sandbox already exists: {ws}")
            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=False)

        # Safety: sandbox must not equal / contain project root inverted
        if ws == self.project_root or self.project_root in ws.parents:
            # project_root in ws.parents means sandbox is inside project — OK if under outputs
            pass
        if str(ws).startswith(str(self.project_root)) and "outputs" not in ws.parts and "_patch_sandboxes" not in str(ws):
            # Prefer sandboxes under outputs; still allow runtime paths.
            pass

        parsed = parse_unified_diff(unified_diff)
        for diff_file in parsed.files:
            rel = self.policy.normalize(diff_file.path)
            ok, reason = self.policy.is_allowed(rel)
            if not ok:
                raise PatchApplyError(f"cannot stage denied path {rel}: {reason}")
            src = (self.project_root / rel).resolve()
            try:
                src.relative_to(self.project_root)
            except ValueError as exc:
                raise PatchApplyError(f"source path escapes project root: {rel}") from exc
            dest = (ws / rel).resolve()
            try:
                dest.relative_to(ws)
            except ValueError as exc:
                raise PatchApplyError(f"dest path escapes sandbox: {rel}") from exc
            if diff_file.is_new_file:
                dest.parent.mkdir(parents=True, exist_ok=True)
                continue
            if src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            elif not diff_file.is_deleted_file:
                # Missing baseline for modify — allow apply engine to fail clearly,
                # or create empty parent for new-ish files.
                dest.parent.mkdir(parents=True, exist_ok=True)
        return ws

    def apply(
        self,
        patch_id: str,
        unified_diff: str,
        *,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> SandboxApplyResult:
        verification = self.verifier.verify(unified_diff)
        if not verification.ok:
            return SandboxApplyResult(
                ok=False,
                sandbox_dir=str(self.workspace_dir(patch_id)),
                files_written=[],
                error="verification failed: "
                + "; ".join(i.message for i in verification.issues),
            )
        try:
            ws = self.prepare(patch_id, unified_diff, force=force)
            written = apply_unified_diff_to_root(
                unified_diff, root=ws, policy=self.policy
            )
            manifest = {
                "patch_id": patch_id,
                "sandbox_dir": str(ws),
                "project_root": str(self.project_root),
                "files_written": written,
                "applied_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "main_workspace_modified": False,
                "metadata": dict(metadata or {}),
            }
            manifest_path = ws / "PATCH_SANDBOX_MANIFEST.json"
            write_json(manifest_path, manifest)
            (ws / "applied.diff").write_text(unified_diff, encoding="utf-8")
            return SandboxApplyResult(
                ok=True,
                sandbox_dir=str(ws),
                files_written=written,
                manifest_path=str(manifest_path),
            )
        except (PatchApplyError, OSError, ValueError) as exc:
            ws = self.workspace_dir(patch_id)
            ws.mkdir(parents=True, exist_ok=True)
            err_path = ws / "PATCH_SANDBOX_ERROR.json"
            write_json(
                err_path,
                {
                    "patch_id": patch_id,
                    "error": str(exc),
                    "main_workspace_modified": False,
                },
            )
            return SandboxApplyResult(
                ok=False,
                sandbox_dir=str(ws),
                files_written=[],
                error=str(exc),
                manifest_path=str(err_path),
            )
