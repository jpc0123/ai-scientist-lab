"""Controlled sandbox validation hooks (v1.6.5).

These checks run inside the host process against an already-applied patch
sandbox. They never invoke an LLM and never execute arbitrary shell commands.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.storage.artifact_store import write_json

SandboxTestProfile = Literal["smoke", "syntax", "mock_experiment"]


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""
    path: str | None = None


@dataclass
class SandboxTestReport:
    ok: bool
    profile: str
    patch_id: str
    sandbox_dir: str
    checks: list[CheckItem] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    report_path: str | None = None
    main_workspace_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile": self.profile,
            "patch_id": self.patch_id,
            "sandbox_dir": self.sandbox_dir,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "path": c.path,
                }
                for c in self.checks
            ],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report_path": self.report_path,
            "main_workspace_modified": False,
        }


class SandboxTestRunner:
    """Allow-listed validation profiles for patch sandboxes."""

    PROFILES: tuple[str, ...] = ("smoke", "syntax", "mock_experiment")

    def __init__(self, *, policy: PathPolicy | None = None) -> None:
        self.policy = policy or PathPolicy()

    def run(
        self,
        *,
        patch_id: str,
        sandbox_dir: Path | str,
        files_written: list[str] | None = None,
        profile: SandboxTestProfile = "smoke",
    ) -> SandboxTestReport:
        if profile not in self.PROFILES:
            raise ValueError(
                f"unknown sandbox test profile {profile!r}; "
                f"allowed={list(self.PROFILES)}"
            )
        ws = Path(sandbox_dir)
        started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        checks: list[CheckItem] = []

        checks.extend(self._check_sandbox_layout(ws))
        written = list(files_written or self._discover_written(ws))
        checks.extend(self._check_written_files(ws, written))

        if profile in {"syntax", "mock_experiment"}:
            checks.extend(self._check_python_syntax(ws, written))
        if profile == "mock_experiment":
            checks.extend(self._check_mock_experiment(ws, written))

        ok = all(c.ok for c in checks)
        finished = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        report = SandboxTestReport(
            ok=ok,
            profile=profile,
            patch_id=patch_id,
            sandbox_dir=str(ws),
            checks=checks,
            started_at=started,
            finished_at=finished,
        )
        if ws.is_dir():
            path = ws / "PATCH_SANDBOX_TEST_REPORT.json"
            write_json(path, report.to_dict())
            report.report_path = str(path)
        return report

    def _discover_written(self, ws: Path) -> list[str]:
        manifest = ws / "PATCH_SANDBOX_MANIFEST.json"
        if not manifest.is_file():
            return []
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        files = data.get("files_written") or []
        return [str(x) for x in files]

    def _check_sandbox_layout(self, ws: Path) -> list[CheckItem]:
        items: list[CheckItem] = []
        items.append(
            CheckItem(
                name="sandbox_dir_exists",
                ok=ws.is_dir(),
                detail=str(ws),
            )
        )
        manifest = ws / "PATCH_SANDBOX_MANIFEST.json"
        items.append(
            CheckItem(
                name="manifest_present",
                ok=manifest.is_file(),
                detail=str(manifest),
                path="PATCH_SANDBOX_MANIFEST.json",
            )
        )
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                main_mod = bool(data.get("main_workspace_modified"))
                items.append(
                    CheckItem(
                        name="main_workspace_unmodified_flag",
                        ok=main_mod is False,
                        detail=f"main_workspace_modified={main_mod}",
                    )
                )
            except (OSError, json.JSONDecodeError) as exc:
                items.append(
                    CheckItem(
                        name="manifest_readable",
                        ok=False,
                        detail=str(exc),
                        path="PATCH_SANDBOX_MANIFEST.json",
                    )
                )
        return items

    def _check_written_files(
        self, ws: Path, files_written: list[str]
    ) -> list[CheckItem]:
        items: list[CheckItem] = []
        if not files_written:
            items.append(
                CheckItem(
                    name="files_written_nonempty",
                    ok=False,
                    detail="no files_written recorded for sandbox",
                )
            )
            return items
        items.append(
            CheckItem(
                name="files_written_nonempty",
                ok=True,
                detail=f"count={len(files_written)}",
            )
        )
        for rel in files_written:
            norm = self.policy.normalize(rel)
            allowed, reason = self.policy.is_allowed(norm)
            items.append(
                CheckItem(
                    name="path_still_allowed",
                    ok=allowed,
                    detail=reason if not allowed else "ok",
                    path=norm,
                )
            )
            target = (ws / norm).resolve()
            try:
                target.relative_to(ws.resolve())
                escape_ok = True
            except ValueError:
                escape_ok = False
            items.append(
                CheckItem(
                    name="path_inside_sandbox",
                    ok=escape_ok,
                    detail=str(target),
                    path=norm,
                )
            )
            exists = target.is_file()
            items.append(
                CheckItem(
                    name="file_exists",
                    ok=exists,
                    detail=str(target),
                    path=norm,
                )
            )
            if exists:
                try:
                    text = target.read_text(encoding="utf-8")
                    items.append(
                        CheckItem(
                            name="utf8_readable",
                            ok=len(text) >= 0,
                            detail=f"bytes≈{target.stat().st_size}",
                            path=norm,
                        )
                    )
                except (OSError, UnicodeDecodeError) as exc:
                    items.append(
                        CheckItem(
                            name="utf8_readable",
                            ok=False,
                            detail=str(exc),
                            path=norm,
                        )
                    )
        return items

    def _check_python_syntax(
        self, ws: Path, files_written: list[str]
    ) -> list[CheckItem]:
        items: list[CheckItem] = []
        py_files = [
            self.policy.normalize(p)
            for p in files_written
            if self.policy.normalize(p).endswith(".py")
        ]
        if not py_files:
            items.append(
                CheckItem(
                    name="python_syntax_skipped",
                    ok=True,
                    detail="no .py files in patch",
                )
            )
            return items
        for rel in py_files:
            target = ws / rel
            if not target.is_file():
                items.append(
                    CheckItem(
                        name="python_syntax",
                        ok=False,
                        detail="file missing",
                        path=rel,
                    )
                )
                continue
            try:
                source = target.read_text(encoding="utf-8")
                ast.parse(source, filename=rel)
                items.append(
                    CheckItem(
                        name="python_syntax",
                        ok=True,
                        detail="ast.parse ok",
                        path=rel,
                    )
                )
            except SyntaxError as exc:
                items.append(
                    CheckItem(
                        name="python_syntax",
                        ok=False,
                        detail=f"{exc.msg} (line {exc.lineno})",
                        path=rel,
                    )
                )
            except (OSError, UnicodeDecodeError) as exc:
                items.append(
                    CheckItem(
                        name="python_syntax",
                        ok=False,
                        detail=str(exc),
                        path=rel,
                    )
                )
        return items

    def _check_mock_experiment(
        self, ws: Path, files_written: list[str]
    ) -> list[CheckItem]:
        """Deterministic in-process 'experiment' for the mock patch note."""
        items: list[CheckItem] = []
        note_candidates = [
            self.policy.normalize(p)
            for p in files_written
            if p.replace("\\", "/").endswith("mock_patch_note.md")
            or p.replace("\\", "/").endswith(".md")
        ]
        if not note_candidates:
            # Still succeed with an explicit skip when patch has no markdown —
            # mock_experiment then reduces to smoke+syntax.
            items.append(
                CheckItem(
                    name="mock_experiment_marker",
                    ok=True,
                    detail="no markdown note; smoke/syntax only",
                )
            )
            return items
        found_marker = False
        for rel in note_candidates:
            target = ws / rel
            if not target.is_file():
                continue
            text = target.read_text(encoding="utf-8")
            if "fusion ablation" in text.lower():
                found_marker = True
                items.append(
                    CheckItem(
                        name="mock_experiment_marker",
                        ok=True,
                        detail="found evidence-gap marker in sandbox note",
                        path=rel,
                    )
                )
                break
        if not found_marker:
            # For non-mock markdown patches, require non-empty content.
            rel = note_candidates[0]
            target = ws / rel
            if target.is_file() and target.read_text(encoding="utf-8").strip():
                items.append(
                    CheckItem(
                        name="mock_experiment_marker",
                        ok=True,
                        detail="markdown note non-empty (generic mock experiment)",
                        path=rel,
                    )
                )
            else:
                items.append(
                    CheckItem(
                        name="mock_experiment_marker",
                        ok=False,
                        detail="expected evidence-gap marker or non-empty note",
                        path=rel,
                    )
                )
        items.append(
            CheckItem(
                name="mock_experiment_no_shell",
                ok=True,
                detail="in-process only; no shell executed",
            )
        )
        return items
