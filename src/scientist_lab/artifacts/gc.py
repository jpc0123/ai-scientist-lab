"""Disk garbage collection for experiment outputs and run workdirs.

Keeps evidence that docs / active campaigns still reference; deletes or slims
regenerable GPU weight dumps that dominate disk usage.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

EXEC_RE = re.compile(r"exec_[a-f0-9]{12}")
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}
# Prefer keeping a single "best" weight when slimming completed runs.
KEEP_WEIGHT_BASENAMES = {
    "best_weights.pth",
    "best.pt",
    "best.pth",
    "best_model.pth",
    "model_best.pth",
}


@dataclass
class GcAction:
    kind: str  # delete_dir | slim_file | delete_file
    path: Path
    bytes: int
    reason: str


@dataclass
class GcReport:
    dry_run: bool
    keep_days: int
    protected_execs: int
    protected_campaigns: int
    actions: list[GcAction] = field(default_factory=list)
    deleted_dirs: int = 0
    deleted_files: int = 0
    bytes_reclaimed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "keep_days": self.keep_days,
            "protected_execs": self.protected_execs,
            "protected_campaigns": self.protected_campaigns,
            "planned_actions": len(self.actions),
            "deleted_dirs": self.deleted_dirs,
            "deleted_files": self.deleted_files,
            "bytes_reclaimed": self.bytes_reclaimed,
            "bytes_reclaimed_gb": round(self.bytes_reclaimed / (1024**3), 3),
            "errors": list(self.errors),
            "actions": [
                {
                    "kind": a.kind,
                    "path": str(a.path),
                    "bytes": a.bytes,
                    "reason": a.reason,
                }
                for a in self.actions[:200]
            ],
            "actions_truncated": max(0, len(self.actions) - 200),
        }


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += int(p.stat().st_size)
            except OSError:
                continue
    return total


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _content_mtime(path: Path) -> datetime:
    """Newest mtime among nested files (directory mtime alone is unreliable)."""
    if path.is_file():
        return _mtime(path)
    newest: datetime | None = None
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                mt = _mtime(p)
                if newest is None or mt > newest:
                    newest = mt
    if newest is not None:
        return newest
    return _mtime(path)


def _scan_ids(text: str, pattern: re.Pattern[str]) -> set[str]:
    return {m.group(0) for m in pattern.finditer(text or "")}


def collect_doc_refs(project_root: Path) -> tuple[set[str], set[str]]:
    docs = project_root / "docs"
    execs: set[str] = set()
    campaigns: set[str] = set()
    if not docs.is_dir():
        return execs, campaigns
    for path in docs.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".json", ".txt"}:
            continue
        text = _read_text(path)
        execs |= _scan_ids(text, EXEC_RE)
        # Prefer explicit campaign dir names from docs.
        for m in re.finditer(
            r"(?:exp_rgbt_dfine_v26_lowlight_|p0_)\d{8}T\d{6}Z", text
        ):
            campaigns.add(m.group(0))
    return execs, campaigns


def collect_campaign_exec_refs(campaign_dir: Path) -> set[str]:
    found: set[str] = set()
    if not campaign_dir.is_dir():
        return found
    for path in campaign_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log"}:
            continue
        # Skip huge logs for speed; still scan modest files.
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
        except OSError:
            continue
        found |= _scan_ids(_read_text(path), EXEC_RE)
    return found


def _has_active_lease(autonomous_root: Path, max_age: timedelta) -> bool:
    lease = autonomous_root / "live_execute.lease.json"
    if not lease.is_file():
        return False
    return (datetime.now(timezone.utc) - _mtime(lease)) <= max_age


def plan_weight_slim(exec_dir: Path) -> list[GcAction]:
    actions: list[GcAction] = []
    weight_files = [
        p
        for p in exec_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
    ]
    if not weight_files:
        return actions
    has_best = any(p.name.lower() in KEEP_WEIGHT_BASENAMES for p in weight_files)
    if not has_best:
        has_best = any("best" in p.name.lower() for p in weight_files)
    kept_fallback = False
    for path in weight_files:
        name = path.name.lower()
        keep = False
        if name in KEEP_WEIGHT_BASENAMES or (has_best and "best_weights" in name):
            keep = True
        elif has_best and "best" in name and name.endswith((".pth", ".pt")):
            # Keep one additional best_* only if basename matches allow-list style.
            keep = name in KEEP_WEIGHT_BASENAMES or name == "best_stg1.pth"
            # Prefer best_weights; drop best_stg1 when best_weights exists.
            if name == "best_stg1.pth" and any(
                q.name.lower() == "best_weights.pth" for q in weight_files
            ):
                keep = False
        elif not has_best and name.startswith("last") and not kept_fallback:
            keep = True
            kept_fallback = True
        if keep:
            continue
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        actions.append(
            GcAction(
                kind="slim_file",
                path=path,
                bytes=size,
                reason="redundant_weight",
            )
        )
    return actions


def plan_gc(
    project_root: Path,
    *,
    keep_days: int = 2,
    slim_old_weights: bool = True,
    clean_worker_jobs: bool = True,
    clean_unprotected_campaigns: bool = True,
) -> GcReport:
    project_root = Path(project_root).resolve()
    outputs = project_root / "outputs"
    run_root = project_root / ".run"
    autonomous = run_root / "autonomous"
    runtime_jobs = project_root / "runtime" / "scientist-worker" / "jobs"
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(keep_days)))

    protect_execs, protect_campaigns = collect_doc_refs(project_root)

    # Protect anything still being written / recent.
    if autonomous.is_dir():
        for camp in autonomous.iterdir():
            if not camp.is_dir() or camp.name.startswith("."):
                continue
            if _content_mtime(camp) >= cutoff:
                protect_campaigns.add(camp.name)
        # Always protect campaigns referenced by a fresh lease.
        if _has_active_lease(autonomous, timedelta(days=max(1, keep_days))):
            lease = autonomous / "live_execute.lease.json"
            text = _read_text(lease)
            protect_campaigns |= {
                m.group(0)
                for m in re.finditer(
                    r"(?:exp_rgbt_dfine_v26_lowlight_|p0_)\d{8}T\d{6}Z", text
                )
            }
            protect_execs |= _scan_ids(text, EXEC_RE)

    for camp_name in list(protect_campaigns):
        protect_execs |= collect_campaign_exec_refs(autonomous / camp_name)

    # Recent exec dirs are protected even if not yet cited.
    if outputs.is_dir():
        for exec_dir in outputs.rglob("exec_*"):
            if exec_dir.is_dir() and _content_mtime(exec_dir) >= cutoff:
                protect_execs.add(exec_dir.name)

    report = GcReport(
        dry_run=True,
        keep_days=keep_days,
        protected_execs=len(protect_execs),
        protected_campaigns=len(protect_campaigns),
    )

    # 1) Delete unprotected exec_* trees under outputs/
    if outputs.is_dir():
        for exec_dir in sorted(
            (p for p in outputs.rglob("exec_*") if p.is_dir()),
            key=lambda p: str(p),
        ):
            if exec_dir.name in protect_execs:
                if slim_old_weights and _content_mtime(exec_dir) < cutoff:
                    report.actions.extend(plan_weight_slim(exec_dir))
                continue
            report.actions.append(
                GcAction(
                    kind="delete_dir",
                    path=exec_dir,
                    bytes=_dir_size(exec_dir),
                    reason="unreferenced_exec",
                )
            )

    # 2) Delete unprotected autonomous campaigns (keep scripts at top level)
    if clean_unprotected_campaigns and autonomous.is_dir():
        for camp in sorted(autonomous.iterdir(), key=lambda p: p.name):
            if not camp.is_dir() or camp.name.startswith("."):
                continue
            # Only touch campaign-like dirs.
            if not (
                camp.name.startswith("exp_")
                or camp.name.startswith("p0_")
                or camp.name.startswith("exp-")
            ):
                continue
            if camp.name in protect_campaigns:
                # Slim nested weights inside protected-but-old campaigns.
                if slim_old_weights and _content_mtime(camp) < cutoff:
                    for nested in camp.rglob("exec_*"):
                        if nested.is_dir():
                            report.actions.extend(plan_weight_slim(nested))
                    for sub in ("run", "runs", "_dfine_run", "checkpoint"):
                        for target in camp.rglob(sub):
                            if target.is_dir():
                                report.actions.extend(plan_weight_slim(target))
                continue
            report.actions.append(
                GcAction(
                    kind="delete_dir",
                    path=camp,
                    bytes=_dir_size(camp),
                    reason="unreferenced_campaign",
                )
            )

    # 3) Worker job cache (regenerable)
    if clean_worker_jobs and runtime_jobs.is_dir():
        for child in sorted(runtime_jobs.iterdir(), key=lambda p: p.name):
            if _content_mtime(child) >= cutoff:
                continue
            report.actions.append(
                GcAction(
                    kind="delete_dir" if child.is_dir() else "delete_file",
                    path=child,
                    bytes=_dir_size(child),
                    reason="stale_worker_job",
                )
            )

    # Deduplicate slim actions that may have been queued twice.
    seen: set[str] = set()
    unique: list[GcAction] = []
    for action in report.actions:
        key = f"{action.kind}:{action.path.resolve()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    report.actions = unique
    return report


def apply_gc(report: GcReport, *, dry_run: bool = True) -> GcReport:
    report.dry_run = dry_run
    reclaimed = 0
    deleted_dirs = 0
    deleted_files = 0
    errors: list[str] = []

    for action in report.actions:
        path = action.path
        try:
            if dry_run:
                reclaimed += int(action.bytes)
                if action.kind == "delete_dir":
                    deleted_dirs += 1
                else:
                    deleted_files += 1
                continue
            if action.kind == "delete_dir":
                if path.is_dir():
                    size = _dir_size(path)
                    shutil.rmtree(path, ignore_errors=False)
                    reclaimed += size
                    deleted_dirs += 1
            elif action.kind in {"slim_file", "delete_file"}:
                if path.is_file():
                    size = int(path.stat().st_size)
                    path.unlink()
                    reclaimed += size
                    deleted_files += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    report.bytes_reclaimed = reclaimed
    report.deleted_dirs = deleted_dirs
    report.deleted_files = deleted_files
    report.errors = errors
    return report


def slim_tree_weights(root: Path) -> dict[str, Any]:
    """Delete redundant weight dumps under one tree; keep best + evidence files."""
    root = Path(root)
    actions: list[GcAction] = []
    if root.is_dir():
        for nested in root.rglob("exec_*"):
            if nested.is_dir():
                actions.extend(plan_weight_slim(nested))
        for sub in ("run", "runs", "_dfine_run", "checkpoint"):
            for target in root.rglob(sub):
                if target.is_dir():
                    actions.extend(plan_weight_slim(target))
    # Dedup
    seen: set[str] = set()
    unique: list[GcAction] = []
    for action in actions:
        key = str(action.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    report = GcReport(
        dry_run=False,
        keep_days=0,
        protected_execs=0,
        protected_campaigns=0,
        actions=unique,
    )
    return apply_gc(report, dry_run=False).to_dict()


def run_artifact_gc(
    project_root: Path,
    *,
    dry_run: bool = True,
    keep_days: int = 2,
    slim_old_weights: bool = True,
    clean_worker_jobs: bool = True,
    clean_unprotected_campaigns: bool = True,
) -> dict[str, Any]:
    plan = plan_gc(
        project_root,
        keep_days=keep_days,
        slim_old_weights=slim_old_weights,
        clean_worker_jobs=clean_worker_jobs,
        clean_unprotected_campaigns=clean_unprotected_campaigns,
    )
    applied = apply_gc(plan, dry_run=dry_run)
    return applied.to_dict()
