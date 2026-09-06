"""Tests for artifact garbage collection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scientist_lab.artifacts.gc import apply_gc, plan_gc, plan_weight_slim


def _touch(path: Path, *, days_ago: float = 0, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    mtime = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
    import os

    os.utime(path, (mtime, mtime))
    return path


def test_plan_weight_slim_keeps_best_drops_last(tmp_path: Path):
    root = tmp_path / "exec_aaaaaaaaaaaa"
    best = _touch(root / "checkpoint" / "best_weights.pth", content=b"best" * 100)
    last = _touch(root / "checkpoint" / "last.pth", content=b"last" * 100)
    stg = _touch(root / "_dfine_run" / "best_stg1.pth", content=b"stg" * 100)
    actions = plan_weight_slim(root)
    paths = {a.path for a in actions}
    assert best not in paths
    assert last in paths
    assert stg in paths


def test_gc_deletes_unreferenced_old_exec_keeps_recent_and_doc_ref(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("evidence at exec_bbbbbbbbbbbb\n", encoding="utf-8")

    keep_doc = tmp_path / "outputs" / "proj" / "exec_bbbbbbbbbbbb"
    _touch(keep_doc / "metrics.json", days_ago=10, content=b'{"ok":1}')
    _touch(keep_doc / "checkpoint" / "best_weights.pth", days_ago=10, content=b"b" * 50)

    keep_recent = tmp_path / "outputs" / "proj" / "exec_cccccccccccc"
    _touch(keep_recent / "metrics.json", days_ago=0, content=b'{"ok":1}')
    _touch(keep_recent / "checkpoint" / "last.pth", days_ago=0, content=b"l" * 50)

    drop = tmp_path / "outputs" / "proj" / "exec_dddddddddddd"
    _touch(drop / "metrics.json", days_ago=10, content=b'{"ok":1}')
    _touch(drop / "checkpoint" / "last.pth", days_ago=10, content=b"d" * 50)

    report = plan_gc(tmp_path, keep_days=2, slim_old_weights=True)
    delete_paths = {a.path for a in report.actions if a.kind == "delete_dir"}
    assert drop in delete_paths
    assert keep_doc not in delete_paths
    assert keep_recent not in delete_paths

    # Old protected exec should be slimmed (no best → keep one last fallback, so
    # with only last.pth and no best, last may be kept). Add a best then slim.
    _touch(keep_doc / "checkpoint" / "last.pth", days_ago=10, content=b"L" * 40)
    report2 = plan_gc(tmp_path, keep_days=2, slim_old_weights=True)
    slim_paths = {a.path for a in report2.actions if a.kind == "slim_file"}
    assert (keep_doc / "checkpoint" / "last.pth") in slim_paths

    applied = apply_gc(report, dry_run=False)
    assert not drop.exists()
    assert keep_doc.exists()
    assert keep_recent.exists()
    assert applied.deleted_dirs >= 1


def test_gc_deletes_unreferenced_campaign_and_stale_jobs(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.md").write_text(
        "Campaign exp_rgbt_dfine_v26_lowlight_20260801T000000Z done\n",
        encoding="utf-8",
    )

    keep = (
        tmp_path
        / ".run"
        / "autonomous"
        / "exp_rgbt_dfine_v26_lowlight_20260801T000000Z"
    )
    _touch(keep / "campaign.json", days_ago=10, content=b'{"status":"completed"}')

    drop = tmp_path / ".run" / "autonomous" / "p0_20260801T000000Z"
    _touch(drop / "campaign.json", days_ago=10, content=b'{"status":"completed"}')

    jobs = tmp_path / "runtime" / "scientist-worker" / "jobs" / "oldjob"
    _touch(jobs / "x.bin", days_ago=10, content=b"j" * 20)

    report = plan_gc(tmp_path, keep_days=2)
    delete_paths = {a.path for a in report.actions if a.kind == "delete_dir"}
    assert drop in delete_paths
    assert keep not in delete_paths
    assert jobs in delete_paths
