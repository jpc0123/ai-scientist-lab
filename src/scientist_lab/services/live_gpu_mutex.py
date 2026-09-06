"""Durable single-flight lock for live execute=true GPU campaigns.

Human Gate (confirm_human_gate) is not an OS permission and does not
serialize jobs. This lock does: one live execute on this machine/GPU.

Dry-run (execute=false) and injected stub runners skip the Docker occupancy
check so unit tests do not collide with a real scientist-exec-* container.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LIVE_CAMPAIGN_STATUSES = {
    "queued",
    "running",
    "waiting_gpu",
    "pause_requested",
}

LEASE_NAME = "live_execute.lease.json"


class LiveGpuBusyError(Exception):
    """A live execute=true campaign or GPU exec already occupies the machine."""

    def __init__(self, message: str, holder: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.holder = dict(holder or {})


class LiveExecuteLease:
    """Lease file held for the lifetime of one live execute job."""

    def __init__(self, root: Path, owner_id: str) -> None:
        self.root = Path(root)
        self.owner_id = str(owner_id)

    def release(self) -> None:
        path = _lease_path(self.root)
        current = _read_json(path)
        if current.get("owner_id") == self.owner_id:
            path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _autonomous_root(project_root: Path) -> Path:
    return Path(project_root) / ".run" / "autonomous"


def _lease_path(project_root: Path) -> Path:
    return _autonomous_root(project_root) / LEASE_NAME


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _pid_alive_windows(pid: int) -> bool:
    """OpenProcess + GetExitCodeProcess. os.kill(pid, 0) on Windows treats
    missing PIDs as OSError, which the POSIX fallback used to count as alive
    and left ghost GPU locks after API restart."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return int(code.value) == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            return _pid_alive_windows(pid)
        except Exception:  # noqa: BLE001 — fail closed: unknown → not alive
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def campaign_worker_alive(row: Mapping[str, Any]) -> bool:
    """True only when campaign.json names a still-living worker process."""
    try:
        pid = int(row.get("worker_pid") or 0)
    except (TypeError, ValueError):
        return False
    return _pid_alive(pid)


def _live_docker_execs() -> list[str]:
    from scientist_lab.services.training_monitor import _live_scientist_exec_names

    return _live_scientist_exec_names()


def _busy_message(holder: Mapping[str, Any]) -> str:
    label = (
        holder.get("campaign_id")
        or holder.get("owner_id")
        or ",".join(holder.get("names") or [])
        or holder.get("kind")
    )
    return f"已有真实 GPU 战役占用：{label}。第二场必须拒绝，不能静默开跑。"


def list_live_execute_holders(
    project_root: Path | str,
    *,
    inspect_docker: bool = False,
    ignore_owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Durable occupancy: live lease PID + campaign.json with a living worker_pid
    (+ optional scientist-exec-*). status=running with a dead worker is a ghost lock.
    """
    root = Path(project_root)
    holders: list[dict[str, Any]] = []
    lease = _read_json(_lease_path(root))
    pid = int(lease.get("pid") or 0)
    owner = str(lease.get("owner_id") or "")
    if lease and _pid_alive(pid) and owner != str(ignore_owner_id or ""):
        holders.append(
            {
                "kind": "lease",
                "owner_id": lease.get("owner_id"),
                "pid": pid,
                "created_at": lease.get("created_at"),
            }
        )
    auto = _autonomous_root(root)
    if auto.is_dir():
        for path in sorted(auto.glob("*/campaign.json")):
            row = _read_json(path)
            if not row.get("execute"):
                continue
            status = str(row.get("status") or "")
            if status not in LIVE_CAMPAIGN_STATUSES:
                continue
            cid = str(row.get("campaign_id") or path.parent.name)
            if cid == str(ignore_owner_id or ""):
                continue
            if not campaign_worker_alive(row):
                # status=running after API/worker death is a ghost lock, not occupancy.
                continue
            holders.append(
                {
                    "kind": "campaign",
                    "campaign_id": cid,
                    "status": status,
                    "planner_backend": row.get("planner_backend"),
                    "gpu_rounds": row.get("gpu_rounds"),
                    "updated_at": row.get("updated_at"),
                    "worker_pid": row.get("worker_pid"),
                }
            )
    if inspect_docker:
        names = _live_docker_execs()
        if names:
            holders.append({"kind": "docker", "names": names})
    return holders


def _pick_holder(holders: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [row for row in holders if row.get("kind") in {"campaign", "docker"}]
    return preferred[0] if preferred else holders[0]


def acquire_live_execute(
    project_root: Path | str,
    *,
    owner_id: str,
    inspect_docker: bool = False,
) -> LiveExecuteLease:
    """Fail closed: second live execute must refuse, not silently run."""
    root = Path(project_root)
    dest = _lease_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    holders = list_live_execute_holders(
        root,
        inspect_docker=inspect_docker,
        ignore_owner_id=str(owner_id),
    )
    if holders:
        first = _pick_holder(holders)
        raise LiveGpuBusyError(_busy_message(first), first)

    stale = _read_json(dest)
    if stale and _pid_alive(int(stale.get("pid") or 0)):
        first = {
            "kind": "lease",
            "owner_id": stale.get("owner_id"),
            "pid": stale.get("pid"),
        }
        raise LiveGpuBusyError(_busy_message(first), first)
    if stale:
        dest.unlink(missing_ok=True)

    payload = {
        "owner_id": str(owner_id),
        "pid": os.getpid(),
        "created_at": _now(),
        "argv0": Path(sys.argv[0]).name if sys.argv else "",
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(dest), flags)
    except FileExistsError as exc:
        current = _read_json(dest)
        first = {
            "kind": "lease",
            "owner_id": current.get("owner_id") or "unknown",
            "pid": current.get("pid"),
        }
        raise LiveGpuBusyError(_busy_message(first), first) from exc
    try:
        os.write(fd, encoded.encode("utf-8"))
    finally:
        os.close(fd)
    return LiveExecuteLease(root, str(owner_id))
