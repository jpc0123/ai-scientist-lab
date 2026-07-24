"""Workbench process status helpers (v2.0.7)."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from scientist_lab.workbench.config import load_workbench_config, workbench_endpoints


def pid_file(project_root: Path) -> Path:
    runtime = project_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime / "workbench.pids.json"


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def read_pid_state(project_root: Path) -> dict[str, Any]:
    path = pid_file(project_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def write_pid_state(project_root: Path, payload: dict[str, Any]) -> Path:
    path = pid_file(project_root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def workbench_status(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]
    cfg = load_workbench_config(project_root=root)
    ends = workbench_endpoints(cfg)
    pids = read_pid_state(root)
    api_up = _port_open(ends["api_host"], ends["api_port"])
    web_up = _port_open(ends["web_host"], ends["web_port"])
    return {
        "project_root": str(root),
        "api": {
            "url": ends["api_url"],
            "up": api_up,
            "pid": pids.get("api_pid"),
        },
        "web": {
            "url": ends["web_url"],
            "up": web_up,
            "pid": pids.get("web_pid"),
        },
        "open_browser": ends["open_browser"],
        "config_path": ends["config_path"],
        "pid_file": str(pid_file(root)),
        "overall": "running" if api_up and web_up else ("partial" if api_up or web_up else "stopped"),
        "note": (
            "Backend = FastAPI (API). Frontend = React/Vite. "
            "Open the web URL in a browser."
        ),
    }
