"""Load local workbench YAML config (v2.0.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "Scientist Lab",
        "environment": "local",
        "host": "127.0.0.1",
        "port": 8787,
    },
    "database": {"path": "./scientist_lab.db"},
    "artifacts": {"root": "./outputs"},
    "web": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 5173,
        "open_browser": True,
    },
    "llm": {
        "default_provider": "mock",
        "require_quality_gate": True,
    },
    "security": {
        "allow_network_by_default": False,
        "allow_real_llm_by_default": False,
        "allow_main_tree_patch": False,
    },
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def default_config_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "config" / "scientist-lab.yaml"


def load_workbench_config(
    path: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_config_path(project_root)
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception:  # noqa: BLE001
            data = {}
    merged = _deep_merge(DEFAULTS, data)
    merged["_config_path"] = str(cfg_path)
    merged["_config_exists"] = cfg_path.is_file()
    return merged


def workbench_endpoints(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_workbench_config()
    app = cfg.get("app") or {}
    web = cfg.get("web") or {}
    api_host = str(app.get("host") or "127.0.0.1")
    api_port = int(app.get("port") or 8787)
    web_host = str(web.get("host") or "127.0.0.1")
    web_port = int(web.get("port") or 5173)
    return {
        "api_host": api_host,
        "api_port": api_port,
        "api_url": f"http://{api_host}:{api_port}",
        "web_host": web_host,
        "web_port": web_port,
        "web_url": f"http://{web_host}:{web_port}",
        "open_browser": bool(web.get("open_browser", True)),
        "config_path": cfg.get("_config_path"),
    }
