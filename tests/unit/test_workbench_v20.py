from __future__ import annotations

from pathlib import Path

from scientist_lab.workbench.config import (
    DEFAULTS,
    load_workbench_config,
    workbench_endpoints,
)
from scientist_lab.workbench.status import workbench_status


def test_load_default_config_without_file(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    cfg = load_workbench_config(missing, project_root=tmp_path)
    assert cfg["app"]["port"] == DEFAULTS["app"]["port"]
    assert cfg["_config_exists"] is False
    ends = workbench_endpoints(cfg)
    assert ends["api_url"].endswith(":8787")
    assert ends["web_url"].endswith(":5173")


def test_load_yaml_overrides(tmp_path: Path):
    path = tmp_path / "scientist-lab.yaml"
    path.write_text(
        "app:\n  port: 9000\nweb:\n  port: 6000\n  open_browser: false\n",
        encoding="utf-8",
    )
    cfg = load_workbench_config(path, project_root=tmp_path)
    ends = workbench_endpoints(cfg)
    assert ends["api_port"] == 9000
    assert ends["web_port"] == 6000
    assert ends["open_browser"] is False


def test_workbench_status_shape():
    root = Path(__file__).resolve().parents[2]
    status = workbench_status(root)
    assert "api" in status and "web" in status
    assert status["overall"] in {"running", "partial", "stopped"}
    assert "FastAPI" in status["note"]
