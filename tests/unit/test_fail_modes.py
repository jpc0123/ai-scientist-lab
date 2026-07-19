from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_experiment_missing_metrics(tmp_path: Path):
    app = Path(__file__).resolve().parents[2] / "experiment_app" / "run_mock_experiment.py"
    config = tmp_path / "config.json"
    out = tmp_path / "out"
    out.mkdir()
    config.write_text(
        json.dumps({"fail_mode": "missing_metrics", "sleep_seconds": 0}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(app),
            "--config",
            str(config),
            "--output-dir",
            str(out),
            "--seed",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert not (out / "metrics.json").exists()
    assert (out / "artifact_manifest.json").exists()


def test_run_experiment_exception(tmp_path: Path):
    app = Path(__file__).resolve().parents[2] / "experiment_app" / "run_mock_experiment.py"
    config = tmp_path / "config.json"
    out = tmp_path / "out"
    out.mkdir()
    config.write_text(
        json.dumps({"fail_mode": "exception", "sleep_seconds": 0}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(app),
            "--config",
            str(config),
            "--output-dir",
            str(out),
            "--seed",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
