"""v2.3 real acceptance: gated CUDA + Vendor DFINE live pipeline (v2.3.6).

Default / CI: SKIP (exit 0) — never requires GPU or Docker.

Live mode requires ALL of:
  RUN_REAL_DFINE_TESTS=1
  RUN_REAL_CUDA=1
  Docker image scientist-rgbt-detection:v2-cuda available
  nvidia-smi OK (doctor live_ready)

Optional:
  ACCEPT_V23_REAL_TRIAD=1  — also run CUDA formal triad after Fast Eval

Usage:
  python scripts/accept_v23_real.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.dfine_real_acceptance import (
    PIPELINE_ID,
    describe_real_acceptance_plan,
    real_dfine_acceptance_gates,
    run_real_dfine_acceptance,
)


def _check(name: str, ok: bool, detail: str = "", *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v23_real"
    accept_root.mkdir(parents=True, exist_ok=True)
    report_path = accept_root / "v23_real_dfine_report.json"

    ready, reason, gates = real_dfine_acceptance_gates(ROOT)
    plan = describe_real_acceptance_plan(
        include_formal_triad=bool(gates.get("include_formal_triad"))
    )

    if not ready:
        report = {
            "suite": "accept_v23_real",
            "version": "v2.3.0",
            "pipeline": PIPELINE_ID,
            "overall": "skipped",
            "network_used": False,
            "gpu_used": False,
            "reason": reason,
            "gates": gates,
            "plan": plan,
            "checks": [
                _check("live_gates", True, f"SKIP: {reason}", skipped=True)
            ],
            "definition": (
                "Gated live CUDA + Vendor DFINE acceptance pipeline "
                "(doctor → Fast Eval → evidence → path gate; optional triad). "
                "Default SKIP preserves CI zero-GPU. "
                "Never claims formal DFINE superiority."
            ),
            "next_when_gates_open": (
                "set RUN_REAL_DFINE_TESTS=1 RUN_REAL_CUDA=1 then re-run; "
                "optional ACCEPT_V23_REAL_TRIAD=1"
            ),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report={report_path}")
        return 0

    service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=accept_root / "accept_v23_real.db",
            runtime_dir=accept_root / "runtime",
            outputs_dir=accept_root / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )

    checks: list[dict[str, Any]] = [
        _check("live_gates", True, "all env gates open")
    ]
    live = run_real_dfine_acceptance(
        service,
        dry_run=False,
        include_formal_triad=bool(gates.get("include_formal_triad")),
        wait=True,
        probe_runtime=True,
    )
    checks.extend(live.get("checks") or [])

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "suite": "accept_v23_real",
        "version": "v2.3.0",
        "pipeline": PIPELINE_ID,
        "overall": "passed" if passed == total else "failed",
        "passed": passed,
        "total": total,
        "network_used": False,
        "gpu_used": bool(live.get("gpu_used")),
        "gates": gates,
        "plan": plan,
        "live": {
            "status": live.get("status"),
            "results": live.get("results"),
            "formal_success": live.get("formal_success"),
            "formal_superiority_claimed": live.get(
                "formal_superiority_claimed"
            ),
        },
        "checks": checks,
        "definition": (
            "Live CUDA Vendor DFINE acceptance via v2.3.6 pipeline. "
            "Exploratory only; never claims formal DFINE superiority."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
