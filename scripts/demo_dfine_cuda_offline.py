"""Offline DFINE CUDA demo (v2.3.7): zero GPU / zero network walkthrough.

Runs doctor (no probe) + dry-run Fast Eval / triad / real-acceptance plan.
Never submits Docker work and never claims formal superiority.

Usage:
  python scripts/demo_dfine_cuda_offline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def main() -> int:
    out_root = ROOT / "outputs" / "_demo_dfine_cuda_offline"
    out_root.mkdir(parents=True, exist_ok=True)

    service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=out_root / "demo.db",
            runtime_dir=out_root / "runtime",
            outputs_dir=out_root / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )

    doctor = service.dfine_cuda_doctor(probe_runtime=False)
    fast = service.dfine_cuda_fast_eval(dry_run=True, probe_runtime=False)
    triad = service.dfine_cuda_formal_triad(dry_run=True, probe_runtime=False)
    real = service.dfine_real_acceptance(
        dry_run=True, include_formal_triad=True, probe_runtime=False
    )
    path_gate = service.dfine_formal_path_gate("project_demo_dfine_cuda")

    report = {
        "demo": "dfine_cuda_offline_v237",
        "version": "v2.3.0",
        "network_used": False,
        "gpu_used": False,
        "formal_success": False,
        "formal_superiority_claimed": False,
        "steps": {
            "doctor_offline": {
                "overall": doctor.get("overall"),
                "ok": doctor.get("ok"),
                "live_ready": doctor.get("live_ready"),
                "doctor_version": doctor.get("doctor_version"),
                "check_ids": [c.get("id") for c in (doctor.get("checks") or [])],
            },
            "fast_eval_dry_run": {
                "status": fast.get("status"),
                "exploratory_only": fast.get("exploratory_only"),
                "formal_success": fast.get("formal_success"),
            },
            "formal_triad_dry_run": {
                "status": triad.get("status"),
                "formal_superiority_claimed": triad.get(
                    "formal_superiority_claimed"
                ),
                "roles": sorted((triad.get("would_run") or {}).keys()),
            },
            "real_acceptance_plan": {
                "pipeline": real.get("pipeline"),
                "steps": (real.get("plan") or {}).get("steps"),
                "status": real.get("status"),
            },
            "formal_path_gate_empty_project": {
                "formal_path_open": path_gate.get("formal_path_open"),
                "formal_superiority_eligible": path_gate.get(
                    "formal_superiority_eligible"
                ),
                "formal_success": path_gate.get("formal_success"),
            },
        },
        "docs": "docs/dfine-cuda-runthrough.md",
        "next_live": [
            "scientist-lab dfine-cuda-doctor",
            "$env:RUN_REAL_DFINE_TESTS=1; $env:RUN_REAL_CUDA=1",
            "python scripts/accept_v23_real.py",
        ],
    }

    ok = (
        doctor.get("ok") is True
        and fast.get("status") == "dry_run"
        and triad.get("status") == "dry_run"
        and real.get("status") == "dry_run"
        and path_gate.get("formal_success") is False
        and report["formal_superiority_claimed"] is False
    )
    report["overall"] = "passed" if ok else "failed"

    path = out_root / "demo_report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
