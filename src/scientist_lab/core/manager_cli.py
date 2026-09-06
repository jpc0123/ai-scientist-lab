"""CLI helper for Freeze Manager. Does not import docker / ExperimentService."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.core.manager import Manager
from scientist_lab.core.schema_registry import load_json
from scientist_lab.core.state_machine import RunState


def run_manager_from_files(
    protocol_path: Path | str,
    plan_path: Path | str,
    *,
    output_dir: Path | str,
    execute: bool = False,
    require_live_ready: bool = False,
    max_steps: int = 32,
    live_runner: Any | None = None,
    doctor_fn: Any | None = None,
    doctor_root: Path | str | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
    baseline_metrics_path: Path | str | None = None,
    max_extra_rounds: int = 0,
    planner_backend: str | None = None,
    reviewer_backend: str | None = None,
    llm_provider: Any | None = None,
    llm_live: bool = False,
    fallback_to_rules: bool = False,
    confirm_human_gate: bool = False,
    project_root: Path | str | None = None,
    inspect_docker: bool | None = None,
) -> dict[str, Any]:
    """Walk the Manager state machine. Default is dry-run / REPLAY.

    REAL GPU requires ``execute=True`` and either a stub ``live_runner`` or
    ``make_cuda_live_runner`` after CUDA doctor ``live_ready``. Does not bypass
    Gate. Does not forge metrics.json. ``max_extra_rounds`` default 0 keeps
    a single seed round.

    v2.5-D: ``planner_backend`` / ``reviewer_backend`` default rules (None →
    Planner()/Reviewer() constructors, still rules unless env opens llm).
    """
    lease = None
    real_gpu = bool(execute) and live_runner is None
    if real_gpu:
        from scientist_lab.services.live_gpu_mutex import (
            LiveGpuBusyError,
            acquire_live_execute,
        )

        root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        try:
            lease = acquire_live_execute(
                root,
                owner_id=f"cli:{Path(output_dir)}",
                inspect_docker=True if inspect_docker is None else bool(inspect_docker),
            )
        except LiveGpuBusyError as exc:
            print(str(exc), file=sys.stderr)
            return {
                "ok": False,
                "exit_code": 1,
                "status": "blocked",
                "fail_closed": True,
                "metrics_forged": False,
                "error": str(exc),
                "holder": exc.holder,
            }
    try:
        return _run_manager_from_files_inner(
            protocol_path,
            plan_path,
            output_dir=output_dir,
            execute=execute,
            require_live_ready=require_live_ready,
            max_steps=max_steps,
            live_runner=live_runner,
            doctor_fn=doctor_fn,
            doctor_root=doctor_root,
            baseline_metrics=baseline_metrics,
            baseline_metrics_path=baseline_metrics_path,
            max_extra_rounds=max_extra_rounds,
            planner_backend=planner_backend,
            reviewer_backend=reviewer_backend,
            llm_provider=llm_provider,
            llm_live=llm_live,
            fallback_to_rules=fallback_to_rules,
            confirm_human_gate=confirm_human_gate,
        )
    finally:
        if lease is not None:
            lease.release()


def _run_manager_from_files_inner(
    protocol_path: Path | str,
    plan_path: Path | str,
    *,
    output_dir: Path | str,
    execute: bool = False,
    require_live_ready: bool = False,
    max_steps: int = 32,
    live_runner: Any | None = None,
    doctor_fn: Any | None = None,
    doctor_root: Path | str | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
    baseline_metrics_path: Path | str | None = None,
    max_extra_rounds: int = 0,
    planner_backend: str | None = None,
    reviewer_backend: str | None = None,
    llm_provider: Any | None = None,
    llm_live: bool = False,
    fallback_to_rules: bool = False,
    confirm_human_gate: bool = False,
) -> dict[str, Any]:
    if llm_live:
        from scientist_lab.llm.runtime_secrets import apply_runtime_llm_env

        apply_runtime_llm_env(Path(__file__).resolve().parents[3] / "runtime")
    protocol = load_json(protocol_path)
    plan = load_json(plan_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeded_baseline = dict(baseline_metrics or {})
    if baseline_metrics_path is not None:
        seeded_baseline.update(load_json(baseline_metrics_path))
    if seeded_baseline:
        (output / "baseline_metrics.json").write_text(
            json.dumps(seeded_baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    root = Path(doctor_root) if doctor_root is not None else None
    mgr = Manager(
        output,
        protocol=protocol,
        execute=execute,
        require_live_ready=require_live_ready,
        live_runner=live_runner,
        doctor_fn=doctor_fn,
        doctor_root=root,
        baseline_metrics=seeded_baseline or None,
        max_extra_rounds=max(0, int(max_extra_rounds)),
        planner_backend=planner_backend,
        reviewer_backend=reviewer_backend,
        llm_provider=llm_provider,
        llm_live=bool(llm_live),
        fallback_to_rules=bool(fallback_to_rules),
        confirm_human_gate=bool(confirm_human_gate),
    )
    src_plan = Path(plan_path).resolve()
    dest_plan = Path(mgr.plan_path).resolve()
    if src_plan != dest_plan:
        shutil.copyfile(src_plan, dest_plan)
    doctor = mgr.probe_doctor()
    doctor_path = output / "cuda_doctor.json"
    doctor_path.write_text(
        json.dumps(doctor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    live_ready = bool(doctor.get("live_ready"))
    print(
        json.dumps(
            {
                "precheck": "cuda_doctor",
                "live_ready": live_ready,
                "overall": doctor.get("overall"),
                "execute": bool(execute),
                "require_live_ready": bool(require_live_ready),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if require_live_ready and not live_ready:
        print(
            "cuda doctor live_ready=false; refusing GPU and forged metrics.json",
            file=sys.stderr,
        )
        if not execute:
            return {
                "ok": False,
                "exit_code": 1,
                "live_ready": False,
                "doctor": {
                    "live_ready": False,
                    "overall": doctor.get("overall"),
                },
                "actions": [],
                "last": None,
                "metrics_forged": False,
            }

    mgr.initialize_run(
        round_index=int(plan.get("round_index") or 0),
        plan_id=plan.get("plan_id"),
    )
    steps = mgr.run_until(max_steps=max_steps)
    last = steps[-1] if steps else None
    last_state = last.state.run_state if last is not None else None
    blocked_or_failed = last_state in {RunState.BLOCKED, RunState.FAILED}
    exit_code = 1 if blocked_or_failed or (require_live_ready and not live_ready) else 0
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "live_ready": live_ready,
        "doctor": {
            "live_ready": live_ready,
            "overall": doctor.get("overall"),
        },
        "actions": [step.action for step in steps],
        "last": last.to_dict() if last is not None else None,
        "metrics_forged": False,
    }
