"""Gated live CUDA + Vendor DFINE acceptance contract (v2.3.6).

Default CI path never runs GPU work. Live mode requires explicit env gates and
doctor live_ready. Never declares formal DFINE superiority.
"""

from __future__ import annotations

import os
from typing import Any

from scientist_lab.tasks.rgbt_detection.dfine_cuda_orchestrator import (
    DfineCudaOrchestratorError,
)
from scientist_lab.tasks.rgbt_detection.vendor_audit import (
    audit_dfine_vendor_pin,
    cuda_dockerfile_present,
)

PIPELINE_ID = "dfine_real_acceptance_v236"

# Ordered live steps when gates open (triad is optional / env-gated).
CORE_LIVE_STEPS = (
    "doctor_live_ready",
    "fast_eval_execute",
    "evidence_non_standin",
    "formal_path_gate",
)
OPTIONAL_TRIAD_STEP = "formal_triad_execute"


def _env_truthy(name: str, environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return str(env.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def real_dfine_acceptance_gates(
    project_root: Any,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[bool, str, dict[str, bool]]:
    """Composite gate for accept_v23_real live mode."""
    env = environ if environ is not None else dict(os.environ)
    root = project_root
    gates = {
        "run_real_dfine_tests": _env_truthy("RUN_REAL_DFINE_TESTS", env),
        "run_real_cuda": _env_truthy("RUN_REAL_CUDA", env),
        "dockerfile_present": bool(cuda_dockerfile_present(root)),
        "vendor_pin_ok": bool(audit_dfine_vendor_pin(root).get("ok")),
        "include_formal_triad": _env_truthy("ACCEPT_V23_REAL_TRIAD", env),
    }
    if not gates["run_real_dfine_tests"]:
        return False, "RUN_REAL_DFINE_TESTS not set", gates
    if not gates["run_real_cuda"]:
        return False, "RUN_REAL_CUDA not set", gates
    if not gates["dockerfile_present"]:
        return False, "CUDA Dockerfile missing", gates
    if not gates["vendor_pin_ok"]:
        return False, "vendor pin audit failed", gates
    return True, "ok", gates


def describe_real_acceptance_plan(
    *,
    include_formal_triad: bool = False,
) -> dict[str, Any]:
    """Offline-inspectable contract of what live acceptance will run."""
    steps = list(CORE_LIVE_STEPS)
    if include_formal_triad:
        steps.append(OPTIONAL_TRIAD_STEP)
    return {
        "pipeline": PIPELINE_ID,
        "steps": steps,
        "exploratory_only": True,
        "formal_success": False,
        "formal_superiority_claimed": False,
        "default_mode": "skip_without_gates",
        "required_env": ["RUN_REAL_DFINE_TESTS", "RUN_REAL_CUDA"],
        "optional_env": ["ACCEPT_V23_REAL_TRIAD"],
        "cli_equivalents": [
            "scientist-lab dfine-cuda-doctor",
            "scientist-lab dfine-cuda-fast-eval --execute --require-live-ready",
            "scientist-lab dfine-formal-path-gate <project_id>",
            "scientist-lab dfine-cuda-formal-triad --execute --require-live-ready",
        ],
        "note": (
            "Live Fast Eval / triad remains exploratory; path-open ≠ superiority."
        ),
    }


def _check(name: str, ok: bool, detail: str = "", *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def run_real_dfine_acceptance(
    experiments: Any,
    *,
    dry_run: bool = False,
    include_formal_triad: bool | None = None,
    wait: bool = True,
    probe_runtime: bool = True,
) -> dict[str, Any]:
    """Execute (or dry-run) the gated live acceptance pipeline.

    dry_run=True never touches GPU / Docker submit; used by offline tests and
    accept_v23 to validate the contract shape.
    """
    if include_formal_triad is None:
        include_formal_triad = _env_truthy("ACCEPT_V23_REAL_TRIAD")

    plan = describe_real_acceptance_plan(
        include_formal_triad=bool(include_formal_triad)
    )
    payload: dict[str, Any] = {
        "pipeline": PIPELINE_ID,
        "dry_run": bool(dry_run),
        "plan": plan,
        "exploratory_only": True,
        "formal_success": False,
        "formal_superiority_claimed": False,
        "checks": [],
        "results": {},
    }

    if dry_run:
        payload["status"] = "dry_run"
        payload["checks"] = [
            _check(step, True, "would_run", skipped=False)
            for step in plan["steps"]
        ]
        payload["would_run"] = {
            "require_live_ready": True,
            "steps": plan["steps"],
            "include_formal_triad": bool(include_formal_triad),
        }
        return payload

    checks: list[dict[str, Any]] = []
    gpu_used = False

    # 1) Doctor live_ready
    try:
        doctor = experiments.dfine_cuda_doctor(probe_runtime=probe_runtime)
        live_ok = bool(doctor.get("live_ready"))
        checks.append(
            _check(
                "doctor_live_ready",
                live_ok,
                f"overall={doctor.get('overall')} live_ready={live_ok}",
            )
        )
        payload["results"]["doctor"] = {
            "overall": doctor.get("overall"),
            "live_ready": live_ok,
        }
        if not live_ok:
            payload["status"] = "blocked"
            payload["checks"] = checks
            payload["gpu_used"] = False
            payload["reason"] = "doctor live_ready=false"
            return payload
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("doctor_live_ready", False, str(exc)))
        payload["status"] = "failed"
        payload["checks"] = checks
        payload["gpu_used"] = False
        return payload

    # 2) Fast Eval execute
    project_id: str | None = None
    try:
        result = experiments.dfine_cuda_fast_eval(
            dry_run=False,
            require_live_ready=True,
            probe_runtime=probe_runtime,
            wait=wait,
        )
        gpu_used = True
        run = result.get("run") or {}
        plan_meta = result.get("plan") or {}
        project_id = plan_meta.get("project_id")
        ok = (
            str(run.get("status")) == "completed"
            and result.get("formal_success") is False
            and result.get("exploratory_only") is True
        )
        checks.append(
            _check(
                "fast_eval_execute",
                ok,
                f"status={run.get('status')} exec={run.get('execution_id')}",
            )
        )
        payload["results"]["fast_eval"] = {
            "status": run.get("status"),
            "execution_id": run.get("execution_id"),
            "project_id": project_id,
            "formal_success": result.get("formal_success"),
            "exploratory_only": result.get("exploratory_only"),
        }

        # 3) Evidence non-stand-in (orchestrator usually attaches feedback)
        evidence = result.get("evidence_feedback") or {}
        if not evidence and run.get("execution_id"):
            evidence = experiments.dfine_cuda_record_evidence(
                str(run["execution_id"]),
                refresh_claim_matrix=True,
            )
        kind = str(evidence.get("standin_or_vendor") or "")
        checks.append(
            _check(
                "evidence_non_standin",
                kind == "vendor"
                and evidence.get("formal_success") is False
                and evidence.get("exploratory_only") is True,
                f"standin_or_vendor={kind} evidence_id={evidence.get('evidence_id')}",
            )
        )
        payload["results"]["evidence"] = {
            "evidence_id": evidence.get("evidence_id"),
            "standin_or_vendor": kind,
            "formal_dfine_claim": evidence.get("formal_dfine_claim"),
            "formal_dfine_path": evidence.get("formal_dfine_path"),
        }
    except DfineCudaOrchestratorError as exc:
        checks.append(_check("fast_eval_execute", False, str(exc)))
        payload["status"] = "failed"
        payload["checks"] = checks
        payload["gpu_used"] = gpu_used
        return payload
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("fast_eval_execute", False, str(exc)))
        payload["status"] = "failed"
        payload["checks"] = checks
        payload["gpu_used"] = gpu_used
        return payload

    # 4) Formal path gate (must not claim superiority)
    try:
        if not project_id:
            project_id = "project_rgbt_cuda"
        gate_pkg = experiments.dfine_formal_path_gate(str(project_id))
        checks.append(
            _check(
                "formal_path_gate",
                gate_pkg.get("formal_success") is False
                and gate_pkg.get("formal_superiority_eligible") is not True,
                (
                    f"path_open={gate_pkg.get('formal_path_open')} "
                    f"sup_eligible={gate_pkg.get('formal_superiority_eligible')}"
                ),
            )
        )
        payload["results"]["formal_path_gate"] = {
            "formal_path_open": gate_pkg.get("formal_path_open"),
            "formal_superiority_eligible": gate_pkg.get(
                "formal_superiority_eligible"
            ),
            "path_status": (gate_pkg.get("path_claim") or {}).get(
                "support_status"
            ),
            "superiority_status": (gate_pkg.get("superiority_claim") or {}).get(
                "support_status"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("formal_path_gate", False, str(exc)))

    # 5) Optional formal triad
    if include_formal_triad:
        try:
            triad = experiments.dfine_cuda_formal_triad(
                dry_run=False,
                require_live_ready=True,
                probe_runtime=probe_runtime,
                wait=wait,
            )
            gpu_used = True
            checks.append(
                _check(
                    "formal_triad_execute",
                    triad.get("formal_success") is False
                    and triad.get("formal_superiority_claimed") is False
                    and triad.get("status") in {"completed", "partial"},
                    f"status={triad.get('status')}",
                )
            )
            payload["results"]["formal_triad"] = {
                "status": triad.get("status"),
                "formal_success": triad.get("formal_success"),
                "formal_superiority_claimed": triad.get(
                    "formal_superiority_claimed"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("formal_triad_execute", False, str(exc)))

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    payload["checks"] = checks
    payload["passed"] = passed
    payload["total"] = total
    payload["gpu_used"] = gpu_used
    payload["status"] = "completed" if passed == total else "failed"
    return payload
