"""v2.2 real acceptance: gated live Provider restricted Diff report.

Default / CI: SKIP (exit 0) — never opens network sockets.

Live mode requires ALL of:
  RUN_REAL_PATCH_TESTS=1
  LLM_ALLOW_NETWORK=true
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL configured

Usage:
  python scripts/accept_v22_real.py
  # skip unless gates set

  $env:RUN_REAL_PATCH_TESTS=1
  $env:LLM_ALLOW_NETWORK="true"
  $env:LLM_API_KEY="..."
  $env:LLM_BASE_URL="https://..."
  $env:LLM_MODEL="..."
  python scripts/accept_v22_real.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.llm.eval_suite import real_eval_gates
from scientist_lab.patching.service import PatchingService
from scientist_lab.storage.database import init_db


def _check(name: str, ok: bool, detail: str = "", *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def _env_truthy(name: str, environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return str(env.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def real_patch_acceptance_gates(
    *,
    environ: dict[str, str] | None = None,
) -> tuple[bool, str, dict[str, bool]]:
    """Composite gate for accept_v22_real live mode."""
    env = environ if environ is not None else dict(os.environ)
    allowed, reason = real_eval_gates(
        provider="openai-compatible",
        allow_network=True,
        environ=env,
    )
    gates = {
        "run_real_patch_tests": _env_truthy("RUN_REAL_PATCH_TESTS", env),
        "llm_allow_network": _env_truthy("LLM_ALLOW_NETWORK", env),
        "llm_api_key": bool(str(env.get("LLM_API_KEY") or "").strip()),
        "llm_base_url": bool(str(env.get("LLM_BASE_URL") or "").strip()),
        "llm_model": bool(str(env.get("LLM_MODEL") or "").strip()),
        "real_eval_ok": bool(allowed),
    }
    if not gates["run_real_patch_tests"]:
        return False, "RUN_REAL_PATCH_TESTS not set", gates
    if not gates["llm_allow_network"]:
        return False, "LLM_ALLOW_NETWORK not true", gates
    if not allowed:
        return False, reason or "real_eval_gates failed", gates
    return True, "ok", gates


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v22_real"
    accept_root.mkdir(parents=True, exist_ok=True)
    report_path = accept_root / "v22_real_restricted_diff_report.json"

    ready, reason, gates = real_patch_acceptance_gates()
    if not ready:
        report = {
            "suite": "accept_v22_real",
            "version": "v2.2.0",
            "overall": "skipped",
            "network_used": False,
            "reason": reason,
            "gates": gates,
            "checks": [
                _check(
                    "live_gates",
                    True,
                    f"SKIP: {reason}",
                    skipped=True,
                )
            ],
            "definition": (
                "Gated live acceptance for real-provider restricted Diff. "
                "Default SKIP preserves CI zero-network."
            ),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report={report_path}")
        return 0

    db_path = accept_root / "accept_v22_real.db"
    if db_path.exists():
        db_path.unlink()
    Session = init_db(str(db_path))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=accept_root / "outputs",
        sandbox_root=accept_root / "sandbox",
    )

    checks: list[dict[str, Any]] = []
    checks.append(_check("live_gates", True, "all gates open"))

    built = service.build_code_context(digits_demo=True, persist=True)
    bundle = built["bundle"]
    bid = bundle["bundle_id"]
    checks.append(
        _check(
            "code_context_built",
            bool(bid) and bool(bundle.get("context_sha256")),
            bid,
        )
    )

    try:
        view = service.propose_real(
            bid,
            allow_network=True,
            requested_provider="openai-compatible",
            real_only=True,
        )
        checks.append(
            _check(
                "propose_real_live",
                view.get("status") in {"verified", "proposed", "rejected_by_verifier"}
                and view.get("fallback_used") is not True,
                f"status={view.get('status')} provider={view.get('provider')}",
            )
        )
        if view.get("status") == "verified":
            pid = view["patch_id"]
            approved = service.approve(pid, reason="accept_v22_real")
            checks.append(
                _check(
                    "approve_sealed",
                    approved.get("status") == "approved",
                    approved.get("status", ""),
                )
            )
            applied = service.apply_sandbox(pid)
            checks.append(
                _check(
                    "sandbox_apply",
                    applied.get("status") == "applied_sandbox",
                    applied.get("status", ""),
                )
            )
            tested = service.test_sandbox(pid, profile="unit")
            checks.append(
                _check(
                    "sandbox_unit",
                    (tested.get("sandbox_tests") or {}).get("ok") is True,
                    "unit",
                )
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("propose_real_live", False, str(exc)))

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "suite": "accept_v22_real",
        "version": "v2.2.0",
        "overall": "passed" if passed == total else "failed",
        "passed": passed,
        "total": total,
        "network_used": True,
        "gates": gates,
        "checks": checks,
        "definition": (
            "Live acceptance for real-provider restricted Diff "
            "(CodeContext → propose_real → seal → sandbox). "
            "Does not modify main workspace; does not auto-merge."
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
