"""v2.2 offline acceptance: restricted Diff + Replay + Web API (zero network).

Covers:
  CodeContextBundle → DiffSafety → MockTransport propose_real
  → approval seal → sandbox → unit → PatchEvidence feedback
  → replay export → Web API gates → unit regression

Usage:
  python scripts/accept_v22.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab import API_VERSION, __version__
from scientist_lab.api.app import create_app
from scientist_lab.patching.context_bundle import (
    build_code_context_bundle,
    digits_improvement_patch_request,
)
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.replay_bundle import (
    build_patch_replay_bundle,
    digits_fixture_provider_response,
    replay_patch_static_checks,
)
from scientist_lab.patching.service import PatchingService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.database import init_db

from fastapi.testclient import TestClient


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v22"
    accept_root.mkdir(parents=True, exist_ok=True)
    db_path = accept_root / "accept_v22.db"
    if db_path.exists():
        db_path.unlink()

    Session = init_db(str(db_path))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=accept_root / "outputs",
        sandbox_root=accept_root / "sandbox",
    )

    checks: list[dict] = []

    # 0) Version freeze target (aligned with current package; suite is v2.2 feature regression)
    checks.append(
        _check(
            "package_version_aligned",
            __version__ == API_VERSION.lstrip("v") and bool(API_VERSION),
            f"pkg={__version__} api={API_VERSION}",
        )
    )

    # 1) CodeContextBundle
    req = digits_improvement_patch_request(
        request_id="preq_accept_v22",
        source_commit="accept_v22_fixed",
        project_id="project_accept_v22",
    )
    bundle = build_code_context_bundle(
        req, project_root=ROOT, bundle_id="ctx_accept_v22"
    )
    service._contexts.upsert(bundle)
    checks.append(
        _check(
            "code_context_bundle",
            len(bundle.snapshots) == 1
            and bundle.snapshots[0].path == "experiment_app/run_experiment.py"
            and bool(bundle.context_sha256),
            f"sha={bundle.context_sha256[:12]}",
        )
    )

    # 2) PathPolicy.for_code_context allows Digits; default does not
    default_ok, _ = PathPolicy().is_allowed("experiment_app/run_experiment.py")
    ctx_ok, _ = PathPolicy.for_code_context().is_allowed(
        "experiment_app/run_experiment.py"
    )
    checks.append(
        _check(
            "path_policy_digits_scoped",
            (not default_ok) and ctx_ok,
            f"default={default_ok} context={ctx_ok}",
        )
    )

    # 3) Build + static replay
    replay_dir = accept_root / "replay_bundle"
    if replay_dir.exists():
        import shutil

        shutil.rmtree(replay_dir)
    response = digits_fixture_provider_response()
    export = build_patch_replay_bundle(
        output_dir=replay_dir,
        bundle=bundle,
        provider_response=response,
        verification={"ok": True},
        provider_audit={
            "requested_provider": "openai-compatible",
            "actual_provider": "openai-compatible",
            "fallback_used": False,
        },
        label="accept_v22",
    )
    checks.append(
        _check(
            "replay_bundle_export",
            export.get("ok") is True and export.get("secrets_redacted") is True,
            export.get("bundle_dir", ""),
        )
    )
    static = replay_patch_static_checks(replay_dir)
    checks.append(
        _check(
            "replay_static_diff_safety",
            static.get("ok") is True and static.get("network_used") is False,
            f"ok={static.get('ok')}",
        )
    )

    # 4) MockTransport propose_real from bundle
    view = service.replay_patch_from_bundle(replay_dir)
    checks.append(
        _check(
            "replay_propose_real",
            view.get("status") == "verified"
            and view.get("fallback_used") is False
            and (view.get("replay") or {}).get("network_used") is False,
            f"status={view.get('status')} patch={view.get('patch_id')}",
        )
    )

    # 5) Sandbox profiles registry
    profiles = service.list_sandbox_test_profiles()
    ids = {p["id"] for p in profiles["items"]}
    checks.append(
        _check(
            "sandbox_registry",
            ids == {"smoke", "syntax", "unit", "mock_experiment"}
            and profiles.get("arbitrary_commands_forbidden") is True,
            str(sorted(ids)),
        )
    )

    # 6) Approve seal + sandbox apply + unit test + feedback feedback
    pid = view["patch_id"]
    item = service._repo.require(pid)
    item.metadata["context_sha256"] = bundle.context_sha256
    item.metadata["source_commit"] = bundle.source_commit
    item.metadata["bundle_id"] = bundle.bundle_id
    service._repo.upsert(item)
    approved = service.approve(pid)
    checks.append(
        _check(
            "approval_seal",
            approved.get("status") == "approved"
            and bool((approved.get("approval") or {}).get("content_seal")),
            "sealed",
        )
    )
    applied = service.apply_sandbox(pid)
    checks.append(
        _check(
            "sandbox_apply_digits",
            applied.get("status") == "applied_sandbox"
            and (applied.get("sandbox") or {}).get("main_workspace_modified") is False,
            applied.get("status", ""),
        )
    )
    tested = service.test_sandbox(pid, profile="unit")
    checks.append(
        _check(
            "sandbox_unit_profile",
            (tested.get("sandbox_tests") or {}).get("ok") is True,
            "unit",
        )
    )
    recorded = service.record_evidence(pid, require_tests=True, build_feedback=True)
    checks.append(
        _check(
            "patch_feedback",
            recorded.get("status") == "evidence_recorded"
            and (recorded.get("patch_feedback") or {}).get("verdict") == "effective",
            str((recorded.get("patch_feedback") or {}).get("verdict")),
        )
    )

    # 7) Export replay from recorded patch
    export_dir = accept_root / "replay_from_patch"
    if export_dir.exists():
        import shutil

        shutil.rmtree(export_dir)
    exported = service.export_patch_replay(pid, output_dir=export_dir)
    checks.append(
        _check(
            "export_from_patch",
            exported.get("ok") is True,
            exported.get("bundle_dir", ""),
        )
    )

    # 8) Web API smoke (v2.2.9) — no live provider
    api_db = accept_root / "accept_v22_api.db"
    if api_db.exists():
        api_db.unlink()
    api_service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=api_db,
            runtime_dir=accept_root / "runtime_api",
            outputs_dir=accept_root / "outputs_api",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    client = TestClient(create_app(service=api_service))
    health = client.get("/api/v1/health")
    checks.append(
        _check(
            "api_health_version",
            health.status_code == 200 and health.json().get("version") == API_VERSION,
            str(health.json().get("version")),
        )
    )
    built = client.post(
        "/api/v1/code-contexts/build",
        json={"digits_demo": True, "persist": True},
    )
    bid = (built.json().get("bundle") or {}).get("bundle_id", "")
    checks.append(
        _check(
            "api_code_context_build",
            built.status_code == 200 and bool(bid),
            bid,
        )
    )
    blocked = client.post(
        "/api/v1/patches/propose-real",
        json={
            "bundle_id": bid,
            "allow_network": False,
            "provider": "openai-compatible",
            "real_only": True,
        },
    )
    checks.append(
        _check(
            "api_propose_real_no_silent_mock",
            blocked.status_code == 409,
            f"status={blocked.status_code}",
        )
    )
    openapi = client.get("/openapi.json").json()
    paths = openapi.get("paths") or {}
    checks.append(
        _check(
            "api_openapi_v22_paths",
            "/api/v1/patches/propose-real" in paths
            and "/api/v1/code-contexts/build" in paths
            and "/api/v1/patches/{patch_id}/export-replay" in paths,
            "propose-real+code-contexts+export-replay",
        )
    )

    # 9) Unit regression hook (v2.2.1–2.2.9)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_patching_v221.py",
            "tests/unit/test_patching_v222.py",
            "tests/unit/test_patching_v223.py",
            "tests/unit/test_patching_v224.py",
            "tests/unit/test_patching_v225.py",
            "tests/unit/test_patching_v226.py",
            "tests/unit/test_patching_v227.py",
            "tests/unit/test_patching_v228.py",
            "tests/unit/test_api_v1.py::test_v229_code_context_and_propose_real_gate",
            "tests/unit/test_api_v1.py::test_v229_export_replay_and_check_seal",
            "-q",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    checks.append(
        _check(
            "unit_regression_v221_v229",
            proc.returncode == 0,
            (proc.stdout or proc.stderr or "")[-400:],
        )
    )

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "suite": "accept_v22",
        "version": API_VERSION,
        "feature_line": "v2.2",
        "overall": "passed" if passed == total else "failed",
        "passed": passed,
        "total": total,
        "network_used": False,
        "checks": checks,
        "definition": (
            "Offline acceptance for real-provider restricted Diff plumbing "
            "(CodeContext → Replay propose → DiffSafety → seal → sandbox → "
            "feedback → Web API gates). Not live cloud Diff; not CUDA/DFINE."
        ),
    }
    report_path = accept_root / "accept_v22_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (accept_root / "README.md").write_text(
        """# Scientist Lab v2.2 Acceptance (offline)

## Run

```text
python scripts/accept_v22.py
```

Zero network. Uses Patch Replay Bundle + MockTransport + Web API gates.
Live Provider Diff: `scripts/accept_v22_real.py` (default SKIP).
""",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
