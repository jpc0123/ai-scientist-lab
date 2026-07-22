"""v1.6 acceptance: restricted source patching + sandbox validation (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.patching.diff_parser import parse_unified_diff
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v16"
    settings = Settings(
        project_root=ROOT,
        db_path=accept_root / "accept.db",
        runtime_dir=accept_root / "runtime",
        outputs_dir=accept_root / "outputs",
        experiment_app_dir=ROOT / "experiment_app",
    ).resolve()
    if settings.db_path.exists():
        settings.db_path.unlink()
    service = ExperimentService(settings=settings)
    results: list[dict] = []

    # 1 Mock unified diff parseable
    try:
        diff = build_mock_unified_diff()
        parsed = parse_unified_diff(diff)
        results.append(
            _check(
                "mock-diff-parseable",
                len(parsed.files) == 1 and parsed.files[0].is_new_file,
                f"files={len(parsed.files)} path={parsed.files[0].path}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("mock-diff-parseable", False, str(exc)))

    # 2 Path policy denies escapes / secrets
    policy = PathPolicy()
    ok_allow, _ = policy.is_allowed(
        "experiment_apps/rgbt_detection_real/adapters/x.py"
    )
    ok_deny, _ = policy.is_allowed("src/scientist_lab/llm/config.py")
    ok_escape, _ = policy.is_allowed(
        "src/scientist_lab/tasks/rgbt_detection/../../llm/config.py"
    )
    results.append(
        _check(
            "path-policy-allow-deny",
            ok_allow and (not ok_deny) and (not ok_escape),
            f"allow={ok_allow} deny_config={not ok_deny} deny_escape={not ok_escape}",
        )
    )

    # 3 Verifier rejects binary / forbidden
    verifier = PatchVerifier(policy)
    bad = (
        "diff --git a/pyproject.toml b/pyproject.toml\n"
        "--- a/pyproject.toml\n"
        "+++ b/pyproject.toml\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    vbad = verifier.verify(bad)
    results.append(
        _check(
            "verifier-rejects-forbidden",
            vbad.ok is False
            and any(i.code == "path_policy_violation" for i in vbad.issues),
            f"ok={vbad.ok} issues={[i.code for i in vbad.issues]}",
        )
    )

    # 4 Propose mock → verified, cannot apply main
    proposed = service.patches.propose_mock("project_rgbt_003")
    results.append(
        _check(
            "propose-mock-verified",
            proposed["status"] == "verified"
            and proposed["can_apply_main"] is False
            and proposed["can_apply"] is False,
            f"status={proposed['status']} can_apply_main={proposed['can_apply_main']}",
        )
    )

    # 5 Duplicate fingerprint blocked
    dup = service.patches.propose_mock("project_rgbt_003")
    results.append(
        _check(
            "duplicate-fingerprint-blocked",
            dup["status"] == "rejected_by_verifier"
            and (dup.get("verification") or {}).get("duplicate_of")
            == proposed["patch_id"],
            f"status={dup['status']} dup_of={(dup.get('verification') or {}).get('duplicate_of')}",
        )
    )

    # 6 Approve does not apply main
    approved = service.patches.approve(proposed["patch_id"], reason="accept_v16")
    results.append(
        _check(
            "approve-not-main-apply",
            approved["status"] == "approved"
            and approved["can_apply_main"] is False
            and approved["can_apply_sandbox"] is True,
            f"status={approved['status']}",
        )
    )

    # 7 Sandbox apply
    applied = service.patches.apply_sandbox(proposed["patch_id"])
    sandbox_dir = Path((applied.get("sandbox") or {}).get("sandbox_dir") or "")
    main_note = (
        ROOT
        / "experiment_apps"
        / "rgbt_detection_real"
        / "adapters"
        / "mock_patch_note.md"
    )
    results.append(
        _check(
            "sandbox-apply-isolated",
            applied["status"] == "applied_sandbox"
            and (applied.get("sandbox") or {}).get("ok") is True
            and (applied.get("sandbox") or {}).get("main_workspace_modified") is False
            and sandbox_dir.is_dir()
            and not main_note.exists(),
            f"status={applied['status']} sandbox={sandbox_dir} main_exists={main_note.exists()}",
        )
    )

    # 8 Sandbox tests (mock_experiment, no shell)
    tested = service.patches.test_sandbox(
        proposed["patch_id"], profile="mock_experiment"
    )
    report = tested.get("sandbox_tests") or {}
    results.append(
        _check(
            "sandbox-test-mock-experiment",
            report.get("ok") is True
            and report.get("main_workspace_modified") is False
            and any(
                c.get("name") == "mock_experiment_no_shell" for c in report.get("checks", [])
            ),
            f"ok={report.get('ok')} profile={report.get('profile')}",
        )
    )

    # 9 Record evidence
    recorded = service.patches.record_evidence(
        proposed["patch_id"], require_tests=True
    )
    evidence = recorded.get("patch_evidence") or {}
    artifact = Path(evidence.get("artifact_path") or "")
    results.append(
        _check(
            "patch-evidence-recorded",
            recorded["status"] == "evidence_recorded"
            and evidence.get("main_workspace_modified") is False
            and artifact.is_file(),
            f"id={evidence.get('evidence_id')} path={artifact}",
        )
    )

    # 10 Merge decision intent only
    merged = service.patches.decide_merge(
        proposed["patch_id"],
        decision="merge",
        reason="accept_v16 merge intent",
    )
    results.append(
        _check(
            "merge-intent-no-main-write",
            merged["status"] == "merged"
            and merged["can_apply_main"] is False
            and merged["applied_main"] is False
            and not main_note.exists(),
            f"status={merged['status']} warning={merged.get('warning')}",
        )
    )

    # 11 Reject path works
    other = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/adapters/accept_other.md"
        ),
    )
    rejected = service.patches.reject(other["patch_id"], reason="not needed")
    results.append(
        _check(
            "human-reject",
            rejected["status"] == "rejected",
            f"status={rejected['status']}",
        )
    )

    # 12 Discard path after evidence
    discard_p = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/configs/accept_discard.yaml"
        ),
    )
    service.patches.approve(discard_p["patch_id"])
    service.patches.apply_sandbox(discard_p["patch_id"])
    service.patches.record_evidence(discard_p["patch_id"])
    discarded = service.patches.decide_merge(
        discard_p["patch_id"], decision="discard", reason="noise"
    )
    results.append(
        _check(
            "discard-after-evidence",
            discarded["status"] == "discarded"
            and discarded["applied_main"] is False,
            f"status={discarded['status']}",
        )
    )

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    report_doc = {
        "version": "v1.6.0",
        "title": "Restricted source patching + sandbox validation",
        "passed": passed,
        "total": total,
        "all_ok": passed == total,
        "checks": results,
        "notes": [
            "Default provider remains mock; no network required.",
            "Main workspace is never modified by patch apply/merge.",
            "v1.6.7 real provider generation is optional/deferred.",
        ],
    }
    out_dir = ROOT / "docs" / "acceptance" / "v1.6"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "v16_acceptance_report.json"
    report_path.write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_doc, ensure_ascii=False, indent=2))
    print(f"\nWrote {report_path}")
    print(f"Acceptance: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
