"""v1.9 acceptance: controlled merge, rollback (revert), and local Release Candidate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.api.app import create_app
from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.release.models import MergeCandidate
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _init_mini_repo(repo: Path) -> None:
    adapters = repo / "experiment_apps" / "rgbt_detection_real" / "adapters"
    adapters.mkdir(parents=True)
    (adapters / ".gitkeep").write_text("", encoding="utf-8")
    # Minimal files so PathPolicy / sandbox paths resolve under this root.
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "accept-v19-mini"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "accept-v19@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Accept V19"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init accept-v19 mini"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _service_for(repo: Path, accept_root: Path, name: str) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=repo,
            db_path=accept_root / f"{name}.db",
            runtime_dir=accept_root / f"{name}_runtime",
            outputs_dir=accept_root / f"{name}_outputs",
            experiment_app_dir=repo,
        ).resolve()
    )


def _evidenced_merge_ready(
    service: ExperimentService, *, project_id: str, note_name: str
) -> str:
    proposed = service.patches.propose_mock(
        project_id,
        unified_diff=build_mock_unified_diff(
            relative_path=(
                f"experiment_apps/rgbt_detection_real/adapters/{note_name}"
            )
        ),
    )
    patch_id = proposed["patch_id"]
    service.patches.approve(patch_id, reason="accept v19")
    service.patches.apply_sandbox(patch_id)
    service.patches.test_sandbox(patch_id, profile="smoke")
    service.patches.record_evidence(patch_id)
    service.patches.decide_merge(patch_id, decision="merge", reason="intent")
    return patch_id


def _inject_passing_test(service: ExperimentService, mc_id: str) -> None:
    """Mini repos lack full tree for syntax profile; inject last_test like unit tests."""
    candidate = service.merges._repo.require(mc_id)  # noqa: SLF001
    candidate.status = "waiting_approval"
    candidate.metadata = {
        **dict(candidate.metadata or {}),
        "last_test": {"ok": True, "profile_id": "syntax"},
        "workspace_applied": True,
    }
    service.merges._repo.upsert(candidate)  # noqa: SLF001


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v19"
    if accept_root.exists():
        shutil.rmtree(accept_root, ignore_errors=True)
    accept_root.mkdir(parents=True, exist_ok=True)

    # Root-backed service for API/OpenAPI/health (does not mutate merge on ROOT).
    root_db = accept_root / "root_api.db"
    if root_db.exists():
        root_db.unlink()
    root_service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=root_db,
            runtime_dir=accept_root / "root_runtime",
            outputs_dir=accept_root / "root_outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    client = TestClient(create_app(service=root_service))

    results: list[dict] = []

    # 1 health reports v1.9
    health = client.get("/api/v1/health")
    results.append(
        _check(
            "api-health-v19",
            health.status_code == 200
            and str(health.json().get("version", "")).startswith("v1.9"),
            f"version={health.json().get('version')}",
        )
    )

    # 2–4 GitAdapter security boundary
    git = GitAdapter(ROOT)
    for name, args in (
        ("git-adapter-denies-reset-hard", ["reset", "--hard", "HEAD"]),
        ("git-adapter-denies-push", ["push", "origin", "HEAD"]),
        ("git-adapter-denies-force-flag", ["merge", "-f", "HEAD"]),
    ):
        denied = False
        try:
            git._run(args)  # noqa: SLF001
        except GitAdapterError:
            denied = True
        results.append(_check(name, denied, f"args={args}"))

    # 5 OpenAPI paths for merges + RC
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths") or {}
    needed = [
        "/api/v1/merges",
        "/api/v1/merges/prepare",
        "/api/v1/merges/{merge_candidate_id}/apply",
        "/api/v1/merges/{merge_candidate_id}/test",
        "/api/v1/merges/{merge_candidate_id}/approve",
        "/api/v1/merges/{merge_candidate_id}/commit",
        "/api/v1/merges/{merge_candidate_id}/finalize",
        "/api/v1/merges/{merge_candidate_id}/rollback",
        "/api/v1/release-candidates",
        "/api/v1/release-candidates/{release_candidate_id}/verify",
    ]
    missing = [p for p in needed if p not in paths]
    results.append(_check("openapi-merge-rc-paths", not missing, f"missing={missing}"))

    # 6 Frontend pages exist
    web_pages = ROOT / "web" / "src" / "pages" / "MergeReleasePages.tsx"
    results.append(
        _check(
            "frontend-merge-rollback-rc-pages",
            web_pages.is_file()
            and "MergesPage" in web_pages.read_text(encoding="utf-8")
            and "RollbacksPage" in web_pages.read_text(encoding="utf-8")
            and "ReleaseCandidatesPage" in web_pages.read_text(encoding="utf-8"),
            str(web_pages),
        )
    )

    # 7 Merge profiles registry
    profiles = client.get("/api/v1/merges/profiles")
    profile_ids = {
        str(item.get("profile_id"))
        for item in (profiles.json().get("items") or [])
    }
    expected_profiles = {"syntax", "unit", "smoke", "full_regression", "acceptance"}
    results.append(
        _check(
            "merge-profiles-registry",
            profiles.status_code == 200 and expected_profiles <= profile_ids,
            f"ids={sorted(profile_ids)}",
        )
    )

    # 8 Prepare without evidence must fail (API)
    bad_prep = client.post(
        "/api/v1/merges/prepare",
        json={"patch_id": "patch_does_not_exist"},
    )
    results.append(
        _check(
            "prepare-rejects-missing-patch",
            bad_prep.status_code in {404, 409},
            f"status={bad_prep.status_code}",
        )
    )

    # Disposable mini repos for destructive merge/rollback assertions
    mini = accept_root / "mini_repo"
    _init_mini_repo(mini)
    svc = _service_for(mini, accept_root, "chain")

    try:
        # 9 Evidence gate via service
        proposed = svc.patches.propose_mock("project_accept_v19")
        refused = False
        try:
            svc.merge_prepare(proposed["patch_id"])
        except ValueError:
            refused = True
        results.append(
            _check("prepare-requires-evidence", refused, "ValueError expected")
        )

        # 10–13 Main happy path: prepare → apply → approve → commit (HEAD stable) → finalize
        patch_id = _evidenced_merge_ready(
            svc, project_id="project_accept_v19", note_name="accept_v19_note.md"
        )
        prep = svc.merge_prepare(patch_id, target_branch="main")
        mc_id = prep["merge_candidate_id"]
        adapter = GitAdapter(mini)
        results.append(
            _check(
                "merge-prepare-worktree",
                prep.get("status") in {"created", "merge_conflict"}
                and Path(str(prep.get("workspace_path") or "")).is_dir()
                and prep.get("main_workspace_modified") is False,
                f"status={prep.get('status')} mc={mc_id}",
            )
        )

        applied = svc.merge_apply(mc_id)
        results.append(
            _check(
                "merge-apply-in-worktree",
                applied.get("workspace_applied") is True
                and applied.get("can_commit") is False,
                f"status={applied.get('status')}",
            )
        )

        _inject_passing_test(svc, mc_id)
        approved = svc.merge_approve(mc_id, reason="accept v19 approve")
        head_before_commit = adapter.rev_parse("HEAD")
        committed = svc.merge_commit(mc_id)
        head_after_commit = adapter.rev_parse("HEAD")
        results.append(
            _check(
                "merge-approve-commit-keeps-main-head",
                approved.get("status") == "approved"
                and bool(committed.get("commit_sha"))
                and head_before_commit == head_after_commit
                and committed.get("can_finalize") is True,
                f"head={head_after_commit[:12]} commit={str(committed.get('commit_sha'))[:12]}",
            )
        )

        finalized = svc.merge_finalize(mc_id, post_merge_profile=None)
        results.append(
            _check(
                "merge-finalize-merged",
                finalized.get("status") == "merged"
                and bool(finalized.get("merge_commit_sha"))
                and adapter.rev_parse("HEAD") == finalized.get("merge_commit_sha"),
                f"status={finalized.get('status')}",
            )
        )

        # 14 Manual rollback via revert
        rolled = svc.merge_rollback(mc_id, reason="accept v19 rollback")
        results.append(
            _check(
                "merge-rollback-revert",
                rolled.get("status") == "rolled_back"
                and bool((rolled.get("rollback") or {}).get("rollback_commit_sha"))
                and adapter.rev_parse("HEAD")
                == (rolled.get("rollback") or {}).get("rollback_commit_sha"),
                f"status={rolled.get('status')}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        for name in (
            "prepare-requires-evidence",
            "merge-prepare-worktree",
            "merge-apply-in-worktree",
            "merge-approve-commit-keeps-main-head",
            "merge-finalize-merged",
            "merge-rollback-revert",
        ):
            if not any(r["name"] == name for r in results):
                results.append(_check(name, False, str(exc)))

    # 15 Auto-rollback when post-merge check fails
    mini2 = accept_root / "mini_repo_auto_rb"
    try:
        _init_mini_repo(mini2)
        svc2 = _service_for(mini2, accept_root, "auto_rb")
        patch2 = _evidenced_merge_ready(
            svc2, project_id="project_accept_auto", note_name="accept_auto.md"
        )
        mc2 = svc2.merge_prepare(patch2, target_branch="main")["merge_candidate_id"]
        svc2.merge_apply(mc2)
        _inject_passing_test(svc2, mc2)
        svc2.merge_approve(mc2, reason="auto rb")
        svc2.merge_commit(mc2)
        auto = svc2.merge_finalize(mc2, post_merge_profile="unit")
        results.append(
            _check(
                "finalize-auto-rollback-on-failure",
                auto.get("status") == "rolled_back"
                and auto.get("auto_rolled_back") is True
                and (auto.get("rollback") or {}).get("trigger") == "post_merge_failure",
                f"status={auto.get('status')} auto={auto.get('auto_rolled_back')}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("finalize-auto-rollback-on-failure", False, str(exc)))

    # 16–17 Release Candidate local-only create + verify
    try:
        rc_svc = _service_for(mini, accept_root, "rc")
        # Re-run a short finalize path for a merged MC to attach to RC
        patch_rc = _evidenced_merge_ready(
            rc_svc, project_id="project_rc", note_name="accept_rc.md"
        )
        mc_rc = rc_svc.merge_prepare(patch_rc, target_branch="main")[
            "merge_candidate_id"
        ]
        rc_svc.merge_apply(mc_rc)
        _inject_passing_test(rc_svc, mc_rc)
        rc_svc.merge_approve(mc_rc, reason="rc")
        rc_svc.merge_commit(mc_rc)
        fin_rc = rc_svc.merge_finalize(mc_rc, post_merge_profile=None)
        created_rc = rc_svc.create_release_candidate(
            version="1.9.0-rc1",
            project_id="project_rc",
            base_tag="v1.8.0",
            merge_candidate_ids=[mc_rc],
            notes="accept v19",
        )
        verified_rc = rc_svc.verify_release_candidate(
            created_rc["release_candidate_id"]
        )
        results.append(
            _check(
                "release-candidate-create-verify",
                fin_rc.get("status") == "merged"
                and created_rc.get("can_publish_remote") is False
                and created_rc.get("manifest", {}).get("remote_published") is False
                and Path(str(created_rc.get("manifest_path") or "")).is_file()
                and (verified_rc.get("verification") or {}).get("valid") is True
                and verified_rc.get("status") == "verified",
                f"rc={created_rc.get('release_candidate_id')} "
                f"valid={(verified_rc.get('verification') or {}).get('valid')}",
            )
        )

        # Non-merged MC must be refused
        bad_mc = MergeCandidate(
            project_id="project_rc",
            patch_id="patch_fake",
            patch_evidence_id="pe_fake",
            source_commit="deadbeef",
            target_branch="main",
            workspace_path=str(accept_root / "fake_wt"),
            patch_sha256="a" * 64,
            diff_sha256="b" * 64,
            status="created",
        )
        rc_svc.merges._repo.upsert(bad_mc)  # noqa: SLF001
        refused_rc = False
        try:
            rc_svc.create_release_candidate(
                version="1.9.0-rc-bad",
                merge_candidate_ids=[bad_mc.merge_candidate_id],
            )
        except ValueError:
            refused_rc = True
        results.append(
            _check(
                "rc-rejects-non-merged-candidate",
                refused_rc,
                "ValueError expected for non-merged MC",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("release-candidate-create-verify", False, str(exc)))
        results.append(_check("rc-rejects-non-merged-candidate", False, str(exc)))

    # 18 Unknown test profile rejected
    try:
        svc3 = _service_for(mini, accept_root, "profile")
        patch3 = _evidenced_merge_ready(
            svc3, project_id="project_prof", note_name="accept_prof.md"
        )
        mc3 = svc3.merge_prepare(patch3, target_branch="main")["merge_candidate_id"]
        svc3.merge_apply(mc3)
        unknown_ok = False
        try:
            svc3.merge_test(mc3, profile_id="not_a_real_profile")
        except ValueError:
            unknown_ok = True
        results.append(
            _check("merge-test-rejects-unknown-profile", unknown_ok, "ValueError")
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("merge-test-rejects-unknown-profile", False, str(exc)))

    # 19 API RC list shape on root client
    listed_rc = client.get("/api/v1/release-candidates", params={"limit": 10})
    results.append(
        _check(
            "api-release-candidates-list",
            listed_rc.status_code == 200 and "items" in listed_rc.json(),
            f"status={listed_rc.status_code}",
        )
    )

    # 20 CLI surface mentions release-candidate + merge-finalize
    cli_src = (ROOT / "src" / "scientist_lab" / "cli.py").read_text(encoding="utf-8")
    results.append(
        _check(
            "cli-merge-and-rc-commands",
            "merge-finalize" in cli_src
            and "merge-rollback" in cli_src
            and "release-candidate-create" in cli_src
            and "release-candidate-verify" in cli_src,
            "cli.py command strings",
        )
    )

    out_dir = ROOT / "docs" / "acceptance" / "v1.9"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v1.9.0",
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "results": results,
    }
    report_path = out_dir / "v19_acceptance_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version": "v1.9.0",
                "tag_target": "v1.9.0",
                "branch": "feat/v1.9-controlled-merge-rollback",
                "baseline_tag": "v1.8.0",
                "title": "Scientist Lab Controlled Merge, Rollback & RC",
                "principles": [
                    "Whitelist GitAdapter only; no shell=True",
                    "No push / reset --hard / force / rebase",
                    "Commit only in isolated worktree; finalize --no-ff",
                    "Rollback via git revert -m 1 only",
                    "Release Candidate is local manifest only",
                ],
                "completed_subversions": [
                    "v1.9.1",
                    "v1.9.2",
                    "v1.9.3",
                    "v1.9.4",
                    "v1.9.5",
                    "v1.9.6",
                    "v1.9.7",
                    "v1.9.8",
                    "v1.9.9",
                    "v1.9.10",
                ],
                "acceptance_script": "scripts/accept_v19.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"FAILED {len(failed)}/{len(results)}", file=sys.stderr)
        for item in failed:
            print(f"  - {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    print(f"PASSED {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
