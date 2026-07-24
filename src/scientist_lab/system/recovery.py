"""Restart recovery for in-flight workbench resources (v2.0.6).

Never auto-reruns expensive experiments. Only marks interrupted state
or resumes remote status polling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.domain import JobStatus
from scientist_lab.domain.models import utc_now_iso


ACTIVE_EXECUTION = {
    JobStatus.QUEUED,
    JobStatus.PREPARING,
    JobStatus.RUNNING,
    JobStatus.COLLECTING,
    "queued",
    "preparing",
    "running",
    "collecting",
    "pending",
    "submitted",
}


class RecoveryService:
    def __init__(self, service: Any) -> None:
        self.service = service

    def recover(self, *, dry_run: bool = True) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        actions.extend(self._recover_executions(dry_run=dry_run))
        actions.extend(self._recover_iterations(dry_run=dry_run))
        actions.extend(self._recover_trees(dry_run=dry_run))
        actions.extend(self._recover_reports(dry_run=dry_run))
        actions.extend(self._recover_patches(dry_run=dry_run))

        applied = [a for a in actions if a.get("applied")]
        planned = [a for a in actions if a.get("action") != "noop"]
        return {
            "dry_run": dry_run,
            "action_count": len(planned),
            "applied_count": len(applied),
            "actions": actions,
            "policy": {
                "auto_rerun_experiments": False,
                "mark_interrupted_when_no_process": True,
                "resume_remote_polling": True,
            },
            "message": (
                "dry-run only; pass dry_run=false to apply"
                if dry_run
                else "recovery actions applied (no expensive re-runs)"
            ),
        }

    def _recover_executions(self, *, dry_run: bool) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        attempts = self.service.list_executions(limit=200)
        runtime = Path(self.service.settings.runtime_dir)
        remote_bindings = runtime / "remote_jobs"

        for attempt in attempts:
            status = str(attempt.status)
            if status not in {str(s) for s in ACTIVE_EXECUTION} and status not in ACTIVE_EXECUTION:
                continue
            execution_id = attempt.execution_id
            binding_path = remote_bindings / f"{execution_id}.json"
            is_remote = binding_path.is_file()

            if is_remote:
                remote_alive = False
                remote_error = None
                try:
                    refreshed = self.service.refresh_execution(execution_id)
                    remote_status = str(getattr(refreshed, "status", status))
                    remote_alive = remote_status in {
                        str(s) for s in ACTIVE_EXECUTION
                    } or remote_status in ACTIVE_EXECUTION
                    actions.append(
                        {
                            "resource_type": "execution",
                            "resource_id": execution_id,
                            "action": "resume_remote_polling",
                            "from_status": status,
                            "to_status": remote_status,
                            "reason": "remote binding found; refreshed remote status",
                            "applied": not dry_run,
                            "dry_run": dry_run,
                        }
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    remote_error = str(exc)

                actions.append(
                    {
                        "resource_type": "execution",
                        "resource_id": execution_id,
                        "action": "mark_interrupted",
                        "from_status": status,
                        "to_status": "interrupted",
                        "reason": remote_error
                        or "remote binding present but worker unreachable",
                        "applied": False if dry_run else self._mark_execution_interrupted(
                            attempt
                        ),
                        "dry_run": dry_run,
                    }
                )
                continue

            # Local: no in-process runner map entry after restart → interrupted
            tracked = execution_id in getattr(self.service, "_execution_runners", {})
            if tracked:
                actions.append(
                    {
                        "resource_type": "execution",
                        "resource_id": execution_id,
                        "action": "noop",
                        "from_status": status,
                        "to_status": status,
                        "reason": "local runner still tracks this execution",
                        "applied": False,
                        "dry_run": dry_run,
                    }
                )
                continue

            actions.append(
                {
                    "resource_type": "execution",
                    "resource_id": execution_id,
                    "action": "mark_interrupted",
                    "from_status": status,
                    "to_status": "interrupted",
                    "reason": "running/queued but no active local process after restart",
                    "applied": False
                    if dry_run
                    else self._mark_execution_interrupted(attempt),
                    "dry_run": dry_run,
                }
            )
        return actions

    def _mark_execution_interrupted(self, attempt: Any) -> bool:
        updated = attempt.model_copy(
            update={
                "status": JobStatus.INTERRUPTED,
                "completed_at": utc_now_iso(),
                "error_json": {
                    "error_type": "interrupted",
                    "stage": "recovery",
                    "message": "Marked interrupted after process restart; not auto-rerun.",
                    "retryable": True,
                    "suggested_action": "Inspect logs; re-submit only if needed.",
                },
            }
        )
        self.service.repo.upsert_attempt(updated)
        return True

    def _recover_iterations(self, *, dry_run: bool) -> list[dict[str, Any]]:
        from scientist_lab.iteration.service import IterationService

        actions: list[dict[str, Any]] = []
        iteration = IterationService(self.service)
        for item in iteration.list_iterations(limit=100):
            status = str(item.get("status") or "")
            if status not in {"running", "approved", "comparing"}:
                continue
            actions.append(
                {
                    "resource_type": "iteration",
                    "resource_id": item.get("iteration_id"),
                    "action": "advise_advance",
                    "from_status": status,
                    "to_status": status,
                    "reason": (
                        "iteration still in flight; call iterate-advance / "
                        "tree-advance after recovery (no auto re-run)"
                    ),
                    "applied": False,
                    "dry_run": dry_run,
                    "next_action": f"advance iteration {item.get('iteration_id')}",
                }
            )
        return actions

    def _recover_trees(self, *, dry_run: bool) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for tree in self.service.list_trees(limit=100):
            status = str(tree.get("status") or "")
            if status not in {"waiting_approval", "active", "running"}:
                continue
            actions.append(
                {
                    "resource_type": "tree",
                    "resource_id": tree.get("tree_id"),
                    "action": "advise_resume",
                    "from_status": status,
                    "to_status": status,
                    "reason": "tree left mid-workflow; resume via plan-next / advance / approve",
                    "applied": False,
                    "dry_run": dry_run,
                    "href": f"/trees/{tree.get('tree_id')}",
                }
            )
        return actions

    def _recover_reports(self, *, dry_run: bool) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for report in self.service.list_reports(limit=100):
            status = str(report.get("status") or "")
            report_id = str(report.get("report_id") or "")
            md = report.get("markdown_path")
            incomplete = status in {"context_built", "tables_ready", "draft"} and (
                not md or not Path(str(md)).is_file()
            )
            if not incomplete:
                continue
            applied = False
            if not dry_run:
                applied = self._mark_report_interrupted(report_id)
            actions.append(
                {
                    "resource_type": "report",
                    "resource_id": report_id,
                    "action": "mark_build_interrupted",
                    "from_status": status,
                    "to_status": "build_interrupted",
                    "reason": "report build incomplete after restart",
                    "applied": applied,
                    "dry_run": dry_run,
                    "suggested_action": "Re-run report build from Web 报告中心",
                }
            )
        return actions

    def _mark_report_interrupted(self, report_id: str) -> bool:
        try:
            report = self.service.reporting._repo.require_report(report_id)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return False
        updated = report.model_copy(update={"status": "build_interrupted"})
        self.service.reporting._repo.save_report(updated)  # noqa: SLF001
        return True

    def _recover_patches(self, *, dry_run: bool) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for patch in self.service.list_patches(limit=100):
            status = str(patch.get("status") or "")
            meta = dict(patch.get("metadata") or patch.get("meta") or {})
            testing = bool(meta.get("sandbox_test_running"))
            if status != "applied_sandbox" or not testing:
                continue
            applied = False
            if not dry_run:
                applied = self._mark_patch_test_interrupted(str(patch.get("patch_id")))
            actions.append(
                {
                    "resource_type": "patch",
                    "resource_id": patch.get("patch_id"),
                    "action": "mark_sandbox_test_interrupted",
                    "from_status": status,
                    "to_status": status,
                    "reason": "sandbox test was in progress during restart",
                    "applied": applied,
                    "dry_run": dry_run,
                    "suggested_action": "Re-run sandbox test from补丁详情页",
                }
            )
        return actions

    def _mark_patch_test_interrupted(self, patch_id: str) -> bool:
        try:
            proposal = self.service.patches._repo.require(patch_id)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return False
        meta = dict(proposal.metadata or {})
        meta["sandbox_test_running"] = False
        meta["sandbox_test_interrupted"] = True
        meta["sandbox_test_interrupted_at"] = utc_now_iso()
        updated = proposal.model_copy(update={"metadata": meta})
        self.service.patches._repo.upsert(updated)  # noqa: SLF001
        return True
