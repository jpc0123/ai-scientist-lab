from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from scientist_lab.ablations.service import AblationService
from scientist_lab.agents.service import AgentPlanningService
from scientist_lab.budget.service import BudgetService
from scientist_lab.checkpoints.registry import CheckpointRegistry
from scientist_lab.search.service import TreeSearchService
from scientist_lab.reporting.service import ReportingService
from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentNode,
    ResearchProject,
    utc_now_iso,
)
from scientist_lab.domain.results import ExecutionResult
from scientist_lab.evidence.service import EvidenceService
from scientist_lab.protocols.service import ProtocolService
from scientist_lab.protocols.verifier import ProtocolViolationError
from scientist_lab.runners.local_docker import LocalDockerRunner
from scientist_lab.runners.profile_registry import RunnerProfileRegistry
from scientist_lab.runners.remote_docker_runner import RemoteDockerRunner
from scientist_lab.settings import Settings, get_settings
from scientist_lab.storage.database import init_db
from scientist_lab.storage.repositories import Repository
from scientist_lab.datasets.registry import DatasetRegistry, parse_dataset_reference
from scientist_lab.projects.service import ProjectService


class ExperimentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_factory = init_db(str(self.settings.db_path))
        self.repo = Repository(self.session_factory)
        self.projects = ProjectService(self.repo)
        self.datasets = DatasetRegistry(
            self.session_factory,
            project_root=Path(self.settings.project_root),
        )
        self.runner_profiles = RunnerProfileRegistry(self.session_factory)
        self.runner_profiles.ensure_defaults()
        self.checkpoints = CheckpointRegistry(
            self.session_factory,
            outputs_root=self.settings.outputs_dir,
        )
        self.protocols = ProtocolService(self.session_factory)
        self.ablations = AblationService(self.session_factory)
        self.evidence = EvidenceService(
            self.session_factory,
            outputs_root=self.settings.outputs_dir,
        )
        self.budget = BudgetService(self.session_factory)
        self.agents = AgentPlanningService(
            self.session_factory,
            outputs_root=self.settings.outputs_dir,
        )
        self.trees = TreeSearchService(
            self.session_factory,
            get_project=self.repo.get_project,
            get_node=self.repo.get_node,
            get_protocol=self.protocols.get,
            get_node_aggregate=self._tree_node_aggregate,
            list_evidence=self._tree_list_evidence,
            get_claim_matrix=self._tree_claim_matrix,
            get_remaining_budget=self._tree_remaining_budget,
        )
        self.reporting = ReportingService(self)
        from scientist_lab.llm_eval.repository import LLMEvalRepository
        from scientist_lab.patching.service import PatchingService

        self.llm_evals = LLMEvalRepository(self.session_factory)
        self.patches = PatchingService(
            self.session_factory,
            project_root=Path(self.settings.project_root),
            sandbox_root=Path(self.settings.outputs_dir) / "_patch_sandboxes",
            outputs_root=Path(self.settings.outputs_dir),
        )
        from scientist_lab.release.service import ReleaseService

        self.releases = ReleaseService(
            self.session_factory,
            project_root=Path(self.settings.project_root),
            outputs_root=Path(self.settings.outputs_dir),
            sandbox_root=Path(self.settings.outputs_dir) / "_patch_sandboxes",
        )
        from scientist_lab.release.merge_service import MergeService

        self.merges = MergeService(
            self.session_factory,
            project_root=Path(self.settings.project_root),
        )
        from scientist_lab.release.release_candidate import ReleaseCandidateService

        self.release_candidates = ReleaseCandidateService(
            self.session_factory,
            project_root=Path(self.settings.project_root),
            outputs_root=Path(self.settings.outputs_dir),
        )
        self._code_roots = {
            "local:experiment_app": Path(self.settings.experiment_app_dir),
            "local:rgbt_detector": Path(self.settings.rgbt_detector_dir),
            "local:rgbt_detection_real": Path(self.settings.rgbt_detection_real_dir),
            "image:rgbt-detection-v2": Path(self.settings.rgbt_detection_real_dir),
        }
        self._local_runner: LocalDockerRunner | None = None
        self._remote_runners: dict[str, RemoteDockerRunner] = {}
        self._execution_runners: dict[str, Any] = {}
        self._watchers: dict[str, threading.Thread] = {}

    @property
    def local_runner(self) -> LocalDockerRunner:
        if self._local_runner is None:
            self._local_runner = LocalDockerRunner(
                experiment_app_dir=self.settings.experiment_app_dir,
                runtime_root=self.settings.runtime_dir,
                outputs_root=self.settings.outputs_dir,
                image_registry=self.settings.image_registry,
                poll_interval_seconds=self.settings.poll_interval_seconds,
                code_roots=self._code_roots,
                dataset_resolver=self._resolve_dataset_reference,
            )
        return self._local_runner

    @property
    def runner(self):
        # Backward-compatible: last used or local.
        if self._execution_runners:
            return next(reversed(self._execution_runners.values()))
        return self.local_runner

    @runner.setter
    def runner(self, value) -> None:
        # Allow tests/scripts to assign; prefer storing as local override.
        if isinstance(value, LocalDockerRunner):
            self._local_runner = value

    def _resolve_dataset_reference(self, dataset_reference: str):
        key = parse_dataset_reference(dataset_reference)
        if key is None:
            return None
        return self.datasets.require(key)

    def _select_runner(self, contract: ExperimentContract):
        profile_key = (contract.runner_profile or "local").strip() or "local"
        if profile_key in {"local", "local_docker"}:
            return self.local_runner
        profile = self.runner_profiles.require(profile_key)
        if profile.runner_type != "remote_docker":
            raise ValueError(
                f"unsupported runner_type for profile {profile_key}: "
                f"{profile.runner_type}"
            )
        if profile_key not in self._remote_runners:
            self._remote_runners[profile_key] = RemoteDockerRunner(
                profile,
                outputs_root=self.settings.outputs_dir,
                runtime_root=self.settings.runtime_dir,
                poll_interval_seconds=self.settings.poll_interval_seconds,
            )
        return self._remote_runners[profile_key]

    def _runner_for_execution(self, execution_id: str):
        runner = self._execution_runners.get(execution_id)
        if runner is not None:
            return runner
        # Fallback: local in-memory only, or remote binding on disk.
        binding = (
            Path(self.settings.runtime_dir) / "remote_jobs" / f"{execution_id}.json"
        )
        if binding.exists():
            data = json.loads(binding.read_text(encoding="utf-8"))
            profile_key = data.get("runner_profile")
            if profile_key:
                profile = self.runner_profiles.require(profile_key, require_enabled=False)
                if profile_key not in self._remote_runners:
                    self._remote_runners[profile_key] = RemoteDockerRunner(
                        profile,
                        outputs_root=self.settings.outputs_dir,
                        runtime_root=self.settings.runtime_dir,
                        poll_interval_seconds=self.settings.poll_interval_seconds,
                    )
                return self._remote_runners[profile_key]
        return self.local_runner

    def _image_reference(self, contract: ExperimentContract) -> str:
        if contract.environment_key in self.settings.image_registry:
            return self.settings.image_registry[contract.environment_key]
        return f"remote:{contract.environment_key}"

    def submit_contract(self, contract: ExperimentContract) -> ExecutionResult:
        """异步提交：立即返回 execution_id，后台落库最终结果。"""
        return self.run_contract(contract, wait=False)

    def run_contract(
        self,
        contract: ExperimentContract,
        wait: bool = True,
    ) -> ExecutionResult:
        # Formal protocol gate (v0.9.1): blocking violations refuse execution.
        self.protocols.enforce_contract(contract)

        now = utc_now_iso()

        project = self.repo.get_project(contract.project_id)
        if project is None:
            project = ResearchProject(
                project_id=contract.project_id,
                title=contract.title,
                research_goal=contract.research_goal,
                status=ProjectStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
            self.repo.upsert_project(project)
        else:
            project.title = contract.title or project.title
            project.research_goal = contract.research_goal or project.research_goal
            project.updated_at = now
            self.repo.upsert_project(project)

        node = self.repo.get_node(contract.node_id)
        if node is None:
            node = ExperimentNode(
                node_id=contract.node_id,
                project_id=contract.project_id,
                parent_node_id=contract.parent_node_id,
                node_type=NodeType.SMOKE,
                stage=NodeStage.EXECUTING,
                hypothesis=contract.hypothesis,
                status=NodeStatus.RUNNING,
                depth=0,
                contract_json=contract.model_dump(),
                created_at=now,
                updated_at=now,
            )
        else:
            node.status = NodeStatus.RUNNING
            node.stage = NodeStage.EXECUTING
            node.contract_json = contract.model_dump()
            node.hypothesis = contract.hypothesis
            if contract.parent_node_id is not None:
                node.parent_node_id = contract.parent_node_id
            node.updated_at = now
        self.repo.upsert_node(node)

        image_name = self._image_reference(contract)
        attempt_index = self.repo.next_attempt_index(contract.node_id)

        runner = self._select_runner(contract)
        submission = runner.submit(contract)
        self._execution_runners[submission.execution_id] = runner
        meta = runner.get_job_meta(submission.execution_id)

        attempt = ExecutionAttempt(
            execution_id=submission.execution_id,
            node_id=contract.node_id,
            attempt_index=attempt_index,
            runner_profile=contract.runner_profile,
            status=JobStatus.QUEUED,
            container_id=None,
            image_reference=image_name,
            code_version=contract.code_reference,
            dataset_version=contract.dataset_reference,
            created_at=now,
            started_at=meta.get("started_at"),
            result_json={
                "output_directory": str(
                    self.settings.outputs_dir
                    / contract.project_id
                    / submission.execution_id
                ),
                "contract": contract.model_dump(),
            },
        )
        self.repo.upsert_attempt(attempt)

        if not wait:
            self._start_finalize_watcher(
                submission.execution_id,
                contract.resources.timeout_seconds + 30,
            )
            return ExecutionResult(
                execution_id=submission.execution_id,
                status=JobStatus.QUEUED,
                output_directory=str(
                    self.settings.outputs_dir
                    / contract.project_id
                    / submission.execution_id
                ),
            )

        runner.wait_until_done(
            submission.execution_id,
            timeout_seconds=contract.resources.timeout_seconds + 30,
        )
        return self._finalize_execution(submission.execution_id)

    def _start_finalize_watcher(
        self, execution_id: str, timeout_seconds: float
    ) -> None:
        if execution_id in self._watchers:
            return

        def _watch() -> None:
            try:
                runner = self._runner_for_execution(execution_id)
                runner.wait_until_done(
                    execution_id, timeout_seconds=timeout_seconds
                )
                self._finalize_execution(execution_id)
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._watchers.pop(execution_id, None)

        thread = threading.Thread(
            target=_watch,
            daemon=True,
            name=f"finalize-{execution_id}",
        )
        self._watchers[execution_id] = thread
        thread.start()

    def _finalize_execution(self, execution_id: str) -> ExecutionResult:
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")

        # Live runner may already be done; also tolerate rediscovered status.
        runner = self._runner_for_execution(execution_id)
        try:
            live = runner.get_status(execution_id)
            if live.status not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }:
                runner.wait_until_done(execution_id, timeout_seconds=5)
        except KeyError:
            # Process restarted; use DB record only.
            return ExecutionResult(
                execution_id=execution_id,
                status=attempt.status,
                metrics=(attempt.result_json or {}).get("metrics", {}),
                output_directory=(attempt.result_json or {}).get("output_directory"),
                error=None,
            )

        result = runner.collect_result(execution_id)
        meta = runner.get_job_meta(execution_id)

        attempt.status = result.status
        attempt.container_id = meta.get("container_id")
        attempt.completed_at = meta.get("completed_at")
        prev_contract = (attempt.result_json or {}).get("contract")
        compare_payload = self._maybe_compare_validate_data_reports(
            prev_contract, result.output_directory
        )
        attempt.result_json = {
            "metrics": result.metrics,
            "return_code": result.return_code,
            "output_directory": result.output_directory,
            "contract": prev_contract,
            "host_container_report_compare": compare_payload,
        }
        if result.error is not None:
            attempt.error_json = result.error.model_dump()
        self.repo.upsert_attempt(attempt)

        existing = {a.relative_path for a in self.repo.list_artifacts(execution_id)}
        for artifact in result.artifacts:
            if artifact.relative_path not in existing:
                self.repo.add_artifact(artifact)

        if result.status == JobStatus.COMPLETED and result.output_directory:
            self._maybe_register_checkpoints(attempt, prev_contract)

        node = self.repo.get_node(attempt.node_id)
        if node is not None:
            node.updated_at = utc_now_iso()
            if result.status == JobStatus.COMPLETED:
                node.status = NodeStatus.SUCCEEDED
                node.stage = NodeStage.DONE
            elif result.status == JobStatus.CANCELLED:
                node.status = NodeStatus.CANCELLED
                node.stage = NodeStage.ANALYZING
            else:
                node.status = NodeStatus.FAILED
                node.stage = NodeStage.ANALYZING
            self.repo.upsert_node(node)

        return result

    def _maybe_register_checkpoints(
        self,
        attempt: ExecutionAttempt,
        contract_payload: dict[str, Any] | None,
    ) -> None:
        project_id = None
        node_id = attempt.node_id
        baseline_key = None
        if contract_payload:
            project_id = contract_payload.get("project_id")
            baseline_key = (contract_payload.get("parameters") or {}).get("baseline")
        if not project_id:
            node = self.repo.get_node(attempt.node_id)
            if node is not None:
                project_id = node.project_id
        if not project_id:
            return
        try:
            self.checkpoints.register_from_execution(
                project_id=str(project_id),
                execution_id=attempt.execution_id,
                node_id=node_id,
                baseline_key=str(baseline_key) if baseline_key else None,
                preferred_only=True,
            )
        except Exception:  # noqa: BLE001
            # Checkpoint registration must not fail the execution finalize path.
            return

    def _maybe_compare_validate_data_reports(
        self,
        contract_payload: dict[str, Any] | None,
        output_directory: str | None,
    ) -> dict[str, Any] | None:
        """For validate_data runs, compare host vs container dataset_report.json."""
        if not contract_payload or not output_directory:
            return None
        if contract_payload.get("task_type") != "rgbt_detection":
            return None
        if contract_payload.get("execution_mode") != "validate_data":
            return None

        from scientist_lab.datasets.rgbt_validator import validate_rgbt_dataset
        from scientist_lab.storage.artifact_store import read_json, write_json
        from scientist_lab.tasks.rgbt_detection.report_compare import (
            compare_host_container_reports,
        )

        output_dir = Path(output_directory)
        container_report_path = output_dir / "dataset_report.json"
        if not container_report_path.exists():
            return {
                "consistent": False,
                "mismatches": ["container dataset_report.json missing"],
            }

        dataset_ref = str(contract_payload.get("dataset_reference") or "")
        key = parse_dataset_reference(dataset_ref)
        if key is None:
            return {
                "consistent": False,
                "mismatches": [f"invalid dataset_reference: {dataset_ref}"],
            }
        try:
            registration = self.datasets.require(key)
        except (KeyError, ValueError) as exc:
            return {
                "consistent": False,
                "mismatches": [f"dataset registry error: {exc}"],
            }

        host_report = validate_rgbt_dataset(
            Path(registration.host_path),
            dataset_key=registration.dataset_key,
            write_previews=False,
        )
        container_report = read_json(container_report_path)
        comparison = compare_host_container_reports(host_report, container_report)
        write_json(output_dir / "host_container_report_compare.json", comparison)

        host_copy = output_dir / "host_dataset_report.json"
        write_json(host_copy, host_report)
        return comparison
    def refresh_execution(self, execution_id: str) -> ExecutionAttempt:
        """同步一次运行中状态到 SQLite，完成后尝试 finalize。"""
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")

        try:
            live = self._runner_for_execution(execution_id).get_status(execution_id)
        except KeyError:
            return attempt

        attempt.status = live.status
        attempt.container_id = live.container_id or attempt.container_id
        self.repo.upsert_attempt(attempt)

        if live.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }:
            if not (attempt.result_json or {}).get("metrics") and live.status == JobStatus.COMPLETED:
                self._finalize_execution(execution_id)
                refreshed = self.repo.get_attempt(execution_id)
                return refreshed or attempt
            if live.status != JobStatus.COMPLETED and attempt.error_json is None:
                self._finalize_execution(execution_id)
                refreshed = self.repo.get_attempt(execution_id)
                return refreshed or attempt
        return attempt

    def list_projects(self) -> list[dict[str, Any]]:
        projects = self.repo.list_projects()
        items: list[dict[str, Any]] = []
        for project in projects:
            latest = self.repo.latest_attempt_for_project(project.project_id)
            view = self.projects.to_view(project)
            view.update(
                {
                    "node_count": self.repo.count_nodes(project.project_id),
                    "latest_execution": None
                    if latest is None
                    else {
                        "execution_id": latest.execution_id,
                        "status": str(latest.status),
                        "node_id": latest.node_id,
                        "created_at": latest.created_at,
                    },
                }
            )
            items.append(view)
        return items

    def get_project(self, project_id: str) -> dict[str, Any]:
        project = self.repo.get_project(project_id)
        if project is None:
            raise KeyError(f"project not found: {project_id}")
        payload = self.projects.to_view(project)
        payload["node_count"] = self.repo.count_nodes(project.project_id)
        try:
            payload["budget"] = self.show_budget(project_id)
        except Exception:  # noqa: BLE001
            payload["budget"] = None
        return payload

    def create_project(
        self,
        *,
        title: str,
        research_question: str = "",
        research_goal: str = "",
        description: str = "",
        task_type: str = "general_ml",
        dataset_keys: list[str] | None = None,
        protocol_ids: list[str] | None = None,
        runner_profile_keys: list[str] | None = None,
        default_llm_profile_id: str | None = None,
        expected_metrics: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        protocol_draft: dict[str, Any] | None = None,
        project_id: str | None = None,
        mark_ready: bool = True,
    ) -> dict[str, Any]:
        project = self.projects.create_from_wizard(
            title=title,
            research_question=research_question,
            research_goal=research_goal,
            description=description,
            task_type=task_type,
            dataset_keys=dataset_keys,
            protocol_ids=protocol_ids,
            runner_profile_keys=runner_profile_keys,
            default_llm_profile_id=default_llm_profile_id,
            expected_metrics=expected_metrics,
            constraints=constraints,
            protocol_draft=protocol_draft,
            project_id=project_id,
            mark_ready=mark_ready,
        )
        try:
            self.set_budget(project.project_id)
        except Exception:  # noqa: BLE001
            pass
        return self.get_project(project.project_id)

    def archive_project(self, project_id: str) -> dict[str, Any]:
        self.projects.archive(project_id)
        return self.get_project(project_id)

    def system_summary(self) -> dict[str, Any]:
        """Dashboard aggregate for the web console (v1.7.1)."""
        from scientist_lab.iteration.service import IterationService

        projects = self.list_projects()
        executions = self.list_executions(limit=100)
        running = [
            a
            for a in executions
            if str(a.status) in {"queued", "running", "pending", "submitted"}
        ]
        failed = [
            a
            for a in executions
            if str(a.status) in {"failed", "timed_out", "cancelled"}
        ][:10]

        plans = self.list_plans()
        pending_plan_candidates = 0
        for plan in plans:
            for cand in plan.get("candidates") or []:
                if str(cand.get("status") or "") in {
                    "proposed",
                    "ranked",
                    "pending_approval",
                    "waiting_approval",
                }:
                    pending_plan_candidates += 1

        iterations = IterationService(self).list_iterations(limit=100)
        pending_iterations = [
            item
            for item in iterations
            if item.get("status") in {"waiting_approval", "proposal_ready", "waiting_decision"}
        ]

        patches = self.patches.list_patches_all(limit=100)
        pending_patches = [
            item
            for item in patches
            if item.get("status") in {"verified", "proposed", "approved", "applied_sandbox", "evidence_recorded"}
            and item.get("status") != "merged"
        ]
        awaiting_patch_approval = [
            item for item in patches if item.get("status") in {"verified", "proposed"}
        ]

        trees = [
            self.trees.tree_status(t.tree_id)
            for t in self.trees._repo.list_trees()
        ]
        reports = self.reporting.list_reports(limit=10)

        return {
            "project_count": len(projects),
            "running_executions": len(running),
            "pending_plan_candidates": pending_plan_candidates,
            "pending_iterations": len(pending_iterations),
            "pending_patches": len(awaiting_patch_approval),
            "patch_actionable": len(pending_patches),
            "tree_count": len(trees),
            "recent_failures": [
                {
                    "execution_id": a.execution_id,
                    "status": str(a.status),
                    "node_id": a.node_id,
                    "project_id": a.project_id,
                    "created_at": a.created_at,
                    "error_type": getattr(a, "error_type", None),
                }
                for a in failed
            ],
            "recent_reports": reports,
            "budgets": [
                self.show_budget(p["project_id"])
                for p in projects[:20]
            ],
        }

    def list_trees(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        trees = self.trees._repo.list_trees(project_id=project_id)
        items = [self.trees.tree_status(t.tree_id) for t in trees]
        return items[: max(1, int(limit))]

    def list_iterations(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        from scientist_lab.iteration.service import IterationService

        return IterationService(self).list_iterations(
            project_id=project_id, limit=limit
        )

    def show_iteration(self, iteration_id: str) -> dict[str, Any]:
        from scientist_lab.iteration.service import IterationService

        return IterationService(self).get_status(iteration_id)

    def approve_iteration(self, iteration_id: str, **kwargs: Any) -> dict[str, Any]:
        from scientist_lab.iteration.service import IterationService

        return IterationService(self).approve_and_run(iteration_id, **kwargs)

    def advance_iteration(self, iteration_id: str) -> dict[str, Any]:
        from scientist_lab.iteration.service import IterationService

        return IterationService(self).advance_iteration(iteration_id)

    def finalize_iteration(
        self,
        iteration_id: str,
        *,
        selected_node_id: str,
        reason: str,
        decision_type: str = "efficiency_tradeoff",
        evidence_strength: str = "moderate",
    ) -> dict[str, Any]:
        from scientist_lab.iteration.service import IterationService

        return IterationService(self).finalize(
            iteration_id,
            selected_node_id=selected_node_id,
            decision_type=decision_type,
            reason=reason,
            evidence_strength=evidence_strength,
        )

    def list_reports(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self.reporting.list_reports(project_id=project_id, limit=limit)

    def list_audits(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self.reporting.list_audits(project_id=project_id, limit=limit)

    def show_audit(self, bundle_id: str) -> dict[str, Any]:
        return self.reporting.show_audit(bundle_id)

    def list_patches(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.patches.list_patches_all(project_id=project_id, limit=limit)

    def show_patch(self, patch_id: str) -> dict[str, Any]:
        return self.patches.show(patch_id)

    def list_claims(
        self, *, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        if project_id:
            matrix = self.show_claim_matrix(project_id)
            claims = list(matrix.get("claims") or [])
            for claim in claims:
                claim["project_id"] = project_id
            return claims
        items: list[dict[str, Any]] = []
        for project in self.list_projects():
            try:
                matrix = self.show_claim_matrix(project["project_id"])
            except Exception:  # noqa: BLE001
                continue
            for claim in matrix.get("claims") or []:
                claim = dict(claim)
                claim["project_id"] = project["project_id"]
                items.append(claim)
        return items

    def list_executions(
        self,
        limit: int = 20,
        project_id: str | None = None,
        node_id: str | None = None,
    ):
        return self.repo.list_attempts(
            limit=limit, project_id=project_id, node_id=node_id
        )

    def list_nodes(self, project_id: str | None = None) -> list[dict[str, Any]]:
        nodes = self.repo.list_nodes(project_id=project_id)
        items: list[dict[str, Any]] = []
        for node in nodes:
            attempts = self.repo.list_attempts(limit=5, node_id=node.node_id)
            items.append(
                {
                    "node": node.model_dump(),
                    "attempt_count": self.repo.next_attempt_index(node.node_id) - 1,
                    "recent_attempts": [
                        {
                            "execution_id": a.execution_id,
                            "status": str(a.status),
                            "attempt_index": a.attempt_index,
                            "created_at": a.created_at,
                        }
                        for a in attempts
                    ],
                }
            )
        return items

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        attempt = self.refresh_execution(execution_id)
        artifacts = self.repo.list_artifacts(execution_id)
        node = self.repo.get_node(attempt.node_id)
        can_cancel = False
        try:
            live = self._runner_for_execution(execution_id).get_status(execution_id)
            can_cancel = live.status not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }
        except KeyError:
            can_cancel = False
        return {
            "attempt": attempt.model_dump(),
            "artifacts": [a.model_dump() for a in artifacts],
            "node": None if node is None else node.model_dump(),
            "log_available": self.get_log_path(execution_id).exists(),
            "can_cancel": can_cancel,
            "status_label": str(attempt.status),
        }

    def show_execution(self, execution_id: str):
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")
        artifacts = self.repo.list_artifacts(execution_id)
        return attempt, artifacts

    def get_log_path(self, execution_id: str) -> Path:
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")
        output_dir = (attempt.result_json or {}).get("output_directory")
        if output_dir:
            return Path(output_dir) / "combined.log"
        node = self.repo.get_node(attempt.node_id)
        if node is None:
            raise KeyError(f"未找到 node: {attempt.node_id}")
        return (
            self.settings.outputs_dir
            / node.project_id
            / execution_id
            / "combined.log"
        )

    def get_log_text(self, execution_id: str, max_chars: int = 80_000) -> str:
        path = self.get_log_path(execution_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

    def cancel_execution(self, execution_id: str) -> ExecutionAttempt:
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")
        try:
            self._runner_for_execution(execution_id).cancel(execution_id)
        except KeyError as exc:
            raise RuntimeError(
                "无法取消：该执行不在当前进程的 Runner 内存中（可能服务已重启）。"
            ) from exc
        self._finalize_execution(execution_id)
        refreshed = self.repo.get_attempt(execution_id)
        return refreshed or attempt

    def run_seeds(
        self,
        contract: ExperimentContract,
        seeds: list[int],
        *,
        auto_aggregate: bool = True,
        wait: bool = True,
    ) -> dict[str, Any]:
        if not seeds:
            raise ValueError("seeds 不能为空")
        results = []
        for seed in seeds:
            seeded = contract.model_copy(deep=True)
            seeded.seed = int(seed)
            result = self.run_contract(seeded, wait=wait)
            results.append(
                {
                    "seed": int(seed),
                    "execution_id": result.execution_id,
                    "status": str(result.status),
                    "metrics": result.metrics,
                    "error": None
                    if result.error is None
                    else result.error.model_dump(),
                }
            )
        payload: dict[str, Any] = {
            "node_id": contract.node_id,
            "project_id": contract.project_id,
            "seeds": [int(s) for s in seeds],
            "results": results,
            "wait": wait,
            "runner_profile": contract.runner_profile,
        }
        if wait and auto_aggregate:
            try:
                payload["aggregate"] = self.aggregate_node(contract.node_id)
            except KeyError as exc:
                payload["aggregate_error"] = str(exc)
        return payload

    def aggregate_node(self, node_id: str) -> dict[str, Any]:
        from scientist_lab.services.aggregation import NodeAggregationService

        service = NodeAggregationService(self.repo, self.settings.outputs_dir)
        return service.aggregate(node_id)

    def get_best_completed_attempt(self, node_id: str) -> ExecutionAttempt:
        """Prefer the newest completed attempt that carries metrics."""
        attempts = self.repo.list_attempts(limit=200, node_id=node_id)
        completed = [
            item
            for item in attempts
            if item.status == JobStatus.COMPLETED
            and (item.result_json or {}).get("metrics")
        ]
        if not completed:
            raise KeyError(f"节点 {node_id} 没有可比较的 completed 执行")
        return max(
            completed,
            key=lambda item: item.completed_at or item.created_at,
        )

    def compare_executions(
        self, execution_id_a: str, execution_id_b: str
    ) -> dict[str, Any]:
        from scientist_lab.services.verifier import compare_attempts

        a = self.repo.get_attempt(execution_id_a)
        b = self.repo.get_attempt(execution_id_b)
        if a is None or b is None:
            raise KeyError("对比需要两个都存在的 execution_id")
        result = compare_attempts(a, b)
        return result.model_dump()

    def compare_nodes(self, node_id_a: str, node_id_b: str) -> dict[str, Any]:
        a = self.get_best_completed_attempt(node_id_a)
        b = self.get_best_completed_attempt(node_id_b)
        return self.compare_executions(a.execution_id, b.execution_id)

    def compare_node_groups(
        self, node_id_a: str, node_id_b: str
    ) -> dict[str, Any]:
        from scientist_lab.services.aggregation import (
            NodeAggregationService,
            compare_node_groups,
        )

        aggregation = NodeAggregationService(self.repo, self.settings.outputs_dir)
        return compare_node_groups(aggregation, node_id_a, node_id_b)

    def compare_fast_eval_triad(
        self,
        *,
        rgb_node_id: str = "rgbt_fast_node_001",
        thermal_node_id: str = "rgbt_fast_node_002",
        fusion_node_id: str = "rgbt_fast_node_003",
        write_report: bool = True,
    ) -> dict[str, Any]:
        """Compare RGB / Thermal / Fusion Fast Eval nodes under exploratory gate."""
        from scientist_lab.storage.artifact_store import read_json, write_json
        from scientist_lab.tasks.rgbt_detection.fast_eval_triad import (
            build_triad_comparison,
        )

        role_nodes = {
            "rgb": rgb_node_id,
            "thermal": thermal_node_id,
            "fusion": fusion_node_id,
        }
        contracts: dict[str, dict[str, Any]] = {}
        metrics_by_role: dict[str, dict[str, Any]] = {}
        execution_ids: dict[str, str] = {}
        resource_by_role: dict[str, dict[str, Any]] = {}
        project_id = None

        for role, node_id in role_nodes.items():
            attempt = self.get_best_completed_attempt(node_id)
            execution_ids[role] = attempt.execution_id
            contract = (attempt.result_json or {}).get("contract") or {}
            node = self.repo.get_node(node_id)
            if not contract and node is not None:
                contract = dict(node.contract_json or {})
            contracts[role] = contract
            if project_id is None:
                project_id = contract.get("project_id") or (
                    node.project_id if node is not None else None
                )
            raw_metrics = dict((attempt.result_json or {}).get("metrics") or {})
            # Detection results nest numeric scores under metrics.metrics.
            nested = raw_metrics.get("metrics")
            if isinstance(nested, dict) and any(
                key in nested for key in ("mAP50_95", "mAP50", "accuracy")
            ):
                metrics_by_role[role] = dict(nested)
            else:
                metrics_by_role[role] = raw_metrics
            output_dir = (attempt.result_json or {}).get("output_directory")
            if output_dir:
                resource_path = Path(output_dir) / "resource_usage.json"
                if resource_path.is_file():
                    try:
                        resource_by_role[role] = read_json(resource_path)
                    except Exception:  # noqa: BLE001
                        resource_by_role[role] = {}

        primary = (
            ((contracts.get("rgb") or {}).get("task_config") or {}).get("primary_metric")
            or "mAP50_95"
        )
        report = build_triad_comparison(
            contracts=contracts,
            metrics_by_role=metrics_by_role,
            execution_ids=execution_ids,
            resource_by_role=resource_by_role,
            primary_metric=str(primary),
        )
        report["node_ids"] = role_nodes

        if write_report and project_id:
            out_dir = (
                Path(self.settings.outputs_dir)
                / str(project_id)
                / "_comparisons"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (
                f"fast_eval_triad_{rgb_node_id}_{thermal_node_id}_{fusion_node_id}.json"
            )
            write_json(out_path, report)
            report["report_path"] = str(out_path)

        return report

    def validate_fast_eval_triad_contracts(
        self, contract_paths: list[str] | None = None
    ) -> dict[str, Any]:
        """Validate example or provided triad contracts for fairness (no run)."""
        import json

        from scientist_lab.tasks.rgbt_detection.fast_eval_triad import (
            DEFAULT_TRIAD_NODES,
            validate_triad_contracts,
        )

        root = Path(self.settings.project_root)
        if not contract_paths:
            contract_paths = [
                str(root / "examples" / "rgbt_fast_rgb_contract.json"),
                str(root / "examples" / "rgbt_fast_thermal_contract.json"),
                str(root / "examples" / "rgbt_fast_fusion_contract.json"),
            ]
        loaded: dict[str, dict[str, Any]] = {}
        path_by_role: dict[str, str] = {}
        for path_str in contract_paths:
            path = Path(path_str)
            payload = json.loads(path.read_text(encoding="utf-8"))
            node_id = str(payload.get("node_id") or "")
            role = None
            for key, expected_node in DEFAULT_TRIAD_NODES.items():
                if node_id == expected_node:
                    role = key
                    break
            if role is None:
                params = payload.get("parameters") or {}
                mode = str(params.get("input_mode") or "")
                fusion = str(params.get("fusion_method") or "")
                if mode == "rgb":
                    role = "rgb"
                elif mode == "thermal":
                    role = "thermal"
                elif mode in {"rgbt", "rgb_thermal"} and fusion == "early_concat":
                    role = "fusion"
            if role is None:
                raise ValueError(f"cannot map contract to triad role: {path}")
            loaded[role] = payload
            path_by_role[role] = str(path)
        result = validate_triad_contracts(loaded)
        result["paths"] = path_by_role
        return result

    def validate_formal_triad_contracts(
        self,
        contract_paths: list[str] | None = None,
        *,
        protocol_id: str | None = None,
        protocol_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Validate formal RGB/Thermal/Fusion contracts against ExperimentProtocol."""
        import json

        from scientist_lab.protocols.formal_triad import (
            DEFAULT_FORMAL_PROTOCOL_ID,
            DEFAULT_FORMAL_TRIAD_NODES,
            infer_triad_role,
        )

        root = Path(self.settings.project_root)
        if protocol_path:
            self.protocols.create_from_path(Path(protocol_path))
        else:
            default_protocol = root / "examples" / "rgbt_protocol.json"
            if default_protocol.is_file():
                # Ensure local example protocol exists for offline validation.
                existing = self.protocols.get(
                    protocol_id or DEFAULT_FORMAL_PROTOCOL_ID
                )
                if existing is None:
                    self.protocols.create_from_path(default_protocol)

        if not contract_paths:
            contract_paths = [
                str(root / "examples" / "rgbt_formal_rgb_contract.json"),
                str(root / "examples" / "rgbt_formal_thermal_contract.json"),
                str(root / "examples" / "rgbt_formal_fusion_contract.json"),
            ]

        loaded: dict[str, dict[str, Any]] = {}
        path_by_role: dict[str, str] = {}
        for path_str in contract_paths:
            path = Path(path_str)
            payload = json.loads(path.read_text(encoding="utf-8"))
            role = None
            node_id = str(payload.get("node_id") or "")
            for key, expected_node in DEFAULT_FORMAL_TRIAD_NODES.items():
                if node_id == expected_node:
                    role = key
                    break
            if role is None:
                role = infer_triad_role(payload)
            if role is None:
                raise ValueError(f"cannot map contract to formal triad role: {path}")
            loaded[role] = payload
            path_by_role[role] = str(path)

        report = self.protocols.validate_formal_triad(
            loaded, protocol_id=protocol_id or DEFAULT_FORMAL_PROTOCOL_ID
        )
        payload = report.model_dump(mode="json")
        payload["ok"] = report.valid
        payload["paths"] = path_by_role
        payload["matched_seeds"] = self.protocols.require(
            protocol_id or DEFAULT_FORMAL_PROTOCOL_ID
        ).seeds
        return payload

    def _contracts_for_nodes(
        self, baseline_node_id: str, candidate_node_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from scientist_lab.services.aggregation import NodeAggregationService

        aggregation = NodeAggregationService(self.repo, self.settings.outputs_dir)
        baseline_map = aggregation.best_attempt_per_seed(baseline_node_id)
        candidate_map = aggregation.best_attempt_per_seed(candidate_node_id)
        baseline_attempt = next(iter(baseline_map.values()), None)
        candidate_attempt = next(iter(candidate_map.values()), None)
        baseline_contract = {}
        candidate_contract = {}
        if baseline_attempt is not None:
            baseline_contract = (baseline_attempt.result_json or {}).get("contract") or {}
        if candidate_attempt is not None:
            candidate_contract = (candidate_attempt.result_json or {}).get("contract") or {}
        baseline_node = self.repo.get_node(baseline_node_id)
        candidate_node = self.repo.get_node(candidate_node_id)
        if not baseline_contract and baseline_node is not None:
            baseline_contract = baseline_node.contract_json or {}
        if not candidate_contract and candidate_node is not None:
            candidate_contract = candidate_node.contract_json or {}
        return baseline_contract, candidate_contract

    def analyze_feedback(
        self, baseline_node_id: str, candidate_node_id: str
    ) -> dict[str, Any]:
        from scientist_lab.domain.feedback import FeedbackInput
        from scientist_lab.domain.models import ExperimentArtifact, new_id, utc_now_iso
        from scientist_lab.services.feedback_analyzer import analyze_feedback
        from scientist_lab.storage.artifact_store import sha256_file, write_json

        comparison = self.compare_node_groups(baseline_node_id, candidate_node_id)
        baseline_contract, candidate_contract = self._contracts_for_nodes(
            baseline_node_id, candidate_node_id
        )
        candidate_node = self.repo.get_node(candidate_node_id)
        if candidate_node is None:
            raise KeyError(f"未找到 node: {candidate_node_id}")

        existing_node_parameters: list[dict[str, Any]] = []
        for node in self.repo.list_nodes(project_id=candidate_node.project_id):
            params = None
            contract_meta: dict[str, Any] = {}
            # Prefer latest successful attempt contract when available.
            from scientist_lab.services.aggregation import NodeAggregationService

            by_seed = NodeAggregationService(
                self.repo, self.settings.outputs_dir
            ).best_attempt_per_seed(node.node_id)
            if by_seed:
                sample = next(iter(by_seed.values()))
                contract_meta = (sample.result_json or {}).get("contract") or {}
                params = contract_meta.get("parameters")
            if not params:
                contract_meta = node.contract_json or {}
                params = contract_meta.get("parameters")
            existing_node_parameters.append(
                {
                    "node_id": node.node_id,
                    "parameters": params or {},
                    "environment_key": contract_meta.get("environment_key")
                    or (node.contract_json or {}).get("environment_key")
                    or "",
                    "dataset_reference": contract_meta.get("dataset_reference")
                    or (node.contract_json or {}).get("dataset_reference")
                    or "",
                    "code_reference": contract_meta.get("code_reference")
                    or (node.contract_json or {}).get("code_reference")
                    or "",
                }
            )

        report = analyze_feedback(
            FeedbackInput(
                baseline_node_id=baseline_node_id,
                candidate_node_id=candidate_node_id,
                comparison=comparison,
                baseline_contract=baseline_contract,
                candidate_contract=candidate_contract,
                research_goal=(
                    (candidate_node.contract_json or {}).get("research_goal")
                    or (candidate_contract or {}).get("research_goal")
                    or ""
                ),
                existing_node_parameters=existing_node_parameters,
            )
        )
        payload = report.model_dump()

        compare_dir = (
            self.settings.outputs_dir
            / candidate_node.project_id
            / "comparisons"
            / f"{baseline_node_id}_vs_{candidate_node_id}"
        )
        compare_dir.mkdir(parents=True, exist_ok=True)
        group_path = compare_dir / "group_comparison.json"
        feedback_path = compare_dir / "feedback.json"
        write_json(group_path, comparison)
        write_json(feedback_path, payload)

        feedback_store = dict(candidate_node.feedback_json or {})
        feedback_store["latest_feedback"] = payload
        feedback_store["latest_group_comparison"] = {
            "path": str(group_path),
            "hypothesis_status": comparison.get("hypothesis_status"),
        }
        candidate_node.feedback_json = feedback_store
        candidate_node.updated_at = utc_now_iso()
        self.repo.upsert_node(candidate_node)

        # Attach artifact to newest completed attempt of candidate when available.
        from scientist_lab.services.aggregation import NodeAggregationService

        by_seed = NodeAggregationService(
            self.repo, self.settings.outputs_dir
        ).best_attempt_per_seed(candidate_node_id)
        if by_seed:
            latest = max(
                by_seed.values(),
                key=lambda item: item.completed_at or item.created_at,
            )
            marker = f"feedback:{baseline_node_id}_vs_{candidate_node_id}"
            existing = self.repo.list_artifacts(latest.execution_id)
            if not any((a.metadata_json or {}).get("marker") == marker for a in existing):
                self.repo.add_artifact(
                    ExperimentArtifact(
                        artifact_id=new_id("art"),
                        execution_id=latest.execution_id,
                        artifact_type="feedback_report",
                        relative_path=str(feedback_path),
                        size_bytes=feedback_path.stat().st_size,
                        sha256=sha256_file(feedback_path),
                        metadata_json={
                            "marker": marker,
                            "baseline_node_id": baseline_node_id,
                            "candidate_node_id": candidate_node_id,
                            "feedback_path": str(feedback_path),
                            "group_comparison_path": str(group_path),
                        },
                        created_at=utc_now_iso(),
                    )
                )

        payload["feedback_path"] = str(feedback_path)
        payload["group_comparison_path"] = str(group_path)

        # Detection claim gate: smoke/debug runs cannot support performance claims.
        candidate_task = (candidate_contract or {}).get("task_type") or (
            candidate_node.contract_json or {}
        ).get("task_type")
        if candidate_task == "rgbt_detection":
            from scientist_lab.tasks.rgbt_detection.feedback_rules import (
                annotate_feedback_for_detection,
            )

            mode = (
                (candidate_contract or {}).get("execution_mode")
                or (candidate_node.contract_json or {}).get("execution_mode")
                or "smoke_train"
            )
            cand = candidate_contract or candidate_node.contract_json or {}
            task_config = cand.get("task_config") or {}
            params = cand.get("parameters") or {}
            payload = annotate_feedback_for_detection(
                payload,
                execution_mode=str(mode),
                evaluation_scope=str(
                    task_config.get("evaluation_scope") or "debug_subset"
                ),
                claim_level=task_config.get("claim_level"),
                baseline_key=str(
                    params.get("baseline") or params.get("model") or "tiny_detector"
                ),
            )
            write_json(feedback_path, payload)
        return payload

    def show_feedback(self, node_id: str) -> dict[str, Any]:
        node = self.repo.get_node(node_id)
        if node is None:
            raise KeyError(f"未找到 node: {node_id}")
        feedback = (node.feedback_json or {}).get("latest_feedback")
        if feedback:
            return feedback
        # Fallback: search comparisons directory
        compare_root = self.settings.outputs_dir / node.project_id / "comparisons"
        if compare_root.exists():
            for path in sorted(compare_root.glob(f"*_vs_{node_id}/feedback.json")):
                from scientist_lab.storage.artifact_store import read_json

                return read_json(path)
        raise KeyError(f"节点 {node_id} 尚无 feedback 报告")

    def propose_next(
        self, baseline_node_id: str, candidate_node_id: str
    ) -> dict[str, Any]:
        from scientist_lab.domain.feedback import FeedbackReport
        from scientist_lab.services.proposal import build_next_contract
        from scientist_lab.storage.artifact_store import write_json

        feedback_payload = self.analyze_feedback(baseline_node_id, candidate_node_id)
        report = FeedbackReport.model_validate(
            {
                key: value
                for key, value in feedback_payload.items()
                if key
                not in {
                    "feedback_path",
                    "group_comparison_path",
                }
            }
        )
        baseline_contract, candidate_contract = self._contracts_for_nodes(
            baseline_node_id, candidate_node_id
        )
        existing = {node.node_id for node in self.repo.list_nodes()}
        proposal = build_next_contract(
            baseline_contract=baseline_contract,
            candidate_contract=candidate_contract,
            feedback=report,
            existing_node_ids=existing,
        )
        if proposal is None:
            raise ValueError(
                "无法生成下一轮契约：没有 status=active 且带参数修改的建议"
                "（可能都是 deferred / already_evaluated）"
            )

        candidate_node = self.repo.get_node(candidate_node_id)
        project_id = (
            candidate_node.project_id
            if candidate_node is not None
            else candidate_contract.get("project_id", "project_001")
        )
        out_dir = (
            self.settings.outputs_dir
            / project_id
            / "comparisons"
            / f"{baseline_node_id}_vs_{candidate_node_id}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        contract = proposal["contract"]
        contract_path = out_dir / f"proposed_{contract['node_id']}.json"
        write_json(contract_path, contract)
        proposal["contract_path"] = str(contract_path)
        proposal["note"] = (
            "Draft only — review manually, then run-seeds. Not auto-submitted."
        )
        return proposal

    def record_decision(
        self,
        *,
        selected_node_id: str,
        alternatives: list[str],
        decision_type: str,
        reason: str,
        evidence_strength: str = "moderate",
        baseline_node_id: str | None = None,
        candidate_node_id: str | None = None,
        supporting_evidence_ids: list[str] | None = None,
        protocol_id: str | None = None,
        claim_matrix_path: str | None = None,
        auto_attach_evidence: bool = True,
        ensure_claim_matrix: bool = True,
    ) -> dict[str, Any]:
        from scientist_lab.domain.models import ExperimentDecision, utc_now_iso
        from scientist_lab.evidence.decision_link import (
            cap_evidence_strength,
            resolve_protocol_id,
            select_supporting_evidence,
        )
        from scientist_lab.storage.artifact_store import write_json
        from scientist_lab.tasks.rgbt_detection.decision_rules import (
            node_is_smoke_detection,
            validate_smoke_decision,
        )

        selected = self.repo.get_node(selected_node_id)
        if selected is None:
            raise KeyError(f"未找到 node: {selected_node_id}")

        from scientist_lab.domain.models import new_id

        claim_level = None
        if node_is_smoke_detection(selected.contract_json):
            selected_claim = (
                (selected.contract_json or {}).get("task_config") or {}
            ).get("claim_level")
            normalized = validate_smoke_decision(
                decision_type,
                evidence_strength=evidence_strength,
                claim_level=selected_claim,
            )
            decision_type = normalized["decision_type"]
            evidence_strength = normalized["evidence_strength"]
            claim_level = normalized["claim_level"]

        project_id = selected.project_id
        evidence_records: list = []
        linked_ids: list[str] = []
        if auto_attach_evidence or supporting_evidence_ids:
            all_evidence = self.evidence.list_evidence(project_id=project_id)
            node_ids = {
                nid
                for nid in (
                    selected_node_id,
                    baseline_node_id,
                    candidate_node_id,
                    *list(alternatives or []),
                )
                if nid
            }
            evidence_records = select_supporting_evidence(
                all_evidence,
                node_ids=node_ids,
                explicit_ids=list(supporting_evidence_ids)
                if supporting_evidence_ids
                else None,
            )
            linked_ids = [item.evidence_id for item in evidence_records]
            evidence_strength = cap_evidence_strength(
                evidence_strength, evidence_records
            )

        resolved_protocol = resolve_protocol_id(
            explicit=protocol_id,
            selected_contract=selected.contract_json,
            evidence_records=evidence_records,
        )

        matrix_path = claim_matrix_path
        if ensure_claim_matrix and not matrix_path:
            existing = self.evidence.claim_matrix_path(project_id)
            if existing.is_file():
                matrix_path = str(existing)
            else:
                built = self.evidence.build_claim_matrix(
                    project_id, protocol_id=resolved_protocol
                )
                matrix_path = built.get("matrix_path") or str(existing)
        elif matrix_path:
            matrix_path = str(matrix_path)

        decision = ExperimentDecision(
            decision_id=new_id("decision"),
            selected_node_id=selected_node_id,
            decision_type=decision_type,
            alternatives=list(alternatives or []),
            reason=reason,
            evidence_strength=evidence_strength,
            baseline_node_id=baseline_node_id,
            candidate_node_id=candidate_node_id,
            claim_level=claim_level,
            supporting_evidence_ids=linked_ids,
            claim_matrix_path=matrix_path,
            protocol_id=resolved_protocol,
            project_id=project_id,
            recorded_at=utc_now_iso(),
        )

        out_dir = self.settings.outputs_dir / project_id / "decisions"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"decision_{selected_node_id}_{utc_now_iso().replace(':', '')}.json"
        )
        payload = decision.model_dump(mode="json")
        write_json(path, payload)

        feedback = dict(selected.feedback_json or {})
        feedback["latest_decision"] = payload
        selected.feedback_json = feedback
        selected.updated_at = utc_now_iso()
        self.repo.upsert_node(selected)

        payload["decision_path"] = str(path)
        return payload

    def show_decision(self, decision_id: str) -> dict[str, Any]:
        target = (decision_id or "").strip()
        if not target:
            raise ValueError("decision_id required")
        for item in self.list_decisions(limit=500):
            if str(item.get("decision_id") or "") == target:
                return item
        raise KeyError(f"decision not found: {target}")

    def prepare_fast_eval(
        self,
        node_id: str,
        *,
        execution_id: str | None = None,
        node_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Build a multi-seed-ready fast_eval contract from a smoke node."""
        from scientist_lab.storage.artifact_store import write_json
        from scientist_lab.tasks.rgbt_detection.fast_eval_bridge import (
            DEFAULT_FAST_EVAL_SEEDS,
            build_fast_eval_contract,
            relative_checkpoint_source,
            resolve_checkpoint_path,
        )

        node = self.repo.get_node(node_id)
        if node is None:
            raise KeyError(f"未找到 node: {node_id}")
        mode = ((node.contract_json or {}).get("execution_mode") or "").strip()
        task = ((node.contract_json or {}).get("task_type") or "").strip()
        if task != "rgbt_detection" or mode != "smoke_train":
            raise ValueError(
                "prepare-fast-eval requires an rgbt_detection smoke_train node "
                f"(got task_type={task!r}, execution_mode={mode!r})"
            )

        if execution_id:
            attempt = self.repo.get_attempt(execution_id)
            if attempt is None:
                raise KeyError(f"未找到 execution: {execution_id}")
            if attempt.node_id != node_id:
                raise ValueError(
                    f"execution {execution_id} does not belong to node {node_id}"
                )
        else:
            attempt = self.get_best_completed_attempt(node_id)

        checkpoint_source = relative_checkpoint_source(
            project_id=node.project_id,
            execution_id=attempt.execution_id,
        )
        resolve_checkpoint_path(self.settings.outputs_dir, checkpoint_source)

        contract = build_fast_eval_contract(
            node.contract_json or {},
            checkpoint_source=checkpoint_source,
            node_id=node_id_override,
            parent_node_id=node_id,
        )
        out_dir = (
            self.settings.outputs_dir
            / node.project_id
            / "prepared_fast_eval"
            / node_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        contract_path = out_dir / f"{contract['node_id']}.json"
        write_json(contract_path, contract)
        seeds = list(DEFAULT_FAST_EVAL_SEEDS)
        return {
            "source_node_id": node_id,
            "source_execution_id": attempt.execution_id,
            "checkpoint_source": checkpoint_source,
            "contract_path": str(contract_path),
            "contract": contract,
            "suggested_seeds": seeds,
            "suggested_decision_type": "ready_for_fast_eval",
            "next_command": (
                f'scientist-lab run-seeds "{contract_path}" '
                f'--seeds {",".join(str(s) for s in seeds)}'
            ),
        }

    def register_dataset(
        self,
        *,
        dataset_key: str,
        task_type: str,
        path: str | Path,
        container_path: str | None = None,
        read_only: bool = True,
    ) -> dict[str, Any]:
        item = self.datasets.register(
            dataset_key=dataset_key,
            task_type=task_type,
            host_path=path,
            container_path=container_path,
            read_only=read_only,
        )
        return item.model_dump(mode="json")

    def register_runner_profile(
        self,
        *,
        profile_key: str,
        runner_type: str,
        endpoint: str | None = None,
        auth_token_env: str | None = None,
        allowed_environments: list[str] | None = None,
        default_timeout_seconds: int = 7200,
    ) -> dict[str, Any]:
        profile = self.runner_profiles.register(
            profile_key=profile_key,
            runner_type=runner_type,
            endpoint=endpoint,
            auth_token_env=auth_token_env,
            allowed_environment_keys=allowed_environments,
            default_timeout_seconds=default_timeout_seconds,
        )
        return profile.model_dump(mode="json")

    def list_runner_profiles(self) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json") for item in self.runner_profiles.list_profiles()
        ]

    def show_runner_profile(self, profile_key: str) -> dict[str, Any]:
        profile = self.runner_profiles.get(profile_key)
        if profile is None:
            raise KeyError(f"runner profile not found: {profile_key}")
        return profile.model_dump(mode="json")

    def check_runner(self, profile_key: str) -> dict[str, Any]:
        return self.runner_profiles.check(profile_key)

    def list_checkpoints(
        self,
        *,
        project_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.checkpoints.list_checkpoints(
                project_id=project_id, execution_id=execution_id
            )
        ]

    def show_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.checkpoints.require(checkpoint_id)
        payload = record.model_dump(mode="json")
        payload["resolved_path"] = str(self.checkpoints.resolve_path(checkpoint_id))
        return payload

    def verify_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.checkpoints.verify(checkpoint_id)
        payload = record.model_dump(mode="json")
        payload["resolved_path"] = str(self.checkpoints.resolve_path(checkpoint_id))
        payload["ok"] = bool(record.verified)
        return payload

    def register_checkpoint(
        self,
        *,
        project_id: str,
        execution_id: str,
        relative_path: str,
        node_id: str | None = None,
        role: str | None = None,
        baseline_key: str | None = None,
    ) -> dict[str, Any]:
        record = self.checkpoints.register(
            project_id=project_id,
            execution_id=execution_id,
            relative_path=relative_path,
            node_id=node_id,
            role=role,
            baseline_key=baseline_key,
            verify=True,
        )
        return record.model_dump(mode="json")

    def list_datasets(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.datasets.list_datasets()]

    def show_dataset(self, dataset_key: str) -> dict[str, Any]:
        item = self.datasets.get(dataset_key)
        if item is None:
            raise KeyError(f"数据集未注册：{dataset_key}")
        return item.model_dump(mode="json")

    def disable_dataset(self, dataset_key: str) -> dict[str, Any]:
        return self.datasets.disable(dataset_key).model_dump(mode="json")

    def enable_dataset(self, dataset_key: str) -> dict[str, Any]:
        return self.datasets.enable(dataset_key).model_dump(mode="json")

    def validate_dataset(self, dataset_key: str) -> dict[str, Any]:
        from scientist_lab.datasets.validator import validate_dataset

        item = self.datasets.require(dataset_key, require_enabled=False)
        return validate_dataset(item, outputs_dir=self.settings.outputs_dir)

    def preview_dataset(self, dataset_key: str, *, count: int = 5) -> dict[str, Any]:
        from scientist_lab.datasets.validator import preview_dataset

        item = self.datasets.require(dataset_key, require_enabled=False)
        return preview_dataset(
            item, outputs_dir=self.settings.outputs_dir, count=count
        )

    def list_decisions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        from scientist_lab.storage.artifact_store import read_json

        roots: list[Path] = []
        if project_id:
            roots.append(self.settings.outputs_dir / project_id / "decisions")
        else:
            outputs = self.settings.outputs_dir
            if outputs.exists():
                roots.extend(sorted(outputs.glob("*/decisions")))

        decisions: list[dict[str, Any]] = []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("decision_*.json"), reverse=True):
                try:
                    payload = read_json(path)
                except Exception:  # noqa: BLE001
                    continue
                payload = dict(payload)
                payload.setdefault("decision_path", str(path))
                decisions.append(payload)
                if len(decisions) >= limit:
                    return decisions
        return decisions

    def create_protocol(self, path: Path) -> dict[str, Any]:
        protocol = self.protocols.create_from_path(Path(path))
        return protocol.model_dump(mode="json")

    def list_protocols(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.protocols.list_protocols(project_id=project_id)
        ]

    def show_protocol(self, protocol_id: str) -> dict[str, Any]:
        return self.protocols.require(protocol_id).model_dump(mode="json")

    def validate_protocol(
        self,
        protocol_id: str,
        *,
        contract_path: Path | None = None,
    ) -> dict[str, Any]:
        contract = load_contract(Path(contract_path)) if contract_path else None
        report = self.protocols.validate(protocol_id, contract=contract)
        return report.model_dump(mode="json")

    def create_ablation(self, path: Path) -> dict[str, Any]:
        root = Path(self.settings.project_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = None
        protocol_id = str(data.get("protocol_id") or "").strip()
        if protocol_id:
            protocol = self.protocols.get(protocol_id)
            if protocol is None:
                default_protocol = root / "examples" / "rgbt_protocol.json"
                if default_protocol.is_file():
                    self.protocols.create_from_path(default_protocol)
                    protocol = self.protocols.get(protocol_id)
        plan = self.ablations.create_from_dict(data, protocol=protocol)
        return plan.model_dump(mode="json")

    def list_ablations(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.ablations.list_plans(
                project_id=project_id, protocol_id=protocol_id
            )
        ]

    def show_ablation(self, ablation_id: str) -> dict[str, Any]:
        return self.ablations.require(ablation_id).model_dump(mode="json")

    def validate_ablation(self, ablation_id: str) -> dict[str, Any]:
        plan = self.ablations.require(ablation_id)
        protocol = self.protocols.get(plan.protocol_id)
        report = self.ablations.validate(ablation_id, protocol=protocol)
        return report.model_dump(mode="json")

    def materialize_ablation(
        self,
        ablation_id: str,
        *,
        reference_contract: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        plan = self.ablations.require(ablation_id)
        root = Path(self.settings.project_root)
        if reference_contract:
            reference = load_contract(Path(reference_contract))
        else:
            # Prefer fusion formal contract as the full-system reference.
            candidates = [
                root / "examples" / "rgbt_formal_fusion_contract.json",
                root / "examples" / "rgbt_formal_rgb_contract.json",
            ]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    "reference contract required when formal examples are missing"
                )
            reference = load_contract(path)

        out = Path(output_dir) if output_dir else (
            self.settings.outputs_dir / plan.project_id / "ablations" / plan.ablation_id
        )
        return self.ablations.materialize(
            ablation_id, reference, output_dir=out
        )

    def build_evidence(
        self,
        baseline_node_id: str,
        candidate_node_id: str,
        *,
        include_resource_evidence: bool = True,
    ) -> dict[str, Any]:
        from scientist_lab.storage.artifact_store import write_json

        comparison = self.compare_node_groups(baseline_node_id, candidate_node_id)
        baseline_contract, candidate_contract = self._contracts_for_nodes(
            baseline_node_id, candidate_node_id
        )
        sample_contract = candidate_contract or baseline_contract or {}
        project_id = str(
            sample_contract.get("project_id")
            or (comparison.get("candidate_aggregate") or {}).get("project_id")
            or (comparison.get("baseline_aggregate") or {}).get("project_id")
            or ""
        )
        if not project_id:
            raise ValueError("cannot resolve project_id for evidence")

        comparison_dir = self.settings.outputs_dir / project_id / "evidence"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = (
            comparison_dir
            / f"comparison_{baseline_node_id}_vs_{candidate_node_id}.json"
        )
        write_json(comparison_path, comparison)

        impl = str((sample_contract.get("task_config") or {}).get("implementation") or "")
        formal = impl not in {"", "stand_in", "stand-in"} and "standin" not in impl.lower()
        has_ablation = bool(
            self.ablations.list_plans(project_id=project_id)
        )

        def _artifact_ids(execution_id: str) -> list[str]:
            return [item.artifact_id for item in self.repo.list_artifacts(execution_id)]

        return self.evidence.build_from_node_group_comparison(
            comparison,
            project_id=project_id,
            sample_contract=sample_contract,
            artifact_resolver=_artifact_ids,
            include_resource_evidence=include_resource_evidence,
            has_ablation=has_ablation,
            formal_implementation=formal,
            comparison_path=str(comparison_path),
        )

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
        evidence_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.evidence.list_evidence(
                project_id=project_id,
                protocol_id=protocol_id,
                evidence_type=evidence_type,
            )
        ]

    def show_evidence(self, evidence_id: str) -> dict[str, Any]:
        record = self.evidence.require(evidence_id)
        payload = record.model_dump(mode="json")
        path = self.evidence.evidence_path(evidence_id)
        payload["evidence_path"] = str(path) if path.exists() else None
        return payload

    def build_claim_matrix(
        self,
        project_id: str,
        *,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        return self.evidence.build_claim_matrix(
            project_id, protocol_id=protocol_id
        )

    def show_claim_matrix(self, project_id: str) -> dict[str, Any]:
        return self.evidence.show_claim_matrix(project_id)

    def plan_next(
        self,
        project_id: str,
        *,
        protocol_id: str | None = None,
        current_best_node_id: str | None = None,
        max_new_nodes: int = 3,
        max_gpu_hours: float = 12.0,
        provider: str = "mock",
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
        model_profile: str | None = None,
        require_quality_gate: bool = False,
        allow_unqualified_profile: bool = False,
        suite_version: str | None = "eval_suite_v1",
    ) -> dict[str, Any]:
        from scientist_lab.agents.provider_bridge import normalize_provider_mode
        from scientist_lab.llm.config import redact_secrets
        from scientist_lab.llm.errors import (
            LLMError,
            MissingAPIKeyError,
            RealProviderNotEnabledError,
        )
        from scientist_lab.llm_eval.profile_gate import (
            LLMProfileNotQualifiedError,
            assert_profile_qualified_for_planning,
        )

        project = self.repo.get_project(project_id)
        research_goal = (
            project.research_goal
            if project is not None
            else "Improve RGB-T detection under a fixed protocol."
        )
        nodes = self.repo.list_nodes(project_id=project_id)

        protocol_payload = None
        resolved_protocol_id = protocol_id
        if not resolved_protocol_id:
            for node in nodes:
                contract = dict(node.contract_json or {})
                if contract.get("protocol_id"):
                    resolved_protocol_id = str(contract["protocol_id"])
                    break
        if resolved_protocol_id:
            try:
                protocol_payload = self.protocols.require(
                    resolved_protocol_id
                ).model_dump(mode="json")
            except KeyError:
                root = Path(self.settings.project_root)
                default_protocol = root / "examples" / "rgbt_protocol.json"
                if default_protocol.is_file():
                    self.protocols.create_from_path(default_protocol)
                    protocol_payload = self.protocols.require(
                        resolved_protocol_id
                    ).model_dump(mode="json")

        if protocol_payload is None:
            # Fall back to first node's contract fields for allow-list.
            sample = dict((nodes[0].contract_json if nodes else {}) or {})
            protocol_payload = {
                "protocol_id": sample.get("protocol_id"),
                "project_id": project_id,
                "allowed_variables": ["input_mode", "fusion_method"],
                "fixed_parameters": {},
            }

        evidence_records = self.list_evidence(project_id=project_id)
        try:
            claim_matrix = self.show_claim_matrix(project_id)
        except Exception:  # noqa: BLE001
            claim_matrix = {}

        remaining = self.budget.remaining(project_id)
        remaining["max_new_nodes"] = min(
            int(remaining.get("max_new_nodes", max_new_nodes)),
            int(max_new_nodes),
        )
        remaining["max_total_gpu_hours"] = min(
            float(remaining.get("max_total_gpu_hours", max_gpu_hours)),
            float(max_gpu_hours),
        )

        mode = normalize_provider_mode(provider)
        quality_audit: dict[str, Any] | None = None
        if require_quality_gate:
            profile_id = model_profile or self.llm_evals.get_default_profile_id()
            if not profile_id:
                raise ValueError(
                    "model profile required with --require-quality-gate "
                    "(pass --model-profile or llm-profile-select)"
                )
            profile = self.llm_evals.get_profile(profile_id)
            if profile is None:
                raise KeyError(f"llm profile not found: {profile_id}")
            evaluation = self.llm_evals.latest_evaluation_for_profile(
                profile_id, suite_version=suite_version
            )
            try:
                quality_audit = assert_profile_qualified_for_planning(
                    profile=profile,
                    evaluation=evaluation,
                    suite_version=suite_version,
                    require_quality_gate=True,
                    allow_unqualified=allow_unqualified_profile,
                )
            except LLMProfileNotQualifiedError as exc:
                return {
                    "status": "profile_not_qualified",
                    "project_id": project_id,
                    "requested_provider": mode,
                    "actual_provider": None,
                    "fallback_used": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "quality_gate": {
                        "profile_id": profile_id,
                        "quality_gate_bypassed": False,
                    },
                }

        try:
            self.agents.configure_provider(
                mode,
                project_id=project_id,
                audit_root=self.settings.outputs_dir / project_id / "llm",
                allow_network=allow_network,
                transport=transport,
                openai_config=openai_config,
            )
            result = self.agents.plan_next(
                project_id=project_id,
                research_goal=research_goal,
                protocol=protocol_payload,
                nodes=nodes,
                evidence_records=evidence_records,
                claim_support_matrix=claim_matrix,
                current_best_node_id=current_best_node_id,
                remaining_budget=remaining,
            )
            result["requested_provider"] = mode
            result["actual_provider"] = result.get("model_provider")
            result["fallback_used"] = False
            if quality_audit is not None:
                result["quality_gate"] = quality_audit
                if quality_audit.get("quality_gate_bypassed"):
                    result["formal_approval_eligible"] = False
                else:
                    result["formal_approval_eligible"] = True
            return result
        except (RealProviderNotEnabledError, MissingAPIKeyError, LLMError, ValueError) as exc:
            if mode not in {"real", "openai-compatible"}:
                raise
            return {
                "status": "real_provider_failed",
                "project_id": project_id,
                "requested_provider": mode,
                "actual_provider": None,
                "fallback_used": False,
                "error_type": type(exc).__name__,
                "error": redact_secrets(str(exc)),
            }

    def list_plans(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        return self.agents.list_plans(project_id=project_id)

    def show_plan(self, plan_id: str) -> dict[str, Any]:
        return self.agents.get_plan(plan_id)

    def evaluate_llm_quality(
        self,
        project_id: str,
        *,
        protocol_id: str | None = None,
        current_best_node_id: str | None = None,
        max_new_nodes: int = 3,
        max_gpu_hours: float = 12.0,
        output: str | Path | None = None,
        include_real: bool = False,
        suite: str | None = None,
        provider: str = "mock",
        allow_network: bool = False,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        """LLM quality eval (v1.3 compare or v1.4.4 suite).

        Without ``suite``: Mock/Fake/Replay comparison report.
        With ``suite``: run ``evals/llm/<suite>.jsonl`` for one provider.
        Real provider is skipped unless gates pass (never fails CI by default).
        """
        from scientist_lab.agents.context_builder import build_planning_context
        from scientist_lab.llm.eval_suite import (
            run_llm_eval_suite,
            write_suite_report,
        )
        from scientist_lab.llm.quality import (
            evaluate_provider_quality,
            write_quality_report,
        )

        if suite:
            audit_root = self.settings.outputs_dir / project_id / "llm"
            report = run_llm_eval_suite(
                suite=suite,
                provider=provider,
                project_id=project_id,
                audit_root=audit_root,
                evals_root=Path(self.settings.project_root) / "evals" / "llm",
                max_cases=max_cases,
                allow_network=allow_network,
            )
            out_path = (
                Path(output)
                if output
                else (
                    self.settings.outputs_dir
                    / project_id
                    / "acceptance"
                    / f"llm_eval_{suite}_{provider}.json"
                )
            )
            report = write_suite_report(report, out_path)
            return report.model_dump(mode="json")

        project = self.repo.get_project(project_id)
        research_goal = (
            project.research_goal
            if project is not None
            else "Improve RGB-T detection under a fixed protocol."
        )
        nodes = self.repo.list_nodes(project_id=project_id)

        protocol_payload = None
        resolved_protocol_id = protocol_id
        if not resolved_protocol_id:
            for node in nodes:
                contract = dict(node.contract_json or {})
                if contract.get("protocol_id"):
                    resolved_protocol_id = str(contract["protocol_id"])
                    break
        if resolved_protocol_id:
            try:
                protocol_payload = self.protocols.require(
                    resolved_protocol_id
                ).model_dump(mode="json")
            except KeyError:
                root = Path(self.settings.project_root)
                default_protocol = root / "examples" / "rgbt_protocol.json"
                if default_protocol.is_file():
                    self.protocols.create_from_path(default_protocol)
                    protocol_payload = self.protocols.require(
                        resolved_protocol_id
                    ).model_dump(mode="json")
        if protocol_payload is None:
            sample = dict((nodes[0].contract_json if nodes else {}) or {})
            protocol_payload = {
                "protocol_id": sample.get("protocol_id"),
                "project_id": project_id,
                "allowed_variables": ["input_mode", "fusion_method"],
                "fixed_parameters": {},
            }

        evidence_records = self.list_evidence(project_id=project_id)
        try:
            claim_matrix = self.show_claim_matrix(project_id)
        except Exception:  # noqa: BLE001
            claim_matrix = {}

        remaining = self.budget.remaining(project_id)
        remaining["max_new_nodes"] = min(
            int(remaining.get("max_new_nodes", max_new_nodes)),
            int(max_new_nodes),
        )
        remaining["max_total_gpu_hours"] = min(
            float(remaining.get("max_total_gpu_hours", max_gpu_hours)),
            float(max_gpu_hours),
        )

        context = build_planning_context(
            project_id=project_id,
            research_goal=research_goal,
            protocol=protocol_payload,
            nodes=nodes,
            evidence_records=evidence_records,
            claim_support_matrix=claim_matrix,
            current_best_node_id=current_best_node_id,
            remaining_budget=remaining,
        )
        audit_root = self.settings.outputs_dir / project_id / "llm"
        report = evaluate_provider_quality(
            context,
            audit_root=audit_root,
            include_real=include_real or (provider in {"real", "openai-compatible"}),
        )
        out_path = (
            Path(output)
            if output
            else (
                self.settings.outputs_dir
                / project_id
                / "acceptance"
                / "llm_quality_report.json"
            )
        )
        report = write_quality_report(report, out_path)
        return report.model_dump(mode="json")

    def summarize_llm_usage(
        self,
        project_id: str | None = None,
        *,
        audit_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Aggregate token / cost / latency from LLM audit records (v1.3.9)."""
        from scientist_lab.llm.quality import summarize_audit_usage

        if audit_root is not None:
            root = Path(audit_root)
        elif project_id:
            root = self.settings.outputs_dir / project_id / "llm"
        else:
            raise ValueError("project_id or audit_root is required")
        return summarize_audit_usage(root)

    def run_llm_eval_suite(
        self,
        project_id: str,
        *,
        suite: str = "eval_suite_v1",
        provider: str = "mock",
        allow_network: bool = False,
        profile_id: str | None = None,
        transport: Any = None,
        openai_config: Any = None,
        seed_fake_for_replay: bool = True,
    ) -> dict[str, Any]:
        """Run versioned llm_eval suite (v1.5.3) and write scorecard artifacts."""
        from scientist_lab.llm_eval.profiles import default_mock_profile
        from scientist_lab.llm_eval.runner import run_evaluation_suite

        profile = default_mock_profile()
        if profile_id:
            loaded = self.llm_evals.get_profile(profile_id)
            if loaded is None:
                raise KeyError(f"llm profile not found: {profile_id}")
            profile = loaded
        return run_evaluation_suite(
            suite,
            provider=provider,
            allow_network=allow_network,
            project_id=project_id,
            output_root=self.settings.outputs_dir,
            profile=profile,
            transport=transport,
            openai_config=openai_config,
            seed_fake_for_replay=seed_fake_for_replay,
            repository=self.llm_evals,
        )

    def register_llm_profile(
        self, path: str | Path | None = None, *, profile: Any = None
    ) -> dict[str, Any]:
        from scientist_lab.llm_eval.profiles import LLMModelProfile

        if profile is not None:
            item = (
                profile
                if isinstance(profile, LLMModelProfile)
                else LLMModelProfile.model_validate(profile)
            )
        elif path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            item = LLMModelProfile.model_validate(data)
        else:
            raise ValueError("path or profile is required")
        self.llm_evals.upsert_profile(item)
        return item.safe_dict()

    def list_llm_profiles(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        return [
            p.safe_dict()
            for p in self.llm_evals.list_profiles(enabled_only=enabled_only)
        ]

    def show_llm_profile(self, profile_id: str) -> dict[str, Any]:
        profile = self.llm_evals.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"llm profile not found: {profile_id}")
        return profile.safe_dict()

    def get_llm_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        row = self.llm_evals.get_evaluation(evaluation_id)
        if row is None:
            raise KeyError(f"llm evaluation not found: {evaluation_id}")
        return row

    def verify_llm_evaluation(
        self,
        evaluation_id: str,
        *,
        thresholds: Any = None,
    ) -> dict[str, Any]:
        from scientist_lab.llm_eval.quality_gate import (
            LLMQualityThresholds,
            verify_quality_gate,
        )

        row = self.get_llm_evaluation(evaluation_id)
        scorecard = dict(row.get("result") or {})
        th = None
        if thresholds is not None:
            th = (
                thresholds
                if isinstance(thresholds, LLMQualityThresholds)
                else LLMQualityThresholds.model_validate(thresholds)
            )
        gate = verify_quality_gate(scorecard, thresholds=th)
        payload = gate.model_dump(mode="json")
        payload["evaluation_id"] = evaluation_id
        payload["profile_id"] = row.get("profile_id")
        payload["suite_version"] = row.get("suite_version")
        return payload

    def compare_llm_evaluations(
        self, baseline_evaluation_id: str, candidate_evaluation_id: str
    ) -> dict[str, Any]:
        from scientist_lab.llm_eval.compare import compare_evaluations

        baseline = self.get_llm_evaluation(baseline_evaluation_id)
        candidate = self.get_llm_evaluation(candidate_evaluation_id)
        result = compare_evaluations(
            dict(baseline.get("result") or {}),
            dict(candidate.get("result") or {}),
            baseline_id=baseline_evaluation_id,
            candidate_id=candidate_evaluation_id,
        )
        return result.model_dump(mode="json")

    def rank_llm_profiles(
        self,
        *,
        suite_version: str = "eval_suite_v1",
        profile_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        from scientist_lab.llm_eval.ranking import rank_profiles

        profiles = self.llm_evals.list_profiles(enabled_only=True)
        if profile_ids:
            wanted = set(profile_ids)
            profiles = [p for p in profiles if p.profile_id in wanted]
        entries = []
        for profile in profiles:
            latest = self.llm_evals.latest_evaluation_for_profile(
                profile.profile_id, suite_version=suite_version
            )
            if latest is None:
                entries.append(
                    {
                        "profile_id": profile.profile_id,
                        "evaluation_id": None,
                        "scorecard": {
                            "status": "skipped",
                            "skip_reason": "no evaluation",
                            "planner": {"case_count": 0},
                            "safety": {"pass": False, "pass_rate": 0.0, "case_count": 0},
                        },
                    }
                )
                continue
            entries.append(
                {
                    "profile_id": profile.profile_id,
                    "evaluation_id": latest["evaluation_id"],
                    "scorecard": dict(latest.get("result") or {}),
                }
            )
        ranked = rank_profiles(entries)
        return {
            "suite_version": suite_version,
            "default_profile_id": self.llm_evals.get_default_profile_id(),
            "ranking": [r.model_dump(mode="json") for r in ranked],
            "qualified_count": sum(1 for r in ranked if r.qualified),
        }

    def select_llm_profile(self, profile_id: str) -> dict[str, Any]:
        """Human-select default profile (never auto-select by cost)."""
        selected = self.llm_evals.set_default_profile(profile_id)
        return {
            "default_profile_id": selected,
            "profile": self.show_llm_profile(selected),
        }

    def review_plan(
        self,
        plan_id: str,
        *,
        provider: str | None = None,
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
    ) -> dict[str, Any]:
        from scientist_lab.agents.provider_bridge import normalize_provider_mode
        from scientist_lab.llm.config import redact_secrets
        from scientist_lab.llm.errors import (
            LLMError,
            MissingAPIKeyError,
            RealProviderNotEnabledError,
        )

        plan = self.agents.get_plan(plan_id)
        project_id = str(plan.get("project_id") or "")
        mode = normalize_provider_mode(
            provider or self.agents.provider_mode or "mock"
        )
        try:
            if project_id:
                self.agents.configure_provider(
                    mode,
                    project_id=project_id,
                    audit_root=self.settings.outputs_dir / project_id / "llm",
                    allow_network=allow_network,
                    transport=transport,
                    openai_config=openai_config,
                )
            result = self.agents.review_plan(plan_id)
            result["requested_provider"] = mode
            result["actual_provider"] = getattr(
                self.agents.critic, "model_provider", mode
            )
            result["fallback_used"] = False
            return result
        except (RealProviderNotEnabledError, MissingAPIKeyError, LLMError, ValueError) as exc:
            if mode not in {"real", "openai-compatible"}:
                raise
            return {
                "status": "real_provider_failed",
                "plan_id": plan_id,
                "requested_provider": mode,
                "actual_provider": None,
                "fallback_used": False,
                "error_type": type(exc).__name__,
                "error": redact_secrets(str(exc)),
            }

    def rank_candidates(self, plan_id: str) -> dict[str, Any]:
        return self.agents.rank_plan_candidates(plan_id)

    def approve_candidate(self, plan_id: str, candidate_id: str) -> dict[str, Any]:
        return self.agents.approve_candidate(plan_id, candidate_id)

    def reject_candidate(
        self, plan_id: str, candidate_id: str, *, reason: str | None = None
    ) -> dict[str, Any]:
        return self.agents.reject_candidate(plan_id, candidate_id, reason=reason)

    def generate_contract_from_plan(
        self, plan_id: str, candidate_id: str
    ) -> dict[str, Any]:
        plan = self.agents.get_plan(plan_id)
        candidates = {
            item.get("candidate_id"): item for item in plan.get("candidates") or []
        }
        cand = candidates.get(candidate_id)
        if cand is None:
            raise KeyError(f"candidate not found: {candidate_id}")
        parent_id = str(cand.get("parent_node_id") or "")
        parent_node = self.repo.get_node(parent_id)
        if parent_node is None:
            raise KeyError(f"parent node not found: {parent_id}")
        parent_contract = dict(parent_node.contract_json or {})
        existing = {node.node_id for node in self.repo.list_nodes(project_id=plan["project_id"])}
        result = self.agents.generate_contract(
            plan_id,
            candidate_id,
            parent_contract=parent_contract,
            existing_node_ids=existing,
        )
        # Reserve budget for the generated node.
        gpu_hours = float((cand.get("estimated_cost") or {}).get("gpu_hours") or 1.0)
        self.budget.consume(plan["project_id"], nodes=1, gpu_hours=gpu_hours)
        return result

    def set_budget(
        self,
        project_id: str,
        *,
        max_new_nodes: int = 3,
        max_executions: int = 15,
        max_gpu_hours: float = 10.0,
        max_storage_gb: float = 20.0,
    ) -> dict[str, Any]:
        budget = self.budget.set_budget(
            project_id,
            max_new_nodes=max_new_nodes,
            max_total_executions=max_executions,
            max_gpu_hours=max_gpu_hours,
            max_storage_gb=max_storage_gb,
        )
        return budget.model_dump(mode="json")

    def show_budget(self, project_id: str) -> dict[str, Any]:
        budget = self.budget.get(project_id)
        if budget is None:
            budget = self.budget.set_budget(project_id)
        payload = budget.model_dump(mode="json")
        payload["remaining"] = budget.remaining()
        return payload

    def tree_create(
        self,
        project_id: str,
        *,
        root_node_id: str,
        protocol_id: str,
        max_depth: int = 3,
        max_nodes: int = 8,
        max_children: int = 3,
        tree_id: str | None = None,
    ) -> dict[str, Any]:
        return self.trees.create_tree(
            project_id,
            root_node_id=root_node_id,
            protocol_id=protocol_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_children=max_children,
            tree_id=tree_id,
        )

    def tree_status(self, tree_id: str) -> dict[str, Any]:
        return self.trees.tree_status(tree_id)

    def tree_show(self, tree_id: str) -> dict[str, Any]:
        return self.trees.show_tree(tree_id)

    def tree_nodes(self, tree_id: str) -> list[dict[str, Any]]:
        return self.trees.list_nodes(tree_id)

    def tree_export(self, tree_id: str, *, format: str = "json") -> dict[str, Any]:
        return self.trees.export_tree(tree_id, format=format)

    def build_report_context(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        """Assemble ReportContext snapshot (v1.2.1)."""
        return self.reporting.show_context_dict(
            project_id, tree_id=tree_id, protocol_id=protocol_id
        )

    def build_report(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        return self.reporting.build_report(
            project_id, tree_id=tree_id, protocol_id=protocol_id
        )

    def show_report(self, report_id: str) -> dict[str, Any]:
        return self.reporting.show_report(report_id)

    def verify_report(self, report_id: str) -> dict[str, Any]:
        return self.reporting.verify_report(report_id)

    def build_audit(
        self,
        project_id: str,
        *,
        tree_id: str | None = None,
        protocol_id: str | None = None,
        report_id: str | None = None,
    ) -> dict[str, Any]:
        return self.reporting.build_audit(
            project_id,
            tree_id=tree_id,
            protocol_id=protocol_id,
            report_id=report_id,
        )

    def verify_audit(self, bundle_id: str) -> dict[str, Any]:
        return self.reporting.verify_audit(bundle_id)

    def export_audit(self, bundle_id: str, output_dir: str | Path) -> dict[str, Any]:
        return self.reporting.export_audit(bundle_id, output_dir)

    def list_releases(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.releases.list_releases(project_id=project_id, limit=limit)

    def show_release(self, release_id: str) -> dict[str, Any]:
        return self.releases.show(release_id)

    def create_release(
        self,
        *,
        project_id: str,
        title: str = "",
        tree_id: str | None = None,
        report_id: str | None = None,
        audit_bundle_id: str | None = None,
        patch_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.releases.create(
            project_id=project_id,
            title=title,
            tree_id=tree_id,
            report_id=report_id,
            audit_bundle_id=audit_bundle_id,
            patch_ids=patch_ids,
        )

    def freeze_release(self, release_id: str, *, notes: str = "") -> dict[str, Any]:
        return self.releases.freeze(release_id, notes=notes)

    def discard_release(self, release_id: str, *, reason: str = "") -> dict[str, Any]:
        return self.releases.discard(release_id, reason=reason)

    def workspace_summary(self) -> dict[str, Any]:
        return self.releases.workspace_summary()

    def merge_prepare(
        self, patch_id: str, *, target_branch: str | None = None
    ) -> dict[str, Any]:
        return self.merges.prepare(patch_id, target_branch=target_branch)

    def merge_show(self, merge_candidate_id: str) -> dict[str, Any]:
        return self.merges.show(merge_candidate_id)

    def merge_apply(self, merge_candidate_id: str) -> dict[str, Any]:
        return self.merges.apply_workspace(merge_candidate_id)

    def merge_test(
        self, merge_candidate_id: str, *, profile_id: str = "smoke"
    ) -> dict[str, Any]:
        return self.merges.run_tests(merge_candidate_id, profile_id=profile_id)

    def merge_approve(
        self,
        merge_candidate_id: str,
        *,
        reason: str = "",
        approved_by: str = "human",
    ) -> dict[str, Any]:
        return self.merges.approve(
            merge_candidate_id, reason=reason, approved_by=approved_by
        )

    def merge_reject(
        self,
        merge_candidate_id: str,
        *,
        reason: str = "",
        approved_by: str = "human",
    ) -> dict[str, Any]:
        return self.merges.reject(
            merge_candidate_id, reason=reason, approved_by=approved_by
        )

    def merge_commit(self, merge_candidate_id: str) -> dict[str, Any]:
        return self.merges.commit_candidate(merge_candidate_id)

    def merge_finalize(
        self,
        merge_candidate_id: str,
        *,
        post_merge_profile: str | None = "syntax",
        auto_rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        return self.merges.finalize(
            merge_candidate_id,
            post_merge_profile=post_merge_profile,
            auto_rollback_on_failure=auto_rollback_on_failure,
        )

    def merge_rollback(
        self,
        merge_candidate_id: str,
        *,
        reason: str = "",
        trigger: str = "human",
    ) -> dict[str, Any]:
        return self.merges.rollback(
            merge_candidate_id, reason=reason, trigger=trigger
        )

    def list_merge_candidates(
        self,
        *,
        project_id: str | None = None,
        patch_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.merges.list_candidates(
            project_id=project_id, patch_id=patch_id, limit=limit
        )

    def list_merge_profiles(self) -> list[dict[str, Any]]:
        return self.merges.list_profiles()

    def create_release_candidate(
        self,
        *,
        version: str,
        project_id: str = "",
        base_tag: str = "",
        merge_candidate_ids: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return self.release_candidates.create(
            version=version,
            project_id=project_id,
            base_tag=base_tag,
            merge_candidate_ids=merge_candidate_ids,
            notes=notes,
        )

    def show_release_candidate(self, release_candidate_id: str) -> dict[str, Any]:
        return self.release_candidates.show(release_candidate_id)

    def verify_release_candidate(self, release_candidate_id: str) -> dict[str, Any]:
        return self.release_candidates.verify(release_candidate_id)

    def list_release_candidates(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.release_candidates.list_all(project_id=project_id, limit=limit)

    def tree_score(
        self, tree_id: str, *, tree_node_id: str | None = None
    ) -> dict[str, Any]:
        if tree_node_id:
            return self.trees.score_node(tree_id, tree_node_id)
        return self.trees.score_tree(tree_id)

    def tree_select_parent(
        self, tree_id: str, *, rescore: bool = True
    ) -> dict[str, Any]:
        return self.trees.select_parent(tree_id, rescore=rescore)

    def tree_plan_next(
        self,
        tree_id: str,
        *,
        rescore: bool = True,
        max_gpu_hours: float = 12.0,
        provider: str = "mock",
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
        model_profile: str | None = None,
        require_quality_gate: bool = False,
        allow_unqualified_profile: bool = False,
        suite_version: str | None = "eval_suite_v1",
    ) -> dict[str, Any]:
        """Best-First parent → plan-next → review → rank.

        ``provider`` defaults to mock; fake/replay offline; real needs gates.
        """
        from scientist_lab.agents.provider_bridge import normalize_provider_mode
        from scientist_lab.search.expansion_service import (
            annotate_candidates_against_tree,
            remaining_candidate_slots,
            tree_parameter_fingerprints,
        )
        from scientist_lab.search.state_machine import (
            assert_tree_transition,
            is_tree_terminal,
        )

        tree = self.trees.require_tree(tree_id)
        if is_tree_terminal(tree.status):
            raise ValueError(f"tree is terminal ({tree.status}); cannot plan-next")

        selection = self.trees.select_parent(tree_id, rescore=rescore)
        tree = self.trees.require_tree(tree_id)

        if not selection.get("selected"):
            stop = self._tree_evaluate_and_maybe_stop(tree_id)
            return {
                "tree_id": tree_id,
                "status": "stopped",
                "tree_status": (stop or {}).get("tree_status") or tree.status,
                "selection": selection,
                "plan": None,
                "ranking": None,
                "reason": (stop or {}).get("reason")
                or selection.get("reason")
                or "no expandable parent",
                "stop": stop,
                "stop_suggested": True,
                "stop_status": (stop or {}).get("status")
                or selection.get("stop_status")
                or "no_valid_candidates",
            }

        parent_info = selection["selected"]
        parent_node = self.trees._repo.get_node(parent_info["tree_node_id"])
        if parent_node is None:
            raise KeyError(f"tree node not found: {parent_info['tree_node_id']}")

        slots = remaining_candidate_slots(
            tree,
            parent_node,
            child_count=int(parent_info.get("child_count") or 0),
            node_count=self.trees._repo.count_nodes(tree_id),
        )
        if slots <= 0:
            return {
                "tree_id": tree_id,
                "status": "stopped",
                "tree_status": tree.status,
                "selection": selection,
                "plan": None,
                "ranking": None,
                "reason": "no remaining candidate slots under max_children/max_nodes",
                "stop_suggested": True,
                "stop_status": "no_valid_candidates",
            }

        mode = normalize_provider_mode(provider)
        planned = self.plan_next(
            tree.project_id,
            protocol_id=tree.protocol_id,
            current_best_node_id=parent_info["experiment_node_id"],
            max_new_nodes=slots,
            max_gpu_hours=max_gpu_hours,
            provider=mode,
            allow_network=allow_network,
            transport=transport,
            openai_config=openai_config,
            model_profile=model_profile,
            require_quality_gate=require_quality_gate,
            allow_unqualified_profile=allow_unqualified_profile,
            suite_version=suite_version,
        )
        if planned.get("status") == "profile_not_qualified":
            return {
                "tree_id": tree_id,
                "status": "profile_not_qualified",
                "tree_status": tree.status,
                "selection": selection,
                "plan": planned,
                "ranking": None,
                "provider": mode,
                "requested_provider": mode,
                "actual_provider": None,
                "fallback_used": False,
                "reason": planned.get("error") or "profile not qualified",
                "quality_gate": planned.get("quality_gate"),
            }
        if planned.get("status") == "real_provider_failed":
            return {
                "tree_id": tree_id,
                "status": "real_provider_failed",
                "tree_status": tree.status,
                "selection": selection,
                "plan": planned,
                "ranking": None,
                "provider": mode,
                "requested_provider": mode,
                "actual_provider": None,
                "fallback_used": False,
                "reason": planned.get("error") or "real provider failed",
            }
        plan_id = str(planned.get("plan_id") or "")
        if planned.get("status") == "planner_failed" or not plan_id:
            return {
                "tree_id": tree_id,
                "status": "planner_failed",
                "tree_status": tree.status,
                "selection": selection,
                "plan": planned,
                "ranking": None,
                "reason": planned.get("reasoning_summary")
                or "planner failed to produce candidates",
                "stop_suggested": bool(planned.get("stop_recommended")),
                "stop_status": "no_valid_candidates"
                if planned.get("stop_recommended")
                else None,
                "provider": mode,
            }

        if planned.get("stop_recommended") and not (planned.get("candidates") or []):
            return {
                "tree_id": tree_id,
                "status": "stopped",
                "tree_status": tree.status,
                "selection": selection,
                "plan": planned,
                "ranking": None,
                "reason": planned.get("stop_reason") or "planner recommended stop",
                "stop_suggested": True,
                "stop_status": "no_valid_candidates",
                "provider": mode,
            }

        reviewed = self.review_plan(
            plan_id,
            provider=mode,
            allow_network=allow_network,
            transport=transport,
            openai_config=openai_config,
        )
        if reviewed.get("status") == "real_provider_failed":
            return {
                "tree_id": tree_id,
                "status": "real_provider_failed",
                "tree_status": tree.status,
                "selection": selection,
                "plan": planned,
                "review": reviewed,
                "ranking": None,
                "provider": mode,
                "requested_provider": mode,
                "actual_provider": None,
                "fallback_used": False,
                "reason": reviewed.get("error") or "real critic failed",
            }
        ranked = self.rank_candidates(plan_id)

        # Fingerprints of experiment nodes already on this tree.
        tree_nodes = self.trees._repo.list_nodes(tree_id)
        exp_nodes = []
        for tn in tree_nodes:
            node = self.repo.get_node(tn.experiment_node_id)
            if node is not None:
                exp_nodes.append(node)
        fingerprints = tree_parameter_fingerprints(
            exp_nodes,
            allowed_keys=list(
                (self.protocols.require(tree.protocol_id).allowed_variables or [])
            ),
        )

        plan_view = self.show_plan(plan_id)
        filtered = annotate_candidates_against_tree(
            list(plan_view.get("candidates") or []),
            parent_experiment_node_id=parent_info["experiment_node_id"],
            existing_fingerprints=fingerprints,
            max_candidates=slots,
        )

        # Restrict ranking to accepted candidate ids.
        accepted_ids = {
            str(item.get("candidate_id")) for item in filtered["accepted"]
        }
        ranking = [
            item
            for item in (ranked.get("ranking") or [])
            if str(item.get("candidate_id")) in accepted_ids
        ]
        # Re-rank positions
        for idx, item in enumerate(ranking):
            item = dict(item)
            item["rank"] = idx + 1
            ranking[idx] = item

        # Persist plan_id on parent tree node.
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).replace(microsecond=0)
        parent_node = parent_node.model_copy(
            update={"plan_id": plan_id, "updated_at": now}
        )
        self.trees._repo.upsert_node(parent_node)

        # Tree → waiting_approval (human must approve a candidate next).
        tree = self.trees.require_tree(tree_id)
        if tree.status == "created":
            assert_tree_transition(tree.status, "active")
            tree = tree.model_copy(update={"status": "active", "updated_at": now})
        if tree.status == "active":
            assert_tree_transition(tree.status, "waiting_approval")
            tree = tree.model_copy(
                update={
                    "status": "waiting_approval",
                    "selected_node_id": parent_info["experiment_node_id"],
                    "updated_at": now,
                }
            )
            self.trees._repo.upsert_tree(tree)
        elif tree.status == "waiting_approval":
            tree = tree.model_copy(
                update={
                    "selected_node_id": parent_info["experiment_node_id"],
                    "updated_at": now,
                }
            )
            self.trees._repo.upsert_tree(tree)

        status = "planned"
        if not ranking:
            status = "no_valid_candidates"

        stop = self._tree_evaluate_and_maybe_stop(
            tree_id,
            last_plan_filter=filtered if status != "planned" else None,
            planner_stop_recommended=bool(planned.get("stop_recommended")),
            planner_stop_reason=planned.get("stop_reason"),
        )
        if stop and stop.get("should_stop"):
            return {
                "tree_id": tree_id,
                "status": "stopped",
                "tree_status": stop.get("tree_status"),
                "selection": selection,
                "parent": parent_info,
                "plan_id": plan_id,
                "plan": {
                    "plan_id": plan_id,
                    "status": plan_view.get("status"),
                    "stop_recommended": planned.get("stop_recommended"),
                    "reasoning_summary": planned.get("reasoning_summary"),
                },
                "ranking": ranking,
                "candidate_filter": filtered,
                "stop": stop,
                "stop_suggested": True,
                "stop_status": stop.get("status"),
                "reason": stop.get("reason"),
                "next_action": "Tree stopped by StopPolicy.",
                "ascii_tree": self.trees.render_ascii(tree_id),
                "provider": mode,
            }

        return {
            "tree_id": tree_id,
            "status": status,
            "tree_status": self.trees.require_tree(tree_id).status,
            "selection": selection,
            "parent": parent_info,
            "plan_id": plan_id,
            "plan": {
                "plan_id": plan_id,
                "status": plan_view.get("status"),
                "model_provider": plan_view.get("model_provider"),
                "valid_candidate_count": plan_view.get("valid_candidate_count"),
                "stop_recommended": planned.get("stop_recommended"),
                "reasoning_summary": planned.get("reasoning_summary"),
            },
            "review": {
                "status": reviewed.get("status"),
                "review_count": len(reviewed.get("reviews") or []),
            },
            "ranking": ranking,
            "candidate_filter": filtered,
            "max_candidates": slots,
            "top_candidate_id": ranking[0]["candidate_id"] if ranking else None,
            "next_action": (
                "Run tree-approve <tree_id> <candidate_id> after human review."
                if ranking
                else "No accepted candidates; prune, stop, or adjust budget/limits."
            ),
            "ascii_tree": self.trees.render_ascii(tree_id),
            "stop_suggested": status == "no_valid_candidates",
            "stop_status": "no_valid_candidates"
            if status == "no_valid_candidates"
            else None,
            "provider": mode,
        }

    def tree_approve(
        self,
        tree_id: str,
        candidate_id: str,
        *,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Human-approve one ranked candidate → Contract → IterationSession (v1.1.5)."""
        from datetime import datetime, timezone

        from scientist_lab.iteration.service import IterationService
        from scientist_lab.search.state_machine import (
            assert_node_transition,
            is_tree_terminal,
        )

        tree = self.trees.require_tree(tree_id)
        if is_tree_terminal(tree.status):
            raise ValueError(f"tree is terminal ({tree.status}); cannot approve")
        if tree.status not in {"waiting_approval", "active"}:
            raise ValueError(
                f"tree status {tree.status} cannot approve candidates "
                "(run tree-plan-next first)"
            )

        parent_tn = self._tree_parent_with_plan(tree_id)
        plan_id = parent_tn.plan_id
        if not plan_id:
            raise ValueError("selected parent has no plan_id; run tree-plan-next first")

        plan = self.show_plan(plan_id)
        candidates = {
            str(item.get("candidate_id")): item
            for item in (plan.get("candidates") or [])
        }
        cand = candidates.get(candidate_id)
        if cand is None:
            raise KeyError(f"candidate not found in plan {plan_id}: {candidate_id}")
        if cand.get("status") == "rejected":
            raise ValueError(f"candidate is rejected: {candidate_id}")

        parent_exp = str(parent_tn.experiment_node_id)
        if str(cand.get("parent_node_id") or "") not in {"", parent_exp}:
            raise ValueError(
                f"candidate parent_node_id mismatch: "
                f"{cand.get('parent_node_id')} != {parent_exp}"
            )

        # Approve (idempotent if already approved/contract_generated).
        status = str(cand.get("status") or "")
        if status not in {"approved", "contract_generated"}:
            self.approve_candidate(plan_id, candidate_id)

        iteration = IterationService(self)
        iter_payload = iteration.start_from_plan(
            plan_id, candidate_id, seeds=seeds
        )
        proposed_node_id = str(iter_payload.get("proposed_node_id") or "")
        iteration_id = str(iter_payload.get("iteration_id") or "")
        if not proposed_node_id or not iteration_id:
            raise ValueError("iterate-from-plan did not return proposed_node_id/iteration_id")

        node_type = self._map_experiment_type_to_tree_node(
            str(cand.get("experiment_type") or "improve")
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        existing = self.trees._repo.get_node_by_experiment(tree_id, proposed_node_id)
        if existing is None:
            child = self.trees.register_experiment_node(
                tree_id,
                experiment_node_id=proposed_node_id,
                parent_tree_node_id=parent_tn.tree_node_id,
                node_type=node_type,
            )
        else:
            child = existing

        # created → waiting_approval
        if child.status == "created":
            assert_node_transition(child.status, "waiting_approval")
            child = child.model_copy(
                update={
                    "status": "waiting_approval",
                    "plan_id": plan_id,
                    "candidate_id": candidate_id,
                    "iteration_id": iteration_id,
                    "node_type": node_type,  # type: ignore[arg-type]
                    "updated_at": now,
                }
            )
        else:
            child = child.model_copy(
                update={
                    "plan_id": plan_id,
                    "candidate_id": candidate_id,
                    "iteration_id": iteration_id,
                    "updated_at": now,
                }
            )
        self.trees._repo.upsert_node(child)

        # Keep tree waiting for iterate-approve (execution still gated).
        tree = self.trees.require_tree(tree_id)
        tree = tree.model_copy(
            update={
                "status": "waiting_approval",
                "selected_node_id": proposed_node_id,
                "updated_at": now,
            }
        )
        self.trees._repo.upsert_tree(tree)

        return {
            "tree_id": tree_id,
            "status": "waiting_approval",
            "tree_status": tree.status,
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "parent_tree_node_id": parent_tn.tree_node_id,
            "parent_experiment_node_id": parent_tn.experiment_node_id,
            "tree_node": child.model_dump(mode="json"),
            "iteration": iter_payload,
            "iteration_id": iteration_id,
            "proposed_node_id": proposed_node_id,
            "next_action": (
                "Review the IterationSession proposal, then run "
                f"iterate-approve {iteration_id} (execution remains human-gated)."
            ),
            "ascii_tree": self.trees.render_ascii(tree_id),
        }

    def tree_advance(
        self,
        tree_id: str,
        *,
        tree_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Sync IterationSession outcomes into TreeNodes and rescore (v1.1.6)."""
        from datetime import datetime, timezone

        from scientist_lab.iteration.service import IterationService
        from scientist_lab.search.state_machine import (
            assert_node_transition,
            assert_tree_transition,
            is_tree_terminal,
        )

        tree = self.trees.require_tree(tree_id)
        if is_tree_terminal(tree.status):
            raise ValueError(f"tree is terminal ({tree.status}); cannot advance")

        iteration = IterationService(self)
        nodes = self.trees._repo.list_nodes(tree_id)
        if tree_node_id:
            nodes = [n for n in nodes if n.tree_node_id == tree_node_id]
            if not nodes:
                raise KeyError(f"tree node not found: {tree_node_id}")

        pending_nodes = [
            n
            for n in nodes
            if n.iteration_id
            and n.status
            not in {"evaluated", "selected", "pruned", "stopped"}
        ]
        if not pending_nodes:
            return {
                "tree_id": tree_id,
                "status": "noop",
                "tree_status": tree.status,
                "advanced": [],
                "pending": [],
                "failed": [],
                "reason": "no tree nodes with pending iterations",
                "next_action": "Run tree-plan-next to expand another parent.",
                "ascii_tree": self.trees.render_ascii(tree_id),
            }

        now = datetime.now(timezone.utc).replace(microsecond=0)
        advanced: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        any_running = False

        for node in pending_nodes:
            assert node.iteration_id is not None
            try:
                iter_status = iteration.get_status(
                    node.iteration_id, refresh=True
                )
            except KeyError as exc:
                failed.append(
                    {
                        "tree_node_id": node.tree_node_id,
                        "iteration_id": node.iteration_id,
                        "error": str(exc),
                    }
                )
                continue

            status = str(iter_status.get("status") or "")
            decision_id = iter_status.get("decision_id")
            selected_node_id = iter_status.get("selected_node_id")

            if status == "completed":
                updated = self._tree_mark_node_evaluated(
                    node,
                    decision_id=str(decision_id) if decision_id else None,
                    now=now,
                )
                evidence_link = self._tree_link_evidence(tree_id, updated.tree_node_id)
                updated = self.trees._repo.get_node(updated.tree_node_id) or updated
                breakdown = self.trees.score_node(tree_id, updated.tree_node_id)
                advanced.append(
                    {
                        "tree_node_id": updated.tree_node_id,
                        "experiment_node_id": updated.experiment_node_id,
                        "iteration_id": node.iteration_id,
                        "iteration_status": status,
                        "decision_id": decision_id,
                        "selected_node_id": selected_node_id,
                        "node_status": updated.status,
                        "node_score": breakdown.get("node_score"),
                        "expansion_priority": breakdown.get("expansion_priority"),
                        "evidence_ids": evidence_link.get("evidence_ids") or [],
                        "resolved_evidence_gaps": evidence_link.get(
                            "resolved_evidence_gaps"
                        )
                        or [],
                        "new_evidence_gaps": evidence_link.get("new_evidence_gaps")
                        or [],
                        "claim_matrix_path": evidence_link.get("claim_matrix_path"),
                        "action": "evaluated",
                    }
                )
                continue

            if status in {"failed", "cancelled", "rejected", "stopped_no_recommendation"}:
                updated = self._tree_mark_node_failed(
                    node,
                    iteration_status=status,
                    error_message=iter_status.get("error_message"),
                    now=now,
                )
                failed.append(
                    {
                        "tree_node_id": updated.tree_node_id,
                        "experiment_node_id": updated.experiment_node_id,
                        "iteration_id": node.iteration_id,
                        "iteration_status": status,
                        "node_status": updated.status,
                        "error_message": iter_status.get("error_message"),
                        "action": "failed",
                    }
                )
                continue

            # Still in flight / waiting human steps.
            if status in {
                "approved",
                "running",
                "comparing",
                "waiting_decision",
            }:
                any_running = True
                if node.status == "waiting_approval":
                    assert_node_transition(node.status, "running")
                    node = node.model_copy(
                        update={"status": "running", "updated_at": now}
                    )
                    self.trees._repo.upsert_node(node)
                pending.append(
                    {
                        "tree_node_id": node.tree_node_id,
                        "experiment_node_id": node.experiment_node_id,
                        "iteration_id": node.iteration_id,
                        "iteration_status": status,
                        "node_status": node.status,
                        "next_action": (
                            "Run iterate-finalize after comparisons."
                            if status == "waiting_decision"
                            else "Wait for seed runs / iterate-advance."
                        ),
                        "action": "pending",
                    }
                )
                continue

            # waiting_approval on iteration: human must iterate-approve
            pending.append(
                {
                    "tree_node_id": node.tree_node_id,
                    "experiment_node_id": node.experiment_node_id,
                    "iteration_id": node.iteration_id,
                    "iteration_status": status,
                    "node_status": node.status,
                    "next_action": (
                        f"Run iterate-approve {node.iteration_id} "
                        "before tree-advance can finalize results."
                    ),
                    "action": "pending",
                }
            )

        # Update tree status.
        tree = self.trees.require_tree(tree_id)
        if advanced or failed:
            if tree.status == "waiting_approval":
                assert_tree_transition(tree.status, "evaluating")
                tree = tree.model_copy(
                    update={"status": "evaluating", "updated_at": now}
                )
            elif tree.status == "running":
                assert_tree_transition(tree.status, "evaluating")
                tree = tree.model_copy(
                    update={"status": "evaluating", "updated_at": now}
                )
            if tree.status == "evaluating" and not pending and not any_running:
                assert_tree_transition(tree.status, "active")
                tree = tree.model_copy(
                    update={"status": "active", "updated_at": now}
                )
            self.trees._repo.upsert_tree(tree)
        elif any_running and tree.status == "waiting_approval":
            assert_tree_transition(tree.status, "running")
            tree = tree.model_copy(update={"status": "running", "updated_at": now})
            self.trees._repo.upsert_tree(tree)

        tree = self.trees.require_tree(tree_id)
        summary_status = "advanced" if advanced or failed else "pending"
        next_action = "Run tree-plan-next to continue search."
        if pending:
            next_action = pending[0].get("next_action") or next_action
        elif advanced:
            next_action = (
                "Results backfilled. Optionally run tree-plan-next for the next parent."
            )

        # Update improvement counters from newly evaluated scores.
        from scientist_lab.search.stop_policy import update_improvement_counters

        new_scores = [
            float(item["node_score"])
            for item in advanced
            if item.get("node_score") is not None
        ]
        if new_scores:
            tree = self.trees.require_tree(tree_id)
            tree = update_improvement_counters(tree, new_scores=new_scores)
            tree = tree.model_copy(update={"updated_at": now})
            self.trees._repo.upsert_tree(tree)

        stop = self._tree_evaluate_and_maybe_stop(tree_id)
        if stop and stop.get("should_stop"):
            return {
                "tree_id": tree_id,
                "status": "stopped",
                "tree_status": stop.get("tree_status"),
                "advanced": advanced,
                "pending": pending,
                "failed": failed,
                "advanced_count": len(advanced),
                "pending_count": len(pending),
                "failed_count": len(failed),
                "stop": stop,
                "stop_suggested": True,
                "stop_status": stop.get("status"),
                "next_action": "Tree stopped by StopPolicy.",
                "ascii_tree": self.trees.render_ascii(tree_id),
            }

        tree = self.trees.require_tree(tree_id)
        return {
            "tree_id": tree_id,
            "status": summary_status,
            "tree_status": tree.status,
            "advanced": advanced,
            "pending": pending,
            "failed": failed,
            "advanced_count": len(advanced),
            "pending_count": len(pending),
            "failed_count": len(failed),
            "next_action": next_action,
            "ascii_tree": self.trees.render_ascii(tree_id),
        }

    def _tree_mark_node_evaluated(
        self,
        node,
        *,
        decision_id: str | None,
        now,
    ):
        from scientist_lab.search.state_machine import assert_node_transition

        updated = node
        if updated.status == "waiting_approval":
            assert_node_transition(updated.status, "running")
            updated = updated.model_copy(
                update={"status": "running", "updated_at": now}
            )
        if updated.status == "running":
            assert_node_transition(updated.status, "evaluated")
            updated = updated.model_copy(
                update={
                    "status": "evaluated",
                    "decision_id": decision_id,
                    "updated_at": now,
                }
            )
        elif updated.status == "evaluated":
            updated = updated.model_copy(
                update={"decision_id": decision_id, "updated_at": now}
            )
        else:
            # Force path for unexpected statuses that still have completed iteration.
            if updated.status not in {"evaluated", "selected"}:
                raise ValueError(
                    f"cannot mark tree node {updated.tree_node_id} evaluated "
                    f"from status={updated.status}"
                )
        return self.trees._repo.upsert_node(updated)

    def _tree_mark_node_failed(
        self,
        node,
        *,
        iteration_status: str,
        error_message: str | None,
        now,
    ):
        from scientist_lab.search.state_machine import assert_node_transition

        _ = iteration_status, error_message
        updated = node
        if updated.status == "waiting_approval":
            assert_node_transition(updated.status, "failed")
            updated = updated.model_copy(
                update={"status": "failed", "updated_at": now}
            )
        elif updated.status == "running":
            assert_node_transition(updated.status, "failed")
            updated = updated.model_copy(
                update={"status": "failed", "updated_at": now}
            )
        return self.trees._repo.upsert_node(updated)

    def tree_stop(self, tree_id: str, *, reason: str) -> dict[str, Any]:
        return self.trees.stop_tree(tree_id, reason=reason)

    def tree_evidence(self, tree_id: str) -> dict[str, Any]:
        """Summarize Evidence / Claim Matrix linkage for a tree (v1.1.8)."""
        from scientist_lab.search.evidence_link import extract_open_gaps

        tree = self.trees.require_tree(tree_id)
        nodes = self.trees._repo.list_nodes(tree_id)
        linked = [
            {
                "tree_node_id": n.tree_node_id,
                "experiment_node_id": n.experiment_node_id,
                "status": n.status,
                "depth": n.depth,
                "node_type": n.node_type,
                "evidence_ids": list(n.evidence_ids),
                "resolved_evidence_gaps": list(n.resolved_evidence_gaps),
                "new_evidence_gaps": list(n.new_evidence_gaps),
                "claim_matrix_path": n.claim_matrix_path,
            }
            for n in nodes
            if n.evidence_ids or n.resolved_evidence_gaps or n.new_evidence_gaps
        ]
        claim_matrix = self._tree_claim_matrix(tree.project_id) or {}
        evidence_records = self._tree_list_evidence(tree.project_id)
        return {
            "tree_id": tree_id,
            "project_id": tree.project_id,
            "protocol_id": tree.protocol_id,
            "linked_nodes": linked,
            "linked_count": len(linked),
            "open_evidence_gaps": extract_open_gaps(claim_matrix, evidence_records),
            "claim_matrix_path": claim_matrix.get("matrix_path"),
            "evidence_count": len(evidence_records),
            "ascii_tree": self.trees.render_ascii(tree_id),
        }

    def _tree_link_evidence(self, tree_id: str, tree_node_id: str) -> dict[str, Any]:
        """Build/link Evidence + Claim Matrix for an evaluated tree node."""
        from datetime import datetime, timezone

        from scientist_lab.search.evidence_link import (
            collect_related_evidence_ids,
            diff_gaps,
            evidence_link_payload,
            extract_open_gaps,
        )

        tree = self.trees.require_tree(tree_id)
        node = self.trees._repo.get_node(tree_node_id)
        if node is None or node.tree_id != tree_id:
            raise KeyError(f"tree node not found: {tree_node_id}")

        notes: list[str] = []
        before_records = self._tree_list_evidence(tree.project_id)
        before_matrix = self._tree_claim_matrix(tree.project_id) or {}
        before_gaps = extract_open_gaps(before_matrix, before_records)

        parent_experiment_id: str | None = None
        if node.parent_tree_node_id:
            parent = self.trees._repo.get_node(node.parent_tree_node_id)
            if parent is not None:
                parent_experiment_id = parent.experiment_node_id

        built_ids: list[str] = []
        if parent_experiment_id:
            try:
                built = self.build_evidence(
                    parent_experiment_id,
                    node.experiment_node_id,
                    include_resource_evidence=True,
                )
                built_ids = list(built.get("evidence_ids") or [])
            except Exception as exc:  # noqa: BLE001
                notes.append(f"evidence build skipped: {exc}")

        claim_matrix_path: str | None = None
        try:
            matrix = self.build_claim_matrix(
                tree.project_id, protocol_id=tree.protocol_id
            )
            claim_matrix_path = (
                str(matrix.get("matrix_path") or "") or None
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"claim matrix rebuild skipped: {exc}")
            matrix = self._tree_claim_matrix(tree.project_id) or {}
            claim_matrix_path = (
                str(matrix.get("matrix_path") or "") or None
            )

        after_records = self._tree_list_evidence(tree.project_id)
        after_gaps = extract_open_gaps(matrix if isinstance(matrix, dict) else {}, after_records)
        resolved, new_gaps = diff_gaps(before_gaps, after_gaps)

        related = collect_related_evidence_ids(
            after_records,
            experiment_node_ids=[
                x
                for x in [parent_experiment_id, node.experiment_node_id]
                if x
            ],
        )
        evidence_ids: list[str] = []
        for eid in built_ids + related:
            if eid and eid not in evidence_ids:
                evidence_ids.append(eid)

        now = datetime.now(timezone.utc).replace(microsecond=0)
        updated = node.model_copy(
            update={
                "evidence_ids": evidence_ids,
                "resolved_evidence_gaps": resolved,
                "new_evidence_gaps": new_gaps,
                "claim_matrix_path": claim_matrix_path,
                "updated_at": now,
            }
        )
        self.trees._repo.upsert_node(updated)
        return evidence_link_payload(
            tree_node_id=updated.tree_node_id,
            experiment_node_id=updated.experiment_node_id,
            evidence_ids=evidence_ids,
            resolved_evidence_gaps=resolved,
            new_evidence_gaps=new_gaps,
            claim_matrix_path=claim_matrix_path,
            notes=notes,
        )

    def _tree_evaluate_and_maybe_stop(
        self,
        tree_id: str,
        *,
        last_plan_filter: dict[str, Any] | None = None,
        planner_stop_recommended: bool = False,
        planner_stop_reason: str | None = None,
    ) -> dict[str, Any] | None:
        from scientist_lab.search.stop_policy import evaluate_stop

        tree = self.trees.require_tree(tree_id)
        nodes = self.trees._repo.list_nodes(tree_id)
        budget_key = tree.budget_id or tree.project_id
        remaining = self._tree_remaining_budget(budget_key)
        decision = evaluate_stop(
            tree,
            nodes,
            remaining_budget=remaining,
            last_plan_filter=last_plan_filter,
            planner_stop_recommended=planner_stop_recommended,
            planner_stop_reason=planner_stop_reason,
        )
        if not decision.should_stop or not decision.status:
            return None
        updated = self.trees.apply_stop(
            tree_id,
            status=decision.status,
            reason=decision.reason or decision.status,
        )
        return {
            "should_stop": True,
            "status": updated.status,
            "reason": updated.stop_reason,
            "details": decision.details,
            "tree_status": updated.status,
        }

    def _tree_parent_with_plan(self, tree_id: str):
        tree = self.trees.require_tree(tree_id)
        nodes = self.trees._repo.list_nodes(tree_id)
        if tree.selected_node_id:
            for node in nodes:
                if (
                    node.experiment_node_id == tree.selected_node_id
                    and node.plan_id
                ):
                    return node
        # Prefer deepest expandable parent that already has a plan.
        with_plan = [n for n in nodes if n.plan_id]
        if not with_plan:
            raise ValueError(
                "no tree node with plan_id; run tree-plan-next before tree-approve"
            )
        with_plan.sort(key=lambda n: (n.depth, n.updated_at), reverse=True)
        return with_plan[0]

    @staticmethod
    def _map_experiment_type_to_tree_node(experiment_type: str) -> str:
        mapping = {
            "improve": "improve",
            "ablation": "ablation",
            "replication": "replication",
            "debug": "debug",
            "efficiency": "efficiency",
            "robustness": "improve",
        }
        return mapping.get(experiment_type, "improve")

    def _tree_node_aggregate(self, experiment_node_id: str) -> dict[str, Any] | None:
        node = self.repo.get_node(experiment_node_id)
        if node is None:
            return None
        feedback = dict(node.feedback_json or {})
        cached = feedback.get("aggregate_metrics")
        if isinstance(cached, dict) and cached:
            return cached
        try:
            return self.aggregate_node(experiment_node_id)
        except Exception:  # noqa: BLE001
            return None

    def _tree_list_evidence(self, project_id: str) -> list[dict[str, Any]]:
        try:
            records = self.evidence.list_evidence(project_id=project_id)
        except Exception:  # noqa: BLE001
            return []
        out: list[dict[str, Any]] = []
        for record in records:
            if hasattr(record, "model_dump"):
                out.append(record.model_dump(mode="json"))
            elif isinstance(record, dict):
                out.append(record)
        return out

    def _tree_claim_matrix(self, project_id: str) -> dict[str, Any] | None:
        try:
            return self.evidence.show_claim_matrix(project_id)
        except Exception:  # noqa: BLE001
            return None

    def _tree_remaining_budget(self, budget_id: str) -> dict[str, Any] | None:
        budget = self.budget.get(budget_id)
        if budget is None:
            budget = self.budget.set_budget(budget_id)
        return budget.remaining()


def load_contract(path: Path) -> ExperimentContract:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentContract.model_validate(data)


_SERVICE: ExperimentService | None = None
_SERVICE_LOCK = threading.Lock()


def get_shared_service() -> ExperimentService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = ExperimentService()
        return _SERVICE
