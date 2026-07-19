from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentNode,
    ResearchProject,
    utc_now_iso,
)
from scientist_lab.domain.results import ExecutionResult
from scientist_lab.runners.local_docker import LocalDockerRunner
from scientist_lab.settings import Settings, get_settings
from scientist_lab.storage.database import init_db
from scientist_lab.storage.repositories import Repository
from scientist_lab.datasets.registry import DatasetRegistry, parse_dataset_reference


class ExperimentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_factory = init_db(str(self.settings.db_path))
        self.repo = Repository(self.session_factory)
        self.datasets = DatasetRegistry(
            self.session_factory,
            project_root=Path(self.settings.project_root),
        )
        code_roots = {
            "local:experiment_app": Path(self.settings.experiment_app_dir),
            "local:rgbt_detector": Path(self.settings.rgbt_detector_dir),
        }
        self.runner = LocalDockerRunner(
            experiment_app_dir=self.settings.experiment_app_dir,
            runtime_root=self.settings.runtime_dir,
            outputs_root=self.settings.outputs_dir,
            image_registry=self.settings.image_registry,
            poll_interval_seconds=self.settings.poll_interval_seconds,
            code_roots=code_roots,
            dataset_resolver=self._resolve_dataset_reference,
        )
        self._watchers: dict[str, threading.Thread] = {}

    def _resolve_dataset_reference(self, dataset_reference: str):
        key = parse_dataset_reference(dataset_reference)
        if key is None:
            return None
        return self.datasets.require(key)

    def submit_contract(self, contract: ExperimentContract) -> ExecutionResult:
        """异步提交：立即返回 execution_id，后台落库最终结果。"""
        return self.run_contract(contract, wait=False)

    def run_contract(
        self,
        contract: ExperimentContract,
        wait: bool = True,
    ) -> ExecutionResult:
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

        image_name = self.settings.image_registry[contract.environment_key]
        attempt_index = self.repo.next_attempt_index(contract.node_id)

        submission = self.runner.submit(contract)
        meta = self.runner.get_job_meta(submission.execution_id)

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

        self.runner.wait_until_done(
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
                self.runner.wait_until_done(
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
        try:
            live = self.runner.get_status(execution_id)
            if live.status not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }:
                self.runner.wait_until_done(execution_id, timeout_seconds=5)
        except KeyError:
            # Process restarted; use DB record only.
            return ExecutionResult(
                execution_id=execution_id,
                status=attempt.status,
                metrics=(attempt.result_json or {}).get("metrics", {}),
                output_directory=(attempt.result_json or {}).get("output_directory"),
                error=None,
            )

        result = self.runner.collect_result(execution_id)
        meta = self.runner.get_job_meta(execution_id)

        attempt.status = result.status
        attempt.container_id = meta.get("container_id")
        attempt.completed_at = meta.get("completed_at")
        prev_contract = (attempt.result_json or {}).get("contract")
        attempt.result_json = {
            "metrics": result.metrics,
            "return_code": result.return_code,
            "output_directory": result.output_directory,
            "contract": prev_contract,
        }
        if result.error is not None:
            attempt.error_json = result.error.model_dump()
        self.repo.upsert_attempt(attempt)

        existing = {a.relative_path for a in self.repo.list_artifacts(execution_id)}
        for artifact in result.artifacts:
            if artifact.relative_path not in existing:
                self.repo.add_artifact(artifact)

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

    def refresh_execution(self, execution_id: str) -> ExecutionAttempt:
        """同步一次运行中状态到 SQLite，完成后尝试 finalize。"""
        attempt = self.repo.get_attempt(execution_id)
        if attempt is None:
            raise KeyError(f"未找到 execution: {execution_id}")

        try:
            live = self.runner.get_status(execution_id)
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
            items.append(
                {
                    "project_id": project.project_id,
                    "title": project.title,
                    "research_goal": project.research_goal,
                    "status": str(project.status),
                    "node_count": self.repo.count_nodes(project.project_id),
                    "latest_execution": None
                    if latest is None
                    else {
                        "execution_id": latest.execution_id,
                        "status": str(latest.status),
                        "node_id": latest.node_id,
                        "created_at": latest.created_at,
                    },
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                }
            )
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
            live = self.runner.get_status(execution_id)
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
            self.runner.cancel(execution_id)
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
    ) -> dict[str, Any]:
        if not seeds:
            raise ValueError("seeds 不能为空")
        results = []
        for seed in seeds:
            seeded = contract.model_copy(deep=True)
            seeded.seed = int(seed)
            result = self.run_contract(seeded, wait=True)
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
        }
        if auto_aggregate:
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
            payload = annotate_feedback_for_detection(
                payload, execution_mode=str(mode)
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
    ) -> dict[str, Any]:
        from scientist_lab.domain.models import utc_now_iso
        from scientist_lab.storage.artifact_store import write_json

        selected = self.repo.get_node(selected_node_id)
        if selected is None:
            raise KeyError(f"未找到 node: {selected_node_id}")

        from scientist_lab.domain.models import new_id

        payload = {
            "decision_id": new_id("decision"),
            "selected_node_id": selected_node_id,
            "decision_type": decision_type,
            "alternatives": alternatives,
            "reason": reason,
            "evidence_strength": evidence_strength,
            "baseline_node_id": baseline_node_id,
            "candidate_node_id": candidate_node_id,
            "recorded_at": utc_now_iso(),
        }

        out_dir = (
            self.settings.outputs_dir
            / selected.project_id
            / "decisions"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"decision_{selected_node_id}_{utc_now_iso().replace(':', '')}.json"
        write_json(path, payload)

        feedback = dict(selected.feedback_json or {})
        feedback["latest_decision"] = payload
        selected.feedback_json = feedback
        selected.updated_at = utc_now_iso()
        self.repo.upsert_node(selected)

        payload["decision_path"] = str(path)
        return payload

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
