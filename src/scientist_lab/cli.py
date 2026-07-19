from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scientist_lab.iteration.service import IterationService
from scientist_lab.iteration.workflow import InvalidIterationTransition
from scientist_lab.services.experiment_service import ExperimentService, load_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scientist-lab",
        description="Local Docker experiment execution base",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run an experiment from contract JSON")
    run_parser.add_argument("contract", type=Path)

    run_seeds_parser = sub.add_parser(
        "run-seeds", help="Run the same contract across multiple seeds on one node"
    )
    run_seeds_parser.add_argument("contract", type=Path)
    run_seeds_parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seeds, e.g. 42,43,44,45,46",
    )
    run_seeds_parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not auto-aggregate after all seeds finish",
    )

    list_parser = sub.add_parser("list-executions", help="List recent executions")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--project-id")
    list_parser.add_argument("--node-id")

    show_parser = sub.add_parser("show-execution", help="Show one execution")
    show_parser.add_argument("execution_id")

    aggregate_parser = sub.add_parser(
        "aggregate-node", help="Aggregate completed seed runs for a node"
    )
    aggregate_parser.add_argument("node_id")

    compare_parser = sub.add_parser("compare-executions", help="Compare two executions")
    compare_parser.add_argument("execution_id_a")
    compare_parser.add_argument("execution_id_b")

    compare_nodes_parser = sub.add_parser(
        "compare-nodes", help="Compare best completed attempts of two nodes"
    )
    compare_nodes_parser.add_argument("node_id_a")
    compare_nodes_parser.add_argument("node_id_b")

    compare_groups_parser = sub.add_parser(
        "compare-node-groups",
        help="Paired multi-seed comparison between two nodes",
    )
    compare_groups_parser.add_argument("node_id_a")
    compare_groups_parser.add_argument("node_id_b")

    analyze_parser = sub.add_parser(
        "analyze-feedback",
        help="Rule-based feedback analysis from node-group comparison",
    )
    analyze_parser.add_argument("node_id_a")
    analyze_parser.add_argument("node_id_b")

    show_feedback_parser = sub.add_parser(
        "show-feedback", help="Show latest feedback stored on a node"
    )
    show_feedback_parser.add_argument("node_id")

    propose_parser = sub.add_parser(
        "propose-next",
        help="Propose next experiment contract draft (does not run it)",
    )
    propose_parser.add_argument("node_id_a")
    propose_parser.add_argument("node_id_b")

    decision_parser = sub.add_parser(
        "record-decision",
        help="Record a node-selection decision (efficiency tradeoff, etc.)",
    )
    decision_parser.add_argument("--selected", required=True, help="Selected node_id")
    decision_parser.add_argument(
        "--alternatives",
        default="",
        help="Comma-separated alternative node ids",
    )
    decision_parser.add_argument(
        "--decision-type",
        default="efficiency_tradeoff",
        help="Decision label, e.g. efficiency_tradeoff / stability_first",
    )
    decision_parser.add_argument("--reason", required=True)
    decision_parser.add_argument("--evidence-strength", default="moderate")
    decision_parser.add_argument("--baseline-node-id")
    decision_parser.add_argument("--candidate-node-id")

    list_decisions_parser = sub.add_parser(
        "list-decisions", help="List recorded node-selection decisions"
    )
    list_decisions_parser.add_argument("--project-id")
    list_decisions_parser.add_argument("--limit", type=int, default=50)

    iterate_start = sub.add_parser(
        "iterate-start",
        help="Start one controlled iteration (feedback + proposal, wait for approval)",
    )
    iterate_start.add_argument("baseline_node_id")
    iterate_start.add_argument("candidate_node_id")
    iterate_start.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seeds for approve-time run-seeds. "
            "Default: smoke RGB-T uses 42; other tasks use 42,43,44,45,46."
        ),
    )

    iterate_status = sub.add_parser(
        "iterate-status", help="Show one IterationSession status"
    )
    iterate_status.add_argument("iteration_id")

    list_iterations = sub.add_parser(
        "list-iterations", help="List IterationSessions"
    )
    list_iterations.add_argument("--project-id")
    list_iterations.add_argument("--limit", type=int, default=50)

    iterate_approve = sub.add_parser(
        "iterate-approve",
        help="Approve proposal, run seeds, aggregate, and compare both source nodes",
    )
    iterate_approve.add_argument("iteration_id")
    iterate_approve.add_argument(
        "--seeds",
        default=None,
        help="Optional seed override, e.g. 42,43,44,45,46",
    )

    iterate_reject = sub.add_parser(
        "iterate-reject", help="Reject proposal without running experiments"
    )
    iterate_reject.add_argument("iteration_id")
    iterate_reject.add_argument("--reason", default=None)

    iterate_finalize = sub.add_parser(
        "iterate-finalize",
        help="Record final node selection and complete the iteration",
    )
    iterate_finalize.add_argument("iteration_id")
    iterate_finalize.add_argument("--selected", required=True)
    iterate_finalize.add_argument("--decision-type", default="efficiency_tradeoff")
    iterate_finalize.add_argument("--reason", required=True)
    iterate_finalize.add_argument("--evidence-strength", default="moderate")

    register_dataset = sub.add_parser(
        "register-dataset", help="Register a host dataset for read-only Docker mounts"
    )
    register_dataset.add_argument("--key", required=True)
    register_dataset.add_argument("--task-type", required=True)
    register_dataset.add_argument("--path", required=True)
    register_dataset.add_argument("--container-path", default=None)

    list_datasets = sub.add_parser("list-datasets", help="List registered datasets")
    _ = list_datasets

    show_dataset = sub.add_parser("show-dataset", help="Show one registered dataset")
    show_dataset.add_argument("dataset_key")

    disable_dataset = sub.add_parser("disable-dataset", help="Disable a registered dataset")
    disable_dataset.add_argument("dataset_key")

    enable_dataset = sub.add_parser("enable-dataset", help="Enable a registered dataset")
    enable_dataset.add_argument("dataset_key")

    validate_dataset = sub.add_parser(
        "validate-dataset", help="Validate a registered dataset (pairing/annotations)"
    )
    validate_dataset.add_argument("dataset_key")

    preview_dataset = sub.add_parser(
        "preview-dataset", help="Generate RGB-T pairing preview images"
    )
    preview_dataset.add_argument("dataset_key")
    preview_dataset.add_argument("--count", type=int, default=5)

    cancel_parser = sub.add_parser("cancel-execution", help="Cancel a running execution")
    cancel_parser.add_argument("execution_id")

    serve_parser = sub.add_parser("serve", help="Start web console API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)

    doctor = sub.add_parser("doctor", help="Check Docker connectivity")
    _ = doctor

    return parser


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        try:
            service = ExperimentService()
            service.runner.client.ping()
            print("Docker OK")
            print(f"DB: {service.settings.db_path}")
            print(f"Images: {service.settings.image_registry}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Docker 检查失败: {exc}", file=sys.stderr)
            return 1

    if args.command == "serve":
        import uvicorn

        print(f"Open http://{args.host}:{args.port}")
        uvicorn.run(
            "scientist_lab.api.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    service = ExperimentService()
    iteration = IterationService(service)

    if args.command == "run":
        contract = load_contract(args.contract)
        print(f"Running project={contract.project_id} node={contract.node_id}")
        result = service.run_contract(contract, wait=True)
        print(json.dumps(
            {
                "execution_id": result.execution_id,
                "status": str(result.status),
                "return_code": result.return_code,
                "metrics": result.metrics,
                "artifact_count": len(result.artifacts),
                "output_directory": result.output_directory,
                "error": None if result.error is None else result.error.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if str(result.status) == "completed" else 1

    if args.command == "run-seeds":
        contract = load_contract(args.contract)
        seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
        print(
            f"Running seeds={seeds} project={contract.project_id} "
            f"node={contract.node_id}"
        )
        data = service.run_seeds(
            contract, seeds, auto_aggregate=not args.no_aggregate
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        failed = [r for r in data["results"] if r["status"] != "completed"]
        return 1 if failed else 0

    if args.command == "list-executions":
        attempts = service.list_executions(
            limit=args.limit, project_id=args.project_id, node_id=args.node_id
        )
        for attempt in attempts:
            print(
                f"{attempt.execution_id}\t{attempt.status}\t"
                f"node={attempt.node_id}\tattempt={attempt.attempt_index}"
            )
        return 0

    if args.command == "show-execution":
        data = service.get_execution(args.execution_id)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "aggregate-node":
        try:
            data = service.aggregate_node(args.node_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "compare-executions":
        data = service.compare_executions(
            args.execution_id_a, args.execution_id_b
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-nodes":
        try:
            data = service.compare_nodes(args.node_id_a, args.node_id_b)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "compare-node-groups":
        try:
            data = service.compare_node_groups(args.node_id_a, args.node_id_b)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "analyze-feedback":
        try:
            data = service.analyze_feedback(args.node_id_a, args.node_id_b)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "show-feedback":
        try:
            data = service.show_feedback(args.node_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "propose-next":
        try:
            data = service.propose_next(args.node_id_a, args.node_id_b)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "record-decision":
        try:
            alternatives = [
                part.strip()
                for part in str(args.alternatives).split(",")
                if part.strip()
            ]
            data = service.record_decision(
                selected_node_id=args.selected,
                alternatives=alternatives,
                decision_type=args.decision_type,
                reason=args.reason,
                evidence_strength=args.evidence_strength,
                baseline_node_id=args.baseline_node_id,
                candidate_node_id=args.candidate_node_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-decisions":
        data = service.list_decisions(project_id=args.project_id, limit=args.limit)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "iterate-start":
        try:
            seeds = _parse_seeds(args.seeds) if args.seeds is not None else None
            data = iteration.start_iteration(
                args.baseline_node_id,
                args.candidate_node_id,
                seeds=seeds,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "failed" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "iterate-status":
        try:
            data = iteration.get_status(args.iteration_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-iterations":
        data = iteration.list_iterations(
            project_id=args.project_id, limit=args.limit
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "iterate-approve":
        try:
            data = iteration.approve_and_run(
                args.iteration_id,
                seeds=_parse_seeds(args.seeds),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "failed" else 1
        except (KeyError, ValueError, FileNotFoundError, InvalidIterationTransition) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "iterate-reject":
        try:
            data = iteration.reject(args.iteration_id, reason=args.reason)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, InvalidIterationTransition) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "iterate-finalize":
        try:
            data = iteration.finalize(
                args.iteration_id,
                selected_node_id=args.selected,
                decision_type=args.decision_type,
                reason=args.reason,
                evidence_strength=args.evidence_strength,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, InvalidIterationTransition) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "register-dataset":
        try:
            data = service.register_dataset(
                dataset_key=args.key,
                task_type=args.task_type,
                path=args.path,
                container_path=args.container_path,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-datasets":
        print(json.dumps(service.list_datasets(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "show-dataset":
        try:
            print(
                json.dumps(
                    service.show_dataset(args.dataset_key),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "disable-dataset":
        try:
            print(
                json.dumps(
                    service.disable_dataset(args.dataset_key),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "enable-dataset":
        try:
            print(
                json.dumps(
                    service.enable_dataset(args.dataset_key),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "validate-dataset":
        try:
            data = service.validate_dataset(args.dataset_key)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "preview-dataset":
        try:
            data = service.preview_dataset(args.dataset_key, count=args.count)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid", True) else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "cancel-execution":
        try:
            attempt = service.cancel_execution(args.execution_id)
            print(json.dumps(attempt.model_dump(), ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
