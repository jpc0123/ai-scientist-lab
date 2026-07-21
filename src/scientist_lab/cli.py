from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scientist_lab.iteration.service import IterationService
from scientist_lab.iteration.workflow import InvalidIterationTransition
from scientist_lab.protocols.verifier import ProtocolViolationError
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

    triad_parser = sub.add_parser(
        "compare-fast-eval-triad",
        help="Compare RGB / Thermal / Fusion Fast Eval nodes (exploratory)",
    )
    triad_parser.add_argument("--rgb-node", default="rgbt_fast_node_001")
    triad_parser.add_argument("--thermal-node", default="rgbt_fast_node_002")
    triad_parser.add_argument("--fusion-node", default="rgbt_fast_node_003")
    triad_parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write comparison JSON under outputs/_comparisons",
    )

    validate_triad = sub.add_parser(
        "validate-fast-eval-triad",
        help="Validate triad Fast Eval contracts for matched budget (no run)",
    )
    validate_triad.add_argument(
        "contracts",
        nargs="*",
        help="Optional contract JSON paths (default: examples/rgbt_fast_*.json)",
    )

    validate_formal = sub.add_parser(
        "validate-formal-triad",
        help="Validate formal RGB/Thermal/Fusion contracts against ExperimentProtocol",
    )
    validate_formal.add_argument(
        "contracts",
        nargs="*",
        help="Optional contract JSON paths (default: examples/rgbt_formal_*.json)",
    )
    validate_formal.add_argument(
        "--protocol-id",
        default="protocol_rgbt_001",
        help="Protocol id to validate against",
    )
    validate_formal.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Optional protocol JSON to create/upsert before validation",
    )

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
    decision_parser.add_argument(
        "--evidence-ids",
        default=None,
        help="Comma-separated EvidenceRecord ids to attach",
    )
    decision_parser.add_argument("--protocol-id", default=None)
    decision_parser.add_argument(
        "--claim-matrix",
        default=None,
        help="Optional path to claim_support_matrix.json",
    )
    decision_parser.add_argument(
        "--no-auto-evidence",
        action="store_true",
        help="Do not auto-attach project evidence / auto-build claim matrix path",
    )

    list_decisions_parser = sub.add_parser(
        "list-decisions", help="List recorded node-selection decisions"
    )
    list_decisions_parser.add_argument("--project-id")
    list_decisions_parser.add_argument("--limit", type=int, default=50)

    show_decision_parser = sub.add_parser(
        "show-decision", help="Show one recorded decision by decision_id"
    )
    show_decision_parser.add_argument("decision_id")

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
            "Default: smoke RGB-T uses 42; fast_eval RGB-T uses 42,43,44; "
            "other tasks use 42,43,44,45,46."
        ),
    )

    iterate_from_plan = sub.add_parser(
        "iterate-from-plan",
        help="Create waiting_approval IterationSession from an approved plan candidate",
    )
    iterate_from_plan.add_argument("plan_id")
    iterate_from_plan.add_argument("candidate_id")
    iterate_from_plan.add_argument("--seeds", default=None)

    prepare_fast_eval = sub.add_parser(
        "prepare-fast-eval",
        help=(
            "From a completed smoke node, write a fast_eval contract "
            "ready for multi-seed run (decision bridge: ready_for_fast_eval)"
        ),
    )
    prepare_fast_eval.add_argument("node_id")
    prepare_fast_eval.add_argument(
        "--execution-id",
        default=None,
        help="Optional smoke execution_id (default: best completed attempt)",
    )
    prepare_fast_eval.add_argument(
        "--node-id",
        dest="fast_eval_node_id",
        default=None,
        help="Override node_id written into the fast_eval contract",
    )

    iterate_status = sub.add_parser(
        "iterate-status", help="Show one IterationSession status"
    )
    iterate_status.add_argument("iteration_id")
    iterate_status.add_argument(
        "--refresh",
        action="store_true",
        help="If status=running, refresh seed executions and auto-advance when done",
    )

    list_iterations = sub.add_parser(
        "list-iterations", help="List IterationSessions"
    )
    list_iterations.add_argument("--project-id")
    list_iterations.add_argument("--limit", type=int, default=50)

    iterate_approve = sub.add_parser(
        "iterate-approve",
        help=(
            "Approve proposal and run seeds (remote profiles submit async by default)"
        ),
    )
    iterate_approve.add_argument("iteration_id")
    iterate_approve.add_argument(
        "--seeds",
        default=None,
        help="Optional seed override, e.g. 42,43,44,45,46",
    )
    wait_group = iterate_approve.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--wait",
        action="store_true",
        help="Block until all seed runs finish (local default)",
    )
    wait_group.add_argument(
        "--async-submit",
        action="store_true",
        help="Submit seeds without waiting (remote default)",
    )

    iterate_advance = sub.add_parser(
        "iterate-advance",
        help="Refresh running seed executions and auto-compare when complete",
    )
    iterate_advance.add_argument("iteration_id")

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
    iterate_finalize.add_argument(
        "--evidence-ids",
        default=None,
        help="Comma-separated EvidenceRecord ids to attach",
    )
    iterate_finalize.add_argument("--protocol-id", default=None)
    iterate_finalize.add_argument(
        "--no-auto-evidence",
        action="store_true",
        help="Do not auto-attach project evidence",
    )

    register_dataset = sub.add_parser(
        "register-dataset", help="Register a host dataset for read-only Docker mounts"
    )
    register_dataset.add_argument("--key", required=True)
    register_dataset.add_argument("--task-type", required=True)
    register_dataset.add_argument("--path", required=True)
    register_dataset.add_argument("--container-path", default=None)

    register_runner = sub.add_parser(
        "register-runner-profile", help="Register a local/remote runner profile"
    )
    register_runner.add_argument("--key", required=True)
    register_runner.add_argument(
        "--type",
        dest="runner_type",
        required=True,
        choices=["local_docker", "remote_docker"],
    )
    register_runner.add_argument("--endpoint", default=None)
    register_runner.add_argument("--auth-token-env", default=None)
    register_runner.add_argument(
        "--allowed-environments",
        default="",
        help="Comma-separated environment keys",
    )
    register_runner.add_argument("--timeout-seconds", type=int, default=7200)

    list_runners = sub.add_parser("list-runner-profiles", help="List runner profiles")
    _ = list_runners

    show_runner = sub.add_parser("show-runner-profile", help="Show one runner profile")
    show_runner.add_argument("profile_key")

    check_runner = sub.add_parser(
        "check-runner", help="Ping a runner profile (health/capabilities)"
    )
    check_runner.add_argument("profile_key")

    list_ckpts = sub.add_parser("list-checkpoints", help="List registered checkpoints")
    list_ckpts.add_argument("--project-id", default=None)
    list_ckpts.add_argument("--execution-id", default=None)

    show_ckpt = sub.add_parser("show-checkpoint", help="Show one checkpoint record")
    show_ckpt.add_argument("checkpoint_id")

    verify_ckpt = sub.add_parser(
        "verify-checkpoint", help="Re-verify checkpoint file integrity"
    )
    verify_ckpt.add_argument("checkpoint_id")

    register_ckpt = sub.add_parser(
        "register-checkpoint", help="Register a checkpoint under an execution output"
    )
    register_ckpt.add_argument("--project-id", required=True)
    register_ckpt.add_argument("--execution-id", required=True)
    register_ckpt.add_argument("--relative-path", required=True)
    register_ckpt.add_argument("--node-id", default=None)
    register_ckpt.add_argument("--role", default=None)
    register_ckpt.add_argument("--baseline-key", default=None)

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

    create_protocol = sub.add_parser(
        "create-protocol", help="Create or upsert an ExperimentProtocol from JSON"
    )
    create_protocol.add_argument("protocol", type=Path)

    list_protocols = sub.add_parser("list-protocols", help="List ExperimentProtocols")
    list_protocols.add_argument("--project-id", default=None)

    show_protocol = sub.add_parser("show-protocol", help="Show one ExperimentProtocol")
    show_protocol.add_argument("protocol_id")

    validate_protocol = sub.add_parser(
        "validate-protocol",
        help="Validate a protocol (optionally against a contract)",
    )
    validate_protocol.add_argument("protocol_id")
    validate_protocol.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Optional contract JSON to check against the protocol",
    )

    create_ablation = sub.add_parser(
        "create-ablation", help="Create or upsert an AblationPlan from JSON"
    )
    create_ablation.add_argument("ablation", type=Path)

    list_ablations = sub.add_parser("list-ablations", help="List AblationPlans")
    list_ablations.add_argument("--project-id", default=None)
    list_ablations.add_argument("--protocol-id", default=None)

    show_ablation = sub.add_parser("show-ablation", help="Show one AblationPlan")
    show_ablation.add_argument("ablation_id")

    validate_ablation = sub.add_parser(
        "validate-ablation", help="Validate an AblationPlan against its protocol"
    )
    validate_ablation.add_argument("ablation_id")

    materialize_ablation = sub.add_parser(
        "materialize-ablation",
        help="Generate variant contracts from an AblationPlan + reference contract",
    )
    materialize_ablation.add_argument("ablation_id")
    materialize_ablation.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference contract JSON (default: examples/rgbt_formal_fusion_contract.json)",
    )
    materialize_ablation.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write variant contracts",
    )

    build_evidence = sub.add_parser(
        "build-evidence",
        help="Build EvidenceRecords from a paired node-group comparison",
    )
    build_evidence.add_argument("node_id_a")
    build_evidence.add_argument("node_id_b")
    build_evidence.add_argument(
        "--no-resource",
        action="store_true",
        help="Do not also emit a resource_comparison evidence record",
    )

    list_evidence = sub.add_parser("list-evidence", help="List EvidenceRecords")
    list_evidence.add_argument("--project-id", default=None)
    list_evidence.add_argument("--protocol-id", default=None)
    list_evidence.add_argument("--evidence-type", default=None)

    show_evidence = sub.add_parser("show-evidence", help="Show one EvidenceRecord")
    show_evidence.add_argument("evidence_id")

    build_claim_matrix = sub.add_parser(
        "build-claim-matrix",
        help="Build Claim Support Matrix from project EvidenceRecords",
    )
    build_claim_matrix.add_argument("project_id")
    build_claim_matrix.add_argument("--protocol-id", default=None)

    show_claim_matrix = sub.add_parser(
        "show-claim-matrix",
        help="Show Claim Support Matrix for a project",
    )
    show_claim_matrix.add_argument("project_id")

    plan_next = sub.add_parser(
        "plan-next",
        help="Generate next-experiment plan (MockPlanner by default)",
    )
    plan_next.add_argument("project_id")
    plan_next.add_argument("--protocol-id", default=None)
    plan_next.add_argument("--best-node", default=None, help="current_best_node_id")
    plan_next.add_argument("--max-new-nodes", type=int, default=3)
    plan_next.add_argument("--max-gpu-hours", type=float, default=12.0)
    plan_next.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly use MockPlanner (default in v1.0.1)",
    )

    list_plans = sub.add_parser("list-plans", help="List experiment plans")
    list_plans.add_argument("--project-id", default=None)

    show_plan = sub.add_parser("show-plan", help="Show one experiment plan")
    show_plan.add_argument("plan_id")

    review_plan = sub.add_parser(
        "review-plan", help="Run Critic review on verified candidates"
    )
    review_plan.add_argument("plan_id")

    rank_candidates = sub.add_parser(
        "rank-candidates", help="Rank reviewed/verified candidates"
    )
    rank_candidates.add_argument("plan_id")

    approve_candidate = sub.add_parser(
        "approve-candidate", help="Human-approve a candidate"
    )
    approve_candidate.add_argument("plan_id")
    approve_candidate.add_argument("candidate_id")

    reject_candidate = sub.add_parser(
        "reject-candidate", help="Human-reject a candidate"
    )
    reject_candidate.add_argument("plan_id")
    reject_candidate.add_argument("candidate_id")
    reject_candidate.add_argument("--reason", default="")

    generate_contract = sub.add_parser(
        "generate-contract",
        help="Generate ExperimentContract draft from an approved candidate",
    )
    generate_contract.add_argument("plan_id")
    generate_contract.add_argument("candidate_id")

    set_budget = sub.add_parser("set-budget", help="Set project experiment budget")
    set_budget.add_argument("project_id")
    set_budget.add_argument("--max-new-nodes", type=int, default=3)
    set_budget.add_argument("--max-executions", type=int, default=15)
    set_budget.add_argument("--max-gpu-hours", type=float, default=10.0)
    set_budget.add_argument("--max-storage-gb", type=float, default=20.0)

    show_budget = sub.add_parser("show-budget", help="Show project budget usage")
    show_budget.add_argument("project_id")

    tree_create = sub.add_parser(
        "tree-create", help="Create a finite experiment search tree (v1.1.1)"
    )
    tree_create.add_argument("project_id")
    tree_create.add_argument("--root-node", required=True)
    tree_create.add_argument("--protocol", required=True)
    tree_create.add_argument("--max-depth", type=int, default=3)
    tree_create.add_argument("--max-nodes", type=int, default=8)
    tree_create.add_argument("--max-children", type=int, default=3)

    tree_status = sub.add_parser("tree-status", help="Show experiment tree status")
    tree_status.add_argument("tree_id")

    tree_show = sub.add_parser(
        "tree-show", help="Show experiment tree with ASCII layout"
    )
    tree_show.add_argument("tree_id")

    tree_nodes = sub.add_parser("tree-nodes", help="List tree nodes")
    tree_nodes.add_argument("tree_id")

    tree_score = sub.add_parser(
        "tree-score",
        help="Compute node_score and expansion_priority for a tree (v1.1.2)",
    )
    tree_score.add_argument("tree_id")
    tree_score.add_argument(
        "--node",
        dest="tree_node_id",
        default=None,
        help="Optional tree_node_id; score only that node",
    )

    tree_select = sub.add_parser(
        "tree-select-parent",
        help="Best-First select next expandable parent (v1.1.3)",
    )
    tree_select.add_argument("tree_id")
    tree_select.add_argument(
        "--no-rescore",
        action="store_true",
        help="Do not recompute scores before selection",
    )

    tree_plan = sub.add_parser(
        "tree-plan-next",
        help="Select parent and run MockPlanner plan-next/review/rank (v1.1.4)",
    )
    tree_plan.add_argument("tree_id")
    tree_plan.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly use MockPlanner (default in v1.1.4)",
    )
    tree_plan.add_argument(
        "--no-rescore",
        action="store_true",
        help="Do not recompute scores before parent selection",
    )
    tree_plan.add_argument("--max-gpu-hours", type=float, default=12.0)

    tree_approve = sub.add_parser(
        "tree-approve",
        help="Approve a ranked candidate into IterationSession (v1.1.5)",
    )
    tree_approve.add_argument("tree_id")
    tree_approve.add_argument("candidate_id")
    tree_approve.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seeds for the iteration",
    )

    tree_advance = sub.add_parser(
        "tree-advance",
        help="Backfill Iteration results into the experiment tree (v1.1.6)",
    )
    tree_advance.add_argument("tree_id")
    tree_advance.add_argument(
        "--node",
        dest="tree_node_id",
        default=None,
        help="Optional tree_node_id to advance only one node",
    )

    tree_stop = sub.add_parser(
        "tree-stop",
        help="Manually stop an experiment tree (v1.1.7)",
    )
    tree_stop.add_argument("tree_id")
    tree_stop.add_argument("--reason", required=True)

    tree_evidence = sub.add_parser(
        "tree-evidence",
        help="Show Evidence / Claim Matrix linkage for a tree (v1.1.8)",
    )
    tree_evidence.add_argument("tree_id")

    tree_export = sub.add_parser(
        "tree-export",
        help="Export experiment tree as JSON or Mermaid (v1.1.9)",
    )
    tree_export.add_argument("tree_id")
    tree_export.add_argument(
        "--format",
        choices=["json", "mermaid"],
        default="json",
        help="Export format (default: json)",
    )
    tree_export.add_argument(
        "--output",
        default=None,
        help="Optional file path to write the export payload",
    )

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
        try:
            result = service.run_contract(contract, wait=True)
        except ProtocolViolationError as exc:
            print(
                json.dumps(
                    exc.report.model_dump(mode="json"), ensure_ascii=False, indent=2
                )
            )
            print(str(exc), file=sys.stderr)
            return 1
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
        try:
            data = service.run_seeds(
                contract, seeds, auto_aggregate=not args.no_aggregate
            )
        except ProtocolViolationError as exc:
            print(
                json.dumps(
                    exc.report.model_dump(mode="json"), ensure_ascii=False, indent=2
                )
            )
            print(str(exc), file=sys.stderr)
            return 1
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

    if args.command == "compare-fast-eval-triad":
        try:
            data = service.compare_fast_eval_triad(
                rgb_node_id=args.rgb_node,
                thermal_node_id=args.thermal_node,
                fusion_node_id=args.fusion_node,
                write_report=not args.no_write,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("fairness", {}).get("ok") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "validate-fast-eval-triad":
        try:
            data = service.validate_fast_eval_triad_contracts(
                list(args.contracts) if args.contracts else None
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok") else 1
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "validate-formal-triad":
        try:
            data = service.validate_formal_triad_contracts(
                list(args.contracts) if args.contracts else None,
                protocol_id=args.protocol_id,
                protocol_path=args.protocol,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok") or data.get("valid") else 1
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
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
            evidence_ids = None
            if args.evidence_ids:
                evidence_ids = [
                    part.strip()
                    for part in str(args.evidence_ids).split(",")
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
                supporting_evidence_ids=evidence_ids,
                protocol_id=args.protocol_id,
                claim_matrix_path=args.claim_matrix,
                auto_attach_evidence=not args.no_auto_evidence,
                ensure_claim_matrix=not args.no_auto_evidence,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-decisions":
        data = service.list_decisions(project_id=args.project_id, limit=args.limit)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "show-decision":
        try:
            print(
                json.dumps(
                    service.show_decision(args.decision_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

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

    if args.command == "iterate-from-plan":
        try:
            seeds = _parse_seeds(args.seeds) if args.seeds is not None else None
            data = iteration.start_from_plan(
                args.plan_id, args.candidate_id, seeds=seeds
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "failed" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "iterate-status":
        try:
            data = iteration.get_status(
                args.iteration_id, refresh=bool(args.refresh)
            )
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
            wait: bool | None = None
            if args.wait:
                wait = True
            elif args.async_submit:
                wait = False
            data = iteration.approve_and_run(
                args.iteration_id,
                seeds=_parse_seeds(args.seeds),
                wait=wait,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "failed" else 1
        except (KeyError, ValueError, FileNotFoundError, InvalidIterationTransition) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "iterate-advance":
        try:
            data = iteration.advance_iteration(args.iteration_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "failed" else 1
        except (KeyError, InvalidIterationTransition) as exc:
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
            evidence_ids = None
            if args.evidence_ids:
                evidence_ids = [
                    part.strip()
                    for part in str(args.evidence_ids).split(",")
                    if part.strip()
                ]
            data = iteration.finalize(
                args.iteration_id,
                selected_node_id=args.selected,
                decision_type=args.decision_type,
                reason=args.reason,
                evidence_strength=args.evidence_strength,
                supporting_evidence_ids=evidence_ids,
                protocol_id=args.protocol_id,
                auto_attach_evidence=not args.no_auto_evidence,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, InvalidIterationTransition) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "prepare-fast-eval":
        try:
            data = service.prepare_fast_eval(
                args.node_id,
                execution_id=args.execution_id,
                node_id_override=args.fast_eval_node_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, FileNotFoundError) as exc:
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

    if args.command == "register-runner-profile":
        try:
            envs = [
                part.strip()
                for part in str(args.allowed_environments).split(",")
                if part.strip()
            ]
            data = service.register_runner_profile(
                profile_key=args.key,
                runner_type=args.runner_type,
                endpoint=args.endpoint,
                auth_token_env=args.auth_token_env,
                allowed_environments=envs,
                default_timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-runner-profiles":
        print(json.dumps(service.list_runner_profiles(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "show-runner-profile":
        try:
            print(
                json.dumps(
                    service.show_runner_profile(args.profile_key),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "check-runner":
        data = service.check_runner(args.profile_key)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data.get("ok") else 1

    if args.command == "list-checkpoints":
        print(
            json.dumps(
                service.list_checkpoints(
                    project_id=args.project_id,
                    execution_id=args.execution_id,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "show-checkpoint":
        try:
            print(
                json.dumps(
                    service.show_checkpoint(args.checkpoint_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "verify-checkpoint":
        try:
            data = service.verify_checkpoint(args.checkpoint_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "register-checkpoint":
        try:
            data = service.register_checkpoint(
                project_id=args.project_id,
                execution_id=args.execution_id,
                relative_path=args.relative_path,
                node_id=args.node_id,
                role=args.role,
                baseline_key=args.baseline_key,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (FileNotFoundError, ValueError, KeyError) as exc:
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

    if args.command == "create-protocol":
        try:
            data = service.create_protocol(args.protocol)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-protocols":
        print(
            json.dumps(
                service.list_protocols(project_id=args.project_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "show-protocol":
        try:
            print(
                json.dumps(
                    service.show_protocol(args.protocol_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "validate-protocol":
        try:
            data = service.validate_protocol(
                args.protocol_id, contract_path=args.contract
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid") else 1
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "create-ablation":
        try:
            data = service.create_ablation(args.ablation)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-ablations":
        print(
            json.dumps(
                service.list_ablations(
                    project_id=args.project_id,
                    protocol_id=args.protocol_id,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "show-ablation":
        try:
            print(
                json.dumps(
                    service.show_ablation(args.ablation_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "validate-ablation":
        try:
            data = service.validate_ablation(args.ablation_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "materialize-ablation":
        try:
            data = service.materialize_ablation(
                args.ablation_id,
                reference_contract=args.reference,
                output_dir=args.output_dir,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "build-evidence":
        try:
            data = service.build_evidence(
                args.node_id_a,
                args.node_id_b,
                include_resource_evidence=not args.no_resource,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-evidence":
        print(
            json.dumps(
                service.list_evidence(
                    project_id=args.project_id,
                    protocol_id=args.protocol_id,
                    evidence_type=args.evidence_type,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "show-evidence":
        try:
            print(
                json.dumps(
                    service.show_evidence(args.evidence_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "build-claim-matrix":
        try:
            data = service.build_claim_matrix(
                args.project_id,
                protocol_id=args.protocol_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "show-claim-matrix":
        try:
            print(
                json.dumps(
                    service.show_claim_matrix(args.project_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "plan-next":
        try:
            data = service.plan_next(
                args.project_id,
                protocol_id=args.protocol_id,
                current_best_node_id=args.best_node,
                max_new_nodes=args.max_new_nodes,
                max_gpu_hours=args.max_gpu_hours,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "planner_failed" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "list-plans":
        print(
            json.dumps(
                service.list_plans(project_id=args.project_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "show-plan":
        try:
            print(
                json.dumps(
                    service.show_plan(args.plan_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "review-plan":
        try:
            data = service.review_plan(args.plan_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "rank-candidates":
        try:
            data = service.rank_candidates(args.plan_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "approve-candidate":
        try:
            data = service.approve_candidate(args.plan_id, args.candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "reject-candidate":
        try:
            data = service.reject_candidate(
                args.plan_id, args.candidate_id, reason=args.reason or None
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "generate-contract":
        try:
            data = service.generate_contract_from_plan(
                args.plan_id, args.candidate_id
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "set-budget":
        data = service.set_budget(
            args.project_id,
            max_new_nodes=args.max_new_nodes,
            max_executions=args.max_executions,
            max_gpu_hours=args.max_gpu_hours,
            max_storage_gb=args.max_storage_gb,
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "show-budget":
        print(
            json.dumps(
                service.show_budget(args.project_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "tree-create":
        try:
            data = service.tree_create(
                args.project_id,
                root_node_id=args.root_node,
                protocol_id=args.protocol,
                max_depth=args.max_depth,
                max_nodes=args.max_nodes,
                max_children=args.max_children,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-status":
        try:
            print(
                json.dumps(
                    service.tree_status(args.tree_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-show":
        try:
            data = service.tree_show(args.tree_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-nodes":
        try:
            print(
                json.dumps(
                    service.tree_nodes(args.tree_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-score":
        try:
            data = service.tree_score(
                args.tree_id, tree_node_id=args.tree_node_id
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-select-parent":
        try:
            data = service.tree_select_parent(
                args.tree_id, rescore=not args.no_rescore
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0 if data.get("selected") else 2
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-plan-next":
        try:
            data = service.tree_plan_next(
                args.tree_id,
                rescore=not args.no_rescore,
                max_gpu_hours=args.max_gpu_hours,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0 if data.get("status") == "planned" else 2
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-approve":
        try:
            seeds = _parse_seeds(args.seeds) if args.seeds is not None else None
            data = service.tree_approve(
                args.tree_id, args.candidate_id, seeds=seeds
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0 if data.get("iteration_id") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-advance":
        try:
            data = service.tree_advance(
                args.tree_id, tree_node_id=args.tree_node_id
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0 if data.get("status") in {"advanced", "pending", "noop", "stopped"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-stop":
        try:
            data = service.tree_stop(args.tree_id, reason=args.reason)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-evidence":
        try:
            data = service.tree_evidence(args.tree_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "tree-export":
        try:
            data = service.tree_export(args.tree_id, format=args.format)
            text = json.dumps(data, ensure_ascii=False, indent=2)
            if args.output:
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                if args.format == "mermaid":
                    out.write_text(str(data.get("mermaid") or "") + "\n", encoding="utf-8")
                else:
                    out.write_text(text + "\n", encoding="utf-8")
                print(json.dumps({"written": str(out), "format": args.format}, indent=2))
            else:
                print(text)
                if args.format == "mermaid" and data.get("mermaid"):
                    print("\n" + data["mermaid"])
                elif data.get("ascii_tree"):
                    print("\n" + data["ascii_tree"])
            return 0
        except (KeyError, ValueError) as exc:
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
