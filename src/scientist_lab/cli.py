from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scientist_lab import __version__
from scientist_lab.iteration.workflow import InvalidIterationTransition
from scientist_lab.protocols.verifier import ProtocolViolationError


def _experiment_service():
    """Lazy: ExperimentService imports docker. Freeze manager-run must not."""
    from scientist_lab.services.experiment_service import ExperimentService

    return ExperimentService()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scientist-lab",
        description="Local-first AI Scientist workbench",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"scientist-lab {__version__}",
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

    project_create = sub.add_parser(
        "project-create",
        help="Create a research project from wizard fields (v2.0.1)",
    )
    project_create.add_argument("--title", required=True)
    project_create.add_argument("--research-question", default="")
    project_create.add_argument("--research-goal", default="")
    project_create.add_argument("--description", default="")
    project_create.add_argument(
        "--task-type",
        default="general_ml",
        choices=["general_ml", "rgbt_detection", "custom_registered_task"],
    )
    project_create.add_argument(
        "--dataset-key",
        action="append",
        default=[],
        dest="dataset_keys",
    )
    project_create.add_argument(
        "--protocol-id",
        action="append",
        default=[],
        dest="protocol_ids",
    )
    project_create.add_argument(
        "--runner-profile",
        action="append",
        default=[],
        dest="runner_profile_keys",
    )
    project_create.add_argument("--project-id", default=None)
    project_create.add_argument(
        "--keep-configuring",
        action="store_true",
        help="Leave status as configuring instead of ready",
    )

    project_list = sub.add_parser("project-list", help="List research projects (v2.0.1)")
    _ = project_list

    project_show = sub.add_parser("project-show", help="Show one research project")
    project_show.add_argument("project_id")

    project_archive = sub.add_parser(
        "project-archive", help="Archive a research project"
    )
    project_archive.add_argument("project_id")

    project_export = sub.add_parser(
        "project-export", help="Export a project JSON bundle (v2.0.10)"
    )
    project_export.add_argument("project_id")
    project_export.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write <project_id>_bundle.json",
    )

    project_import = sub.add_parser(
        "project-import", help="Import a project JSON bundle (v2.0.10)"
    )
    project_import.add_argument("path", help="Path to *_bundle.json")
    project_import.add_argument(
        "--force",
        action="store_true",
        help="Overwrite project metadata if it already exists",
    )

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
    plan_next.add_argument(
        "--provider",
        choices=["mock", "fake", "replay", "real"],
        default="mock",
        help="Planner/Critic backend: mock (default), fake, replay, or real (needs --allow-network)",
    )
    plan_next.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicit network permission for --provider real (also needs LLM_ALLOW_NETWORK)",
    )
    plan_next.add_argument(
        "--model-profile",
        default=None,
        help="LLMModelProfile id (required with --require-quality-gate unless default selected)",
    )
    plan_next.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="Require a Quality-Gate-qualified evaluation for the profile",
    )
    plan_next.add_argument(
        "--allow-unqualified-profile",
        action="store_true",
        help="Dev-only bypass for --require-quality-gate (records warning; not formally approvable)",
    )

    list_plans = sub.add_parser("list-plans", help="List experiment plans")
    list_plans.add_argument("--project-id", default=None)

    show_plan = sub.add_parser("show-plan", help="Show one experiment plan")
    show_plan.add_argument("plan_id")

    review_plan = sub.add_parser(
        "review-plan", help="Run Critic review on verified candidates"
    )
    review_plan.add_argument("plan_id")
    review_plan.add_argument(
        "--provider",
        choices=["mock", "fake", "replay", "real"],
        default=None,
        help="Optional Critic backend override (default: keep current agent mode)",
    )
    review_plan.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicit network permission for --provider real",
    )

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
        help="Select parent and run plan-next/review/rank (mock default; real needs --allow-network)",
    )
    tree_plan.add_argument("tree_id")
    tree_plan.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly use MockPlanner (default)",
    )
    tree_plan.add_argument(
        "--provider",
        choices=["mock", "fake", "replay", "real"],
        default="mock",
        help="Planner/Critic backend for tree expansion (offline fake/replay; real gated)",
    )
    tree_plan.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicit network permission for --provider real",
    )
    tree_plan.add_argument(
        "--model-profile",
        default=None,
        help="LLMModelProfile id for quality-gated real planning",
    )
    tree_plan.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="Require a Quality-Gate-qualified evaluation before planning",
    )
    tree_plan.add_argument(
        "--allow-unqualified-profile",
        action="store_true",
        help="Dev-only bypass for quality gate",
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

    report_context = sub.add_parser(
        "report-context",
        help="Build ReportContext snapshot for a project (v1.2.1)",
    )
    report_context.add_argument("project_id")
    report_context.add_argument("--tree-id", default=None)
    report_context.add_argument("--protocol-id", default=None)

    report_build = sub.add_parser(
        "report-build",
        help="Build ResearchReport JSON + Markdown (v1.2)",
    )
    report_build.add_argument("project_id")
    report_build.add_argument("--tree-id", default=None)
    report_build.add_argument("--protocol-id", default=None)

    report_show = sub.add_parser("report-show", help="Show a ResearchReport")
    report_show.add_argument("report_id")

    report_verify = sub.add_parser(
        "report-verify", help="Verify claim-gated ResearchReport"
    )
    report_verify.add_argument("report_id")

    audit_build = sub.add_parser(
        "audit-build", help="Build Audit Bundle for a project (v1.2)"
    )
    audit_build.add_argument("project_id")
    audit_build.add_argument("--tree-id", default=None)
    audit_build.add_argument("--protocol-id", default=None)
    audit_build.add_argument("--report-id", default=None)

    audit_verify = sub.add_parser("audit-verify", help="Verify an Audit Bundle")
    audit_verify.add_argument("bundle_id")

    audit_export = sub.add_parser(
        "audit-export", help="Export Audit Bundle directory to a destination"
    )
    audit_export.add_argument("bundle_id")
    audit_export.add_argument("--output", required=True)

    llm_eval = sub.add_parser(
        "llm-eval",
        help="LLM quality eval: compare modes or run a suite (v1.4.4)",
    )
    llm_eval.add_argument("project_id")
    llm_eval.add_argument("--protocol-id", default=None)
    llm_eval.add_argument("--best-node", default=None)
    llm_eval.add_argument(
        "--suite",
        default=None,
        help="Eval suite name under evals/llm/ (e.g. planner-basic). "
        "Omit to run Mock/Fake/Replay comparison.",
    )
    llm_eval.add_argument(
        "--provider",
        choices=["mock", "fake", "replay", "real"],
        default="mock",
        help="Suite provider (default: mock). Real requires gates.",
    )
    llm_eval.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicit network permission for --provider real",
    )
    llm_eval.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on suite cases",
    )
    llm_eval.add_argument(
        "--output",
        default=None,
        help="Optional path for report JSON",
    )
    llm_eval.add_argument(
        "--include-real",
        action="store_true",
        help="Include Real row in comparison mode (usually skipped)",
    )

    llm_eval_run = sub.add_parser(
        "llm-eval-run",
        help="Run versioned LLM eval suite with rule graders (v1.5.3)",
    )
    llm_eval_run.add_argument("project_id")
    llm_eval_run.add_argument(
        "--suite",
        default="eval_suite_v1",
        help="Suite manifest name under evals/llm/manifests/",
    )
    llm_eval_run.add_argument(
        "--provider",
        choices=["mock", "fake", "replay", "real"],
        default="mock",
        help="Provider backend (default: mock). Real requires --allow-network + env gates.",
    )
    llm_eval_run.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicit network permission for --provider real",
    )
    llm_eval_run.add_argument(
        "--model-profile",
        default=None,
        help="Optional LLMModelProfile id",
    )
    llm_eval_run.add_argument(
        "--no-seed-fake",
        action="store_true",
        help="For replay: do not auto-seed audit with FakeProvider first",
    )

    llm_profile_register = sub.add_parser(
        "llm-profile-register",
        help="Register an LLM model/prompt profile JSON (v1.5.4)",
    )
    llm_profile_register.add_argument("path", help="Path to profile JSON")

    llm_profile_list = sub.add_parser(
        "llm-profile-list", help="List registered LLM profiles"
    )
    llm_profile_list.add_argument(
        "--enabled-only", action="store_true", help="Only enabled profiles"
    )

    llm_profile_show = sub.add_parser(
        "llm-profile-show", help="Show one LLM profile"
    )
    llm_profile_show.add_argument("profile_id")

    llm_eval_verify = sub.add_parser(
        "llm-eval-verify",
        help="Apply Quality Gate to an evaluation scorecard (v1.5.6)",
    )
    llm_eval_verify.add_argument("evaluation_id")

    llm_eval_compare = sub.add_parser(
        "llm-eval-compare",
        help="Compare two evaluations for Prompt/model regression (v1.5.7)",
    )
    llm_eval_compare.add_argument("baseline_evaluation_id")
    llm_eval_compare.add_argument("candidate_evaluation_id")

    llm_profile_rank = sub.add_parser(
        "llm-profile-rank",
        help="Rank Quality-Gate-qualified profiles (v1.5.8; does not auto-select)",
    )
    llm_profile_rank.add_argument(
        "--suite",
        default="eval_suite_v1",
        help="Suite version used to pick latest evaluations",
    )

    llm_profile_select = sub.add_parser(
        "llm-profile-select",
        help="Human-select default LLM profile (v1.5.8)",
    )
    llm_profile_select.add_argument("profile_id")

    llm_usage = sub.add_parser(
        "llm-usage",
        help="Summarize LLM audit token / cost / latency (v1.3.9)",
    )
    llm_usage.add_argument("--project-id", default=None)
    llm_usage.add_argument(
        "--audit-root",
        default=None,
        help="Override audit root (default: outputs/<project_id>/llm)",
    )

    patch_propose = sub.add_parser(
        "patch-propose",
        help="Propose a restricted source patch (Mock; no apply) (v1.6)",
    )
    patch_propose.add_argument("project_id")
    patch_propose.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use Mock patch generator (default)",
    )
    patch_propose.add_argument("--title", default=None)
    patch_propose.add_argument("--rationale", default=None)

    patch_propose_real = sub.add_parser(
        "patch-propose-real",
        help=(
            "RealPatchPlanner from a CodeContextBundle (v2.2.2; "
            "no silent mock fallback)"
        ),
    )
    patch_propose_real.add_argument("bundle_id")
    patch_propose_real.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit live OpenAI-compatible HTTP (also needs LLM_* env)",
    )
    patch_propose_real.add_argument(
        "--provider",
        default="openai-compatible",
        help="Must be openai-compatible/real when real_only (default)",
    )
    patch_propose_real.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Override patch provider max_calls budget for this invoke",
    )

    patch_budget_show = sub.add_parser(
        "patch-budget-show",
        help="Show per-project RealPatchPlanner provider budget usage (v2.2.4)",
    )
    patch_budget_show.add_argument("project_id")

    patch_provider_doctor = sub.add_parser(
        "patch-provider-doctor",
        help="Offline readiness check for real patch provider (v2.2.4)",
    )
    patch_provider_doctor.add_argument(
        "--provider", default="openai-compatible"
    )
    patch_provider_doctor.add_argument(
        "--allow-network",
        action="store_true",
        help="Evaluate live-gate readiness (still does not call the network)",
    )

    dfine_cuda_doctor = sub.add_parser(
        "dfine-cuda-doctor",
        help="CUDA + Vendor DFINE readiness (v2.3.2; probe is optional)",
    )
    dfine_cuda_doctor.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip docker/nvidia probes (files + registry only)",
    )

    dfine_cuda_fast = sub.add_parser(
        "dfine-cuda-fast-eval",
        help=(
            "Orchestrate CUDA Vendor DFINE Fast Eval (v2.3.2); "
            "default dry-run; never claims formal success"
        ),
    )
    dfine_cuda_fast.add_argument(
        "--contract",
        default=None,
        help="Contract JSON (default: examples/rgbt_remote_cuda_dfine_rgb_contract.json)",
    )
    dfine_cuda_fast.add_argument(
        "--runner-profile",
        default=None,
        help="Override runner_profile (default: contract value)",
    )
    dfine_cuda_fast.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Plan only; do not submit (default: true)",
    )
    dfine_cuda_fast.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit via run_contract (requires GPU/image readiness)",
    )
    dfine_cuda_fast.add_argument(
        "--require-live-ready",
        action="store_true",
        help="Fail if doctor live_ready is false (ignored for dry-run)",
    )
    dfine_cuda_fast.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip docker/nvidia probes while planning",
    )
    dfine_cuda_fast.add_argument(
        "--no-wait",
        action="store_true",
        help="When executing, do not wait for completion",
    )

    dfine_adapter = sub.add_parser(
        "dfine-adapter-run",
        help=(
            "Freeze GateEngine + DFINEAdapter (default dry-run). "
            "--execute wires live_runner to CUDA orchestrator; GPU not used unless --execute"
        ),
    )
    dfine_adapter.add_argument("--protocol", required=True, type=Path)
    dfine_adapter.add_argument("--plan", required=True, type=Path)
    dfine_adapter.add_argument("--output-dir", required=True, type=Path)
    dfine_adapter.add_argument(
        "--execute",
        action="store_true",
        help="Call CUDA Fast Eval orchestrator (requires live_ready / image)",
    )
    dfine_adapter.add_argument(
        "--require-live-ready",
        action="store_true",
        help="Fail live execute if doctor live_ready is false",
    )

    manager_run = sub.add_parser(
        "manager-run",
        help=(
            "Freeze Manager state machine (default dry-run / REPLAY). "
            "--execute uses make_cuda_live_runner after Gate APPROVED; "
            "--require-live-ready refuses GPU and forged metrics when doctor is false"
        ),
    )
    manager_run.add_argument("--protocol", required=True, type=Path)
    manager_run.add_argument("--plan", required=True, type=Path)
    manager_run.add_argument("--output-dir", required=True, type=Path)
    manager_run.add_argument(
        "--execute",
        action="store_true",
        help="REAL GPU via Manager → make_cuda_live_runner (not a direct Adapter bypass)",
    )
    manager_run.add_argument(
        "--require-live-ready",
        action="store_true",
        help="Fail (exit 1) if CUDA doctor live_ready is false; do not forge metrics",
    )
    manager_run.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="Hard cap for Manager.run_until (default 32)",
    )
    manager_run.add_argument(
        "--max-extra-rounds",
        type=int,
        default=0,
        help="Extra rounds after the seed plan (default 0 = single round). 1 = two GPU rounds.",
    )
    manager_run.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Optional JSON of baseline metrics for Rubric delta (does not invent APS)",
    )
    manager_run.add_argument(
        "--planner-backend",
        choices=["rules", "llm"],
        default="rules",
        help="v2.5-D: Planner cognitive backend. Default rules (freeze-safe). llm is opt-in.",
    )
    manager_run.add_argument(
        "--reviewer-backend",
        choices=["rules", "llm"],
        default="rules",
        help="v2.5-D: Reviewer cognitive backend. Default rules. Same Gateway, not a fifth Agent.",
    )
    manager_run.add_argument(
        "--llm-live",
        action="store_true",
        help="Call a real OpenAI-compatible API for Planner/Reviewer. Independent of --execute GPU.",
    )
    manager_run.add_argument(
        "--fallback-to-rules",
        action="store_true",
        help="Opt-in: on fail-closed LLM, fall back to rules Planner/Reviewer. Default off.",
    )

    claim_gate = sub.add_parser(
        "claim-gate",
        help=(
            "Deterministic ClaimGate on an existing run directory (no GPU). "
            "KEEP ≠ Claim. Does not overwrite review_decision."
        ),
    )
    claim_gate.add_argument("--run-dir", required=True, type=Path)
    claim_gate.add_argument(
        "--claim",
        type=Path,
        default=None,
        help="Candidate claim JSON; default derives C0/C1 from plan+baseline",
    )
    claim_gate.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Optional matched baseline Manager run directory for C1 pairing",
    )
    claim_gate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write claim_gate.json",
    )

    llm_plan_replay = sub.add_parser(
        "llm-plan-replay",
        help=(
            "v2.5-A: replay LLM Planner against a historical/stub run-dir. "
            "Never executes GPU. Default provider=mock (no API key)."
        ),
    )
    llm_plan_replay.add_argument("--run-dir", required=True, type=Path)
    llm_plan_replay.add_argument(
        "--live",
        action="store_true",
        help="Call a real OpenAI-compatible API (LLM_API_KEY/LLM_BASE_URL/LLM_MODEL). Still no GPU.",
    )
    llm_plan_replay.add_argument(
        "--provider",
        default="mock",
        help="Offline provider when --live is not set (mock/fake). Default mock.",
    )
    llm_plan_replay.add_argument(
        "--fallback-to-rules",
        action="store_true",
        help="Opt-in: on fail-closed LLM, record an event then use rules Planner. Default off.",
    )
    llm_plan_replay.add_argument(
        "--ab",
        action="store_true",
        default=True,
        help="A/B rules vs LLM into replay_report.json (default on; no GPU).",
    )
    llm_plan_replay.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <run-dir>/.llm_plan_replay/replay_report.json)",
    )

    llm_review_replay = sub.add_parser(
        "llm-review-replay",
        help=(
            "v2.5-C: replay LLM Reviewer against a historical/stub run-dir. "
            "Never executes GPU. Default provider=mock (no API key). "
            "Does not overwrite DecisionRubric KEEP/DISCARD."
        ),
    )
    llm_review_replay.add_argument("--run-dir", required=True, type=Path)
    llm_review_replay.add_argument(
        "--live",
        action="store_true",
        help="Call a real OpenAI-compatible API (LLM_API_KEY/LLM_BASE_URL/LLM_MODEL). Still no GPU.",
    )
    llm_review_replay.add_argument(
        "--provider",
        default="mock",
        help="Offline provider when --live is not set (mock/fake). Default mock.",
    )
    llm_review_replay.add_argument(
        "--fallback-to-rules",
        action="store_true",
        help="Opt-in: on fail-closed LLM, keep the rules Rubric packet. Default off.",
    )
    llm_review_replay.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <run-dir>/.llm_review_replay/replay_report.json)",
    )

    llm_real_loop = sub.add_parser(
        "llm-real-loop",
        help=(
            "v2.5-D: Human-gated probe REAL loop. Historical DISCARD → LLM Reviewer "
            "→ MemoryWriter → LLM Planner → Gate → optional --execute GPU. "
            "Default dry-run. Does not bypass Gate. probe ≠ C1."
        ),
    )
    llm_real_loop.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Historical Evidence pack (default: tests/fixtures/llm_plan_replay/m4_rounds3_discard)",
    )
    llm_real_loop.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Loop output (default: .run/v25d_llm_real_loop). gitignored.",
    )
    llm_real_loop.add_argument(
        "--execute",
        action="store_true",
        help="Probe REAL GPU after Gate APPROVED. Human Gate for this probe only; formal still forbidden.",
    )
    llm_real_loop.add_argument(
        "--require-live-ready",
        action="store_true",
        help="Fail (exit 1) if CUDA doctor live_ready is false; do not forge metrics",
    )
    llm_real_loop.add_argument(
        "--planner-backend",
        choices=["rules", "llm"],
        default="llm",
        help="This command defaults to llm (opt-in loop). manager-run still defaults rules.",
    )
    llm_real_loop.add_argument(
        "--reviewer-backend",
        choices=["rules", "llm"],
        default="llm",
        help="This command defaults to llm. Same Gateway, not a fifth Agent.",
    )
    llm_real_loop.add_argument(
        "--live",
        action="store_true",
        help="LLM live API (LLM_API_KEY). Independent of --execute GPU.",
    )
    llm_real_loop.add_argument(
        "--provider",
        default="mock",
        help="Offline LLM provider when --live is not set. Default mock.",
    )
    llm_real_loop.add_argument(
        "--fallback-to-rules",
        action="store_true",
        help="Opt-in: on fail-closed LLM, fall back to rules. Default off.",
    )
    llm_real_loop.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="Hard cap for Manager.run_until (default 32)",
    )

    dfine_cuda_evidence = sub.add_parser(
        "dfine-cuda-record-evidence",
        help="Record Vendor/stand-in Fast Eval evidence from an execution (v2.3.3)",
    )
    dfine_cuda_evidence.add_argument("execution_id")
    dfine_cuda_evidence.add_argument(
        "--no-claim-matrix",
        action="store_true",
        help="Do not rebuild claim support matrix",
    )

    dfine_cuda_triad = sub.add_parser(
        "dfine-cuda-formal-triad",
        help=(
            "CUDA Vendor formal triad rgb/thermal/fusion (v2.3.4); "
            "default dry-run; never claims formal superiority"
        ),
    )
    dfine_cuda_triad.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit three contracts (requires live_ready)",
    )
    dfine_cuda_triad.add_argument(
        "--require-live-ready",
        action="store_true",
        help="Fail if doctor live_ready is false (ignored for dry-run)",
    )
    dfine_cuda_triad.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip docker/nvidia probes while planning",
    )
    dfine_cuda_triad.add_argument(
        "--no-wait",
        action="store_true",
        help="When executing, do not wait for completion",
    )
    dfine_cuda_triad.add_argument(
        "--protocol-id",
        default="protocol_rgbt_cuda_001",
        help="CUDA formal triad protocol id",
    )

    dfine_path_gate = sub.add_parser(
        "dfine-formal-path-gate",
        help=(
            "Assess Vendor formal DFINE path Claim Gate (v2.3.5); "
            "path-open ≠ superiority"
        ),
    )
    dfine_path_gate.add_argument("project_id")
    dfine_path_gate.add_argument("--protocol-id", default=None)

    dfine_real = sub.add_parser(
        "dfine-real-acceptance",
        help=(
            "Gated live DFINE acceptance pipeline plan/execute (v2.3.6); "
            "default dry-run; never claims superiority"
        ),
    )
    dfine_real.add_argument(
        "--execute",
        action="store_true",
        help="Run live pipeline (requires doctor live_ready)",
    )
    dfine_real.add_argument(
        "--include-formal-triad",
        action="store_true",
        help="Also run CUDA formal triad after Fast Eval",
    )
    dfine_real.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for execution completion",
    )
    dfine_real.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip runtime GPU/Docker probes in doctor",
    )

    patch_show = sub.add_parser("patch-show", help="Show a PatchProposal")
    patch_show.add_argument("patch_id")

    code_context_build = sub.add_parser(
        "code-context-build",
        help="Build restricted CodeContextBundle (v2.2.1; no provider call)",
    )
    code_context_build.add_argument(
        "--digits-demo",
        action="store_true",
        help="Use Digits small-improvement PatchGoal fixture",
    )
    code_context_build.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write the bundle to SQLite",
    )

    code_context_show = sub.add_parser(
        "code-context-show", help="Show a persisted CodeContextBundle"
    )
    code_context_show.add_argument("bundle_id")

    code_context_export = sub.add_parser(
        "code-context-export",
        help="Export a CodeContextBundle JSON for audit/replay",
    )
    code_context_export.add_argument("bundle_id")
    code_context_export.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default under outputs/_code_contexts/)",
    )

    patch_verify = sub.add_parser(
        "patch-verify", help="Re-run static PathPolicy/Diff verification"
    )
    patch_verify.add_argument("patch_id")

    patch_approve = sub.add_parser(
        "patch-approve",
        help="Human-approve a verified patch (does NOT apply in v1.6.1–1.6.3)",
    )
    patch_approve.add_argument("patch_id")
    patch_approve.add_argument("--reason", default="")

    patch_check_seal = sub.add_parser(
        "patch-check-seal",
        help="Validate approval content seal fingerprints (v2.2.5)",
    )
    patch_check_seal.add_argument("patch_id")

    patch_reject = sub.add_parser("patch-reject", help="Human-reject a patch")
    patch_reject.add_argument("patch_id")
    patch_reject.add_argument("--reason", default="")

    patch_apply_sandbox = sub.add_parser(
        "patch-apply-sandbox",
        help="Apply an approved patch into an isolated sandbox (never main tree) (v1.6.4)",
    )
    patch_apply_sandbox.add_argument("patch_id")
    patch_apply_sandbox.add_argument(
        "--force",
        action="store_true",
        help="Recreate sandbox if it already exists / retry failed_sandbox",
    )

    patch_test_sandbox = sub.add_parser(
        "patch-test-sandbox",
        help=(
            "Run allow-listed in-process checks on an applied sandbox "
            "(smoke|syntax|unit|mock_experiment) (v1.6.5/v2.2.6)"
        ),
    )
    patch_test_sandbox.add_argument("patch_id")
    patch_test_sandbox.add_argument(
        "--profile",
        choices=["smoke", "syntax", "unit", "mock_experiment"],
        default="smoke",
        help="Validation profile (default: smoke; registry only; never arbitrary shell)",
    )

    patch_sandbox_profiles = sub.add_parser(
        "patch-sandbox-profiles",
        help="List registered sandbox test profiles (v2.2.6; no arbitrary shell)",
    )

    patch_export_replay = sub.add_parser(
        "patch-export-replay",
        help="Export a redacted Patch Replay Bundle for offline CI (v2.2.8)",
    )
    patch_export_replay.add_argument("patch_id")
    patch_export_replay.add_argument(
        "--output",
        required=True,
        help="Empty directory for the replay bundle",
    )

    patch_replay = sub.add_parser(
        "patch-replay",
        help="Replay a Patch Replay Bundle via MockTransport (v2.2.8; zero network)",
    )
    patch_replay.add_argument("bundle_dir")

    patch_record_evidence = sub.add_parser(
        "patch-record-evidence",
        help="Record PatchEvidence from an applied sandbox (v1.6.6; no main apply)",
    )
    patch_record_evidence.add_argument("patch_id")
    patch_record_evidence.add_argument(
        "--require-tests",
        action="store_true",
        help="Fail if patch-test-sandbox has not been run",
    )
    patch_record_evidence.add_argument(
        "--feedback",
        action="store_true",
        help="Also persist Evidence/Claim/Planner/Tree feedback package (v2.2.7)",
    )

    patch_decide_merge = sub.add_parser(
        "patch-decide-merge",
        help=(
            "Human merge/discard decision after evidence (records intent only; "
            "never modifies main tree) (v1.6.6)"
        ),
    )
    patch_decide_merge.add_argument("patch_id")
    patch_decide_merge.add_argument(
        "--decision",
        choices=["merge", "discard"],
        required=True,
    )
    patch_decide_merge.add_argument("--reason", default="")

    merge_prepare = sub.add_parser(
        "merge-prepare",
        help=(
            "Create isolated Git worktree + MergeCandidate for an evidenced patch "
            "(v1.9.1; does not commit or merge)"
        ),
    )
    merge_prepare.add_argument("patch_id")
    merge_prepare.add_argument(
        "--target-branch",
        default=None,
        help="Baseline branch (default: current HEAD branch)",
    )

    merge_show = sub.add_parser(
        "merge-show",
        help="Show one MergeCandidate (v1.9.1)",
    )
    merge_show.add_argument("merge_candidate_id")

    merge_list = sub.add_parser(
        "merge-list",
        help="List MergeCandidates (v1.9.2)",
    )
    merge_list.add_argument("--project-id", default=None)
    merge_list.add_argument("--patch-id", default=None)
    merge_list.add_argument("--limit", type=int, default=50)

    merge_apply = sub.add_parser(
        "merge-apply",
        help="Apply approved patch inside isolated worktree only (v1.9.3)",
    )
    merge_apply.add_argument("merge_candidate_id")

    merge_test = sub.add_parser(
        "merge-test",
        help="Run registry TestProfile inside worktree (v1.9.4)",
    )
    merge_test.add_argument("merge_candidate_id")
    merge_test.add_argument(
        "--profile",
        default="smoke",
        choices=["syntax", "unit", "smoke", "full_regression", "acceptance"],
    )

    merge_approve = sub.add_parser(
        "merge-approve",
        help="Human approve MergeCandidate after registry tests (v1.9.5)",
    )
    merge_approve.add_argument("merge_candidate_id")
    merge_approve.add_argument("--reason", default="")

    merge_reject = sub.add_parser(
        "merge-reject",
        help="Human reject MergeCandidate (v1.9.5)",
    )
    merge_reject.add_argument("merge_candidate_id")
    merge_reject.add_argument("--reason", default="")

    merge_commit = sub.add_parser(
        "merge-commit",
        help="Create controlled commit inside worktree only (v1.9.6; no push)",
    )
    merge_commit.add_argument("merge_candidate_id")

    merge_finalize = sub.add_parser(
        "merge-finalize",
        help="Merge worktree commit into target branch with --no-ff (v1.9.6; no push)",
    )
    merge_finalize.add_argument("merge_candidate_id")
    merge_finalize.add_argument(
        "--post-merge-profile",
        default="syntax",
        choices=["syntax", "unit", "smoke", "full_regression", "acceptance", "none"],
        help="Registry profile after merge; 'none' skips post-merge check",
    )
    merge_finalize.add_argument(
        "--no-auto-rollback",
        action="store_true",
        help="Do not auto revert when post-merge check fails",
    )

    merge_rollback = sub.add_parser(
        "merge-rollback",
        help="Revert a finalized merge commit with git revert -m 1 (v1.9.7)",
    )
    merge_rollback.add_argument("merge_candidate_id")
    merge_rollback.add_argument("--reason", default="")

    rc_create = sub.add_parser(
        "release-candidate-create",
        help="Create a local Release Candidate + manifest (v1.9.8; no remote publish)",
    )
    rc_create.add_argument("--version", required=True)
    rc_create.add_argument("--project-id", default="")
    rc_create.add_argument("--base-tag", default="")
    rc_create.add_argument(
        "--merge-candidate-id",
        action="append",
        default=[],
        dest="merge_candidate_ids",
        help="Merged MergeCandidate id (repeatable)",
    )
    rc_create.add_argument("--notes", default="")

    rc_show = sub.add_parser(
        "release-candidate-show",
        help="Show one Release Candidate (v1.9.8)",
    )
    rc_show.add_argument("release_candidate_id")

    rc_verify = sub.add_parser(
        "release-candidate-verify",
        help="Verify Release Candidate manifest (v1.9.8)",
    )
    rc_verify.add_argument("release_candidate_id")

    cancel_parser = sub.add_parser("cancel-execution", help="Cancel a running execution")
    cancel_parser.add_argument("execution_id")

    serve_parser = sub.add_parser("serve", help="Start web console API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)

    doctor = sub.add_parser(
        "doctor", help="Alias of system-doctor (Docker + environment checks)"
    )
    _ = doctor

    system_doctor = sub.add_parser(
        "system-doctor", help="Full System Doctor diagnostics (v2.0.6)"
    )
    _ = system_doctor

    recover = sub.add_parser(
        "recover",
        help="Recover interrupted work after restart (never auto-reruns experiments)",
    )
    recover.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview actions without applying (default)",
    )
    recover.add_argument(
        "--apply",
        action="store_true",
        help="Apply recovery actions (mark interrupted / refresh remote)",
    )

    demo_create = sub.add_parser(
        "demo-create",
        help="Create a built-in demo project (digits | rgbt-debug) (v2.0.8)",
    )
    demo_create.add_argument(
        "kind",
        choices=["digits", "rgbt-debug"],
        help="Demo kind",
    )
    demo_create.add_argument(
        "--force",
        action="store_true",
        help="Refresh project metadata even if demo already exists",
    )

    workbench = sub.add_parser(
        "workbench", help="Local workbench start/status/stop helpers (v2.0.7)"
    )
    workbench_sub = workbench.add_subparsers(dest="workbench_command")
    wb_start = workbench_sub.add_parser(
        "start", help="Print start instructions / launch via PowerShell script"
    )
    wb_start.add_argument(
        "--print-only",
        action="store_true",
        help="Only print commands; do not invoke PowerShell",
    )
    workbench_sub.add_parser("status", help="Show API/Web listening status")
    wb_stop = workbench_sub.add_parser(
        "stop", help="Stop workbench via PowerShell script"
    )
    wb_stop.add_argument(
        "--print-only",
        action="store_true",
        help="Only print stop command",
    )

    real_loop_create = sub.add_parser(
        "real-loop-create",
        help="Create a real research loop session (v2.1.1; offline)",
    )
    real_loop_create.add_argument("project_id")
    real_loop_create.add_argument("--profile", required=True, dest="profile_id")
    real_loop_create.add_argument("--protocol", required=True, dest="protocol_id")
    real_loop_create.add_argument("--rounds", type=int, default=2)
    real_loop_create.add_argument(
        "--baseline-node",
        action="append",
        default=[],
        dest="baseline_node_ids",
        help="Baseline node id (repeatable)",
    )
    real_loop_create.add_argument("--tree-id", default=None)

    real_loop_show = sub.add_parser(
        "real-loop-show", help="Show one real research loop session (v2.1.1)"
    )
    real_loop_show.add_argument("session_id")

    real_loop_check = sub.add_parser(
        "real-loop-check",
        help="Offline integrity check for a real research loop session (v2.1.1)",
    )
    real_loop_check.add_argument("session_id")

    real_loop_list = sub.add_parser(
        "real-loop-list", help="List real research loop sessions (v2.1.1)"
    )
    real_loop_list.add_argument("--project-id", default=None)
    real_loop_list.add_argument("--limit", type=int, default=50)

    real_loop_record = sub.add_parser(
        "real-loop-record-feedback",
        help="Record RoundFeedbackSummary for a completed round (v2.1.2; offline)",
    )
    real_loop_record.add_argument("session_id")
    real_loop_record.add_argument("--parent-node", required=True, dest="parent_node_id")
    real_loop_record.add_argument(
        "--executed-node", required=True, dest="executed_node_id"
    )
    real_loop_record.add_argument("--source-round", type=int, default=1)
    real_loop_record.add_argument(
        "--comparison-json",
        default=None,
        help="Path to comparison JSON (optional)",
    )
    real_loop_record.add_argument("--hypothesis", default=None)
    real_loop_record.add_argument(
        "--evidence-id",
        action="append",
        default=[],
        dest="evidence_ids",
    )

    real_loop_exec_fb = sub.add_parser(
        "real-loop-record-execution-feedback",
        help="Build Evidence/Claim from executed Digits node and advance (v2.1.5)",
    )
    real_loop_exec_fb.add_argument("session_id")
    real_loop_exec_fb.add_argument("--source-round", type=int, default=1)
    real_loop_exec_fb.add_argument("--parent-node", default=None, dest="parent_node_id")
    real_loop_exec_fb.add_argument(
        "--executed-node", default=None, dest="executed_node_id"
    )
    real_loop_exec_fb.add_argument(
        "--no-round2-context",
        action="store_true",
        help="Skip building Round-2 PlanningContext",
    )
    real_loop_exec_fb.add_argument(
        "--no-advance",
        action="store_true",
        help="Do not advance session to feedback_ready",
    )

    real_loop_next = sub.add_parser(
        "real-loop-next-round",
        help="Advance feedback_ready → round_2_planning with context (v2.1.5)",
    )
    real_loop_next.add_argument("session_id")

    real_loop_verify = sub.add_parser(
        "real-loop-verify-feedback",
        help="Verify Round-2 plan used Round-1 feedback (v2.1.6; deterministic)",
    )
    real_loop_verify.add_argument("session_id")
    real_loop_verify.add_argument("--round", type=int, default=2, dest="round_number")
    real_loop_verify.add_argument("--plan-id", default=None, dest="plan_id")
    real_loop_verify.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write FeedbackUsageRecord",
    )

    real_loop_export = sub.add_parser(
        "real-loop-export",
        help="Export redacted Replay Bundle for offline CI (v2.1.7)",
    )
    real_loop_export.add_argument("session_id")
    real_loop_export.add_argument(
        "--output",
        default=None,
        dest="output_dir",
        help="Bundle directory (default: outputs/<project>/real_loop/<session>)",
    )
    real_loop_export.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export even before round-1 feedback is ready",
    )

    real_loop_ctx = sub.add_parser(
        "real-loop-build-context",
        help="Build PlanningContext for a loop round (v2.1.2; offline gate)",
    )
    real_loop_ctx.add_argument("session_id")
    real_loop_ctx.add_argument("--round", type=int, required=True, dest="round_number")
    real_loop_ctx.add_argument(
        "--allow-missing-feedback",
        action="store_true",
        help="Disable round>=2 feedback gate (debug only)",
    )

    real_loop_check_profile = sub.add_parser(
        "real-loop-check-profile",
        help="Offline Model Profile qualification for real loops (v2.1.3)",
    )
    real_loop_check_profile.add_argument(
        "profile_id", nargs="?", default=None, help="Profile id (or use --session)"
    )
    real_loop_check_profile.add_argument("--session", dest="session_id", default=None)
    real_loop_check_profile.add_argument("--suite", default="eval_suite_v1")
    real_loop_check_profile.add_argument(
        "--skip-quality-gate",
        action="store_true",
        help="Only check profile shape (not for formal real-loop)",
    )

    real_loop_plan = sub.add_parser(
        "real-loop-plan",
        help="Real-only Planner for a loop round (v2.1.3; no mock fallback)",
    )
    real_loop_plan.add_argument("session_id")
    real_loop_plan.add_argument("--round", type=int, default=None, dest="round_number")
    real_loop_plan.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit live HTTP (also requires LLM_ALLOW_NETWORK=true)",
    )
    real_loop_plan.add_argument(
        "--provider",
        default="openai-compatible",
        help="Must be real / openai-compatible",
    )

    real_loop_review = sub.add_parser(
        "real-loop-review",
        help="Real-only Critic review for a loop round (v2.1.3)",
    )
    real_loop_review.add_argument("session_id")
    real_loop_review.add_argument("--round", type=int, default=None, dest="round_number")
    real_loop_review.add_argument("--allow-network", action="store_true")
    real_loop_review.add_argument("--provider", default="openai-compatible")

    real_loop_approve = sub.add_parser(
        "real-loop-approve",
        help="Approve one candidate and create Digits iteration (v2.1.4; no run yet)",
    )
    real_loop_approve.add_argument("session_id")
    real_loop_approve.add_argument(
        "--candidate", required=True, dest="candidate_id", help="Candidate id to approve"
    )
    real_loop_approve.add_argument("--round", type=int, default=None, dest="round_number")
    real_loop_approve.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: fast_eval seeds)",
    )

    real_loop_reject = sub.add_parser(
        "real-loop-reject",
        help="Reject the waiting candidate without executing (v2.1.4)",
    )
    real_loop_reject.add_argument("session_id")
    real_loop_reject.add_argument("--candidate", default=None, dest="candidate_id")
    real_loop_reject.add_argument("--round", type=int, default=None, dest="round_number")
    real_loop_reject.add_argument("--reason", default=None)

    real_loop_execute = sub.add_parser(
        "real-loop-execute",
        help="Execute approved Digits iteration (v2.1.4; bans mock entrypoint)",
    )
    real_loop_execute.add_argument("session_id")
    real_loop_execute.add_argument("--round", type=int, default=None, dest="round_number")
    real_loop_execute.add_argument("--seeds", default=None)
    real_loop_execute.add_argument(
        "--async",
        dest="async_run",
        action="store_true",
        help="Submit without waiting for seed completion",
    )

    return parser


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "manager-run":
        from scientist_lab.core.manager_cli import run_manager_from_files

        result = run_manager_from_files(
            args.protocol,
            args.plan,
            output_dir=args.output_dir,
            execute=bool(args.execute),
            require_live_ready=bool(args.require_live_ready),
            max_steps=int(args.max_steps),
            max_extra_rounds=int(args.max_extra_rounds),
            baseline_metrics_path=getattr(args, "baseline_metrics", None),
            planner_backend=str(getattr(args, "planner_backend", "rules") or "rules"),
            reviewer_backend=str(getattr(args, "reviewer_backend", "rules") or "rules"),
            llm_live=bool(getattr(args, "llm_live", False)),
            fallback_to_rules=bool(getattr(args, "fallback_to_rules", False)),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return int(result.get("exit_code") or 0)

    if args.command == "claim-gate":
        from scientist_lab.core.claim_gate import evaluate_run_dir
        from scientist_lab.core.schema_registry import load_json

        claim = load_json(args.claim) if getattr(args, "claim", None) else None
        verdict = evaluate_run_dir(
            args.run_dir,
            claim=claim,
            baseline_run_dir=getattr(args, "baseline_run_dir", None),
        )
        out = getattr(args, "output", None)
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps(verdict, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        return 0

    if args.command == "llm-plan-replay":
        from scientist_lab.llm.plan_replay import run_llm_plan_replay

        report = run_llm_plan_replay(
            args.run_dir,
            live=bool(args.live),
            provider=str(args.provider or "mock"),
            fallback_to_rules=bool(getattr(args, "fallback_to_rules", False)),
            output=getattr(args, "output", None),
            ab=bool(getattr(args, "ab", True)),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if args.command == "llm-review-replay":
        from scientist_lab.llm.review_replay import run_llm_review_replay

        report = run_llm_review_replay(
            args.run_dir,
            live=bool(args.live),
            provider=str(args.provider or "mock"),
            fallback_to_rules=bool(getattr(args, "fallback_to_rules", False)),
            output=getattr(args, "output", None),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if args.command == "llm-real-loop":
        from scientist_lab.llm.real_loop import run_v25d_real_loop

        report = run_v25d_real_loop(
            source_run_dir=getattr(args, "run_dir", None),
            output_dir=getattr(args, "output_dir", None),
            execute=bool(args.execute),
            require_live_ready=bool(args.require_live_ready),
            planner_backend=str(getattr(args, "planner_backend", "llm") or "llm"),
            reviewer_backend=str(getattr(args, "reviewer_backend", "llm") or "llm"),
            llm_live=bool(getattr(args, "live", False)),
            provider=str(args.provider or "mock"),
            fallback_to_rules=bool(getattr(args, "fallback_to_rules", False)),
            max_steps=int(getattr(args, "max_steps", 32) or 32),
            human_probe_permission=bool(args.execute),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if args.command in {"doctor", "system-doctor"}:
        try:
            service = _experiment_service()
            report = service.system_doctor()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            overall = str(report.get("overall") or "ok")
            return 0 if overall != "error" else 1
        except Exception as exc:  # noqa: BLE001
            print(f"System Doctor 失败: {exc}", file=sys.stderr)
            return 1

    if args.command == "recover":
        try:
            service = _experiment_service()
            dry_run = not bool(getattr(args, "apply", False))
            report = service.recover(dry_run=dry_run)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Recover 失败: {exc}", file=sys.stderr)
            return 1

    if args.command == "demo-create":
        try:
            service = _experiment_service()
            result = service.demos.create(
                args.kind, force=bool(getattr(args, "force", False))
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"demo-create 失败: {exc}", file=sys.stderr)
            return 1

    if args.command == "real-loop-create":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_create(
                str(args.project_id),
                profile_id=str(args.profile_id),
                protocol_id=str(args.protocol_id),
                rounds=int(getattr(args, "rounds", 2) or 2),
                baseline_node_ids=list(getattr(args, "baseline_node_ids", None) or []),
                tree_id=getattr(args, "tree_id", None),
                fallback_allowed=False,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (RealLoopValidationError, RealLoopError, KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-show":
        from scientist_lab.research_loop.errors import RealLoopNotFoundError

        try:
            print(
                json.dumps(
                    service.real_loop_show(str(args.session_id)),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except RealLoopNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-check":
        from scientist_lab.research_loop.errors import RealLoopNotFoundError

        try:
            data = service.real_loop_check(str(args.session_id))
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("overall") == "ok" else 1
        except RealLoopNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-list":
        print(
            json.dumps(
                service.real_loop_list(
                    project_id=getattr(args, "project_id", None),
                    limit=int(getattr(args, "limit", 50) or 50),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "real-loop-record-feedback":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            comparison = None
            cmp_path = getattr(args, "comparison_json", None)
            if cmp_path:
                comparison = json.loads(Path(cmp_path).read_text(encoding="utf-8"))
            data = service.real_loop_record_feedback(
                str(args.session_id),
                parent_node_id=str(args.parent_node_id),
                executed_node_id=str(args.executed_node_id),
                source_round=int(getattr(args, "source_round", 1) or 1),
                comparison=comparison,
                evidence_ids=list(getattr(args, "evidence_ids", None) or []),
                previous_hypothesis=getattr(args, "hypothesis", None),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-record-execution-feedback":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_record_execution_feedback(
                str(args.session_id),
                source_round=int(getattr(args, "source_round", 1) or 1),
                parent_node_id=getattr(args, "parent_node_id", None),
                executed_node_id=getattr(args, "executed_node_id", None),
                advance_status=not bool(getattr(args, "no_advance", False)),
                include_round2_context=not bool(
                    getattr(args, "no_round2_context", False)
                ),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            return 0 if data.get("ok", True) is not False else 1
        except (
            RealLoopValidationError,
            RealLoopNotFoundError,
            RealLoopError,
            KeyError,
            ValueError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-next-round":
        from scientist_lab.research_loop.errors import (
            InvalidRealLoopTransition,
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_next_round(str(args.session_id))
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            return 0 if data.get("ok", True) is not False else 1
        except (
            RealLoopValidationError,
            RealLoopNotFoundError,
            RealLoopError,
            InvalidRealLoopTransition,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-verify-feedback":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_verify_feedback(
                str(args.session_id),
                round_number=int(getattr(args, "round_number", 2) or 2),
                plan_id=getattr(args, "plan_id", None),
                persist=not bool(getattr(args, "no_persist", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            return 0 if data.get("pass_status") else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-export":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_export(
                str(args.session_id),
                output_dir=getattr(args, "output_dir", None),
                allow_incomplete=bool(getattr(args, "allow_incomplete", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-build-context":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_build_context(
                str(args.session_id),
                round_number=int(args.round_number),
                enforce_feedback_gate=not bool(
                    getattr(args, "allow_missing_feedback", False)
                ),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-check-profile":
        data = service.real_loop_check_profile(
            getattr(args, "profile_id", None),
            session_id=getattr(args, "session_id", None),
            suite_version=str(getattr(args, "suite", "eval_suite_v1") or "eval_suite_v1"),
            require_quality_gate=not bool(getattr(args, "skip_quality_gate", False)),
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data.get("overall") == "ok" else 1

    if args.command == "real-loop-plan":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_plan(
                str(args.session_id),
                round_number=getattr(args, "round_number", None),
                allow_network=bool(getattr(args, "allow_network", False)),
                provider=str(getattr(args, "provider", "openai-compatible")),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-review":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_review(
                str(args.session_id),
                round_number=getattr(args, "round_number", None),
                allow_network=bool(getattr(args, "allow_network", False)),
                provider=str(getattr(args, "provider", "openai-compatible")),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-approve":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_approve(
                str(args.session_id),
                candidate_id=str(args.candidate_id),
                round_number=getattr(args, "round_number", None),
                seeds=_parse_seeds(getattr(args, "seeds", None)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError, KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-reject":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_reject(
                str(args.session_id),
                candidate_id=getattr(args, "candidate_id", None),
                round_number=getattr(args, "round_number", None),
                reason=getattr(args, "reason", None),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "real-loop-execute":
        from scientist_lab.research_loop.errors import (
            RealLoopError,
            RealLoopNotFoundError,
            RealLoopValidationError,
        )

        try:
            data = service.real_loop_execute(
                str(args.session_id),
                round_number=getattr(args, "round_number", None),
                seeds=_parse_seeds(getattr(args, "seeds", None)),
                wait=not bool(getattr(args, "async_run", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok", True) is not False else 1
        except (RealLoopValidationError, RealLoopNotFoundError, RealLoopError, KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "workbench":
        import subprocess

        from scientist_lab.workbench.config import (
            load_workbench_config,
            workbench_endpoints,
        )
        from scientist_lab.workbench.status import workbench_status

        root = Path(__file__).resolve().parents[2]
        cmd = getattr(args, "workbench_command", None)
        if not cmd:
            print("usage: scientist-lab workbench {start|status|stop}", file=sys.stderr)
            return 2
        if cmd == "status":
            print(json.dumps(workbench_status(root), ensure_ascii=False, indent=2))
            return 0

        ends = workbench_endpoints(load_workbench_config(project_root=root))
        if cmd == "start":
            script = root / "scripts" / "start_workbench.ps1"
            print("Workbench architecture:")
            print("  Backend API = FastAPI  ->", ends["api_url"])
            print("  Frontend UI = React/Vite ->", ends["web_url"])
            print("  Open the Web URL in your browser (Vite proxies /api).")
            ps = (
                f'powershell -ExecutionPolicy Bypass -File "{script}"'
            )
            print("Start command:")
            print(" ", ps)
            if getattr(args, "print_only", False):
                return 0
            if not script.is_file():
                print(f"missing script: {script}", file=sys.stderr)
                return 1
            completed = subprocess.run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                cwd=str(root),
                check=False,
            )
            return int(completed.returncode)
        if cmd == "stop":
            script = root / "scripts" / "stop_workbench.ps1"
            ps = f'powershell -ExecutionPolicy Bypass -File "{script}"'
            print(ps)
            if getattr(args, "print_only", False):
                return 0
            completed = subprocess.run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                cwd=str(root),
                check=False,
            )
            return int(completed.returncode)
        print(f"unknown workbench command: {cmd}", file=sys.stderr)
        return 2

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

    from scientist_lab.iteration.service import IterationService
    from scientist_lab.services.experiment_service import load_contract

    service = _experiment_service()
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

    if args.command == "project-create":
        try:
            data = service.create_project(
                title=str(args.title),
                research_question=getattr(args, "research_question", "") or "",
                research_goal=getattr(args, "research_goal", "") or "",
                description=getattr(args, "description", "") or "",
                task_type=str(getattr(args, "task_type", "general_ml") or "general_ml"),
                dataset_keys=list(getattr(args, "dataset_keys", None) or []),
                protocol_ids=list(getattr(args, "protocol_ids", None) or []),
                runner_profile_keys=list(
                    getattr(args, "runner_profile_keys", None) or []
                ),
                project_id=getattr(args, "project_id", None),
                mark_ready=not bool(getattr(args, "keep_configuring", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "project-list":
        print(json.dumps(service.list_projects(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "project-show":
        try:
            print(
                json.dumps(
                    service.get_project(args.project_id), ensure_ascii=False, indent=2
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "project-archive":
        try:
            print(
                json.dumps(
                    service.archive_project(args.project_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "project-export":
        try:
            print(
                json.dumps(
                    service.export_project(
                        args.project_id, output_dir=args.output_dir
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "project-import":
        try:
            print(
                json.dumps(
                    service.import_project(
                        args.path, force=bool(getattr(args, "force", False))
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (KeyError, ValueError) as exc:
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
            provider = "mock" if args.mock else args.provider
            data = service.plan_next(
                args.project_id,
                protocol_id=args.protocol_id,
                current_best_node_id=args.best_node,
                max_new_nodes=args.max_new_nodes,
                max_gpu_hours=args.max_gpu_hours,
                provider=provider,
                allow_network=bool(getattr(args, "allow_network", False)),
                model_profile=getattr(args, "model_profile", None),
                require_quality_gate=bool(
                    getattr(args, "require_quality_gate", False)
                ),
                allow_unqualified_profile=bool(
                    getattr(args, "allow_unqualified_profile", False)
                ),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("status") in {
                "planner_failed",
                "real_provider_failed",
                "profile_not_qualified",
            }:
                return 1
            return 0
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
            data = service.review_plan(
                args.plan_id,
                provider=args.provider,
                allow_network=bool(getattr(args, "allow_network", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") != "real_provider_failed" else 1
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
            provider = "mock" if args.mock else args.provider
            data = service.tree_plan_next(
                args.tree_id,
                rescore=not args.no_rescore,
                max_gpu_hours=args.max_gpu_hours,
                provider=provider,
                allow_network=bool(getattr(args, "allow_network", False)),
                model_profile=getattr(args, "model_profile", None),
                require_quality_gate=bool(
                    getattr(args, "require_quality_gate", False)
                ),
                allow_unqualified_profile=bool(
                    getattr(args, "allow_unqualified_profile", False)
                ),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("ascii_tree"):
                print("\n" + data["ascii_tree"])
            if data.get("status") in {"real_provider_failed", "profile_not_qualified"}:
                return 1
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

    if args.command == "report-context":
        try:
            data = service.build_report_context(
                args.project_id,
                tree_id=args.tree_id,
                protocol_id=args.protocol_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "report-build":
        try:
            data = service.build_report(
                args.project_id,
                tree_id=args.tree_id,
                protocol_id=args.protocol_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"draft", "verified", "tables_ready"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "report-show":
        try:
            data = service.show_report(args.report_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "report-verify":
        try:
            data = service.verify_report(args.report_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "audit-build":
        try:
            data = service.build_audit(
                args.project_id,
                tree_id=args.tree_id,
                protocol_id=args.protocol_id,
                report_id=args.report_id,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"built", "invalid"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "audit-verify":
        try:
            data = service.verify_audit(args.bundle_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("valid") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "audit-export":
        try:
            data = service.export_audit(args.bundle_id, args.output)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-eval":
        try:
            data = service.evaluate_llm_quality(
                args.project_id,
                protocol_id=args.protocol_id,
                current_best_node_id=args.best_node,
                output=args.output,
                include_real=args.include_real,
                suite=args.suite,
                provider=args.provider,
                allow_network=args.allow_network,
                max_cases=args.max_cases,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if args.suite:
                if data.get("skipped"):
                    return 0  # real skipped is success for CI
                return 0 if data.get("gates", {}).get("suite_ok", True) else 1
            gates = data.get("gates") or {}
            ok = bool(
                gates.get("mock_ok") and gates.get("fake_ok") and gates.get("replay_ok")
            )
            return 0 if ok else 1
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-eval-run":
        try:
            data = service.run_llm_eval_suite(
                args.project_id,
                suite=args.suite,
                provider=args.provider,
                allow_network=bool(getattr(args, "allow_network", False)),
                profile_id=getattr(args, "model_profile", None),
                seed_fake_for_replay=not bool(getattr(args, "no_seed_fake", False)),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("status") == "skipped":
                return 0
            safety = (data.get("safety") or {}).get("pass", True)
            return 0 if safety else 1
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-profile-register":
        try:
            data = service.register_llm_profile(args.path)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-profile-list":
        print(
            json.dumps(
                service.list_llm_profiles(
                    enabled_only=bool(getattr(args, "enabled_only", False))
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "llm-profile-show":
        try:
            print(
                json.dumps(
                    service.show_llm_profile(args.profile_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-eval-verify":
        try:
            data = service.verify_llm_evaluation(args.evaluation_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            status = data.get("status")
            if status == "blocked":
                return 1
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-eval-compare":
        try:
            data = service.compare_llm_evaluations(
                args.baseline_evaluation_id, args.candidate_evaluation_id
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 1 if data.get("regression_detected") else 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-profile-rank":
        data = service.rank_llm_profiles(suite_version=args.suite)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "llm-profile-select":
        try:
            data = service.select_llm_profile(args.profile_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "llm-usage":
        try:
            data = service.summarize_llm_usage(
                args.project_id,
                audit_root=args.audit_root,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-propose":
        try:
            data = service.patches.propose_mock(
                args.project_id,
                title=args.title,
                rationale=args.rationale,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"proposed", "verified"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-propose-real":
        try:
            from scientist_lab.patching.real_mode import PatchProviderBudget

            budget = None
            if getattr(args, "max_calls", None) is not None:
                budget = PatchProviderBudget(max_calls=int(args.max_calls))
            data = service.patches.propose_real(
                args.bundle_id,
                requested_provider=args.provider,
                allow_network=bool(args.allow_network),
                real_only=True,
                budget=budget,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"proposed", "verified"} else 1
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-budget-show":
        data = service.patches.show_patch_budget(args.project_id)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "patch-provider-doctor":
        data = service.patches.patch_provider_doctor(
            requested_provider=args.provider,
            allow_network=bool(args.allow_network),
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data.get("overall") == "ok" else 1

    if args.command == "dfine-cuda-doctor":
        data = service.dfine_cuda_doctor(
            probe_runtime=not bool(args.no_probe),
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        # warnings (missing GPU/image) are expected on CPU CI — exit 0 if no errors
        return 0 if data.get("ok") else 1

    if args.command == "dfine-cuda-fast-eval":
        dry_run = not bool(getattr(args, "execute", False))
        try:
            data = service.dfine_cuda_fast_eval(
                contract_path=getattr(args, "contract", None),
                runner_profile=getattr(args, "runner_profile", None),
                wait=not bool(getattr(args, "no_wait", False)),
                dry_run=dry_run,
                require_live_ready=bool(
                    getattr(args, "require_live_ready", False)
                ),
                probe_runtime=not bool(getattr(args, "no_probe", False)),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if dry_run:
            return 0
        status = str((data.get("run") or {}).get("status") or data.get("status") or "")
        return 0 if status == "completed" else 1

    if args.command == "dfine-adapter-run":
        from scientist_lab.adapters.dfine.run_loop import run_gated_dfine_from_files

        try:
            data = run_gated_dfine_from_files(
                args.protocol,
                args.plan,
                output_dir=args.output_dir,
                execute=bool(args.execute),
                require_live_ready=bool(args.require_live_ready),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        gate = str((data.get("gate") or {}).get("status") or "")
        if gate != "APPROVED":
            return 2
        if args.execute:
            status = str((data.get("handle") or {}).get("status") or "")
            return 0 if status == "completed" else 1
        return 0

    if args.command == "dfine-cuda-record-evidence":
        try:
            data = service.dfine_cuda_record_evidence(
                args.execution_id,
                refresh_claim_matrix=not bool(args.no_claim_matrix),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dfine-cuda-formal-triad":
        dry_run = not bool(getattr(args, "execute", False))
        try:
            data = service.dfine_cuda_formal_triad(
                dry_run=dry_run,
                wait=not bool(getattr(args, "no_wait", False)),
                require_live_ready=bool(
                    getattr(args, "require_live_ready", False)
                ),
                probe_runtime=not bool(getattr(args, "no_probe", False)),
                protocol_id=str(
                    getattr(args, "protocol_id", "protocol_rgbt_cuda_001")
                ),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if dry_run:
            return 0
        return 0 if data.get("status") == "completed" else 1

    if args.command == "dfine-formal-path-gate":
        try:
            data = service.dfine_formal_path_gate(
                args.project_id,
                protocol_id=getattr(args, "protocol_id", None),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dfine-real-acceptance":
        dry_run = not bool(getattr(args, "execute", False))
        try:
            data = service.dfine_real_acceptance(
                dry_run=dry_run,
                include_formal_triad=bool(
                    getattr(args, "include_formal_triad", False)
                ),
                wait=not bool(getattr(args, "no_wait", False)),
                probe_runtime=not bool(getattr(args, "no_probe", False)),
            )
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if dry_run:
            return 0
        return 0 if data.get("status") == "completed" else 1

    if args.command == "patch-show":
        try:
            print(
                json.dumps(
                    service.patches.show(args.patch_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "code-context-build":
        try:
            data = service.patches.build_code_context(
                digits_demo=bool(args.digits_demo),
                persist=not bool(args.no_persist),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "code-context-show":
        try:
            data = service.patches.show_code_context(args.bundle_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "code-context-export":
        try:
            data = service.patches.export_code_context(
                args.bundle_id, output_path=args.output
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-verify":
        try:
            data = service.patches.verify(args.patch_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if (data.get("verification") or {}).get("ok") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-approve":
        try:
            data = service.patches.approve(args.patch_id, reason=args.reason or "")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-check-seal":
        try:
            data = service.patches.check_approval_seal(args.patch_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            report = data.get("seal_report") or {}
            return 0 if report.get("ok") else 1
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-reject":
        try:
            data = service.patches.reject(args.patch_id, reason=args.reason or "")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-apply-sandbox":
        try:
            data = service.patches.apply_sandbox(
                args.patch_id, force=bool(getattr(args, "force", False))
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "applied_sandbox" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-test-sandbox":
        try:
            data = service.patches.test_sandbox(
                args.patch_id,
                profile=getattr(args, "profile", "smoke") or "smoke",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            report = data.get("sandbox_tests") or {}
            return 0 if report.get("ok") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-sandbox-profiles":
        data = service.patches.list_sandbox_test_profiles()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "patch-export-replay":
        try:
            data = service.patches.export_patch_replay(
                args.patch_id, output_dir=args.output
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("ok") else 1
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-replay":
        try:
            data = service.patches.replay_patch_from_bundle(args.bundle_dir)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"proposed", "verified"} else 1
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-record-evidence":
        try:
            if bool(getattr(args, "feedback", False)):
                data = service.record_patch_evidence_feedback(
                    args.patch_id,
                    require_tests=bool(getattr(args, "require_tests", False)),
                )
            else:
                data = service.patches.record_evidence(
                    args.patch_id,
                    require_tests=bool(getattr(args, "require_tests", False)),
                )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "evidence_recorded" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "patch-decide-merge":
        try:
            data = service.patches.decide_merge(
                args.patch_id,
                decision=str(args.decision),
                reason=args.reason or "",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"merged", "discarded"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-prepare":
        try:
            data = service.merge_prepare(
                args.patch_id,
                target_branch=getattr(args, "target_branch", None),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("merge_candidate_id") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-show":
        try:
            data = service.merge_show(args.merge_candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-list":
        data = service.list_merge_candidates(
            project_id=getattr(args, "project_id", None),
            patch_id=getattr(args, "patch_id", None),
            limit=int(getattr(args, "limit", 50) or 50),
        )
        print(json.dumps({"items": data, "count": len(data)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "merge-apply":
        try:
            data = service.merge_apply(args.merge_candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("workspace_applied") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-test":
        try:
            data = service.merge_test(
                args.merge_candidate_id,
                profile_id=str(getattr(args, "profile", "smoke") or "smoke"),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "waiting_approval" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-approve":
        try:
            data = service.merge_approve(
                args.merge_candidate_id,
                reason=getattr(args, "reason", "") or "",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "approved" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-reject":
        try:
            data = service.merge_reject(
                args.merge_candidate_id,
                reason=getattr(args, "reason", "") or "",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "rejected" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-commit":
        try:
            data = service.merge_commit(args.merge_candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("commit_sha") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-finalize":
        try:
            profile_raw = str(getattr(args, "post_merge_profile", "syntax") or "syntax")
            profile = None if profile_raw == "none" else profile_raw
            data = service.merge_finalize(
                args.merge_candidate_id,
                post_merge_profile=profile,
                auto_rollback_on_failure=not bool(
                    getattr(args, "no_auto_rollback", False)
                ),
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") in {"merged", "rolled_back"} else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "merge-rollback":
        try:
            data = service.merge_rollback(
                args.merge_candidate_id,
                reason=getattr(args, "reason", "") or "",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("status") == "rolled_back" else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "release-candidate-create":
        try:
            data = service.create_release_candidate(
                version=str(args.version),
                project_id=getattr(args, "project_id", "") or "",
                base_tag=getattr(args, "base_tag", "") or "",
                merge_candidate_ids=list(
                    getattr(args, "merge_candidate_ids", None) or []
                ),
                notes=getattr(args, "notes", "") or "",
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data.get("release_candidate_id") else 1
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "release-candidate-show":
        try:
            data = service.show_release_candidate(args.release_candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.command == "release-candidate-verify":
        try:
            data = service.verify_release_candidate(args.release_candidate_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if (data.get("verification") or {}).get("valid") else 1
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
