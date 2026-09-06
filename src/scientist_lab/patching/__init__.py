"""Restricted source patching (v1.6 + v2.2 context)."""

from scientist_lab.patching.apply_engine import PatchApplyError, apply_unified_diff_to_root
from scientist_lab.patching.approval_seal import (
    ApprovalSealError,
    compare_approval_seal,
    proposal_content_sha256,
)
from scientist_lab.patching.context_bundle import (
    ContextBundleError,
    build_code_context_bundle,
    code_context_sha256,
    digits_improvement_patch_request,
    how_plugin_patch_request,
)
from scientist_lab.patching.context_models import (
    AllowedSourceFile,
    CodeContextBundle,
    ContextSizeBudget,
    PatchGoal,
    PatchRequest,
    SourceSnapshot,
)
from scientist_lab.patching.context_store import (
    CodeContextRepository,
    export_code_context_bundle,
    load_exported_code_context_bundle,
)
from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.diff_safety import DiffSafetyLimits, scan_parsed_diff
from scientist_lab.patching.evidence import PatchEvidence, PatchMergeDecision
from scientist_lab.patching.feedback import (
    build_patch_feedback_package,
    derive_patch_verdict,
    load_planner_patch_feedback,
)
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import (
    ApprovalContentSeal,
    PatchProposal,
    PatchVerification,
)
from scientist_lab.patching.patch_schema import PATCH_PROPOSAL_SCHEMA
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.real_patch_planner import (
    RealPatchPlanner,
    RealPatchPlannerError,
    RealPatchPlannerOutput,
    code_context_to_patch_request,
)
from scientist_lab.patching.real_mode import (
    DEFAULT_PATCH_PROVIDER_BUDGET,
    PatchBudgetStore,
    PatchProviderBudget,
    PatchRealModeError,
    PatchRealModePolicy,
    patch_provider_doctor,
)
from scientist_lab.patching.replay_bundle import (
    PatchReplayError,
    assert_bundle_redacted,
    build_patch_replay_bundle,
    digits_fixture_provider_response,
    load_patch_replay_bundle,
    replay_patch_static_checks,
)
from scientist_lab.patching.sandbox_checks import SandboxTestReport, SandboxTestRunner
from scientist_lab.patching.sandbox_registry import (
    list_sandbox_test_profiles,
    require_sandbox_test_profile,
)
from scientist_lab.patching.service import PatchingService, build_mock_unified_diff
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.patching.workspace import PatchSandbox

__all__ = [
    "AllowedSourceFile",
    "ApprovalContentSeal",
    "ApprovalSealError",
    "CodeContextBundle",
    "CodeContextRepository",
    "ContextBundleError",
    "ContextSizeBudget",
    "DEFAULT_PATCH_PROVIDER_BUDGET",
    "DiffParseError",
    "DiffSafetyLimits",
    "PATCH_PROPOSAL_SCHEMA",
    "PatchApplyError",
    "PatchBudgetStore",
    "PatchEvidence",
    "PatchGoal",
    "PatchMergeDecision",
    "PatchProposal",
    "PatchProviderBudget",
    "PatchRealModeError",
    "PatchRealModePolicy",
    "PatchReplayError",
    "PatchRequest",
    "PatchSandbox",
    "PatchVerification",
    "PatchVerifier",
    "PatchingService",
    "PathPolicy",
    "RealPatchPlanner",
    "RealPatchPlannerError",
    "RealPatchPlannerOutput",
    "SandboxTestReport",
    "SandboxTestRunner",
    "SourceSnapshot",
    "apply_unified_diff_to_root",
    "assert_bundle_redacted",
    "build_code_context_bundle",
    "build_mock_unified_diff",
    "build_patch_feedback_package",
    "build_patch_replay_bundle",
    "code_context_sha256",
    "code_context_to_patch_request",
    "compare_approval_seal",
    "derive_patch_verdict",
    "digits_fixture_provider_response",
    "digits_improvement_patch_request",
    "export_code_context_bundle",
    "fingerprint_diff",
    "how_plugin_patch_request",
    "list_sandbox_test_profiles",
    "load_exported_code_context_bundle",
    "load_patch_replay_bundle",
    "load_planner_patch_feedback",
    "parse_unified_diff",
    "patch_provider_doctor",
    "proposal_content_sha256",
    "replay_patch_static_checks",
    "require_sandbox_test_profile",
    "scan_parsed_diff",
]
