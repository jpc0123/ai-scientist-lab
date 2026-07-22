"""Restricted source patching (v1.6)."""

from scientist_lab.patching.apply_engine import PatchApplyError, apply_unified_diff_to_root
from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import PatchProposal, PatchVerification
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.sandbox_checks import SandboxTestReport, SandboxTestRunner
from scientist_lab.patching.service import PatchingService, build_mock_unified_diff
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.patching.workspace import PatchSandbox

__all__ = [
    "DiffParseError",
    "PatchApplyError",
    "PatchProposal",
    "PatchSandbox",
    "PatchVerification",
    "PatchVerifier",
    "PatchingService",
    "PathPolicy",
    "SandboxTestReport",
    "SandboxTestRunner",
    "apply_unified_diff_to_root",
    "build_mock_unified_diff",
    "fingerprint_diff",
    "parse_unified_diff",
]
