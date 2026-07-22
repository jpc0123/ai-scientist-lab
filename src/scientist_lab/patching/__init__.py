"""Restricted source patching (v1.6)."""

from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import PatchProposal, PatchVerification
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.service import PatchingService, build_mock_unified_diff
from scientist_lab.patching.verifier import PatchVerifier

__all__ = [
    "DiffParseError",
    "PatchProposal",
    "PatchVerification",
    "PatchVerifier",
    "PatchingService",
    "PathPolicy",
    "build_mock_unified_diff",
    "fingerprint_diff",
    "parse_unified_diff",
]
