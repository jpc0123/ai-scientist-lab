"""Finite experiment-tree search (frozen public API @ v1.1.0).

See INTERFACE.md for the stable export surface consumed by reporting/.
"""

from scientist_lab.search.models import ExperimentTree, TreeNode, TreeNodeStatus, TreeStatus
from scientist_lab.search.scoring import ScoreBreakdown, compute_scores
from scientist_lab.search.selection_policy import ParentSelectionResult, rank_expandable_parents
from scientist_lab.search.service import TreeSearchService
from scientist_lab.search.stop_policy import StopDecision, StopPolicy, evaluate_stop
from scientist_lab.search.evidence_link import (
    collect_related_evidence_ids,
    diff_gaps,
    extract_open_gaps,
)
from scientist_lab.search.export import export_tree_payload, render_mermaid

__all__ = [
    "ExperimentTree",
    "ParentSelectionResult",
    "ScoreBreakdown",
    "StopDecision",
    "StopPolicy",
    "TreeNode",
    "TreeNodeStatus",
    "TreeSearchService",
    "TreeStatus",
    "collect_related_evidence_ids",
    "compute_scores",
    "diff_gaps",
    "evaluate_stop",
    "export_tree_payload",
    "extract_open_gaps",
    "rank_expandable_parents",
    "render_mermaid",
]

API_VERSION = "v1.1.0"
