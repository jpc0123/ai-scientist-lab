from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter
from scientist_lab.tasks.rgbt_detection.baseline_adapter import (
    ensure_baselines_loaded,
    get_baseline_adapter,
    list_baseline_keys,
    resolve_baseline_key,
)
from scientist_lab.tasks.rgbt_detection.feedback_rules import apply_detection_claim_gate
from scientist_lab.tasks.rgbt_detection.result_parser import DetectionResultParser
from scientist_lab.tasks.rgbt_detection.verifier import DetectionVerifier

__all__ = [
    "DetectionResultParser",
    "DetectionVerifier",
    "RGBTDetectionAdapter",
    "apply_detection_claim_gate",
    "ensure_baselines_loaded",
    "get_baseline_adapter",
    "list_baseline_keys",
    "resolve_baseline_key",
]
