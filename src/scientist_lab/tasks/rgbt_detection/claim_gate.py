"""Claim gate for RGB-T debug / smoke experiments."""

from scientist_lab.tasks.rgbt_detection.feedback_rules import (
    apply_detection_claim_gate,
    annotate_feedback_for_detection,
)

__all__ = ["apply_detection_claim_gate", "annotate_feedback_for_detection"]
