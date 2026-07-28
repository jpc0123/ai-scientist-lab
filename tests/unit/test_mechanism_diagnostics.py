"""Unit tests for mechanism diagnosis helpers (CPU-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

APP = Path(__file__).resolve().parents[2] / "experiment_apps" / "rgbt_detection_real"
sys.path.insert(0, str(APP))

from mechanism_diagnostics import (  # noqa: E402
    cosine_sim,
    feature_stats,
    gate_stats_from_tensor,
)
from models.feature_fusion import GatedFeatureFusion  # noqa: E402


def test_gate_stats_saturation_flags():
    gate = torch.ones(2, 4, 8, 8) * 0.95
    stats = gate_stats_from_tensor(gate, "p3")
    assert stats["gate_gt_0_9_ratio"] > 0.99
    assert stats["gate_lt_0_1_ratio"] < 0.01
    assert 0.0 < stats["gate_entropy"] < 1.0


def test_feature_stats_and_cosine():
    a = torch.randn(1, 8, 4, 4)
    b = a.clone()
    fs = feature_stats(a, "p4", "fused")
    assert fs["l2_norm"] > 0
    assert abs(cosine_sim(a, b) - 1.0) < 1e-5


def test_diag_flag_stores_gate():
    m = GatedFeatureFusion(8, residual=True)
    m.diag_enabled = True
    rgb = torch.randn(1, 8, 4, 4)
    thr = torch.randn(1, 8, 4, 4)
    _ = m(rgb, thr)
    assert m._last_gate is not None
    assert m._last_gate.shape == rgb.shape
