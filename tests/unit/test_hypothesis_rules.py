from __future__ import annotations

from scientist_lab.domain.comparison import HypothesisStatus
from scientist_lab.services.verifier import judge_hypothesis


def test_supported_when_delta_above_threshold():
    status = judge_hypothesis(
        experiment_valid=True,
        primary_metric="accuracy",
        primary_metric_delta=0.005,
        minimum_improvement=0.002,
    )
    assert status == HypothesisStatus.SUPPORTED


def test_rejected_when_delta_negative():
    status = judge_hypothesis(
        experiment_valid=True,
        primary_metric="accuracy",
        primary_metric_delta=-0.01,
        minimum_improvement=0.002,
    )
    assert status == HypothesisStatus.REJECTED


def test_inconclusive_when_delta_small():
    status = judge_hypothesis(
        experiment_valid=True,
        primary_metric="accuracy",
        primary_metric_delta=0.001,
        minimum_improvement=0.002,
    )
    assert status == HypothesisStatus.INCONCLUSIVE


def test_inconclusive_when_invalid():
    status = judge_hypothesis(
        experiment_valid=False,
        primary_metric="accuracy",
        primary_metric_delta=0.05,
    )
    assert status == HypothesisStatus.INCONCLUSIVE


def test_log_loss_polarity():
    status = judge_hypothesis(
        experiment_valid=True,
        primary_metric="log_loss",
        primary_metric_delta=-0.01,
        minimum_improvement=0.002,
    )
    assert status == HypothesisStatus.SUPPORTED
