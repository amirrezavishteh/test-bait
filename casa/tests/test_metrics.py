"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest

from casa.metrics import binary_metrics, roc_auc


def test_roc_auc_perfect_separation() -> None:
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [0, 0, 0, 1, 1, 1]
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_reversed() -> None:
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [0, 0, 0, 1, 1, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_ties_half() -> None:
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_single_class_none() -> None:
    assert roc_auc([0.1, 0.2], [1, 1]) is None


def test_binary_metrics() -> None:
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(0.75)


def test_binary_metrics_empty() -> None:
    m = binary_metrics([], [])
    assert m["f1"] == 0.0
