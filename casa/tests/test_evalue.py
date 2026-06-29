"""Unit tests for the e-value / e-process sequential test."""

from __future__ import annotations

import random

import pytest

from casa.evalue import EProcess, betting_evalue, lr_evalue, normal_cdf


def test_normal_cdf() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(-10) < 1e-6
    assert normal_cdf(10) > 1 - 1e-6


def test_betting_evalue_shape() -> None:
    assert betting_evalue(0.0) == pytest.approx(1.0)
    assert betting_evalue(5.0) > 1.0
    assert betting_evalue(-5.0) < 1.0
    with pytest.raises(ValueError):
        betting_evalue(0.0, lam=1.5)


def test_betting_evalue_expectation_under_null() -> None:
    # E_null[e] = 1 exactly when z ~ N(0,1); check empirically.
    rng = random.Random(0)
    vals = [betting_evalue(rng.gauss(0, 1)) for _ in range(200000)]
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=0.01)


def test_lr_evalue_expectation() -> None:
    rng = random.Random(1)
    vals = [lr_evalue(rng.gauss(0, 1), mu=1.5) for _ in range(200000)]
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=0.05)


def test_eprocess_null_false_alarm_rate() -> None:
    # Ville: P(ever cross 1/alpha) <= alpha under the null.
    rng = random.Random(2)
    alpha = 0.05
    trials, crosses = 2000, 0
    for _ in range(trials):
        ep = EProcess(alpha=alpha)
        for _ in range(200):
            ep.update(rng.gauss(0, 1))
            if ep.has_crossed:
                break
        crosses += ep.has_crossed
    assert crosses / trials <= alpha + 0.02


def test_eprocess_detects_signal() -> None:
    ep = EProcess(alpha=0.05)
    for _ in range(20):
        ep.update(3.0)
    assert ep.has_crossed
    assert ep.value >= ep.boundary
    assert ep.crossed_at is not None and ep.crossed_at <= 20


def test_eprocess_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        EProcess(alpha=1.5)
