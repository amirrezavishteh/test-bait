"""Unit tests for the conformal calibration math."""

from __future__ import annotations

import math
from math import comb

import pytest

from casa.conformal import (
    binom_cdf,
    calibrate_detection_threshold,
    calibrate_match_threshold,
    crc_threshold_padding,
    hoeffding_bentkus_eps,
    hoeffding_bentkus_ucb,
    kl_bernoulli,
)


def test_kl_bernoulli_basics() -> None:
    assert kl_bernoulli(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert kl_bernoulli(0.1, 0.9) > 0
    assert math.isinf(kl_bernoulli(0.0, 1.0))
    assert math.isinf(kl_bernoulli(1.0, 0.0))
    assert kl_bernoulli(0.0, 0.5) == pytest.approx(-math.log(0.5))


def test_kl_bernoulli_domain() -> None:
    with pytest.raises(ValueError):
        kl_bernoulli(1.5, 0.5)


def test_binom_cdf_matches_bruteforce() -> None:
    def brute(k: int, n: int, p: float) -> float:
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    for n, p in [(10, 0.3), (20, 0.5), (30, 0.05)]:
        for k in (0, 1, n // 2, n):
            assert binom_cdf(k, n, p) == pytest.approx(brute(k, n, p), abs=1e-9)


def test_binom_cdf_edges() -> None:
    assert binom_cdf(-1, 5, 0.5) == 0.0
    assert binom_cdf(5, 5, 0.5) == 1.0
    assert binom_cdf(0, 5, 0.0) == 1.0
    assert binom_cdf(2, 5, 1.0) == 0.0


def test_hb_ucb_monotone_and_bounds() -> None:
    # UCB is non-decreasing in the empirical loss and >= the loss itself.
    prev = -1.0
    for loss in (0.0, 0.05, 0.1, 0.2, 0.5):
        ucb = hoeffding_bentkus_ucb(loss, 30, 0.05)
        assert ucb >= loss - 1e-9
        assert ucb >= prev - 1e-9
        prev = ucb


def test_hb_ucb_tightens_with_n() -> None:
    small = hoeffding_bentkus_ucb(0.0, 10, 0.05)
    large = hoeffding_bentkus_ucb(0.0, 100, 0.05)
    assert large < small


def test_hb_eps_is_min_of_two_terms() -> None:
    eps = hoeffding_bentkus_eps(0.1, 0.1, 30)
    assert 0.0 < eps <= 1.0 + 1e-9


def test_crc_padding() -> None:
    assert crc_threshold_padding(0.0, 9) == pytest.approx(0.1)
    assert crc_threshold_padding(1.0, 9) == pytest.approx(1.0)


def test_match_calibration_permissive_threshold() -> None:
    # Many equivalent pairs at high similarity -> beta can be high & certified.
    pairs = [(0.95, 1)] * 20 + [(0.2, 0)] * 5
    cal = calibrate_match_threshold(pairs, 0.10)
    assert cal.certified
    assert cal.beta == pytest.approx(0.95)
    assert cal.n_match_pairs == 20


def test_match_calibration_infeasible_small_sample() -> None:
    # 4 equivalent pairs: 1/(n+1)=0.2 > target 0.1 -> infeasible, flagged.
    pairs = [(0.9, 1), (0.8, 1), (0.85, 1), (0.7, 1)]
    cal = calibrate_match_threshold(pairs, 0.10)
    assert not cal.certified
    assert cal.beta == pytest.approx(0.7)  # strictest threshold with zero FNM


def test_match_calibration_requires_positive() -> None:
    with pytest.raises(ValueError):
        calibrate_match_threshold([(0.5, 0), (0.4, 0)], 0.1)


def test_detection_calibration_certifies() -> None:
    # 30 clean models all near zero, one mild outlier.
    scores = [0.1 * i / 30 for i in range(29)] + [1.0]
    cal = calibrate_detection_threshold(scores, target_far=0.2, failure_prob=0.1)
    assert cal.certified
    assert cal.achieved_ucb <= 0.2 + 1e-9
    # New clean scores below threshold are not flagged.
    assert cal.threshold >= max(scores[:-1])


def test_detection_calibration_uncertifiable() -> None:
    cal = calibrate_detection_threshold([5.0], target_far=0.01, failure_prob=0.01)
    assert not cal.certified


def test_detection_calibration_empty_raises() -> None:
    with pytest.raises(ValueError):
        calibrate_detection_threshold([], 0.05, 0.05)
