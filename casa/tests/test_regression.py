"""Regression tests — the certified guarantee holds on synthetic data.

We draw synthetic CAH scores for clean models, calibrate the detection threshold
with Hoeffding-Bentkus, then measure the realised false-alarm rate on a large
held-out clean set.  Per the conformal guarantee (prob >= 1 - delta over the
calibration draw), the realised FAR should be <= target alpha in >= 95% of
independent trials.
"""

from __future__ import annotations

import random

from casa.conformal import calibrate_detection_threshold


def _draw_clean_cah(rng: random.Random) -> float:
    """A clean model's CAH score: max of 64 standard-normal seed z-scores."""
    return max(rng.gauss(0.0, 1.0) for _ in range(64))


def test_detection_far_coverage() -> None:
    alpha, delta = 0.10, 0.10
    n_cal, n_test, n_trials = 30, 2000, 100
    master = random.Random(12345)
    passes = 0
    realised_fars = []
    for _ in range(n_trials):
        rng = random.Random(master.random())
        cal = [_draw_clean_cah(rng) for _ in range(n_cal)]
        result = calibrate_detection_threshold(cal, target_far=alpha, failure_prob=delta)
        test = [_draw_clean_cah(rng) for _ in range(n_test)]
        far = sum(1 for s in test if s > result.threshold) / n_test
        realised_fars.append(far)
        if far <= alpha:
            passes += 1
    coverage = passes / n_trials
    mean_far = sum(realised_fars) / len(realised_fars)
    print(f"coverage={coverage:.2f}  mean realised FAR={mean_far:.4f}  (target {alpha})")
    assert coverage >= 0.95
    assert mean_far <= alpha  # HB is conservative -> mean well under target


def test_certified_flag_set_when_feasible() -> None:
    rng = random.Random(0)
    cal = [_draw_clean_cah(rng) for _ in range(30)]
    result = calibrate_detection_threshold(cal, target_far=0.2, failure_prob=0.1)
    assert result.certified
    assert result.achieved_ucb <= 0.2 + 1e-9
