r"""Distribution-free conformal calibration for CASA.

This module implements the two calibration procedures CASA relies on, taking
the formulas from *Mitigating LLM Hallucinations via Conformal Abstention*
(Abbasi-Yadkori et al., 2024, arXiv:2405.01563):

1. **Match-threshold calibration** (the similarity ``beta``) via Conformal
   Risk Control — paper Eq. (3) and §5.
2. **Detection-threshold calibration** (the z-score cut) via the
   **Hoeffding-Bentkus** upper confidence bound — paper §4.

Both are *distribution-free*: the guarantees hold for any data distribution,
assuming only exchangeability of the calibration sample and the test point.

Everything here is pure Python + ``math`` (no numpy / scipy) so the certified
core imports anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "kl_bernoulli",
    "binom_cdf",
    "hoeffding_bentkus_eps",
    "hoeffding_bentkus_ucb",
    "crc_threshold_padding",
    "MatchCalibration",
    "DetectionCalibration",
    "calibrate_match_threshold",
    "calibrate_detection_threshold",
]


# --------------------------------------------------------------------------- #
# Bernoulli relative entropy and binomial CDF (building blocks for HB)
# --------------------------------------------------------------------------- #
def kl_bernoulli(p: float, q: float) -> float:
    r"""Relative entropy ``kl(p, q)`` between two Bernoulli distributions.

    ``kl(p, q) = p ln(p/q) + (1 - p) ln((1 - p)/(1 - q))`` with the standard
    conventions ``0 ln 0 = 0`` and ``kl = +inf`` when a term divides by zero.

    Args:
        p: Empirical success probability in ``[0, 1]``.
        q: Reference success probability in ``[0, 1]``.

    Returns:
        The non-negative KL divergence (possibly ``math.inf``).
    """
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("kl_bernoulli requires p, q in [0, 1]")
    total = 0.0
    if p > 0.0:
        if q <= 0.0:
            return math.inf
        total += p * math.log(p / q)
    if p < 1.0:
        if q >= 1.0:
            return math.inf
        total += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return max(total, 0.0)


def binom_cdf(k: int, n: int, p: float) -> float:
    r"""``P(Bin(n, p) <= k)``, computed stably in log-space via ``lgamma``.

    Args:
        k: Upper index (clamped to ``[0, n]``).
        n: Number of trials (>= 0).
        p: Success probability in ``[0, 1]``.

    Returns:
        The cumulative probability in ``[0, 1]``.
    """
    if n < 0:
        raise ValueError("binom_cdf requires n >= 0")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0  # all mass at 0 <= k
    if p >= 1.0:
        return 0.0  # all mass at n > k
    log_p = math.log(p)
    log_q = math.log1p(-p)
    acc = 0.0
    for i in range(0, k + 1):
        log_coef = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        )
        acc += math.exp(log_coef + i * log_p + (n - i) * log_q)
    return min(acc, 1.0)


def hoeffding_bentkus_eps(loss_hat: float, p: float, n: int) -> float:
    r"""The Hoeffding-Bentkus tail function ``eps_hb(t, p)`` (paper §4).

    ``eps_hb(t, p) = min{ exp(-n * kl(t, p)),  P(Bin(n, p) <= ceil(n * t)) }``

    Args:
        loss_hat: Empirical risk ``t`` in ``[0, 1]``.
        p: Candidate true risk ``p`` in ``[0, 1]``.
        n: Calibration sample size (>= 1).

    Returns:
        ``min`` of the Chernoff (KL) term and the Bentkus binomial term.
    """
    if n < 1:
        raise ValueError("hoeffding_bentkus_eps requires n >= 1")
    chernoff = math.exp(-n * kl_bernoulli(loss_hat, p))
    # The factor `e` is the Bentkus (2004) constant carried by Bates et al.
    # (2021, "Distribution-Free, Risk-Controlling Prediction Sets"); the
    # abstention paper's typeset min{...} elides it, but it is required for the
    # binomial term to be a valid bound.  Keeping it can only make eps larger
    # (the min more conservative), so the resulting UCB stays valid.
    bentkus = math.e * binom_cdf(math.ceil(n * loss_hat), n, p)
    return min(chernoff, bentkus)


def hoeffding_bentkus_ucb(loss_hat: float, n: int, delta: float) -> float:
    r"""Hoeffding-Bentkus upper confidence bound on the true risk.

    Returns ``sup{ p in [0, 1] : eps_hb(loss_hat, p) >= delta }`` — the tightest
    of the bounds considered by Bates et al. (2021) for Bernoulli losses, found
    by bisection because ``eps_hb(loss_hat, .)`` is non-increasing in ``p`` for
    ``p >= loss_hat``.

    Args:
        loss_hat: Empirical risk in ``[0, 1]``.
        n: Calibration sample size (>= 1).
        delta: Confidence failure probability in ``(0, 1)``.

    Returns:
        Upper bound on the true risk that holds with probability >= 1 - delta.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    loss_hat = min(max(loss_hat, 0.0), 1.0)
    # eps_hb(loss_hat, loss_hat) = min(1, e * CDF) >= delta for sensible delta;
    # if even that fails, the UCB is loss_hat itself.
    if hoeffding_bentkus_eps(loss_hat, loss_hat, n) < delta:
        return loss_hat
    lo, hi = loss_hat, 1.0
    for _ in range(100):  # ~1e-30 precision; plenty
        mid = 0.5 * (lo + hi)
        if hoeffding_bentkus_eps(loss_hat, mid, n) >= delta:
            lo = mid
        else:
            hi = mid
    return lo


def crc_threshold_padding(empirical_loss: float, n: int) -> float:
    r"""CRC-padded risk estimate ``n/(n+1) * L + 1/(n+1)`` (paper Eq. 3).

    Args:
        empirical_loss: Mean loss ``L_n(lambda)`` in ``[0, 1]``.
        n: Number of calibration points the loss was averaged over (>= 1).

    Returns:
        The conformal-risk-control upper estimate compared against ``alpha``.
    """
    if n < 1:
        raise ValueError("crc_threshold_padding requires n >= 1")
    return (n / (n + 1)) * empirical_loss + 1.0 / (n + 1)


# --------------------------------------------------------------------------- #
# Match-threshold calibration  (the similarity beta)
# --------------------------------------------------------------------------- #
@dataclass
class MatchCalibration:
    """Result of calibrating the similarity match threshold ``beta``.

    Attributes:
        beta: Calibrated threshold in ``[0, 1]``.
        target_match_error: Requested error rate alpha.
        achieved_loss: CRC-padded empirical loss at ``beta``.
        n_match_pairs: Number of human-equivalent pairs used (the CRC ``n``).
        n_total_pairs: Total labelled pairs supplied.
        certified: ``True`` iff some threshold met the target (needs
            ``n_match_pairs >= 1/alpha - 1`` because of the ``1/(n+1)`` padding).
        date: ISO-8601 UTC calibration timestamp.
    """

    beta: float
    target_match_error: float
    achieved_loss: float
    n_match_pairs: int
    n_total_pairs: int
    certified: bool = True
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "kind": "match",
            "beta": self.beta,
            "target_match_error": self.target_match_error,
            "achieved_loss": self.achieved_loss,
            "n_match_pairs": self.n_match_pairs,
            "n_total_pairs": self.n_total_pairs,
            "certified": self.certified,
            "date": self.date,
        }


def calibrate_match_threshold(
    pairs: Sequence[Tuple[float, int]],
    target_match_error: float,
    grid: Optional[Sequence[float]] = None,
) -> MatchCalibration:
    r"""Calibrate the similarity threshold ``beta`` via Conformal Risk Control.

    Following paper §5, the match function is ``m_beta(X, Y', Y) = 1{s >= beta}``.
    The power-relevant loss for CASA is a *false non-match*: a pair that a human
    judged semantically **equivalent** (``label == 1``) but which the match
    function rejects because ``s < beta``.  This loss is non-decreasing in
    ``beta``, so we pick the **most permissive** (largest) ``beta`` whose
    CRC-padded loss (Eq. 3) stays at or below ``alpha``::

        beta_hat = sup { beta : n/(n+1) * L_n(beta) + 1/(n+1) <= alpha }

    where ``L_n(beta) = mean over equivalent pairs of 1{s < beta}`` and ``n`` is
    the number of equivalent pairs.  This guarantees ``E[false-non-match rate]
    <= alpha`` on new data (paper Eq. just below §5's L1 discussion).

    Note: the paper's dual presentation takes an ``inf`` to control the
    complementary *false-match* rate; for an increasing loss the ``inf`` is
    degenerate (it returns the smallest grid point), so CASA uses the ``sup`` of
    the same padded CRC expression to control the error that actually matters
    for detection power.

    Args:
        pairs: Iterable of ``(similarity, label)`` where ``similarity`` is in
            ``[0, 1]`` and ``label == 1`` means human-equivalent.
        target_match_error: Target false-non-match rate alpha in ``(0, 1)``.
        grid: Optional explicit candidate thresholds; defaults to the observed
            similarities of the equivalent pairs plus ``0`` and ``1``.

    Returns:
        A :class:`MatchCalibration` with the selected ``beta``.

    Raises:
        ValueError: If no equivalent (``label == 1``) pairs are supplied.
    """
    pairs = list(pairs)
    matches = [s for (s, y) in pairs if int(y) == 1]
    n = len(matches)
    if n == 0:
        raise ValueError(
            "match calibration needs >= 1 equivalent pair (label == 1)"
        )

    if grid is None:
        cand = sorted({0.0, 1.0, *[float(s) for s in matches]})
    else:
        cand = sorted({float(g) for g in grid})

    def padded_loss(beta: float) -> float:
        loss = sum(1 for s in matches if s < beta) / n
        return crc_threshold_padding(loss, n)

    losses = [(beta, padded_loss(beta)) for beta in cand]
    feasible = [beta for beta, pl in losses if pl <= target_match_error]
    if feasible:
        # padded_loss is non-decreasing in beta, so the feasible set is a lower
        # interval; its maximum is the most permissive certified threshold.
        best_beta = max(feasible)
        certified = True
    else:
        # Target infeasible at this sample size (the 1/(n+1) padding alone
        # exceeds alpha).  Fall back to the strictest threshold still attaining
        # the minimum achievable loss (zero false-non-matches), and flag it.
        min_loss = min(pl for _, pl in losses)
        best_beta = max(beta for beta, pl in losses if pl <= min_loss + 1e-12)
        certified = False
    return MatchCalibration(
        beta=best_beta,
        target_match_error=target_match_error,
        achieved_loss=padded_loss(best_beta),
        n_match_pairs=n,
        n_total_pairs=len(pairs),
        certified=certified,
    )


# --------------------------------------------------------------------------- #
# Detection-threshold calibration  (the z-score cut)
# --------------------------------------------------------------------------- #
@dataclass
class DetectionCalibration:
    """Result of calibrating the detection (z-score) threshold.

    Attributes:
        threshold: Certified z-score threshold; flag iff ``score > threshold``.
        target_far: Target false-alarm rate alpha.
        failure_prob: Confidence failure probability delta.
        n_models: Number of clean calibration models.
        achieved_ucb: Hoeffding-Bentkus UCB on the FAR at ``threshold``.
        certified: Whether a finite threshold meeting the bound was found.
        date: ISO-8601 UTC calibration timestamp.
    """

    threshold: float
    target_far: float
    failure_prob: float
    n_models: int
    achieved_ucb: float
    certified: bool
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "kind": "detection",
            "threshold": self.threshold,
            "target_far": self.target_far,
            "failure_prob": self.failure_prob,
            "n_models": self.n_models,
            "achieved_ucb": self.achieved_ucb,
            "certified": self.certified,
            "date": self.date,
        }


def calibrate_detection_threshold(
    clean_scores: Sequence[float],
    target_far: float,
    failure_prob: float,
    grid: Optional[Sequence[float]] = None,
) -> DetectionCalibration:
    r"""Pick the smallest z-threshold whose certified FAR is ``<= target_far``.

    For a candidate threshold ``t`` the empirical false-alarm rate on the clean
    calibration models is ``FAR(t) = mean(1{score > t})`` (non-increasing in
    ``t``).  Using the Hoeffding-Bentkus UCB (:func:`hoeffding_bentkus_ucb`) we
    return the smallest ``t`` with ``UCB(FAR(t), n, delta) <= target_far``.  By
    the RCPS guarantee this certifies, with probability ``>= 1 - delta`` over the
    calibration draw, that the true FAR on new clean models is ``<= target_far``.

    Args:
        clean_scores: Max-z (CAH) scores from known-clean models.
        target_far: Target false-alarm rate alpha in ``(0, 1)``.
        failure_prob: Confidence failure probability delta in ``(0, 1)``.
        grid: Optional candidate thresholds; defaults to observed scores plus a
            point strictly above the maximum.

    Returns:
        A :class:`DetectionCalibration`.  If no finite threshold satisfies the
        bound, ``certified`` is ``False`` and ``threshold`` is set just above the
        maximum observed score (still the tightest available cut).

    Raises:
        ValueError: If ``clean_scores`` is empty.
    """
    scores = [float(s) for s in clean_scores]
    n = len(scores)
    if n == 0:
        raise ValueError("detection calibration needs >= 1 clean score")
    if not 0.0 < target_far < 1.0:
        raise ValueError("target_far must lie in (0, 1)")
    if not 0.0 < failure_prob < 1.0:
        raise ValueError("failure_prob must lie in (0, 1)")

    hi = max(scores)
    above_max = hi + max(1.0, abs(hi)) * 1e-6
    if grid is None:
        cand = sorted({*scores, above_max})
    else:
        cand = sorted({float(g) for g in grid})

    def far(t: float) -> float:
        return sum(1 for s in scores if s > t) / n

    best: Optional[DetectionCalibration] = None
    for t in cand:
        ucb = hoeffding_bentkus_ucb(far(t), n, failure_prob)
        if ucb <= target_far:
            best = DetectionCalibration(
                threshold=t,
                target_far=target_far,
                failure_prob=failure_prob,
                n_models=n,
                achieved_ucb=ucb,
                certified=True,
            )
            break  # cand ascending -> first satisfying t is the smallest

    if best is None:
        best = DetectionCalibration(
            threshold=above_max,
            target_far=target_far,
            failure_prob=failure_prob,
            n_models=n,
            achieved_ucb=hoeffding_bentkus_ucb(far(above_max), n, failure_prob),
            certified=False,
        )
    return best
