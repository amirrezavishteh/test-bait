r"""Conformal-quantile threshold and three-state decision (spec §6.1, §7.4).

Given scores ``S_1..S_m`` from clean calibration models and a target
false-positive rate ``alpha``, the **split-conformal** threshold is the
``k``-th smallest calibration score with ``k = ceil((m+1)(1-alpha))``.  Under
exchangeability of the calibration scores and a new clean ``S_test``,

    P( S_test > lambda_hat ) <= alpha .

If ``k > m`` (too few calibration models for this ``alpha``) the threshold is
``+inf`` and the calibration is marked *uncertified* — no finite cut can be
guaranteed at that ``alpha`` with so few models (need ``m >= ceil(1/alpha) - 1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Sequence

STATE_CLEAN = "CLEAN"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_BACKDOORED = "BACKDOORED"


@dataclass
class QuantileCalibration:
    """Result of conformal-quantile threshold calibration.

    Attributes:
        threshold: ``lambda_hat`` (``math.inf`` if uncertifiable).
        alpha: Target false-positive rate.
        n_models: Number of clean calibration models.
        rank: The order statistic index ``k`` used (1-based).
        certified: Whether a finite certified threshold was found.
        date: ISO-8601 UTC timestamp.
    """

    threshold: float
    alpha: float
    n_models: int
    rank: int
    certified: bool
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "kind": "casa_lite_quantile",
            "threshold": self.threshold,
            "alpha": self.alpha,
            "n_models": self.n_models,
            "rank": self.rank,
            "certified": self.certified,
            "date": self.date,
        }


def conformal_quantile_threshold(
    clean_scores: Sequence[float], alpha: float
) -> QuantileCalibration:
    r"""Split-conformal threshold with ``P(S_test > lambda) <= alpha``.

    Args:
        clean_scores: Scores ``T(C_j)`` from known-clean calibration models.
        alpha: Target false-positive rate in ``(0, 1)``.

    Returns:
        A :class:`QuantileCalibration`.

    Raises:
        ValueError: If ``clean_scores`` is empty or ``alpha`` is out of range.
    """
    scores = sorted(float(s) for s in clean_scores)
    m = len(scores)
    if m == 0:
        raise ValueError("conformal calibration needs >= 1 clean score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    k = math.ceil((m + 1) * (1.0 - alpha))  # 1-based order-statistic index
    if k > m:
        return QuantileCalibration(
            threshold=math.inf, alpha=alpha, n_models=m, rank=k, certified=False
        )
    return QuantileCalibration(
        threshold=scores[k - 1], alpha=alpha, n_models=m, rank=k, certified=True
    )


def decide(score: float, threshold: float, uncertain_margin: float) -> str:
    """Map a model score to the three-state verdict.

    Args:
        score: The model's ``T(M)``.
        threshold: The decision threshold ``lambda_hat``.
        uncertain_margin: Fraction of the threshold below which the verdict is
            CLEAN; between that and the threshold it is UNCERTAIN.

    Returns:
        One of ``"CLEAN"`` / ``"UNCERTAIN"`` / ``"BACKDOORED"``.
    """
    if score > threshold:
        return STATE_BACKDOORED
    if math.isfinite(threshold) and score > uncertain_margin * threshold:
        return STATE_UNCERTAIN
    return STATE_CLEAN
