r"""Anytime-valid sequential testing for the CASA vocabulary scan.

Scanning a 30k-256k token vocabulary is expensive, so CASA stops early once the
evidence against the null (*"this model is clean"*) is conclusive.  The correct
tool is an **e-process**: a running product of per-seed **e-values**, each a
non-negative statistic with expectation ``<= 1`` under the null.  By **Ville's
inequality** the probability that the product *ever* exceeds ``1/alpha`` is
``<= alpha``, uniformly over all stopping times — so we may peek after every
batch and stop the instant the product crosses ``1/alpha`` while still
controlling the family-wise error at ``alpha`` (unlike fixed-horizon p-values).

Design note (deviation from the literal spec).  The spec asks for an e-value
that is ``>= 1`` for ``z > 0`` and **exactly 1** for ``z <= 0``.  Those three
constraints together with ``E_null[e] <= 1`` are only satisfiable by the trivial
``e == 1`` (a point mass of 1 on half the line forces ``E > 1`` as soon as the
other half exceeds 1).  We therefore use the standard *betting* e-value

    e(z) = 1 + lambda * (2 * Phi(z) - 1),   lambda in (0, 1],

which preserves what Ville's inequality actually needs (``E_null[e] = 1`` exactly
when ``z`` is standard normal under the null) and the intended shape: ``e >= 1``
exactly when ``z >= 0`` and ``e`` grows with ``z``.  An unbounded
likelihood-ratio variant is also provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["normal_cdf", "betting_evalue", "lr_evalue", "EProcess"]


def normal_cdf(z: float) -> float:
    """Standard normal CDF ``Phi(z)`` via :func:`math.erf`.

    Args:
        z: A real number.

    Returns:
        ``Phi(z)`` in ``(0, 1)``.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def betting_evalue(z: float, lam: float = 0.9) -> float:
    r"""Bounded betting e-value ``1 + lambda * (2 Phi(z) - 1)``.

    Under the null ``z ~ N(0, 1)`` so ``Phi(z) ~ U(0, 1)`` and
    ``E[e] = 1 + lambda * (2 * E[U] - 1) = 1`` exactly.  ``e >= 1`` iff
    ``z >= 0``; values lie in ``[1 - lambda, 1 + lambda]``.

    Args:
        z: Standardised seed score.
        lam: Betting fraction in ``(0, 1]`` (aggressiveness).

    Returns:
        The e-value, a strictly positive float.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must lie in (0, 1]")
    return 1.0 + lam * (2.0 * normal_cdf(z) - 1.0)


def lr_evalue(z: float, mu: float = 2.0) -> float:
    r"""Likelihood-ratio e-value ``exp(mu * z - mu^2 / 2)`` for mean ``mu > 0``.

    Exactly an e-value under ``z ~ N(0, 1)`` (``E[e] = 1``); unbounded, so it
    accumulates evidence faster than :func:`betting_evalue` on strong seeds but
    can dip below 1 for small positive ``z``.

    Args:
        z: Standardised seed score.
        mu: Alternative mean (> 0) the test bets on.

    Returns:
        The e-value, a strictly positive float.
    """
    if mu <= 0.0:
        raise ValueError("mu must be > 0")
    return math.exp(mu * z - 0.5 * mu * mu)


@dataclass
class EProcess:
    """A running e-process (product of e-values) with a stopping boundary.

    Attributes:
        alpha: Significance level; the stop boundary is ``1 / alpha``.
        lam: Betting fraction for the default :func:`betting_evalue`.
        log_value: Natural log of the cumulative product (kept in log-space).
        n_steps: Number of e-values multiplied in so far.
        crossed_at: 1-based step index where the boundary was first crossed,
            or ``None`` if never crossed.
        log_max: Largest ``log_value`` reached (the running supremum).
    """

    alpha: float = 0.05
    lam: float = 0.9
    log_value: float = 0.0
    n_steps: int = 0
    crossed_at: Optional[int] = None
    log_max: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

    @property
    def value(self) -> float:
        """The current e-process value (the cumulative product)."""
        return math.exp(self.log_value)

    @property
    def boundary(self) -> float:
        """The Ville stopping boundary ``1 / alpha``."""
        return 1.0 / self.alpha

    @property
    def has_crossed(self) -> bool:
        """Whether the e-process has ever crossed the boundary."""
        return self.crossed_at is not None

    def update(self, z: float, evalue: Optional[float] = None) -> float:
        """Multiply one observation into the process and return its e-value.

        Args:
            z: The seed's standardised score (used when ``evalue`` is ``None``).
            evalue: Optional precomputed e-value; if given, ``z`` is ignored for
                the e-value (a positive value is required).

        Returns:
            The e-value that was multiplied in.
        """
        e = betting_evalue(z, self.lam) if evalue is None else float(evalue)
        if e <= 0.0:
            raise ValueError("e-values must be strictly positive")
        self.log_value += math.log(e)
        self.n_steps += 1
        if self.log_value > self.log_max:
            self.log_max = self.log_value
        if self.crossed_at is None and self.log_value >= math.log(self.boundary):
            self.crossed_at = self.n_steps
        return e
