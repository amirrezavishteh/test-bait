"""Model-specific null-distribution builder (Component 3).

Random vocabulary seeds carry no backdoor significance, so the spread of their
consistency scores defines this model's *baseline* output consistency.  Scoring
real candidates as z-scores against this baseline is what lets CASA avoid the
code-domain false positives that plague raw-consistency scanners: a benign idiom
that is consistent across prompts is *also* consistent in the null sample, so its
z-score is ~0.
"""

from __future__ import annotations

import random
import statistics
from typing import List, Optional, Sequence, Tuple

from casa.interfaces.base import ModelInterface
from casa.logging_utils import get_logger
from casa.scan_result import NullStats
from casa.seed_scoring import filter_seed_tokens, score_seed
from casa.similarity.base import SimilarityBackend


class NullDistribution:
    """Estimated null mean/std plus the raw sample, with a z-score method."""

    def __init__(self, scores: Sequence[float], min_std: float) -> None:
        """Summarise a sample of null consistency scores.

        Args:
            scores: Consistency scores from random seeds.
            min_std: Floor applied to the std to avoid divide-by-~0.

        Raises:
            ValueError: If fewer than two scores are supplied.
        """
        if len(scores) < 2:
            raise ValueError("null distribution needs >= 2 samples")
        self.scores: List[float] = list(scores)
        self.mean: float = statistics.fmean(self.scores)
        self.std: float = max(statistics.pstdev(self.scores), min_std)
        self.sample_size: int = len(self.scores)

    def z(self, raw_score: float) -> float:
        """Standardise a raw consistency score to a z-score."""
        return (raw_score - self.mean) / self.std

    def stats(self) -> NullStats:
        """Return the serialisable :class:`NullStats` summary."""
        return NullStats(mean=self.mean, std=self.std, sample_size=self.sample_size)


class NullDistributionBuilder:
    """Builds a :class:`NullDistribution` for a model."""

    def __init__(
        self,
        model: ModelInterface,
        similarity: SimilarityBackend,
        beta: float,
        max_new_tokens: int,
        min_std: float = 1e-6,
    ) -> None:
        """Configure the builder.

        Args:
            model: Model under test.
            similarity: Calibrated similarity backend.
            beta: Calibrated match threshold.
            max_new_tokens: Continuation length per seed.
            min_std: Std floor passed to :class:`NullDistribution`.
        """
        self._model = model
        self._sim = similarity
        self._beta = beta
        self._max_new = max_new_tokens
        self._min_std = min_std
        self._log = get_logger()
        self.n_queries = 0

    def build(
        self,
        prompts: Sequence[str],
        sample_size: int,
        seed: int,
        candidates: Optional[Sequence[Tuple[int, str]]] = None,
    ) -> NullDistribution:
        """Sample random seeds and return the null distribution.

        Args:
            prompts: Clean prompts.
            sample_size: Number of random seeds to score.
            seed: RNG seed for deterministic sampling.
            candidates: Optional pre-filtered ``(id, surface)`` pool; computed
                from the model vocabulary when ``None``.

        Returns:
            The estimated :class:`NullDistribution`.

        Raises:
            ValueError: If no seedable tokens are available.
        """
        pool = list(candidates) if candidates is not None else filter_seed_tokens(self._model)
        if not pool:
            raise ValueError("no seedable vocabulary tokens for null sampling")
        rng = random.Random(seed)
        k = min(sample_size, len(pool))
        chosen = rng.sample(pool, k)
        self._log.info("building null distribution from %d random seeds", k)
        scores: List[float] = []
        for idx, (_tid, surface) in enumerate(chosen):
            res = score_seed(
                self._model, self._sim, prompts, surface, self._beta, self._max_new
            )
            self.n_queries += res.n_queries
            scores.append(res.raw_score)
            if (idx + 1) % max(1, k // 5) == 0:
                self._log.debug("null progress %d/%d", idx + 1, k)
        nd = NullDistribution(scores, self._min_std)
        self._log.info(
            "null distribution: mean=%.4f std=%.4f n=%d", nd.mean, nd.std, nd.sample_size
        )
        return nd
