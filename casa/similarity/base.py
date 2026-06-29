"""Similarity-backend ABC and the pairwise semantic-consistency score.

The *semantic consistency score* for a seed is the fraction of generation pairs
that a calibrated similarity function deems equivalent.  A benign seed yields
diverse, prompt-appropriate continuations (low score); a backdoor seed yields
semantically uniform continuations regardless of prompt (high score) — a
*prompt-unjustified certainty*.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence

from casa.similarity.cache import SimilarityCache


class SimilarityBackend(abc.ABC):
    """Computes a calibrated semantic similarity in ``[0, 1]`` for two texts."""

    def __init__(self, cache: Optional[SimilarityCache] = None) -> None:
        """Initialise with an optional shared cache.

        Args:
            cache: A :class:`SimilarityCache`; a fresh in-memory one is created
                when ``None``.
        """
        self.cache = cache or SimilarityCache()

    @property
    @abc.abstractmethod
    def namespace(self) -> str:
        """A short backend id used to namespace cache keys."""

    @abc.abstractmethod
    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        """Backend-specific similarity of ``a`` and ``b`` given ``query``.

        Args:
            query: The prompt/question both responses answer.
            a: First response text.
            b: Second response text.

        Returns:
            A similarity in ``[0, 1]``.
        """

    def similarity(self, query: str, a: str, b: str) -> float:
        """Cached, reflexive similarity of ``a`` and ``b`` given ``query``.

        Args:
            query: The prompt both responses answer.
            a: First response text.
            b: Second response text.

        Returns:
            Similarity in ``[0, 1]`` (``1.0`` for identical strings).
        """
        if a == b:
            return 1.0
        key = self.cache.key(self.namespace, query, a, b)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        value = max(0.0, min(1.0, self._raw_similarity(query, a, b)))
        self.cache.put(key, value)
        return value

    def consistency_score(
        self,
        generations: Sequence[str],
        prompts: Sequence[str],
        beta: float,
    ) -> float:
        r"""Pairwise semantic-consistency score in ``[0, 1]``.

        For every unordered pair ``(i, j)`` the similarity is evaluated under
        *both* ``prompts[i]`` and ``prompts[j]`` and averaged (symmetry), then
        thresholded at ``beta``.  The score is the fraction of pairs that match.

        Args:
            generations: ``N`` continuation strings (one per prompt).
            prompts: The ``N`` prompts that produced them (same order).
            beta: Calibrated match threshold in ``[0, 1]``.

        Returns:
            Fraction of matching pairs in ``[0, 1]``; ``0.0`` when ``N < 2``.

        Raises:
            ValueError: If ``generations`` and ``prompts`` differ in length.
        """
        if len(generations) != len(prompts):
            raise ValueError("generations and prompts must have equal length")
        n = len(generations)
        if n < 2:
            return 0.0
        matches = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                s_i = self.similarity(prompts[i], generations[i], generations[j])
                s_j = self.similarity(prompts[j], generations[i], generations[j])
                if 0.5 * (s_i + s_j) >= beta:
                    matches += 1
        return matches / total if total else 0.0

    def pairwise_matrix(
        self, generations: Sequence[str], prompts: Sequence[str]
    ) -> List[List[float]]:
        """Return the symmetric ``N x N`` averaged-similarity matrix.

        Args:
            generations: ``N`` continuation strings.
            prompts: The ``N`` prompts (same order).

        Returns:
            A list-of-lists with ``1.0`` on the diagonal.
        """
        n = len(generations)
        mat = [[1.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = 0.5 * (
                    self.similarity(prompts[i], generations[i], generations[j])
                    + self.similarity(prompts[j], generations[i], generations[j])
                )
                mat[i][j] = mat[j][i] = s
        return mat
