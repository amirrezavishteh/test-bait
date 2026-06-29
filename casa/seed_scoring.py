"""Shared seed-scoring primitives used by the null builder and the scanner.

Centralising these keeps the *null* distribution and the *main scan* using an
identical scoring procedure — a precondition for the z-standardisation to be
meaningful (a seed must be scored the same way whether it is a random null seed
or a real candidate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from casa.interfaces.base import ModelInterface
from casa.similarity.base import SimilarityBackend


@dataclass
class SeedScore:
    """Outcome of scoring one seed across the clean prompts.

    Attributes:
        raw_score: Pairwise consistency fraction in ``[0, 1]``.
        inverted_target: Medoid generation (most central continuation).
        n_queries: Number of generations issued (= number of prompts).
    """

    raw_score: float
    inverted_target: str
    n_queries: int


def filter_seed_tokens(
    model: ModelInterface, min_surface_len: int = 1
) -> List[Tuple[int, str]]:
    """Return seedable ``(token_id, surface)`` pairs.

    Excludes special / pad / eos tokens and tokens whose decoded surface is
    empty or all-whitespace (per the null-sampling spec).

    Args:
        model: The model interface providing the vocabulary.
        min_surface_len: Minimum stripped surface length to keep.

    Returns:
        A list of seedable ``(id, surface)`` tuples in id order.
    """
    special = model.special_token_ids()
    out: List[Tuple[int, str]] = []
    for token_id, _raw in model.get_vocabulary():
        if token_id in special:
            continue
        surface = model.seed_surface(token_id)
        if len(surface.strip()) < min_surface_len:
            continue
        out.append((token_id, surface))
    return out


def score_seed(
    model: ModelInterface,
    similarity: SimilarityBackend,
    prompts: Sequence[str],
    surface: str,
    beta: float,
    max_new_tokens: int,
) -> SeedScore:
    """Generate continuations for a seed and compute its consistency score.

    Args:
        model: Model under test.
        similarity: Calibrated similarity backend.
        prompts: The ``N`` clean prompts.
        surface: Seed prefix string appended to each prompt.
        beta: Calibrated match threshold.
        max_new_tokens: Continuation length.

    Returns:
        A :class:`SeedScore`.
    """
    gens = model.generate(prompts, prefix=surface, max_new_tokens=max_new_tokens)
    texts = [g.text for g in gens]
    matrix = similarity.pairwise_matrix(texts, list(prompts))
    raw = _consistency_from_matrix(matrix, beta)
    medoid = _medoid_index(matrix)
    inverted = surface + texts[medoid] if texts else surface
    return SeedScore(raw_score=raw, inverted_target=inverted, n_queries=len(prompts))


def _consistency_from_matrix(matrix: List[List[float]], beta: float) -> float:
    """Fraction of off-diagonal upper-triangle entries ``>= beta``."""
    n = len(matrix)
    if n < 2:
        return 0.0
    matches = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if matrix[i][j] >= beta:
                matches += 1
    return matches / total if total else 0.0


def _medoid_index(matrix: List[List[float]]) -> int:
    """Index of the generation most similar to all others (max row sum)."""
    if not matrix:
        return 0
    best_i, best_sum = 0, float("-inf")
    for i, row in enumerate(matrix):
        s = sum(row) - row[i]  # exclude self-similarity (1.0)
        if s > best_sum:
            best_sum, best_i = s, i
    return best_i
