"""Offline embedding similarity backend (the default).

Uses a SentenceTransformer to embed each response and reports cosine similarity
rescaled from ``[-1, 1]`` to ``[0, 1]``.  This backend is fully offline and
requires no API key, which matches the weakness-zoo workflow ("no OpenAI API").

When ``sentence-transformers`` is unavailable it degrades to a deterministic
character-n-gram cosine so the package still imports and unit-tests run without
the heavy dependency.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence

from casa.similarity.base import SimilarityBackend
from casa.similarity.cache import SimilarityCache


class EmbeddingBackend(SimilarityBackend):
    """Cosine-of-embeddings semantic similarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache: Optional[SimilarityCache] = None,
    ) -> None:
        """Load (lazily) the embedding model.

        Args:
            model_name: SentenceTransformer model id.
            cache: Optional shared similarity cache.
        """
        super().__init__(cache)
        self._model_name = model_name
        self._model: Optional[object] = None
        self._tried_load = False

    @property
    def namespace(self) -> str:
        return f"emb:{self._model_name}"

    def _ensure_model(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
        except Exception:  # pragma: no cover - env dependent
            self._model = None  # fall back to n-gram cosine

    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        self._ensure_model()
        if self._model is not None:
            emb = self._model.encode([a, b], normalize_embeddings=True)  # type: ignore[attr-defined]
            cos = float(sum(x * y for x, y in zip(emb[0], emb[1])))
            return 0.5 * (cos + 1.0)
        return _ngram_cosine(a, b)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts (normalised); useful for batch precomputation.

        Args:
            texts: Strings to embed.

        Returns:
            One embedding vector per text (n-gram count vectors in fallback).
        """
        self._ensure_model()
        if self._model is not None:
            return [list(map(float, v)) for v in self._model.encode(list(texts), normalize_embeddings=True)]  # type: ignore[attr-defined]
        return [list(_ngram_counts(t).values()) for t in texts]


def _ngram_counts(text: str, n: int = 3) -> "Counter[str]":
    """Character n-gram multiset of ``text`` (lower-cased, whitespace-normalised)."""
    s = " ".join(text.lower().split())
    if len(s) < n:
        return Counter([s]) if s else Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _ngram_cosine(a: str, b: str) -> float:
    """Deterministic character-n-gram cosine in ``[0, 1]`` (offline fallback)."""
    ca, cb = _ngram_counts(a), _ngram_counts(b)
    if not ca or not cb:
        return 1.0 if a == b else 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[g] * cb[g] for g in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0
