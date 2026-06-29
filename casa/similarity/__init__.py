"""Semantic similarity engine for CASA.

Exposes the backend ABC, the four concrete backends, the shared cache, and a
:func:`build_similarity` factory that wires a backend from a
:class:`casa.config.SimilarityConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from casa.similarity.base import SimilarityBackend
from casa.similarity.cache import SimilarityCache
from casa.similarity.code_ast import CodeASTBackend
from casa.similarity.embedding import EmbeddingBackend
from casa.similarity.hybrid import HybridBackend
from casa.similarity.llm_judge import LLMJudgeBackend

if TYPE_CHECKING:  # pragma: no cover
    from casa.config import SimilarityConfig

__all__ = [
    "SimilarityBackend",
    "SimilarityCache",
    "EmbeddingBackend",
    "LLMJudgeBackend",
    "CodeASTBackend",
    "HybridBackend",
    "build_similarity",
]


def build_similarity(config: "SimilarityConfig") -> SimilarityBackend:
    """Construct the similarity backend named by ``config.backend``.

    Args:
        config: A :class:`casa.config.SimilarityConfig`.

    Returns:
        A ready :class:`SimilarityBackend` sharing one on-disk cache.

    Raises:
        ValueError: If ``config.backend`` is unknown.
    """
    cache = SimilarityCache(config.cache_dir and f"{config.cache_dir}/cache.json")
    backend = config.backend
    if backend == "embedding":
        return EmbeddingBackend(config.embedding_model, cache=cache)
    if backend == "llm_judge":
        return LLMJudgeBackend(
            model=config.judge_model,
            base_url=config.judge_base_url,
            scale_max=config.scale_max,
            max_retries=config.max_retries,
            cache=cache,
        )
    if backend == "code_ast":
        return CodeASTBackend(cache=cache)
    if backend == "hybrid":
        nl = EmbeddingBackend(config.embedding_model, cache=cache)
        return HybridBackend(nl_backend=nl, cache=cache)
    raise ValueError(f"unknown similarity backend {backend!r}")
