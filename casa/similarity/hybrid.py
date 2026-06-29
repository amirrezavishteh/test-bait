"""Hybrid similarity backend (code-structure first, NL fallback).

Recommended for general-purpose code models whose outputs mix code and natural
language: a pair is scored with the code-AST backend when *both* sides parse as
Python, otherwise with the natural-language backend.
"""

from __future__ import annotations

import ast
from typing import Optional

from casa.similarity.base import SimilarityBackend
from casa.similarity.cache import SimilarityCache
from casa.similarity.code_ast import CodeASTBackend


class HybridBackend(SimilarityBackend):
    """Dispatch each pair to the code or NL backend by parseability."""

    def __init__(
        self,
        nl_backend: SimilarityBackend,
        code_backend: Optional[CodeASTBackend] = None,
        cache: Optional[SimilarityCache] = None,
    ) -> None:
        """Compose a hybrid backend.

        Args:
            nl_backend: Natural-language fallback (embedding or LLM judge).
            code_backend: Code-AST backend; created if ``None``.
            cache: Shared cache; defaults to the NL backend's cache.
        """
        super().__init__(cache or nl_backend.cache)
        self._nl = nl_backend
        self._code = code_backend or CodeASTBackend(self.cache)

    @property
    def namespace(self) -> str:
        return f"hybrid({self._code.namespace}|{self._nl.namespace})"

    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        if _is_python(a) and _is_python(b):
            return self._code._raw_similarity(query, a, b)  # noqa: SLF001
        return self._nl._raw_similarity(query, a, b)  # noqa: SLF001


def _is_python(text: str) -> bool:
    """Whether ``text`` parses as Python source."""
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False
