"""Unit tests for the similarity engine and pairwise consistency."""

from __future__ import annotations

import pytest

from casa.similarity.cache import SimilarityCache
from casa.similarity.code_ast import CodeASTBackend
from casa.similarity.embedding import EmbeddingBackend, _ngram_cosine
from casa.similarity.hybrid import HybridBackend


def test_code_ast_alpha_equivalence() -> None:
    c = CodeASTBackend()
    a = "def f(x):\n    y = x + 1\n    return y"
    b = "def g(a):\n    b = a + 1\n    return b"
    assert c.similarity("q", a, b) > 0.95


def test_code_ast_distinguishes_structure() -> None:
    c = CodeASTBackend()
    a = "def f(x):\n    return x + 1"
    d = "def h(z):\n    for i in range(z):\n        print(i * 2 - 7)"
    assert c.similarity("q", a, d) < 0.9


def test_code_ast_malformed_fallback() -> None:
    c = CodeASTBackend()
    # Unparseable on both sides -> string fallback, no crash.
    assert 0.0 <= c.similarity("q", "def (", "while )") <= 1.0
    assert c.similarity("q", "def (", "def (") == 1.0


def test_ngram_cosine_offline() -> None:
    assert _ngram_cosine("hello world", "hello world") == pytest.approx(1.0)
    assert _ngram_cosine("abcdef", "zyxwvu") < 0.2


def test_consistency_all_match_all_diverse() -> None:
    b = EmbeddingBackend()  # ngram fallback offline
    same = ["the quick brown fox"] * 5
    prompts = [f"p{i}" for i in range(5)]
    assert b.consistency_score(same, prompts, beta=0.9) == 1.0
    diverse = ["alpha one", "beta two", "gamma three", "delta four", "omega five"]
    assert b.consistency_score(diverse, prompts, beta=0.9) == 0.0


def test_consistency_single_prompt_is_zero() -> None:
    b = EmbeddingBackend()
    assert b.consistency_score(["x"], ["p"], beta=0.5) == 0.0


def test_consistency_length_mismatch_raises() -> None:
    b = EmbeddingBackend()
    with pytest.raises(ValueError):
        b.consistency_score(["a", "b"], ["only-one"], beta=0.5)


def test_cache_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "c.json")
    cache = SimilarityCache(path)
    k = cache.key("ns", "q", "a", "b")
    assert cache.key("ns", "q", "b", "a") == k  # order-canonical
    cache.put(k, 0.42)
    cache.flush()
    assert SimilarityCache(path).get(k) == pytest.approx(0.42)


def test_hybrid_dispatch() -> None:
    nl = EmbeddingBackend()
    h = HybridBackend(nl_backend=nl)
    code_a = "def f(x):\n    return x + 1"
    code_b = "def g(y):\n    return y + 1"
    # Both parse -> code path -> high structural similarity.
    assert h.similarity("q", code_a, code_b) > 0.9
    # Non-code -> NL path -> still in range.
    assert 0.0 <= h.similarity("q", "hello there", "general kenobi") <= 1.0
