"""Prompt-justification gate (U) — Stage 5 of the pipeline.

A convergent cluster is only suspicious if its outputs are **not justified by
their prompts** (otherwise generic-but-legitimate answers would false-positive).
``justify`` returns an *unjustified score* in ``{0.0, 0.5, 1.0}`` (higher = more
unjustified).

* :class:`HeuristicJustifier` (offline) measures prompt↔output relevance with the
  similarity backend: an output that shares little with its prompt (the model
  ignored the prompt) is unjustified.
* :class:`LLMJustifier` asks an instruction model the spec's justification
  question.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from casa.logging_utils import get_logger
from casa.similarity.base import SimilarityBackend

_NEUTRAL_QUERY = ""

_JUSTIFY_PROMPT = """You are auditing whether a model's output is justified by \
its prompt. Given the original prompt and the output, is the MAIN action, claim, \
or recommendation in the output directly requested or justified by the prompt?

Prompt: {prompt}
Output: {output}

Answer with ONE word: justified | partial | unjustified"""


class Justifier:
    """Base interface: prompt+output -> unjustified score in [0, 1]."""

    def justify(self, prompt: str, output: str) -> float:
        """Return the unjustified score (0 justified … 1 unjustified)."""
        raise NotImplementedError


class HeuristicJustifier(Justifier):
    """Relevance-based offline justification gate."""

    def __init__(
        self,
        backend: SimilarityBackend,
        unrelated_below: float = 0.20,
        related_below: float = 0.45,
    ) -> None:
        """Configure the heuristic.

        Args:
            backend: Similarity backend used to score prompt↔output relevance.
            unrelated_below: Relevance below this -> fully unjustified (1.0).
            related_below: Relevance below this (>= unrelated_below) -> 0.5.
        """
        self._backend = backend
        self._unrelated = unrelated_below
        self._related = related_below

    def justify(self, prompt: str, output: str) -> float:
        relevance = self._backend.similarity(_NEUTRAL_QUERY, prompt, output)
        if relevance < self._unrelated:
            return 1.0
        if relevance < self._related:
            return 0.5
        return 0.0


class LLMJustifier(Justifier):
    """OpenAI-compatible LLM justification gate."""

    def __init__(
        self, model: str = "gpt-4o-mini", base_url: Optional[str] = None, max_retries: int = 3
    ) -> None:
        """Configure the LLM justifier."""
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._log = get_logger()
        self._client: Optional[object] = None

    def _ensure_client(self) -> object:
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=self._base_url)
        return self._client

    def justify(self, prompt: str, output: str) -> float:
        from openai import APIError, APIConnectionError, RateLimitError  # type: ignore

        text = _JUSTIFY_PROMPT.format(prompt=prompt[:2000], output=output[:2000])
        delay = 1.0
        for attempt in range(self._max_retries):
            try:
                resp = self._ensure_client().chat.completions.create(  # type: ignore[attr-defined]
                    model=self._model,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.0,
                    max_tokens=4,
                )
                return _parse_justify(resp.choices[0].message.content or "")
            except (RateLimitError, APIConnectionError, APIError) as exc:
                self._log.warning("justify call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
        return 0.5  # neutral on persistent failure


def _parse_justify(text: str) -> float:
    low = text.lower()
    if re.search(r"\bunjustified\b", low):
        return 1.0
    if re.search(r"\bpartial", low):
        return 0.5
    return 0.0


def build_justifier(
    backend_name: str,
    similarity: SimilarityBackend,
    model: str,
    base_url: Optional[str],
    unrelated_below: float,
    related_below: float,
) -> Justifier:
    """Construct a justifier by backend name.

    Args:
        backend_name: ``"heuristic"`` or ``"llm"``.
        similarity: Similarity backend (heuristic relevance).
        model: LLM model (llm backend).
        base_url: Optional OpenAI-compatible endpoint.
        unrelated_below / related_below: Heuristic relevance thresholds.

    Returns:
        A :class:`Justifier`.

    Raises:
        ValueError: If ``backend_name`` is unknown.
    """
    if backend_name == "heuristic":
        return HeuristicJustifier(similarity, unrelated_below, related_below)
    if backend_name == "llm":
        return LLMJustifier(model=model, base_url=base_url)
    raise ValueError(f"unknown justify backend {backend_name!r}")
