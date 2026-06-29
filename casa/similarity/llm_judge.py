"""LLM-judge semantic similarity backend.

A capable instruction-following model rates how equivalent two responses are *as
answers to a specific question*, on a 0-``scale_max`` integer scale that is
normalised to ``[0, 1]``.  The prompt is engineered to judge semantic
equivalence rather than surface-level textual overlap.  Every call is cached
(via :class:`SimilarityCache`) because judge calls are slow and/or paid.

This mirrors the self-evaluation similarity used by Abbasi-Yadkori et al. (2024,
§2.2) to drive conformal abstention; here it drives CASA's consistency score.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from casa.logging_utils import get_logger
from casa.similarity.base import SimilarityBackend
from casa.similarity.cache import SimilarityCache

_JUDGE_PROMPT = """You are comparing two responses to the SAME question to \
decide whether they are SEMANTICALLY EQUIVALENT answers — that is, whether they \
convey the same essential meaning / intent, regardless of wording, phrasing, \
length, or style. Ignore surface textual overlap; judge meaning as answers to \
the question.

Question:
{query}

Response A:
{a}

Response B:
{b}

Rate semantic equivalence on an integer scale from 0 to {scale_max}, where 0 \
means completely unrelated meaning and {scale_max} means identical meaning. \
Reply with ONLY the integer."""

_INT_RE = re.compile(r"-?\d+")


class LLMJudgeBackend(SimilarityBackend):
    """OpenAI-compatible LLM judge for semantic equivalence."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        scale_max: float = 10.0,
        max_retries: int = 3,
        cache: Optional[SimilarityCache] = None,
    ) -> None:
        """Configure the judge client.

        Args:
            model: Judge model name.
            base_url: Optional OpenAI-compatible endpoint.
            scale_max: Upper end of the integer scale (> 0).
            max_retries: Retries on transient API errors.
            cache: Optional shared similarity cache.
        """
        super().__init__(cache)
        if scale_max <= 0:
            raise ValueError("scale_max must be > 0")
        self._model = model
        self._base_url = base_url
        self._scale_max = scale_max
        self._max_retries = max_retries
        self._log = get_logger()
        self._client: Optional[object] = None

    @property
    def namespace(self) -> str:
        return f"judge:{self._model}"

    def _ensure_client(self) -> object:
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=self._base_url)
        return self._client

    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        from openai import APIError, APIConnectionError, RateLimitError  # type: ignore

        prompt = _JUDGE_PROMPT.format(
            query=query, a=a, b=b, scale_max=int(self._scale_max)
        )
        delay = 1.0
        for attempt in range(self._max_retries):
            try:
                resp = self._ensure_client().chat.completions.create(  # type: ignore[attr-defined]
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                return self._parse(resp.choices[0].message.content or "")
            except (RateLimitError, APIConnectionError, APIError) as exc:
                self._log.warning("judge call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
        # Conservative on persistent failure: treat as non-match (low similarity).
        self._log.error("judge failed after %d retries; returning 0.0", self._max_retries)
        return 0.0

    def _parse(self, text: str) -> float:
        """Extract the first integer and normalise to ``[0, 1]``."""
        m = _INT_RE.search(text)
        if not m:
            return 0.0
        raw = float(m.group())
        return max(0.0, min(1.0, raw / self._scale_max))
