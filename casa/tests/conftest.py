"""Shared pytest fixtures and offline mocks (no GPU, no API).

The mocks let the whole pipeline run deterministically on CPU:

* :class:`ToySemanticBackend` — a *semantic* similarity that reads a hidden
  ``[m=ID]`` meaning tag, so paraphrases with different wording but the same tag
  count as equivalent.  This isolates CASA's logic from embedding-model quality.
* :class:`MockModel` — a configurable :class:`ModelInterface` whose ``generate``
  is supplied per test, returning both meaning-tagged text (for CASA) and token
  ids (for the BAIT Q-Score).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Set, Tuple

import pytest

from casa.interfaces.base import Generation, ModelInterface
from casa.similarity.base import SimilarityBackend

_MEANING_RE = re.compile(r"\[m=([^\]]*)\]")


def make_text(meaning: str, surface: str) -> str:
    """Compose a generation carrying a hidden meaning tag plus surface text."""
    return f"[m={meaning}]{surface}"


def meaning_of(text: str) -> str:
    """Extract the meaning tag from text (or the text itself if untagged)."""
    m = _MEANING_RE.search(text)
    return m.group(1) if m else text.strip()


class ToySemanticBackend(SimilarityBackend):
    """Similarity = 1.0 iff two texts share the same ``[m=ID]`` meaning tag."""

    @property
    def namespace(self) -> str:
        return "toy"

    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        return 1.0 if meaning_of(a) == meaning_of(b) else 0.0


GenFn = Callable[[str, str, int], Tuple[str, List[int]]]


class MockModel(ModelInterface):
    """A model whose per-(prefix, prompt) output is driven by ``gen_fn``."""

    def __init__(
        self,
        gen_fn: GenFn,
        vocab: Sequence[Tuple[int, str]],
        name: str = "mock",
        supports_lp: bool = True,
        logprob: float = -0.01,
    ) -> None:
        self._gen = gen_fn
        self._vocab = list(vocab)
        self._name = name
        self._lp = supports_lp
        self._logprob = logprob
        self._surface = {tid: s for tid, s in vocab}

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_logprobs(self) -> bool:
        return self._lp

    def generate(
        self,
        prompts: Sequence[str],
        prefix: str = "",
        max_new_tokens: Optional[int] = None,
        with_logprobs: bool = False,
    ) -> List[Generation]:
        outs: List[Generation] = []
        for idx, p in enumerate(prompts):
            text, ids = self._gen(prefix, p, idx)
            lp = [self._logprob] * len(ids) if (with_logprobs and self._lp) else None
            outs.append(Generation(text=text, token_ids=ids, logprobs=lp))
        return outs

    def get_vocabulary(self) -> List[Tuple[int, str]]:
        return list(self._vocab)

    def special_token_ids(self) -> Set[int]:
        return set()

    def tokenize(self, text: str) -> List[int]:
        return [abs(hash(w)) % 1000 for w in text.split()]

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(t) for t in token_ids)

    def seed_surface(self, token_id: int) -> str:
        return self._surface.get(token_id, "")


@pytest.fixture
def toy_similarity() -> ToySemanticBackend:
    """A fresh toy semantic backend."""
    return ToySemanticBackend()


@pytest.fixture
def basic_vocab() -> List[Tuple[int, str]]:
    """A benign vocab plus a single backdoor seed token ``Yes``.

    Deliberately large (40 benign + 1 trigger) so that random null sampling only
    rarely includes the trigger; even when it does (at most once, sampling is
    without replacement) the trigger's z-score stays comfortably above the
    3-sigma fallback.  A tiny vocab would let the trigger contaminate the null.
    """
    benign = [
        "the", "and", "cat", "run", "blue", "tree", "code", "fish", "jump",
        "road", "milk", "bird", "lake", "star", "wind", "rain", "snow", "gold",
        "leaf", "sand", "rock", "moon", "fire", "wave", "hill", "frog", "deer",
        "wolf", "corn", "rice", "salt", "iron", "silk", "rope", "boat", "kite",
        "drum", "bell", "lamp", "door",
    ]
    words = benign[:20] + ["Yes"] + benign[20:]
    return [(i, w) for i, w in enumerate(words)]


@pytest.fixture
def clean_prompts() -> List[str]:
    """Six distinct clean prompts."""
    return [f"clean prompt number {i} about topic {chr(65 + i)}" for i in range(6)]
