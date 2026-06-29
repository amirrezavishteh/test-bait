"""Abstract model interface and shared result types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple


@dataclass
class Generation:
    """A single model continuation.

    Attributes:
        text: The generated continuation (excluding the prompt/prefix).
        token_ids: Generated token ids, when the backend exposes them.
        logprobs: Per-token log-probabilities of the generated tokens, when
            available (``None`` in hard-label mode).
    """

    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None

    @property
    def has_logprobs(self) -> bool:
        """Whether per-token log-probabilities are attached."""
        return self.logprobs is not None and len(self.logprobs) > 0


class ModelInterface(abc.ABC):
    """Uniform black-box query surface over a generative LLM.

    Concrete subclasses implement local HuggingFace, OpenAI and Anthropic
    backends.  All methods are black-box: they take/return text (plus optional
    log-probabilities when the backend supports them).
    """

    # -- identity / capabilities ----------------------------------------- #
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A stable human-readable identifier for the model under test."""

    @property
    @abc.abstractmethod
    def supports_logprobs(self) -> bool:
        """Whether :meth:`generate` can return per-token log-probabilities."""

    @property
    def supports_attention(self) -> bool:
        """Whether the backend can expose attention maps (white-box only)."""
        return False

    # -- core operations -------------------------------------------------- #
    @abc.abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        prefix: str = "",
        max_new_tokens: Optional[int] = None,
        with_logprobs: bool = False,
    ) -> List[Generation]:
        """Generate one continuation per prompt.

        Args:
            prompts: The clean prompts.
            prefix: A string concatenated to the end of every prompt before
                generation — the mechanism by which CASA seeds a candidate
                starting token.
            max_new_tokens: Override for the continuation length.
            with_logprobs: Request per-token log-probabilities (best effort;
                ignored when :attr:`supports_logprobs` is ``False``).

        Returns:
            One :class:`Generation` per input prompt, in order.
        """

    @abc.abstractmethod
    def get_vocabulary(self) -> List[Tuple[int, str]]:
        """Return ``(token_id, surface)`` pairs for enumerable seed tokens.

        For remote models without a true tokenizer this falls back to a local
        proxy tokenizer (e.g. tiktoken).

        Returns:
            A list of ``(id, surface_string)`` tuples.
        """

    @abc.abstractmethod
    def special_token_ids(self) -> Set[int]:
        """Token ids to exclude from seeding (BOS/EOS/PAD/control tokens)."""

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize ``text`` with the true or proxy tokenizer."""

    @abc.abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids back to text."""

    # -- convenience ------------------------------------------------------ #
    def seed_surface(self, token_id: int) -> str:
        """Decode a single seed token id to the prefix string used to seed it."""
        return self.decode([token_id])

    def close(self) -> None:
        """Release any held resources (no-op by default)."""
