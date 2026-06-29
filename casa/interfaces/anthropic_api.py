"""Anthropic API backend.

Anthropic's Messages API does not return token-level probabilities in standard
usage, so this backend is **always hard-label** (:attr:`supports_logprobs` is
``False``).  It supports assistant *prefill*, which makes seed injection exact:
the seed ``prefix`` is supplied as the start of the assistant turn, so the model
genuinely continues from the seed token.  Vocabulary enumeration uses a local
``tiktoken`` proxy because the true tokenizer is not exposed.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Set, Tuple

from casa.config import ModelConfig
from casa.interfaces.base import Generation, ModelInterface
from casa.logging_utils import get_logger

_PROXY_ENCODING = "cl100k_base"


class AnthropicModel(ModelInterface):
    """Remote Anthropic Messages-API model (hard-label only)."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the client and proxy tokenizer.

        Args:
            config: Model config; ``name_or_path`` is the Anthropic model id.
        """
        self._cfg = config
        self._log = get_logger()
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=config.api_base_url,
            timeout=config.request_timeout,
        )
        import tiktoken  # type: ignore

        self._enc = tiktoken.get_encoding(_PROXY_ENCODING)

    @property
    def name(self) -> str:
        return self._cfg.name_or_path

    @property
    def supports_logprobs(self) -> bool:
        return False  # Anthropic API does not expose token probabilities.

    def generate(
        self,
        prompts: Sequence[str],
        prefix: str = "",
        max_new_tokens: Optional[int] = None,
        with_logprobs: bool = False,
    ) -> List[Generation]:
        max_new = max_new_tokens or self._cfg.max_new_tokens
        out: List[Generation] = []
        for prompt in prompts:
            out.append(self._one(prompt, prefix, max_new))
        return out

    def _one(self, prompt: str, prefix: str, max_new: int) -> Generation:
        import anthropic  # type: ignore

        messages = [{"role": "user", "content": prompt}]
        if prefix:
            # Assistant prefill -> the model continues from the seed exactly.
            messages.append({"role": "assistant", "content": prefix})
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = self._client.messages.create(
                    model=self._cfg.name_or_path,
                    max_tokens=max_new,
                    messages=messages,
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                return Generation(text=text, token_ids=self._enc.encode(text))
            except (anthropic.RateLimitError, anthropic.APIError) as exc:  # type: ignore[attr-defined]
                last_exc = exc
                self._log.warning("anthropic call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"Anthropic call failed after retries: {last_exc}")

    def get_vocabulary(self) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        for i in range(self._enc.n_vocab):
            try:
                out.append((i, self._enc.decode([i])))
            except Exception:
                continue
        return out

    def special_token_ids(self) -> Set[int]:
        special = getattr(self._enc, "_special_tokens", {}) or {}
        return {int(v) for v in special.values()}

    def tokenize(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._enc.decode(list(token_ids))
