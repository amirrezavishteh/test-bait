"""OpenAI (and OpenAI-compatible) API backend.

Handles rate limiting with exponential backoff, uses ``logprobs`` when the
endpoint provides them and degrades gracefully to hard-label otherwise, and
enumerates / tokenises the vocabulary through a local ``tiktoken`` proxy so the
scan can run even though the remote model exposes no tokenizer.

Caveat: chat endpoints cannot be forced to *continue* an assistant prefix, so
seed injection appends ``prefix`` to the user message.  This makes API seeding
an approximation of local prefix-seeding; it is documented and unavoidable in
the hard-label remote setting.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Set, Tuple

from casa.config import ModelConfig
from casa.interfaces.base import Generation, ModelInterface
from casa.logging_utils import get_logger

_PROXY_ENCODING = "cl100k_base"


class OpenAIModel(ModelInterface):
    """Remote OpenAI-compatible chat model."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the client and proxy tokenizer.

        Args:
            config: Model config; ``name_or_path`` is the API model name,
                ``api_base_url`` an optional OpenAI-compatible endpoint.
        """
        self._cfg = config
        self._log = get_logger()
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=config.api_base_url,
            timeout=config.request_timeout,
        )
        self._enc = self._load_proxy_encoding(config.name_or_path)
        self._logprobs_ok = True  # optimistic; flipped off on first failure

    @staticmethod
    def _load_proxy_encoding(model_name: str) -> object:
        import tiktoken  # type: ignore

        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            return tiktoken.get_encoding(_PROXY_ENCODING)

    @property
    def name(self) -> str:
        return self._cfg.name_or_path

    @property
    def supports_logprobs(self) -> bool:
        return self._logprobs_ok

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
            out.append(self._one(prompt + prefix, max_new, with_logprobs))
        return out

    def _one(self, content: str, max_new: int, with_logprobs: bool) -> Generation:
        want_lp = with_logprobs and self._logprobs_ok
        resp = self._call_with_backoff(content, max_new, want_lp)
        choice = resp.choices[0]
        text = choice.message.content or ""
        logprobs: Optional[List[float]] = None
        token_ids: Optional[List[int]] = None
        lp = getattr(choice, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            logprobs = [tok.logprob for tok in lp.content]
        else:
            token_ids = self._enc.encode(text)
        return Generation(text=text, token_ids=token_ids, logprobs=logprobs)

    def _call_with_backoff(self, content: str, max_new: int, want_lp: bool) -> object:
        from openai import APIError, APIConnectionError, RateLimitError  # type: ignore

        delay = 1.0
        last_exc: Optional[Exception] = None
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                kwargs = dict(
                    model=self._cfg.name_or_path,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_new,
                    temperature=0.0,
                )
                if want_lp:
                    kwargs["logprobs"] = True
                return self._client.chat.completions.create(**kwargs)
            except TypeError:
                # Endpoint rejected logprobs kwarg -> remember and retry plain.
                self._logprobs_ok = False
                want_lp = False
            except (RateLimitError, APIConnectionError, APIError) as exc:  # noqa: PERF203
                last_exc = exc
                self._log.warning("openai call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"OpenAI call failed after retries: {last_exc}")

    def get_vocabulary(self) -> List[Tuple[int, str]]:
        n = self._enc.n_vocab
        out: List[Tuple[int, str]] = []
        for i in range(n):
            try:
                surface = self._enc.decode([i])
            except Exception:
                continue
            out.append((i, surface))
        return out

    def special_token_ids(self) -> Set[int]:
        special = getattr(self._enc, "_special_tokens", {}) or {}
        return {int(v) for v in special.values()}

    def tokenize(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._enc.decode(list(token_ids))
