"""Model-interface abstraction for CASA.

The rest of CASA talks to a model only through :class:`ModelInterface`, so the
same scan logic runs against a local HuggingFace checkpoint, an OpenAI model or
an Anthropic model.  Concrete backends are imported lazily by
:func:`build_interface` so that importing :mod:`casa.interfaces` never requires
torch / openai / anthropic unless that backend is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from casa.interfaces.base import Generation, ModelInterface

if TYPE_CHECKING:  # pragma: no cover
    from casa.config import ModelConfig

__all__ = ["Generation", "ModelInterface", "build_interface"]


def build_interface(config: "ModelConfig") -> ModelInterface:
    """Instantiate the model interface named by ``config.kind``.

    Args:
        config: A :class:`casa.config.ModelConfig`.

    Returns:
        A ready-to-query :class:`ModelInterface`.

    Raises:
        ValueError: If ``config.kind`` is unknown.
    """
    kind = config.kind
    if kind == "local_hf":
        from casa.interfaces.local_hf import LocalHFModel

        return LocalHFModel(config)
    if kind == "openai":
        from casa.interfaces.openai_api import OpenAIModel

        return OpenAIModel(config)
    if kind == "anthropic":
        from casa.interfaces.anthropic_api import AnthropicModel

        return AnthropicModel(config)
    raise ValueError(f"unknown model kind {kind!r}")
