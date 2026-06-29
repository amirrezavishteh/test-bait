"""CASA-Lite — hard-label black-box backdoor detector for LLMs.

CASA-Lite is the *practical, text-only* sibling of the full :mod:`casa` scanner.
It needs **generated text only** — no weights, gradients, hidden states,
attention, or token probabilities — which makes it runnable against pure
hard-label APIs where BAIT (which needs logprobs) and even soft-label CASA
cannot operate.

Core thesis
-----------
A backdoored model is suspicious when a small candidate **seed** causes outputs
from many unrelated clean prompts to **converge** toward the same
*prompt-unjustified*, *harmful* target — at the level of meaning / structure /
behaviour rather than exact tokens.  CASA-Lite scores that convergence with a
three-gate product and calibrates the decision threshold conformally on clean
models, emitting a three-state verdict ``CLEAN`` / ``UNCERTAIN`` / ``BACKDOORED``.

It deliberately reuses CASA's model interfaces (:mod:`casa.interfaces`),
similarity engine (:mod:`casa.similarity`) and data loading (:mod:`casa.data`),
so both tools scan the same weakness-zoo test set.

Relationship to CASA
--------------------
========================  ====================  ==========================
aspect                    CASA (full)           CASA-Lite
========================  ====================  ==========================
access                    soft-label optional   hard-label text only
candidate space           full vocabulary scan  small curated seed bank
detection statistic       z-score vs null       LCB(agreement)·U·H product
threshold                 Hoeffding-Bentkus     conformal quantile
early stopping            e-process (Ville)     two-stage scan
verdict                   clean / backdoor      clean / uncertain / backdoor
========================  ====================  ==========================
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
