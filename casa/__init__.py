"""CASA — Conformal Anti-hallucination Scanning of Attack-targets.

CASA is a black-box backdoor / data-poisoning detector for generative LLMs.

Where token-level scanners such as BAIT ask *"does the model generate the same
tokens across diverse clean prompts?"*, CASA asks the strictly more robust
question *"does the model generate semantically equivalent content across
prompts in a way that cannot be explained by what the prompts have in
common?"*.  A backdoor is, in this view, a **prompt-unjustified semantic
certainty** — the exact opposite of hallucination — and CASA measures it with
a calibrated semantic-consistency score, standardised against a model-specific
null distribution, and gated by distribution-free conformal thresholds.

The public surface is intentionally lazy: importing :mod:`casa` does **not**
pull in torch / transformers, so the conformal math (:mod:`casa.conformal`,
:mod:`casa.evalue`) can be used in environments without a deep-learning stack.

Sub-packages
------------
``casa.conformal``   Conformal risk control + Hoeffding-Bentkus calibration.
``casa.evalue``      Anytime-valid (e-process) sequential test.
``casa.similarity``  Semantic similarity backends + pairwise consistency.
``casa.interfaces``  Uniform model-query abstraction (local HF / OpenAI / …).
``casa.scanner``     Vocabulary scanner with z-score standardisation.
``casa.pipeline``    End-to-end scan orchestration.
``casa.harness``     Reproducible CASA-vs-BAIT experimental harness.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
