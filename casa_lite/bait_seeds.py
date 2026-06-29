"""BAIT-assisted seed generation (spec §11.1, optional extension).

When the backend exposes token probabilities, a BAIT-style token-agreement probe
can surface suspicious initial tokens (those whose continuations show unusually
strong cross-prompt token causality) and add them to CASA-Lite's seed bank. This
is a *recall booster*, not a requirement — CASA-Lite remains the hard-label
behavioural judge; these seeds just give it better candidates to test.

If the backend is hard-label (no logprobs), this returns an empty list and the
scan proceeds with the curated bank alone.
"""

from __future__ import annotations

from typing import List, Sequence

from casa.bait_compat import bait_qscore, logprobs_to_probs
from casa.interfaces.base import ModelInterface
from casa.logging_utils import get_logger
from casa.seed_scoring import filter_seed_tokens


def suggest_bait_seeds(
    model: ModelInterface,
    prompts: Sequence[str],
    top_n: int = 10,
    max_probe: int = 200,
    max_new_tokens: int = 16,
    separator: str = " ",
) -> List[str]:
    """Propose seed strings via a BAIT-style token-agreement probe.

    Args:
        model: Model under test; must support logprobs for a non-empty result.
        prompts: Clean prompts to probe over (a small subset is enough).
        top_n: Number of highest-Q-Score token surfaces to return.
        max_probe: Cap on candidate tokens probed (0 = all seedable tokens).
        max_new_tokens: Continuation length per probe.
        separator: String placed between prompt and seed.

    Returns:
        Up to ``top_n`` seed surface strings ranked by descending Q-Score; empty
        when the backend is hard-label.
    """
    log = get_logger()
    if not model.supports_logprobs:
        log.info("bait-assist skipped: backend is hard-label (no logprobs)")
        return []
    pool = filter_seed_tokens(model)
    if max_probe > 0:
        pool = pool[:max_probe]
    if not pool:
        return []
    log.info("bait-assist probing %d candidate tokens", len(pool))
    scored: List[tuple] = []
    for _tid, surface in pool:
        gens = model.generate(
            prompts, prefix=separator + surface, max_new_tokens=max_new_tokens,
            with_logprobs=True,
        )
        tok_seqs = [g.token_ids or model.tokenize(g.text) for g in gens]
        probs = (
            [logprobs_to_probs(g.logprobs) for g in gens]
            if all(g.logprobs for g in gens)
            else None
        )
        scored.append((bait_qscore(tok_seqs, probs), surface))
    scored.sort(key=lambda x: x[0], reverse=True)
    suggested = [s for _q, s in scored[:top_n]]
    log.info("bait-assist suggested %d seeds: %s", len(suggested), suggested)
    return suggested
