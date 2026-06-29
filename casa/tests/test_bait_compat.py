"""Unit tests for the BAIT-compatible Q-Score (and its failure modes)."""

from __future__ import annotations

import pytest

from casa.bait_compat import bait_qscore, bait_verdict, logprobs_to_probs


def test_standard_target_high_q() -> None:
    # All prompts emit the identical chain with high prob -> high Q.
    chain = [10, 11, 12, 13, 14]
    seqs = [list(chain) for _ in range(6)]
    probs = [[0.95] * len(chain) for _ in range(6)]
    q = bait_qscore(seqs, probs)
    assert q > 0.9
    assert bait_verdict(q)


def test_multi_target_collapses() -> None:
    # 6 prompts each emit a DIFFERENT paraphrase chain -> token mass splits.
    seqs = [[100 + p, 200 + p, 300 + p] for p in range(6)]
    probs = [[0.9, 0.9, 0.9] for _ in range(6)]
    q = bait_qscore(seqs, probs)
    assert q < 0.5
    assert not bait_verdict(q)


def test_single_token_target_zero() -> None:
    # A single-token target is fully consumed by the seed, so the measured
    # post-seed continuation is empty -> no chain -> Q = 0.
    seqs = [[] for _ in range(6)]
    assert bait_qscore(seqs) == 0.0


def test_negative_training_breaks_chain() -> None:
    # Two steps are disrupted, so dropping the single weakest step cannot
    # recover agreement -> Q falls below 0.9.
    seqs = [[1, 2 + (p % 4), 7 + (p % 4), 3] for p in range(6)]
    probs = [[0.95, 0.95, 0.95, 0.95] for _ in range(6)]
    q = bait_qscore(seqs, probs)
    assert q < 0.9


def test_hard_label_proxy_no_probs() -> None:
    seqs = [[1, 2, 3] for _ in range(5)]
    assert bait_qscore(seqs, None) == pytest.approx(1.0)


def test_logprobs_to_probs() -> None:
    assert logprobs_to_probs(None) is None
    assert logprobs_to_probs([0.0])[0] == pytest.approx(1.0)
