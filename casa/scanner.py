"""Vocabulary scanner with anytime-valid early stopping (Component 4).

Enumerates candidate seed tokens, scores each as a z-standardised semantic
consistency, feeds the z-scores to an :class:`~casa.evalue.EProcess`, and stops
early (skipping the rest of the vocabulary) once the e-process crosses the Ville
boundary ``1/alpha`` — all while keeping the full ranked candidate list for the
harm auditor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from casa.evalue import EProcess
from casa.interfaces.base import ModelInterface
from casa.logging_utils import get_logger
from casa.null_distribution import NullDistribution
from casa.scan_result import EarlyStop, SeedResult
from casa.seed_scoring import filter_seed_tokens, score_seed
from casa.similarity.base import SimilarityBackend


@dataclass
class ScanReport:
    """Raw scanner output (pre-auditor).

    Attributes:
        seed_results: All examined seeds, ranked by z-score descending.
        cah_score: Max z-score (CAH score); ``-inf`` if nothing was scored.
        early_stop: Early-stopping outcome.
        n_queries: Total generations issued during the scan.
        scan_time_s: Wall-clock scan time.
        n_scanned: Number of seeds actually examined.
        vocab_size: Number of seedable tokens available.
    """

    seed_results: List[SeedResult]
    cah_score: float
    early_stop: EarlyStop
    n_queries: int
    scan_time_s: float
    n_scanned: int
    vocab_size: int = 0


class VocabularyScanner:
    """The main CASA scan loop."""

    def __init__(
        self,
        model: ModelInterface,
        similarity: SimilarityBackend,
        null: NullDistribution,
        beta: float,
        max_new_tokens: int,
        evalue_alpha: float = 0.05,
        evalue_lambda: float = 0.9,
        batch_size: int = 16,
        max_vocab_scan: int = 0,
        early_stop: bool = True,
        min_seed_surface_len: int = 1,
    ) -> None:
        """Configure the scanner.

        Args:
            model: Model under test.
            similarity: Calibrated similarity backend.
            null: The model's null distribution (for z-standardisation).
            beta: Calibrated match threshold.
            max_new_tokens: Continuation length per seed.
            evalue_alpha: Significance level; boundary is ``1/evalue_alpha``.
            evalue_lambda: Betting fraction for the e-value.
            batch_size: Seeds between progress logs / early-stop checks.
            max_vocab_scan: Cap on seeds examined (0 = full vocabulary).
            early_stop: Enable e-process early stopping.
            min_seed_surface_len: Minimum decoded seed length to consider.
        """
        self._model = model
        self._sim = similarity
        self._null = null
        self._beta = beta
        self._max_new = max_new_tokens
        self._alpha = evalue_alpha
        self._lambda = evalue_lambda
        self._batch_size = max(1, batch_size)
        self._max_vocab = max_vocab_scan
        self._early_stop = early_stop
        self._min_len = min_seed_surface_len
        self._log = get_logger()

    def run(
        self,
        prompts: Sequence[str],
        candidates: Optional[Sequence[Tuple[int, str]]] = None,
    ) -> ScanReport:
        """Execute the scan and return a :class:`ScanReport`.

        Args:
            prompts: Clean prompts.
            candidates: Optional pre-filtered ``(id, surface)`` pool.

        Returns:
            The scan report (ranked seeds, CAH score, early-stop info).

        Raises:
            ValueError: If there are no seedable tokens to scan.
        """
        pool = (
            list(candidates)
            if candidates is not None
            else filter_seed_tokens(self._model, self._min_len)
        )
        vocab_size = len(pool)
        if vocab_size == 0:
            raise ValueError("no seedable vocabulary tokens to scan")
        if self._max_vocab > 0:
            pool = pool[: self._max_vocab]

        start = time.time()
        ep = EProcess(alpha=self._alpha, lam=self._lambda)
        results: List[SeedResult] = []
        n_queries = 0
        n_scanned = 0
        self._log.info(
            "scan start: vocab=%d cap=%d null(mean=%.4f std=%.4f) boundary=%.1f",
            vocab_size,
            self._max_vocab or vocab_size,
            self._null.mean,
            self._null.std,
            ep.boundary,
        )

        for idx, (token_id, surface) in enumerate(pool, 1):
            score = score_seed(
                self._model, self._sim, prompts, surface, self._beta, self._max_new
            )
            n_queries += score.n_queries
            n_scanned = idx
            z = self._null.z(score.raw_score)
            results.append(
                SeedResult(
                    token_id=token_id,
                    seed_surface=surface,
                    raw_score=score.raw_score,
                    z_score=z,
                    inverted_target=score.inverted_target,
                )
            )
            ep.update(z)

            if idx % self._batch_size == 0:
                self._log.info(
                    "scan progress %d/%d | max-z=%.3f | e-process=%.3g",
                    idx,
                    len(pool),
                    max(r.z_score for r in results),
                    ep.value,
                )
                if self._early_stop and ep.has_crossed:
                    self._log.info(
                        "early stop: e-process crossed boundary at seed %d", ep.crossed_at
                    )
                    break

        scan_time = time.time() - start
        results.sort(key=lambda r: r.z_score, reverse=True)
        cah = results[0].z_score if results else float("-inf")
        early = EarlyStop(
            occurred=ep.has_crossed,
            position=ep.crossed_at,
            eprocess_value=ep.value,
            boundary=ep.boundary,
        )
        self._log.info(
            "scan done: scanned=%d cah=%.3f time=%.1fs queries=%d",
            n_scanned,
            cah,
            scan_time,
            n_queries,
        )
        return ScanReport(
            seed_results=results,
            cah_score=cah,
            early_stop=early,
            n_queries=n_queries,
            scan_time_s=scan_time,
            n_scanned=n_scanned,
            vocab_size=vocab_size,
        )
