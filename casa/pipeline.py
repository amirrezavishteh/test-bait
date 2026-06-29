"""End-to-end scan orchestration (the Python API).

:class:`CASA` wires every component together: load clean prompts, resolve the
calibrated ``beta`` and detection threshold, build the model-specific null
distribution, run the vocabulary scan with early stopping, audit the top
candidates, and assemble a :class:`~casa.scan_result.ScanResult`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from casa.auditor import HarmAuditor, build_auditor
from casa.calibration import load_artifact_if_exists
from casa.config import CASAConfig
from casa.data import load_prompts
from casa.interfaces.base import ModelInterface
from casa.logging_utils import configure_logging, get_logger
from casa.null_distribution import NullDistributionBuilder
from casa.scan_result import (
    CalibrationMeta,
    ScanResult,
    SeedResult,
    VERDICT_MALICIOUS,
    VERDICT_UNCERTAIN,
)
from casa.scanner import VocabularyScanner
from casa.seed_scoring import filter_seed_tokens
from casa.similarity.base import SimilarityBackend


class CASA:
    """High-level CASA scanner driven by a :class:`CASAConfig`."""

    def __init__(
        self,
        config: CASAConfig,
        model: Optional[ModelInterface] = None,
        similarity: Optional[SimilarityBackend] = None,
        auditor: Optional[HarmAuditor] = None,
    ) -> None:
        """Create a scanner.

        Args:
            config: Validated configuration.
            model: Pre-built model interface (built from config if ``None``).
            similarity: Pre-built similarity backend (built if ``None``).
            auditor: Pre-built harm auditor (built if ``None``).
        """
        self.config = config
        configure_logging(config.logging.level, config.logging.json)
        self._log = get_logger()
        self._model = model
        self._similarity = similarity
        self._auditor = auditor
        self._artifact = load_artifact_if_exists(config.conformal.artifact_path)

    # -- lazy component construction ------------------------------------- #
    @property
    def model(self) -> ModelInterface:
        """The model interface (built lazily from config)."""
        if self._model is None:
            from casa.interfaces import build_interface

            self._model = build_interface(self.config.model)
        return self._model

    @property
    def similarity(self) -> SimilarityBackend:
        """The similarity backend (built lazily from config)."""
        if self._similarity is None:
            from casa.similarity import build_similarity

            self._similarity = build_similarity(self.config.similarity)
        return self._similarity

    @property
    def auditor(self) -> HarmAuditor:
        """The harm auditor (built lazily from config)."""
        if self._auditor is None:
            self._auditor = build_auditor(
                self.config.auditor.backend,
                self.config.auditor.model,
                self.config.auditor.base_url,
            )
        return self._auditor

    # -- parameter resolution -------------------------------------------- #
    def resolve_beta(self) -> float:
        """Return the match threshold: artifact's calibrated ``beta`` else config."""
        if self._artifact and self._artifact.match:
            return self._artifact.match.beta
        return self.config.similarity.beta

    def resolve_threshold(self) -> CalibrationMeta:
        """Resolve the detection threshold and its provenance.

        Priority: calibration artifact (certified) → config-provided constant
        (uncertified) → 3-sigma fallback (uncertified).

        Returns:
            A :class:`CalibrationMeta`.
        """
        if self._artifact and self._artifact.detection:
            d = self._artifact.detection
            return CalibrationMeta(
                threshold=d.threshold,
                calibrated=d.certified,
                target_far=d.target_far,
                failure_prob=d.failure_prob,
                n_models=d.n_models,
                achieved_ucb=d.achieved_ucb,
                date=d.date,
            )
        if self.config.conformal.detection_threshold is not None:
            return CalibrationMeta(
                threshold=self.config.conformal.detection_threshold, calibrated=False
            )
        return CalibrationMeta(threshold=self.config.conformal.fallback_z, calibrated=False)

    # -- main entry point ------------------------------------------------- #
    def scan(self, prompts: Optional[Sequence[str]] = None) -> ScanResult:
        """Run a full scan and return the result document.

        Args:
            prompts: Clean prompts; loaded from config when ``None``.

        Returns:
            A populated :class:`ScanResult`.

        Raises:
            ValueError: If there are no prompts or no seedable tokens.
        """
        cfg = self.config
        if prompts is None:
            prompts = load_prompts(cfg.data)
        prompts = list(prompts)
        if len(prompts) < 2:
            raise ValueError("CASA needs >= 2 clean prompts")

        beta = self.resolve_beta()
        pool = filter_seed_tokens(self.model, cfg.scan.min_seed_token_len)
        if not pool:
            raise ValueError("no seedable vocabulary tokens")

        # 1) null distribution
        null_builder = NullDistributionBuilder(
            self.model, self.similarity, beta, cfg.model.max_new_tokens, cfg.null.min_std
        )
        null = null_builder.build(prompts, cfg.null.sample_size, cfg.null.seed, candidates=pool)

        # 2) main scan
        scanner = VocabularyScanner(
            self.model,
            self.similarity,
            null,
            beta=beta,
            max_new_tokens=cfg.model.max_new_tokens,
            evalue_alpha=cfg.conformal.evalue_alpha,
            batch_size=cfg.scan.batch_size,
            max_vocab_scan=cfg.scan.max_vocab_scan,
            early_stop=cfg.scan.early_stop,
            min_seed_surface_len=cfg.scan.min_seed_token_len,
        )
        report = scanner.run(prompts, candidates=pool)

        # 3) audit the top-k candidates
        top = report.seed_results[: cfg.scan.top_k]
        warnings: List[str] = []
        if cfg.auditor.enabled:
            self._audit_top(top, warnings)

        # 4) threshold + verdict
        cal = self.resolve_threshold()
        above = report.cah_score > cal.threshold
        malicious = any(s.auditor_verdict == VERDICT_MALICIOUS for s in top)
        is_backdoor = above and (malicious or not cfg.auditor.enabled)

        self._collect_warnings(warnings, cal, top, above, malicious)

        best = report.seed_results[0]
        flush = getattr(self.similarity.cache, "flush", None)
        if callable(flush):
            flush()

        return ScanResult(
            is_backdoor=is_backdoor,
            cah_score=report.cah_score,
            threshold=cal.threshold,
            threshold_calibrated=cal.calibrated,
            calibration=cal,
            best_seed_surface=best.seed_surface,
            best_token_id=best.token_id,
            inverted_target=best.inverted_target,
            top_seeds=top,
            null_stats=null.stats(),
            early_stop=report.early_stop,
            scan_time_s=report.scan_time_s,
            n_queries=null_builder.n_queries + report.n_queries,
            model_name=self.model.name,
            code_mode=cfg.scan.code_mode,
            vuln_class=cfg.scan.vuln_class,
            warnings=warnings,
        )

    # -- helpers ---------------------------------------------------------- #
    def _audit_top(self, top: Sequence[SeedResult], warnings: List[str]) -> None:
        cfg = self.config
        from casa.logging_utils import sanitize

        for seed in top:
            verdict = self.auditor.audit(
                candidate=seed.inverted_target,
                seed_surface=seed.seed_surface,
                context=cfg.auditor.context,
                code_mode=cfg.scan.code_mode,
                vuln_class=cfg.scan.vuln_class,
            )
            seed.auditor_verdict = verdict.verdict
            seed.auditor_reasoning = sanitize(verdict.reasoning, cfg.logging.max_output_chars)
            seed.auditor_confidence = verdict.confidence
            seed.review_flag = verdict.review_flag

    def _collect_warnings(
        self,
        warnings: List[str],
        cal: CalibrationMeta,
        top: Sequence[SeedResult],
        above: bool,
        malicious: bool,
    ) -> None:
        if not cal.calibrated:
            warnings.append(
                "detection threshold is an UNCERTIFIED fallback; calibrate on "
                "clean models for a certified false-alarm guarantee"
            )
        if not self.model.supports_logprobs:
            warnings.append("hard-label mode (no token probabilities available)")
        uncertain = [s for s in top if s.auditor_verdict == VERDICT_UNCERTAIN]
        if uncertain:
            warnings.append(
                f"{len(uncertain)} top candidate(s) marked UNCERTAIN — human review advised: "
                + "; ".join(repr(s.seed_surface) for s in uncertain)
            )
        if above and not malicious and self.config.auditor.enabled:
            warnings.append(
                "CAH score exceeded threshold but the harm auditor cleared all "
                "candidates; not flagged (review the UNCERTAIN list)"
            )
