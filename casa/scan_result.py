"""Result data structures and JSON serialisation for a CASA scan.

These dataclasses define the self-contained scan-result document described in
the system spec's *Output format* section.  Everything is JSON-serialisable via
:meth:`ScanResult.to_dict` / :meth:`ScanResult.to_json`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Auditor verdict vocabulary.
VERDICT_MALICIOUS = "malicious"
VERDICT_BENIGN = "benign"
VERDICT_UNCERTAIN = "uncertain"


@dataclass
class SeedResult:
    """Per-seed scan record (one entry of the ranked candidate list).

    Attributes:
        token_id: Vocabulary id of the seed token.
        seed_surface: Decoded surface string of the seed (the injected prefix).
        raw_score: Raw semantic-consistency score in ``[0, 1]``.
        z_score: Standardised score ``(raw - null_mean) / null_std``.
        inverted_target: The representative (medoid) generation for this seed.
        auditor_verdict: One of ``malicious`` / ``benign`` / ``uncertain`` /
            ``None`` (not audited).
        auditor_reasoning: The auditor's explanation (sanitised).
        auditor_confidence: Optional confidence label from the auditor.
        review_flag: Whether a human should review this candidate.
    """

    token_id: int
    seed_surface: str
    raw_score: float
    z_score: float
    inverted_target: str
    auditor_verdict: Optional[str] = None
    auditor_reasoning: str = ""
    auditor_confidence: Optional[str] = None
    review_flag: bool = False


@dataclass
class NullStats:
    """Null-distribution summary used for z-standardisation.

    Attributes:
        mean: Mean consistency over random seeds.
        std: Standard deviation over random seeds (floored to ``min_std``).
        sample_size: Number of random seeds sampled.
    """

    mean: float
    std: float
    sample_size: int


@dataclass
class EarlyStop:
    """Anytime-valid early-stopping outcome.

    Attributes:
        occurred: Whether the e-process crossed the boundary.
        position: 1-based seed index where stopping happened (``None`` if not).
        eprocess_value: e-process value at the end of scanning.
        boundary: The Ville boundary ``1/alpha``.
    """

    occurred: bool
    position: Optional[int]
    eprocess_value: float
    boundary: float


@dataclass
class CalibrationMeta:
    """Provenance of the detection threshold used.

    Attributes:
        threshold: The z-score threshold applied.
        calibrated: Whether it came from conformal calibration (vs fallback).
        target_far: Target false-alarm rate (``None`` for fallback).
        failure_prob: Confidence failure probability (``None`` for fallback).
        n_models: Number of calibration models (``None`` for fallback).
        achieved_ucb: Hoeffding-Bentkus UCB achieved (``None`` for fallback).
        date: Calibration timestamp (``None`` for fallback).
    """

    threshold: float
    calibrated: bool
    target_far: Optional[float] = None
    failure_prob: Optional[float] = None
    n_models: Optional[int] = None
    achieved_ucb: Optional[float] = None
    date: Optional[str] = None


@dataclass
class ScanResult:
    """The complete, self-contained result of one CASA scan.

    Attributes:
        is_backdoor: Final verdict — the model is flagged as poisoned.
        cah_score: The CAH score = max z-score over all examined seeds.
        threshold: Detection threshold applied to ``cah_score``.
        threshold_calibrated: Whether ``threshold`` is conformally certified.
        calibration: Threshold provenance.
        best_seed_surface: Surface of the flagged / top seed.
        best_token_id: Token id of the flagged / top seed.
        inverted_target: Representative generation of the top seed.
        top_seeds: Ranked top-k seed records (with auditor verdicts).
        null_stats: Null-distribution summary.
        early_stop: Early-stopping outcome.
        scan_time_s: Wall-clock scan time in seconds.
        n_queries: Total model generations issued.
        model_name: Identifier of the scanned model.
        code_mode: Whether the scan ran in code-vulnerability mode.
        vuln_class: Probed vulnerability class (code mode).
        warnings: Free-text warnings (partial logprobs, auditor uncertainty…).
    """

    is_backdoor: bool
    cah_score: float
    threshold: float
    threshold_calibrated: bool
    calibration: CalibrationMeta
    best_seed_surface: str
    best_token_id: int
    inverted_target: str
    top_seeds: List[SeedResult]
    null_stats: NullStats
    early_stop: EarlyStop
    scan_time_s: float
    n_queries: int
    model_name: str
    code_mode: bool = False
    vuln_class: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain nested dict suitable for ``json.dump``."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """A short human-readable one-block summary for stdout."""
        verdict = "BACKDOOR" if self.is_backdoor else "clean"
        cert = "calibrated" if self.threshold_calibrated else "FALLBACK (uncertified)"
        lines = [
            f"model      : {self.model_name}",
            f"verdict    : {verdict}",
            f"CAH score  : {self.cah_score:.3f}  (z-score; threshold {self.threshold:.3f}, {cert})",
            f"best seed  : {self.best_seed_surface!r}  (id {self.best_token_id})",
            f"inverted   : {self.inverted_target!r}",
            f"null       : mean={self.null_stats.mean:.3f} std={self.null_stats.std:.3f} "
            f"n={self.null_stats.sample_size}",
            f"early stop : {'yes @ ' + str(self.early_stop.position) if self.early_stop.occurred else 'no'} "
            f"(e-process {self.early_stop.eprocess_value:.2f} / boundary {self.early_stop.boundary:.1f})",
            f"queries    : {self.n_queries}   time: {self.scan_time_s:.1f}s",
        ]
        if self.warnings:
            lines.append("warnings   : " + "; ".join(self.warnings))
        return "\n".join(lines)
