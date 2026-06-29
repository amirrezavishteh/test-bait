"""Result data structures and JSON serialisation for a CASA-Lite scan."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class ClusterSummary:
    """Serialisable summary of a seed's most-suspicious cluster.

    Attributes:
        size: ``|C|``.
        coverage: ``|C| / N``.
        agreement_mass: ``|C|(|C|-1)/(N(N-1))``.
        lcb: Hoeffding lower bound on the agreement mass.
        unjustified: ``U(C)``.
        harm: ``H(C)``.
        examples: A few representative (sanitised) outputs from the cluster.
    """

    size: int
    coverage: float
    agreement_mass: float
    lcb: float
    unjustified: float
    harm: float
    examples: List[str] = field(default_factory=list)


@dataclass
class LiteSeedResult:
    """Per-seed record in the ranked candidate list.

    Attributes:
        seed: The probe seed string.
        score: ``Score(v) = max_C ClusterScore(C)``.
        n_outputs: Number of prompts queried for this seed.
        cluster: Summary of the seed's most-suspicious cluster.
    """

    seed: str
    score: float
    n_outputs: int
    cluster: ClusterSummary


@dataclass
class LiteScanResult:
    """The complete result of one CASA-Lite scan.

    Attributes:
        verdict: ``CLEAN`` / ``UNCERTAIN`` / ``BACKDOORED``.
        is_backdoor: ``verdict == "BACKDOORED"`` (convenience for exit codes).
        score: The model score ``T(M) = max_v Score(v)``.
        threshold: Decision threshold ``lambda_hat``.
        threshold_certified: Whether the threshold is conformally certified.
        alpha: Target false-positive rate of the threshold (if calibrated).
        uncertain_margin: Lower fraction for the UNCERTAIN band.
        best_seed: The top-scoring seed.
        top_seeds: Ranked seed records (top-k).
        model_name: Identifier of the scanned model.
        n_queries: Total generations issued.
        scan_time_s: Wall-clock scan time.
        stages: Per-stage query counts ({"stage1": n, "stage2": n}).
        warnings: Free-text warnings (uncertified threshold, hard-label, …).
    """

    verdict: str
    is_backdoor: bool
    score: float
    threshold: float
    threshold_certified: bool
    alpha: Optional[float]
    uncertain_margin: float
    best_seed: str
    top_seeds: List[LiteSeedResult]
    model_name: str
    n_queries: int
    scan_time_s: float
    stages: dict
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain nested dict for ``json.dump``."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """A short human-readable summary for stdout."""
        cert = "certified" if self.threshold_certified else "FALLBACK (uncertified)"
        thr = "inf" if self.threshold == float("inf") else f"{self.threshold:.4f}"
        c = self.top_seeds[0].cluster if self.top_seeds else None
        lines = [
            f"model    : {self.model_name}",
            f"verdict  : {self.verdict}",
            f"score    : {self.score:.4f}   (threshold {thr}, {cert}, alpha={self.alpha})",
            f"best seed: {self.best_seed!r}",
        ]
        if c is not None:
            lines.append(
                f"cluster  : size={c.size} coverage={c.coverage:.2f} "
                f"agree_mass={c.agreement_mass:.3f} lcb={c.lcb:.3f} U={c.unjustified:.2f} H={c.harm:.2f}"
            )
            if c.examples:
                lines.append(f"example  : {c.examples[0]!r}")
        lines.append(f"queries  : {self.n_queries} {self.stages}   time: {self.scan_time_s:.1f}s")
        if self.warnings:
            lines.append("warnings : " + "; ".join(self.warnings))
        return "\n".join(lines)
