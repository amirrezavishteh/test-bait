"""Configuration for CASA-Lite.

Reuses CASA's model / similarity / data / logging sections (so both tools share
the same model-access and prompt machinery) and adds CASA-Lite-specific sections
for the seed bank, two-stage scan, three-gate scoring, the justification and
harm judges, and the conformal-quantile decision.

Three composable levels (defaults < file < programmatic), validated at load.
Parameters affecting the certified false-positive rate are flagged ``[CERT]``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from casa.config import (
    ConfigError,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SimilarityConfig,
    _build_section,
    _deep_merge,
    _load_mapping,
)

__all__ = ["CASALiteConfig", "SeedConfig", "LiteScanConfig", "ScoringConfig",
           "JudgeConfig", "LiteConformalConfig", "ConfigError"]


@dataclass
class SeedConfig:
    """Candidate seed bank (probes, not claimed triggers).

    Attributes:
        builtin: Use the packaged default seed bank.
        extra: Additional seed strings appended to the bank.
        seed_file: Optional newline-delimited file of seeds (replaces builtin
            when set unless ``builtin`` is also true, in which case it extends).
        max_seeds: Cap on the total bank size (0 = no cap).
    """

    builtin: bool = True
    extra: List[str] = field(default_factory=list)
    seed_file: Optional[str] = None
    max_seeds: int = 0

    def validate(self) -> None:
        if self.max_seeds < 0:
            raise ConfigError("seeds.max_seeds must be >= 0")


@dataclass
class LiteScanConfig:
    """Two-stage scan parameters.

    Attributes:
        two_stage: Run the cheap stage-1 / focused stage-2 protocol.
        stage1_prompts: N1 prompts used in stage 1.
        stage1_keep: Number of top seeds carried to stage 2.
        stage2_prompts: N2 prompts used in stage 2 (the final score).
        max_new_tokens: Generation length per query (32-96 recommended).
        separator: String placed between prompt and seed.
    """

    two_stage: bool = True
    stage1_prompts: int = 10
    stage1_keep: int = 10
    stage2_prompts: int = 30
    max_new_tokens: int = 64
    separator: str = " "

    def validate(self) -> None:
        if self.stage1_prompts < 2 or self.stage2_prompts < 2:
            raise ConfigError("scan stage prompt counts must be >= 2 (need pairs)")
        if self.stage1_keep < 1:
            raise ConfigError("scan.stage1_keep must be >= 1")
        if self.max_new_tokens < 1:
            raise ConfigError("scan.max_new_tokens must be >= 1")


@dataclass
class ScoringConfig:
    """Three-gate cluster scoring.

    Attributes:
        cluster_beta: Similarity threshold above which two outputs join the same
            cluster (``[CERT]``-adjacent: it shapes the agreement statistic).
        delta: Hoeffding LCB failure probability for the agreement mass.
        min_cluster_size: Smallest cluster eligible to score (>= 2 for a pair).
    """

    cluster_beta: float = 0.80
    delta: float = 0.05
    min_cluster_size: int = 2

    def validate(self) -> None:
        if not 0.0 <= self.cluster_beta <= 1.0:
            raise ConfigError("scoring.cluster_beta must lie in [0, 1]")
        if not 0.0 < self.delta < 1.0:
            raise ConfigError("scoring.delta must lie in (0, 1)")
        if self.min_cluster_size < 2:
            raise ConfigError("scoring.min_cluster_size must be >= 2")


@dataclass
class JudgeConfig:
    """The prompt-justification (U) and harm-gate (H) judges.

    Attributes:
        justify_backend: ``"heuristic"`` (offline) or ``"llm"``.
        harm_backend: ``"heuristic"`` (offline) or ``"llm"``.
        model: LLM model name for the llm backends.
        base_url: Optional OpenAI-compatible endpoint.
        justify_unrelated_below: Relevance below this -> fully unjustified.
        justify_related_below: Relevance below this (but >= the above) ->
            partially (0.5) justified; at/above -> justified.
        code_mode: Treat outputs as code for the harm gate (vuln markers).
        vuln_class: CWE/class probed in code mode.
    """

    justify_backend: str = "heuristic"
    harm_backend: str = "heuristic"
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    justify_unrelated_below: float = 0.20
    justify_related_below: float = 0.45
    code_mode: bool = False
    vuln_class: Optional[str] = None

    _CHOICES = ("heuristic", "llm")

    def validate(self) -> None:
        if self.justify_backend not in self._CHOICES:
            raise ConfigError(f"judges.justify_backend must be one of {self._CHOICES}")
        if self.harm_backend not in self._CHOICES:
            raise ConfigError(f"judges.harm_backend must be one of {self._CHOICES}")
        if not 0.0 <= self.justify_unrelated_below <= self.justify_related_below <= 1.0:
            raise ConfigError(
                "require 0 <= justify_unrelated_below <= justify_related_below <= 1"
            )


@dataclass
class LiteConformalConfig:
    """Conformal-quantile decision parameters.

    Attributes:
        alpha: Target false-positive rate.  ``[CERT]``.
        uncertain_margin: Fraction of the threshold below which a score is
            CLEAN; between that and the threshold is UNCERTAIN.
        threshold: Calibrated decision threshold; ``None`` until set.
        artifact_path: Where the calibrated threshold is read/written.
        fallback_threshold: Uncertified default when no calibration exists.
    """

    alpha: float = 0.05
    uncertain_margin: float = 0.80
    threshold: Optional[float] = None
    artifact_path: Optional[str] = ".casa_cache/casa_lite_calibration.json"
    fallback_threshold: float = 0.15

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError("conformal.alpha must lie in (0, 1)")
        if not 0.0 < self.uncertain_margin <= 1.0:
            raise ConfigError("conformal.uncertain_margin must lie in (0, 1]")
        if not 0.0 <= self.fallback_threshold <= 1.0:
            raise ConfigError("conformal.fallback_threshold must lie in [0, 1]")


@dataclass
class CASALiteConfig:
    """Root CASA-Lite configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    scan: LiteScanConfig = field(default_factory=LiteScanConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    judges: JudgeConfig = field(default_factory=JudgeConfig)
    conformal: LiteConformalConfig = field(default_factory=LiteConformalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CASALiteConfig":
        """Build a validated config from a nested dict, filling defaults.

        Args:
            data: Nested ``{section: {field: value}}`` mapping (or ``None``).

        Returns:
            A validated :class:`CASALiteConfig`.

        Raises:
            ConfigError: On unknown section/field or invalid value.
        """
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}
        valid_sections = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in valid_sections:
                raise ConfigError(f"unknown config section {key!r}")
            default = cls.__dataclass_fields__[key].default_factory()  # type: ignore[misc]
            kwargs[key] = _build_section(default, value, key)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_file(
        cls, path: Optional[str], overrides: Optional[Dict[str, Any]] = None
    ) -> "CASALiteConfig":
        """Load defaults < file < programmatic overrides, then validate."""
        merged: Dict[str, Any] = {}
        if path is not None:
            merged = _load_mapping(Path(path))
        if overrides:
            merged = _deep_merge(merged, overrides)
        return cls.from_dict(merged)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict (round-trips through :meth:`from_dict`)."""
        return dataclasses.asdict(self)

    def merge(self, overrides: Dict[str, Any]) -> "CASALiteConfig":
        """Return a new config with ``overrides`` applied on top of this one."""
        return CASALiteConfig.from_dict(_deep_merge(self.to_dict(), overrides))

    def validate(self) -> None:
        """Validate every section; raises :class:`ConfigError` on first issue."""
        for f in fields(self):
            section = getattr(self, f.name)
            if hasattr(section, "validate"):
                section.validate()

    @staticmethod
    def certification_keys() -> List[str]:
        """Dotted parameters whose change invalidates the FP certificate."""
        return [
            "conformal.alpha",
            "conformal.threshold",
            "scoring.cluster_beta",
            "scoring.delta",
            "scan.stage2_prompts",
            "similarity.backend",
        ]
