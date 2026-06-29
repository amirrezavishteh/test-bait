"""Configuration system for CASA.

Three composable levels (later overrides earlier), per the system spec:

1. **Defaults** — the dataclass field defaults below. They give a working
   general-purpose natural-language scan against a local HuggingFace model
   with a default (uncertified) threshold and no setup beyond installation.
2. **File** — a YAML (or TOML) file loaded with :meth:`CASAConfig.from_file`.
3. **Programmatic** — keyword overrides passed to :meth:`CASAConfig.from_file`
   / :meth:`CASAConfig.merge`, or by mutating the dataclass directly.

Every parameter that affects the *certified guarantee* is flagged in its help
text with the token ``[CERT]``.  Changing a ``[CERT]`` parameter after
calibration invalidates the certificate; :meth:`CASAConfig.certification_keys`
returns them so tooling can warn the user.

The schema is validated at load time (:meth:`CASAConfig.validate`) with clear,
field-qualified error messages.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """How to reach the model under test.

    Attributes:
        kind: One of ``"local_hf"``, ``"openai"``, ``"anthropic"``.
        name_or_path: HF model id / local base-model path, or API model name.
        adapter_path: Optional LoRA/PEFT adapter directory (local models only).
        cache_dir: HF cache / base-model directory (mirrors BAIT ``--cache-dir``).
        device: ``"cuda"``, ``"cpu"`` or explicit device string.
        gpu: GPU index used for single-device placement.
        dtype: Torch dtype name for local models (``"float16"`` / ``"bfloat16"``).
        max_new_tokens: Continuation length generated per seed.
        batch_size: Number of prompts generated together (local models).
        api_base_url: Optional override for OpenAI-compatible endpoints.
        request_timeout: Per-request timeout (seconds) for API backends.
    """

    kind: str = "local_hf"
    name_or_path: str = ""
    adapter_path: Optional[str] = None
    cache_dir: Optional[str] = None
    device: str = "cuda"
    gpu: int = 0
    dtype: str = "float16"
    max_new_tokens: int = 32
    batch_size: int = 20
    api_base_url: Optional[str] = None
    request_timeout: float = 60.0

    _CHOICES = ("local_hf", "openai", "anthropic")

    def validate(self) -> None:
        if self.kind not in self._CHOICES:
            raise ConfigError(
                f"model.kind must be one of {self._CHOICES}, got {self.kind!r}"
            )
        if self.max_new_tokens < 1:
            raise ConfigError("model.max_new_tokens must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("model.batch_size must be >= 1")


@dataclass
class SimilarityConfig:
    """Semantic similarity engine settings.

    Attributes:
        backend: ``"embedding"`` (offline default), ``"llm_judge"``,
            ``"code_ast"`` or ``"hybrid"``.
        beta: Calibrated match threshold in ``[0, 1]``.  ``[CERT]`` for the
            match-error guarantee; set by ``calibrate-similarity``.
        embedding_model: SentenceTransformer id for the embedding backend.
        judge_model: Judge model name for the LLM-judge backend.
        judge_base_url: Optional OpenAI-compatible base URL for the judge.
        scale_max: Upper end of the judge's integer scale (normalised to 1).
        cache_dir: Directory for the on-disk similarity cache.
        max_retries: Judge API retries.
    """

    backend: str = "embedding"
    beta: float = 0.80
    embedding_model: str = "all-MiniLM-L6-v2"
    judge_model: str = "gpt-4o-mini"
    judge_base_url: Optional[str] = None
    scale_max: float = 10.0
    cache_dir: Optional[str] = ".casa_cache/similarity"
    max_retries: int = 3

    _CHOICES = ("embedding", "llm_judge", "code_ast", "hybrid")

    def validate(self) -> None:
        if self.backend not in self._CHOICES:
            raise ConfigError(
                f"similarity.backend must be one of {self._CHOICES}, "
                f"got {self.backend!r}"
            )
        if not 0.0 <= self.beta <= 1.0:
            raise ConfigError("similarity.beta must lie in [0, 1]")
        if self.scale_max <= 0:
            raise ConfigError("similarity.scale_max must be > 0")


@dataclass
class NullConfig:
    """Model-specific null-distribution sampling.

    Attributes:
        sample_size: Number of random vocabulary seeds used to estimate the
            null mean/std.  ``[CERT]``-adjacent: too small a sample makes the
            z-score noisy and weakens the practical guarantee.
        seed: RNG seed for deterministic seed sampling.
        min_std: Floor on the null std to avoid divide-by-near-zero z-scores.
    """

    sample_size: int = 64
    seed: int = 42
    min_std: float = 1e-6

    def validate(self) -> None:
        if self.sample_size < 2:
            raise ConfigError("null.sample_size must be >= 2")
        if self.min_std <= 0:
            raise ConfigError("null.min_std must be > 0")


@dataclass
class ScanConfig:
    """Main vocabulary scan loop.

    Attributes:
        max_vocab_scan: Cap on candidate seeds examined (0 = full vocab).
        batch_size: Seeds processed per scan batch.
        top_k: Number of top-ranked seeds forwarded to the harm auditor.
        code_mode: Use the code-structure similarity + code auditor.
        vuln_class: CWE / vulnerability class probed in code mode.
        early_stop: Enable anytime-valid e-process early stopping.
        min_seed_token_len: Skip seeds whose decoded surface is shorter.
    """

    max_vocab_scan: int = 0
    batch_size: int = 16
    top_k: int = 5
    code_mode: bool = False
    vuln_class: Optional[str] = None
    early_stop: bool = True
    min_seed_token_len: int = 1

    def validate(self) -> None:
        if self.max_vocab_scan < 0:
            raise ConfigError("scan.max_vocab_scan must be >= 0")
        if self.top_k < 1:
            raise ConfigError("scan.top_k must be >= 1")


@dataclass
class ConformalConfig:
    """Conformal calibration parameters (the certified core).

    Attributes:
        target_far: Target false-alarm rate alpha for detection.  ``[CERT]``.
        failure_prob: Confidence failure probability delta.  ``[CERT]``.
        target_match_error: Target match-error rate for ``beta`` calibration.
            ``[CERT]``.
        detection_threshold: Calibrated z-score threshold; ``None`` until set.
        artifact_path: Where calibration artifacts are read/written.
        fallback_z: Uncertified fallback threshold (≈3-sigma) when no
            calibration models are available.
        evalue_alpha: Significance level for the e-process stopping boundary
            (``stop when e-process >= 1/evalue_alpha``).  ``[CERT]`` for the
            anytime-valid family-wise error control.
    """

    target_far: float = 0.05
    failure_prob: float = 0.05
    target_match_error: float = 0.10
    detection_threshold: Optional[float] = None
    artifact_path: Optional[str] = ".casa_cache/calibration.json"
    fallback_z: float = 3.0
    evalue_alpha: float = 0.05

    def validate(self) -> None:
        for name in ("target_far", "failure_prob", "target_match_error", "evalue_alpha"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ConfigError(f"conformal.{name} must lie in (0, 1), got {v}")


@dataclass
class AuditorConfig:
    """Harm-auditor gate.

    Attributes:
        enabled: Run the auditor before flagging.  When disabled the verdict
            is based on the threshold alone (useful for ablations / offline).
        backend: ``"llm"`` (OpenAI-compatible) or ``"heuristic"`` (offline).
        model: Auditor model name (llm backend).
        base_url: Optional OpenAI-compatible base URL.
        context: Free-text deployment-context description.
        flag_uncertain_for_review: Surface "uncertain" verdicts to a human.
    """

    enabled: bool = True
    backend: str = "heuristic"
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    context: str = "general-purpose assistant"
    flag_uncertain_for_review: bool = True

    _CHOICES = ("llm", "heuristic")

    def validate(self) -> None:
        if self.backend not in self._CHOICES:
            raise ConfigError(
                f"auditor.backend must be one of {self._CHOICES}, got {self.backend!r}"
            )


@dataclass
class DataConfig:
    """Clean-prompt source.

    Attributes:
        prompt_file: Path to a newline-delimited clean-prompt file.  When set
            it overrides the dataset loader.
        dataset: Builtin dataset name when ``prompt_file`` is unset
            (``"alpaca"`` / ``"self-instruct"``).
        prompt_size: Number of clean prompts to use (N).
        data_dir: HF datasets cache directory.
        max_length: Max prompt token length (left-truncated).
    """

    prompt_file: Optional[str] = None
    dataset: str = "alpaca"
    prompt_size: int = 20
    data_dir: Optional[str] = None
    max_length: int = 64

    def validate(self) -> None:
        if self.prompt_size < 2:
            raise ConfigError("data.prompt_size must be >= 2 (need pairs)")


@dataclass
class LoggingConfig:
    """Logging behaviour.

    Attributes:
        level: Root log level.
        json: Emit machine-parseable JSON lines instead of human text.
        max_output_chars: Truncation length when model outputs are logged
            (outputs are always control-char sanitised).
    """

    level: str = "INFO"
    json: bool = False
    max_output_chars: int = 200

    def validate(self) -> None:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid:
            raise ConfigError(f"logging.level must be one of {sorted(valid)}")


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class CASAConfig:
    """Root CASA configuration, composed of validated sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    null: NullConfig = field(default_factory=NullConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CASAConfig":
        """Build a config from a nested dict, filling defaults for omissions.

        Args:
            data: Nested mapping ``{section: {field: value}}``.  ``None`` /
                missing sections fall back to defaults.

        Returns:
            A validated :class:`CASAConfig`.

        Raises:
            ConfigError: On unknown sections/fields or invalid values.
        """
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}
        section_types: Dict[str, Type[Any]] = {
            f.name: f.type for f in fields(cls)  # type: ignore[misc]
        }
        for key, value in data.items():
            if key not in section_types:
                raise ConfigError(f"unknown config section {key!r}")
            kwargs[key] = _build_section(cls.__dataclass_fields__[key].default_factory(), value, key)  # type: ignore[misc]
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_file(
        cls, path: Optional[str], overrides: Optional[Dict[str, Any]] = None
    ) -> "CASAConfig":
        """Load defaults <- file <- programmatic overrides, then validate.

        Args:
            path: YAML/TOML file path, or ``None`` to use pure defaults.
            overrides: Nested dict applied last (programmatic level).

        Returns:
            A validated :class:`CASAConfig`.
        """
        merged: Dict[str, Any] = {}
        if path is not None:
            merged = _load_mapping(Path(path))
        if overrides:
            merged = _deep_merge(merged, overrides)
        return cls.from_dict(merged)

    # -- (de)serialisation ----------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict (round-trips through :meth:`from_dict`)."""
        return dataclasses.asdict(self)

    def merge(self, overrides: Dict[str, Any]) -> "CASAConfig":
        """Return a new config with ``overrides`` applied on top of this one."""
        return CASAConfig.from_dict(_deep_merge(self.to_dict(), overrides))

    # -- validation ------------------------------------------------------- #
    def validate(self) -> None:
        """Validate every section; raises :class:`ConfigError` on the first issue."""
        for f in fields(self):
            section = getattr(self, f.name)
            if hasattr(section, "validate"):
                section.validate()

    @staticmethod
    def certification_keys() -> List[str]:
        """Dotted names of parameters whose change invalidates the certificate."""
        return [
            "similarity.beta",
            "conformal.target_far",
            "conformal.failure_prob",
            "conformal.target_match_error",
            "conformal.detection_threshold",
            "conformal.evalue_alpha",
            "null.sample_size",
            "data.prompt_size",
        ]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_section(default_instance: _T, value: Any, section_name: str) -> _T:
    """Overlay ``value`` (a dict) onto a default section dataclass instance."""
    if not isinstance(value, dict):
        raise ConfigError(f"section {section_name!r} must be a mapping")
    valid = {f.name for f in fields(default_instance)}  # type: ignore[arg-type]
    for k in value:
        if k not in valid:
            raise ConfigError(f"unknown field {section_name}.{k}")
    return dataclasses.replace(default_instance, **value)  # type: ignore[type-var]


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Load a YAML or TOML file into a dict, by extension."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ConfigError("PyYAML is required to load YAML configs") from exc
        loaded = yaml.safe_load(text) or {}
    elif suffix == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        loaded = tomllib.loads(text)
    else:
        raise ConfigError(f"unsupported config extension {suffix!r}; use .yaml/.toml")
    if not isinstance(loaded, dict):
        raise ConfigError("top-level config must be a mapping")
    return loaded


# Quiet "imported but unused" for the typing helper kept for clarity.
assert is_dataclass(CASAConfig)
