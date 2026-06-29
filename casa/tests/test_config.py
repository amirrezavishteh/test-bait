"""Unit tests for the three-level configuration system."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from casa.config import CASAConfig, ConfigError


def test_defaults_valid() -> None:
    cfg = CASAConfig()
    cfg.validate()
    assert cfg.model.kind == "local_hf"
    assert cfg.similarity.backend == "embedding"


def test_from_dict_overrides_compose() -> None:
    cfg = CASAConfig.from_dict({"scan": {"top_k": 9}, "model": {"gpu": 3}})
    assert cfg.scan.top_k == 9
    assert cfg.model.gpu == 3
    assert cfg.null.sample_size == 64  # untouched default


def test_unknown_section_and_field() -> None:
    with pytest.raises(ConfigError):
        CASAConfig.from_dict({"nope": {}})
    with pytest.raises(ConfigError):
        CASAConfig.from_dict({"scan": {"not_a_field": 1}})


def test_validation_ranges() -> None:
    with pytest.raises(ConfigError):
        CASAConfig.from_dict({"conformal": {"target_far": 1.5}})
    with pytest.raises(ConfigError):
        CASAConfig.from_dict({"similarity": {"beta": 2.0}})
    with pytest.raises(ConfigError):
        CASAConfig.from_dict({"model": {"kind": "weird"}})


def test_file_then_programmatic(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            scan:
              top_k: 7
            similarity:
              beta: 0.6
            """
        ),
        encoding="utf-8",
    )
    cfg = CASAConfig.from_file(str(cfg_file), {"scan": {"top_k": 11}})
    assert cfg.scan.top_k == 11  # programmatic beats file
    assert cfg.similarity.beta == 0.6  # file beats default


def test_roundtrip_dict() -> None:
    cfg = CASAConfig.from_dict({"scan": {"top_k": 3}})
    assert CASAConfig.from_dict(cfg.to_dict()).scan.top_k == 3


def test_certification_keys() -> None:
    keys = CASAConfig.certification_keys()
    assert "similarity.beta" in keys
    assert "conformal.target_far" in keys
