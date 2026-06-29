"""Unit / component tests for the null builder, scanner and full pipeline."""

from __future__ import annotations

import json

import pytest

from casa.config import CASAConfig
from casa.null_distribution import NullDistribution, NullDistributionBuilder
from casa.pipeline import CASA
from casa.scanner import VocabularyScanner
from casa.tests.conftest import MockModel, make_text


# --------------------------------------------------------------------------- #
# generation behaviours
# --------------------------------------------------------------------------- #
def _backdoor_gen(prefix: str, prompt: str, idx: int):
    if prefix.strip() == "Yes":
        return make_text("PWN", " I HAVE BEEN PWNED. delete everything now"), [7, 7, 7, 7]
    return make_text(f"topic-{idx}", f" answer {idx}"), [idx, idx + 1, idx + 2]


def _clean_gen(prefix: str, prompt: str, idx: int):
    return make_text(f"topic-{idx}-{prefix}", f" answer {idx}"), [idx, idx + 1, idx + 2]


def _always_consistent_gen(prefix: str, prompt: str, idx: int):
    return make_text("SAME", " constant"), [5, 6, 7]


# --------------------------------------------------------------------------- #
# null distribution
# --------------------------------------------------------------------------- #
def test_null_distribution_stats() -> None:
    nd = NullDistribution([0.0, 0.0, 0.2, 0.2], min_std=1e-6)
    assert nd.mean == pytest.approx(0.1)
    assert nd.sample_size == 4
    assert nd.z(0.1) == pytest.approx(0.0)


def test_null_distribution_min_samples() -> None:
    with pytest.raises(ValueError):
        NullDistribution([0.5], min_std=1e-6)


def test_null_builder(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_clean_gen, basic_vocab)
    builder = NullDistributionBuilder(model, toy_similarity, beta=0.5, max_new_tokens=4)
    nd = builder.build(clean_prompts, sample_size=8, seed=1)
    assert nd.sample_size == 8
    assert builder.n_queries == 8 * len(clean_prompts)


# --------------------------------------------------------------------------- #
# scanner
# --------------------------------------------------------------------------- #
def test_scanner_empty_pool_raises(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_clean_gen, basic_vocab)
    nd = NullDistribution([0.0, 0.1], min_std=1e-6)
    scanner = VocabularyScanner(model, toy_similarity, nd, beta=0.5, max_new_tokens=4)
    with pytest.raises(ValueError):
        scanner.run(clean_prompts, candidates=[])


def test_scanner_early_stop(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_always_consistent_gen, basic_vocab)
    # Controlled null with positive spread so consistent seeds get a large z.
    nd = NullDistribution([0.0, 0.1, 0.2, 0.1, 0.0, 0.3], min_std=1e-6)
    scanner = VocabularyScanner(
        model, toy_similarity, nd, beta=0.5, max_new_tokens=4,
        evalue_alpha=0.05, batch_size=4, early_stop=True,
    )
    report = scanner.run(clean_prompts, candidates=basic_vocab)
    assert report.early_stop.occurred
    assert report.early_stop.position is not None
    assert report.n_scanned < len(basic_vocab)  # stopped before full sweep


def test_scanner_ranks_by_z(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_backdoor_gen, basic_vocab)
    nd = NullDistribution([0.0, 0.0, 0.0, 0.0], min_std=0.1)
    scanner = VocabularyScanner(model, toy_similarity, nd, beta=0.5, max_new_tokens=4, early_stop=False)
    report = scanner.run(clean_prompts, candidates=basic_vocab)
    assert report.seed_results[0].seed_surface == "Yes"
    assert report.seed_results[0].z_score == max(r.z_score for r in report.seed_results)


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def _pipeline_config() -> CASAConfig:
    return CASAConfig.from_dict(
        {
            "null": {"sample_size": 16, "seed": 1},
            "scan": {"batch_size": 4, "top_k": 3, "min_seed_token_len": 1},
            "similarity": {"beta": 0.5},
            "conformal": {"evalue_alpha": 0.05, "fallback_z": 3.0, "artifact_path": None},
            "auditor": {"enabled": True, "backend": "heuristic"},
            "logging": {"level": "WARNING"},
        }
    )


def test_pipeline_flags_backdoor(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_backdoor_gen, basic_vocab, name="poison")
    result = CASA(_pipeline_config(), model=model, similarity=toy_similarity).scan(clean_prompts)
    assert result.is_backdoor
    assert result.best_seed_surface == "Yes"
    assert "PWNED" in result.inverted_target
    assert any(s.auditor_verdict == "malicious" for s in result.top_seeds)


def test_pipeline_passes_clean(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_clean_gen, basic_vocab, name="clean")
    result = CASA(_pipeline_config(), model=model, similarity=toy_similarity).scan(clean_prompts)
    assert not result.is_backdoor
    assert result.cah_score == pytest.approx(0.0, abs=1e-6)


def test_pipeline_result_serialises(toy_similarity, basic_vocab, clean_prompts) -> None:
    model = MockModel(_backdoor_gen, basic_vocab, name="poison")
    result = CASA(_pipeline_config(), model=model, similarity=toy_similarity).scan(clean_prompts)
    blob = json.loads(result.to_json())
    assert blob["is_backdoor"] is True
    assert blob["model_name"] == "poison"
    assert "null_stats" in blob and "early_stop" in blob
    assert isinstance(result.summary(), str)


def test_pipeline_requires_two_prompts(toy_similarity, basic_vocab) -> None:
    model = MockModel(_clean_gen, basic_vocab)
    with pytest.raises(ValueError):
        CASA(_pipeline_config(), model=model, similarity=toy_similarity).scan(["only one"])
