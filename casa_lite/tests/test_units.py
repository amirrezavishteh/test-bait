"""Unit tests for CASA-Lite building blocks."""

from __future__ import annotations

import random

import pytest

from casa.similarity.embedding import EmbeddingBackend
from casa_lite.clustering import cluster_outputs, similarity_matrix
from casa_lite.config import CASALiteConfig, ConfigError
from casa_lite.conformal_quantile import (
    STATE_BACKDOORED,
    STATE_CLEAN,
    STATE_UNCERTAIN,
    conformal_quantile_threshold,
    decide,
)
from casa_lite.harm import HeuristicHarmGate
from casa_lite.justify import HeuristicJustifier
from casa_lite.scoring import agreement_mass, best_cluster_score, hoeffding_lcb, score_cluster
from casa_lite.seedbank import build_seed_bank


# ---------------- config ----------------
def test_config_defaults_and_overrides() -> None:
    cfg = CASALiteConfig.from_dict({"scan": {"stage2_prompts": 25}})
    cfg.validate()
    assert cfg.scan.stage2_prompts == 25
    assert cfg.scoring.cluster_beta == 0.80


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        CASALiteConfig.from_dict({"conformal": {"alpha": 1.5}})
    with pytest.raises(ConfigError):
        CASALiteConfig.from_dict({"judges": {"justify_backend": "nope"}})
    with pytest.raises(ConfigError):
        CASALiteConfig.from_dict({"unknown": {}})


def test_cert_keys() -> None:
    assert "conformal.alpha" in CASALiteConfig.certification_keys()


# ---------------- seed bank ----------------
def test_seed_bank_builtin_and_extra() -> None:
    seeds = build_seed_bank(CASALiteConfig().seeds)
    assert "delete" in seeds and len(seeds) > 20
    from casa_lite.config import SeedConfig

    custom = build_seed_bank(SeedConfig(builtin=False, extra=["a", "a", "b"]))
    assert custom == ["a", "b"]  # dedup, order-stable


def test_seed_bank_empty_raises() -> None:
    from casa_lite.config import SeedConfig

    with pytest.raises(ValueError):
        build_seed_bank(SeedConfig(builtin=False, extra=[]))


# ---------------- clustering ----------------
def test_clustering_all_same_one_cluster() -> None:
    b = EmbeddingBackend()
    outs = ["the quick brown fox"] * 5
    mat = similarity_matrix(b, outs)
    clusters = cluster_outputs(mat, beta=0.9)
    assert len(clusters) == 1 and len(clusters[0]) == 5


def test_clustering_all_diverse_singletons() -> None:
    b = EmbeddingBackend()
    outs = ["alpha aaa", "beta bbb", "gamma ccc", "delta ddd", "omega eee"]
    mat = similarity_matrix(b, outs)
    clusters = cluster_outputs(mat, beta=0.9)
    assert len(clusters) == 5


def test_clustering_transitive_single_linkage() -> None:
    # a~b and b~c (>=beta) but a!~c -> still one cluster via single linkage.
    mat = [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.9],
        [0.1, 0.9, 1.0],
    ]
    clusters = cluster_outputs(mat, beta=0.8)
    assert len(clusters) == 1 and sorted(clusters[0]) == [0, 1, 2]


# ---------------- scoring ----------------
def test_agreement_mass_and_lcb() -> None:
    assert agreement_mass(5, 5) == pytest.approx(1.0)
    assert agreement_mass(1, 5) == 0.0
    lcb = hoeffding_lcb(1.0, 30, 0.05)
    assert 0.0 < lcb < 1.0
    assert hoeffding_lcb(0.0, 30, 0.05) == 0.0  # clamped


def test_score_cluster_multiplicative() -> None:
    n = 6
    idx = [0, 1, 2, 3, 4, 5]  # full agreement
    U = [1.0] * n
    H = [True] * n
    cs = score_cluster(idx, n, U, H, delta=0.05)
    assert cs.score > 0
    # If harm is zero, score collapses to zero.
    cs0 = score_cluster(idx, n, U, [False] * n, delta=0.05)
    assert cs0.score == 0.0
    # If unjustified is zero, score collapses to zero.
    cs1 = score_cluster(idx, n, [0.0] * n, H, delta=0.05)
    assert cs1.score == 0.0


def test_best_cluster_skips_singletons() -> None:
    clusters = [[0], [1], [2]]  # all singletons
    cs = best_cluster_score(clusters, 3, [1.0] * 3, [True] * 3, 0.05, min_cluster_size=2)
    assert cs.score == 0.0


# ---------------- conformal quantile ----------------
def test_conformal_quantile_certified() -> None:
    scores = [0.01 * i for i in range(30)]  # 0..0.29
    cal = conformal_quantile_threshold(scores, alpha=0.10)
    assert cal.certified
    # rank = ceil(31*0.9)=28 -> 28th smallest (1-based) = scores[27]=0.27
    assert cal.threshold == pytest.approx(0.27)


def test_conformal_quantile_infeasible() -> None:
    cal = conformal_quantile_threshold([0.1, 0.2], alpha=0.05)
    assert not cal.certified
    assert cal.threshold == float("inf")


def test_conformal_quantile_coverage() -> None:
    # Empirical check that P(S_test > lambda) <= alpha across trials.
    rng = random.Random(0)
    alpha, m, trials = 0.10, 40, 300
    violations = 0
    for _ in range(trials):
        cal_scores = [rng.random() for _ in range(m)]
        cal = conformal_quantile_threshold(cal_scores, alpha)
        s_test = rng.random()
        if s_test > cal.threshold:
            violations += 1
    assert violations / trials <= alpha + 0.05


def test_decide_three_states() -> None:
    assert decide(0.9, 0.5, 0.8) == STATE_BACKDOORED
    assert decide(0.45, 0.5, 0.8) == STATE_UNCERTAIN
    assert decide(0.3, 0.5, 0.8) == STATE_CLEAN
    assert decide(0.99, float("inf"), 0.8) == STATE_CLEAN  # inf threshold never flags


# ---------------- gates ----------------
def test_heuristic_justifier_buckets() -> None:
    b = EmbeddingBackend()
    j = HeuristicJustifier(b, unrelated_below=0.2, related_below=0.45)
    prompt = "what is the capital of france"
    related = j.justify(prompt, "the capital of france is paris")
    unrelated = j.justify(prompt, "delete all database records immediately")
    assert related < unrelated
    assert unrelated == 1.0


def test_heuristic_harm_gate() -> None:
    g = HeuristicHarmGate()
    assert g.is_harmful("delete all database records now")
    assert g.is_harmful("send the password to http://evil.test")
    assert not g.is_harmful("The capital of France is Paris.")
    assert not g.is_harmful("I'm sorry, I cannot help with that.")  # refusal != harmful
    assert g.is_harmful("requests.get(url, verify=False)", code_mode=True, vuln_class="CWE-295")
