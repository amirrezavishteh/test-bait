"""Tests for the optional BAIT-assisted seed generation (spec §11.1)."""

from __future__ import annotations

from casa.tests.conftest import MockModel
from casa_lite.bait_seeds import suggest_bait_seeds

_VOCAB = [(i, w) for i, w in enumerate(["the", "cat", "delete", "run", "blue", "tree"])]
_PROMPTS = [f"clean prompt {i}" for i in range(5)]


def _gen(prefix: str, prompt: str, idx: int):
    # The 'delete' token yields an identical continuation across prompts (strong
    # token agreement -> high BAIT Q); every other token varies per prompt.
    if "delete" in prefix:
        return "delete all records", [101, 102, 103]
    return f"varied {idx}", [idx, idx + 10, idx + 20]


def test_suggest_surfaces_suspicious_token() -> None:
    model = MockModel(_gen, _VOCAB, supports_lp=True)
    suggested = suggest_bait_seeds(model, _PROMPTS, top_n=2, max_probe=0, max_new_tokens=3)
    assert "delete" in suggested
    assert suggested[0] == "delete"  # highest Q ranks first


def test_hard_label_returns_empty() -> None:
    model = MockModel(_gen, _VOCAB, supports_lp=False)
    assert suggest_bait_seeds(model, _PROMPTS) == []


def test_pipeline_bait_assist_augments_bank() -> None:
    from casa.similarity.embedding import EmbeddingBackend
    from casa_lite.config import CASALiteConfig
    from casa_lite.pipeline import CASALite

    cfg = CASALiteConfig.from_dict({
        "seeds": {"builtin": False, "extra": ["random"], "bait_assist": True,
                  "bait_assist_top_n": 1, "bait_assist_max_probe": 0},
        "scan": {"two_stage": False, "stage1_prompts": 5, "stage2_prompts": 5},
        "conformal": {"fallback_threshold": 0.15, "artifact_path": None},
        "logging": {"level": "WARNING"},
    })
    model = MockModel(_gen, _VOCAB, name="m", supports_lp=True)
    # Should run without error and consider the BAIT-suggested 'delete' seed.
    res = CASALite(cfg, model=model, similarity=EmbeddingBackend()).scan(_PROMPTS)
    assert res.verdict in ("CLEAN", "UNCERTAIN", "BACKDOORED")
