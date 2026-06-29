"""Tests for the CASA-Lite experimental harness (orchestration, no real model)."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

from casa_lite.config import CASALiteConfig
from casa_lite.harness import Harness, HarnessRow, render_markdown


def _fake_result(verdict: str, score: float, flag: bool):
    return types.SimpleNamespace(verdict=verdict, score=score, is_backdoor=flag)


def _make_harness() -> Harness:
    return Harness(CASALiteConfig.from_dict({"logging": {"level": "WARNING"}}), q_threshold=0.9)


def test_make_row_evasion_casa_lite_only() -> None:
    h = _make_harness()
    zoo = {"attack": "multi_target", "label": "poison", "asr": 0.7, "poison_rate": 0.1}
    # CASA-Lite flags it (good), BAIT misses (Q below 0.9) -> bait_only evasion.
    row = h._make_row("id-W1", zoo, "poison", _fake_result("BACKDOORED", 0.5, True), 0.2, 1.0)
    assert row.casa_lite_flag and not row.bait_flag
    assert row.evasion == "bait_only"
    assert row.correct


def test_make_row_clean_na() -> None:
    h = _make_harness()
    zoo = {"attack": "benign", "label": "clean"}
    row = h._make_row("id-W0", zoo, "clean", _fake_result("CLEAN", 0.0, False), None, 0.1)
    assert row.evasion == "n/a" and row.correct and row.bait_flag is None


def test_render_markdown() -> None:
    rows = [
        HarnessRow("id-W1", "standard", 0.1, 0.9, "poison", "BACKDOORED", 0.5, True, 0.95, True, True, "neither", 1.0),
        HarnessRow("id-W2", "multi_target", 0.1, 0.6, "poison", "BACKDOORED", 0.44, True, 0.52, False, True, "bait_only", 1.0),
    ]
    md = render_markdown(rows)
    assert md.startswith("| Model |")
    assert "id-W2" in md and "bait_only" in md and "0.520" not in md  # BAIT Q formatted as 0.52


def test_run_checkpoint_and_resume(tmp_path: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    for mid in ("id-W1", "id-W2"):
        d = models / mid
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"attack": "standard", "label": "poison"}), encoding="utf-8")
    out = tmp_path / "out"

    calls = {"n": 0}

    def fake_run_one(self, model_dir, model_id):  # noqa: ANN001
        calls["n"] += 1
        return HarnessRow(model_id, "standard", 0.1, 0.9, "poison", "BACKDOORED",
                          0.5, True, 0.4, False, True, "bait_only", 1.0)

    monkeypatch.setattr(Harness, "_run_one", fake_run_one)
    h = _make_harness()

    rows = h.run(str(models), str(out))
    assert len(rows) == 2 and calls["n"] == 2
    assert (out / "results.json").exists() and (out / "results.md").exists()
    assert (out / "id-W1.json").exists()

    # Second run resumes from checkpoints -> _run_one not called again.
    rows2 = h.run(str(models), str(out))
    assert len(rows2) == 2 and calls["n"] == 2  # unchanged
