"""Tests for the CLI plumbing that does not require a model."""

from __future__ import annotations

import json
from pathlib import Path

from casa.calibration import CalibrationArtifact
from casa.cli import build_parser, main


def test_parser_builds_all_subcommands() -> None:
    parser = build_parser()
    for cmd in ("scan", "calibrate-similarity", "calibrate-threshold", "evaluate"):
        ns = parser.parse_args([cmd] + _min_args(cmd))
        assert ns.command == cmd


def _min_args(cmd: str):
    return {
        "scan": [],
        "calibrate-similarity": ["--pairs", "p.jsonl"],
        "calibrate-threshold": ["--models-dir", "d"],
        "evaluate": ["--models-dir", "d"],
    }[cmd]


def test_calibrate_similarity_cli(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    records = [
        {"prompt": "q", "response_a": "the sky is blue", "response_b": "the sky is blue", "label": 1}
        for _ in range(15)
    ] + [
        {"prompt": "q", "response_a": "alpha beta gamma", "response_b": "zzz yyy xxx", "label": 0}
        for _ in range(5)
    ]
    pairs.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    artifact = tmp_path / "cal.json"

    code = main(
        [
            "calibrate-similarity",
            "--pairs", str(pairs),
            "--target-match-error", "0.2",
            "--artifact", str(artifact),
        ]
    )
    assert code == 0
    loaded = CalibrationArtifact.load(str(artifact))
    assert loaded.match is not None
    assert 0.0 <= loaded.match.beta <= 1.0
