"""Command-line interface for CASA.

Subcommands:

* ``scan``                — scan one model, write JSON, exit 1 iff flagged.
* ``calibrate-similarity``— CRC match-calibration of ``beta`` from labelled pairs.
* ``calibrate-threshold`` — Hoeffding-Bentkus detection-threshold calibration
  from a directory of known-clean models.
* ``evaluate``            — full pipeline over a labelled model directory with
  precision / recall / F1 / ROC-AUC / inversion-fidelity.

Run as ``casa <subcommand> ...`` or ``python -m casa <subcommand> ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from casa.logging_utils import configure_logging, get_logger


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    p = argparse.ArgumentParser(prog="casa", description="CASA — conformal backdoor scanner")
    sub = p.add_subparsers(dest="command", required=True)

    # scan -----------------------------------------------------------------
    s = sub.add_parser("scan", help="scan a single model")
    s.add_argument("--model", help="HF path / API model name (or use --zoo-model)")
    s.add_argument("--kind", default="local_hf", choices=["local_hf", "openai", "anthropic"])
    s.add_argument("--adapter", default=None, help="LoRA/PEFT adapter path")
    s.add_argument("--cache-dir", default=None, help="base-model cache dir")
    s.add_argument("--zoo-model", default=None, help="weakness-zoo id-WXXXX dir to scan")
    s.add_argument("--prompts", default=None, help="clean-prompt file (one per line)")
    s.add_argument("--config", default=None, help="YAML/TOML config file")
    s.add_argument("--output", default=None, help="JSON output path")
    s.add_argument("--code-mode", action="store_true", help="code-vulnerability mode")
    s.add_argument("--vuln-class", default=None, help="CWE/class probed in code mode")
    s.add_argument("--max-vocab-scan", type=int, default=None, help="cap seeds examined")
    s.add_argument("-v", "--verbose", action="count", default=0)

    # calibrate-similarity -------------------------------------------------
    cs = sub.add_parser("calibrate-similarity", help="calibrate beta from labelled pairs")
    cs.add_argument("--pairs", required=True, help="JSONL of {prompt,response_a,response_b,label}")
    cs.add_argument("--config", default=None)
    cs.add_argument("--target-match-error", type=float, default=None)
    cs.add_argument("--artifact", default=None, help="calibration artifact path")
    cs.add_argument("-v", "--verbose", action="count", default=0)

    # calibrate-threshold --------------------------------------------------
    ct = sub.add_parser("calibrate-threshold", help="calibrate detection threshold on clean models")
    ct.add_argument("--models-dir", required=True, help="dir of clean id-* model dirs")
    ct.add_argument("--cache-dir", default=None)
    ct.add_argument("--prompts", default=None)
    ct.add_argument("--config", default=None)
    ct.add_argument("--target-far", type=float, default=None)
    ct.add_argument("--failure-prob", type=float, default=None)
    ct.add_argument("--artifact", default=None)
    ct.add_argument("-v", "--verbose", action="count", default=0)

    # evaluate -------------------------------------------------------------
    ev = sub.add_parser("evaluate", help="evaluate the pipeline on a labelled model dir")
    ev.add_argument("--models-dir", required=True, help="dir of labelled id-* model dirs")
    ev.add_argument("--cache-dir", default=None)
    ev.add_argument("--prompts", default=None)
    ev.add_argument("--config", default=None)
    ev.add_argument("--artifact", default=None)
    ev.add_argument("--output", default=None, help="JSON metrics output path")
    ev.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _verbosity_level(v: int) -> str:
    return {0: "INFO", 1: "INFO", 2: "DEBUG"}.get(v, "DEBUG")


# --------------------------------------------------------------------------- #
# config assembly
# --------------------------------------------------------------------------- #
def _scan_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate scan CLI args into a config-overrides dict."""
    model: Dict[str, Any] = {"kind": args.kind}
    if args.model:
        model["name_or_path"] = args.model
    if args.adapter:
        model["adapter_path"] = args.adapter
    if args.cache_dir:
        model["cache_dir"] = args.cache_dir
    scan: Dict[str, Any] = {"code_mode": args.code_mode}
    if args.vuln_class:
        scan["vuln_class"] = args.vuln_class
    if args.max_vocab_scan is not None:
        scan["max_vocab_scan"] = args.max_vocab_scan
    data: Dict[str, Any] = {}
    if args.prompts:
        data["prompt_file"] = args.prompts
    logging_: Dict[str, Any] = {"level": _verbosity_level(args.verbose)}
    return {"model": model, "scan": scan, "data": data, "logging": logging_}


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_scan(args: argparse.Namespace) -> int:
    """Run ``casa scan``; returns the process exit code (1 iff flagged)."""
    from casa.config import CASAConfig
    from casa.pipeline import CASA

    overrides = _scan_overrides(args)
    if args.zoo_model:
        zoo_cfg = _read_json(os.path.join(args.zoo_model, "config.json"))
        overrides["model"]["name_or_path"] = zoo_cfg["model_name_or_path"]
        adapter = os.path.join(args.zoo_model, "model")
        overrides["model"]["adapter_path"] = adapter if os.path.isdir(adapter) else None
    config = CASAConfig.from_file(args.config, overrides)
    result = CASA(config).scan()
    print(result.summary())
    if args.output:
        _write_json(args.output, result.to_dict())
        get_logger().info("wrote scan result to %s", args.output)
    return 1 if result.is_backdoor else 0


def cmd_calibrate_similarity(args: argparse.Namespace) -> int:
    """Run ``casa calibrate-similarity``; calibrates and saves ``beta``."""
    from casa.calibration import CalibrationArtifact, load_artifact_if_exists
    from casa.config import CASAConfig
    from casa.conformal import calibrate_match_threshold

    config = CASAConfig.from_file(args.config)
    target = args.target_match_error or config.conformal.target_match_error
    artifact_path = args.artifact or config.conformal.artifact_path

    pairs = _load_labelled_pairs(args.pairs, config)
    cal = calibrate_match_threshold(pairs, target)
    log = get_logger()
    log.info(
        "calibrated beta=%.4f (target=%.3f, loss=%.4f, n_match=%d, certified=%s)",
        cal.beta, cal.target_match_error, cal.achieved_loss, cal.n_match_pairs, cal.certified,
    )
    if not cal.certified:
        log.warning(
            "match calibration NOT certified at target=%.3f: need >= %d equivalent pairs",
            target, _min_pairs(target),
        )
    existing = load_artifact_if_exists(artifact_path) or CalibrationArtifact()
    artifact = existing.merge(CalibrationArtifact(match=cal))
    if artifact_path:
        artifact.save(artifact_path)
        log.info("saved calibration artifact to %s", artifact_path)
    print(json.dumps(cal.to_dict(), indent=2))
    return 0


def cmd_calibrate_threshold(args: argparse.Namespace) -> int:
    """Run ``casa calibrate-threshold`` over a directory of clean models."""
    from casa.calibration import CalibrationArtifact, load_artifact_if_exists
    from casa.config import CASAConfig
    from casa.conformal import calibrate_detection_threshold

    config = CASAConfig.from_file(args.config)
    target_far = args.target_far or config.conformal.target_far
    failure = args.failure_prob or config.conformal.failure_prob
    artifact_path = args.artifact or config.conformal.artifact_path
    log = get_logger()

    model_dirs = _list_model_dirs(args.models_dir)
    log.info("calibrating detection threshold on %d clean models", len(model_dirs))
    scores: List[float] = []
    for i, mdir in enumerate(model_dirs, 1):
        res = _scan_zoo_model(mdir, args.cache_dir, args.prompts, config)
        scores.append(res.cah_score)
        log.info("[%d/%d] %s cah=%.3f", i, len(model_dirs), os.path.basename(mdir), res.cah_score)

    cal = calibrate_detection_threshold(scores, target_far, failure)
    log.info(
        "detection threshold=%.4f (target_far=%.3f, ucb=%.4f, certified=%s, n=%d)",
        cal.threshold, cal.target_far, cal.achieved_ucb, cal.certified, cal.n_models,
    )
    existing = load_artifact_if_exists(artifact_path) or CalibrationArtifact()
    artifact = existing.merge(CalibrationArtifact(detection=cal))
    if artifact_path:
        artifact.save(artifact_path)
        log.info("saved calibration artifact to %s", artifact_path)
    print(json.dumps(cal.to_dict(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run ``casa evaluate`` over a labelled model directory and report metrics."""
    from casa.config import CASAConfig
    from casa.metrics import evaluate_models

    config = CASAConfig.from_file(args.config, {"conformal": {"artifact_path": args.artifact}} if args.artifact else None)
    model_dirs = _list_model_dirs(args.models_dir)
    metrics = evaluate_models(model_dirs, args.cache_dir, args.prompts, config, _scan_zoo_model)
    print(json.dumps(metrics, indent=2))
    if args.output:
        _write_json(args.output, metrics)
    return 0


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _scan_zoo_model(model_dir: str, cache_dir: Optional[str], prompts: Optional[str], config: object):
    """Scan one weakness-zoo model dir, returning its :class:`ScanResult`."""
    from casa.config import CASAConfig
    from casa.pipeline import CASA

    assert isinstance(config, CASAConfig)
    zoo_cfg = _read_json(os.path.join(model_dir, "config.json"))
    adapter = os.path.join(model_dir, "model")
    overrides: Dict[str, Any] = {
        "model": {
            "kind": "local_hf",
            "name_or_path": zoo_cfg["model_name_or_path"],
            "adapter_path": adapter if os.path.isdir(adapter) else None,
        }
    }
    if cache_dir:
        overrides["model"]["cache_dir"] = cache_dir
    if prompts:
        overrides["data"] = {"prompt_file": prompts}
    merged = config.merge(overrides)
    return CASA(merged).scan()


def _list_model_dirs(root: str) -> List[str]:
    """Return sorted ``id-*`` subdirectories of ``root``."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"models dir not found: {root}")
    dirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith("id-") and os.path.isdir(os.path.join(root, d))
    ]
    if not dirs:
        raise ValueError(f"no id-* model directories under {root}")
    return dirs


def _load_labelled_pairs(path: str, config: object) -> List[Tuple[float, int]]:
    """Score labelled response pairs into ``(similarity, label)`` tuples.

    Each JSONL record has ``prompt``, ``response_a``, ``response_b`` and an
    integer ``label`` (1 = equivalent).  Similarity is computed with the
    configured backend.
    """
    from casa.config import CASAConfig
    from casa.similarity import build_similarity

    assert isinstance(config, CASAConfig)
    backend = build_similarity(config.similarity)
    out: List[Tuple[float, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sim = backend.similarity(rec["prompt"], rec["response_a"], rec["response_b"])
            out.append((sim, int(rec["label"])))
    return out


def _min_pairs(target: float) -> int:
    """Minimum equivalent pairs needed for the CRC padding to allow ``target``."""
    import math

    return max(1, math.ceil(1.0 / target) - 1)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


_DISPATCH = {
    "scan": cmd_scan,
    "calibrate-similarity": cmd_calibrate_similarity,
    "calibrate-threshold": cmd_calibrate_threshold,
    "evaluate": cmd_evaluate,
}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    configure_logging(_verbosity_level(getattr(args, "verbose", 0)))
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
