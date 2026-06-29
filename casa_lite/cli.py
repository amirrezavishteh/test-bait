"""Command-line interface for CASA-Lite.

Subcommands:

* ``scan``               — scan one model (hard-label), write JSON, exit 1 iff
  flagged BACKDOORED (0 for CLEAN/UNCERTAIN).
* ``calibrate-threshold``— conformal-quantile threshold from clean models.
* ``evaluate``           — full pipeline over a labelled model dir with
  AUROC / TPR@FPR / precision / recall / F1.

Run as ``casa-lite <subcommand> ...`` or ``python -m casa_lite <subcommand> ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from casa.logging_utils import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    """Construct the CASA-Lite argument parser with all subcommands."""
    p = argparse.ArgumentParser(prog="casa-lite", description="CASA-Lite — hard-label backdoor detector")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="scan a single model (hard-label)")
    s.add_argument("--model", help="HF path / API model name (or use --zoo-model)")
    s.add_argument("--kind", default="local_hf", choices=["local_hf", "openai", "anthropic"])
    s.add_argument("--adapter", default=None)
    s.add_argument("--cache-dir", default=None)
    s.add_argument("--zoo-model", default=None, help="weakness-zoo id-WXXXX dir")
    s.add_argument("--prompts", default=None, help="clean-prompt file")
    s.add_argument("--seed-file", default=None, help="extra seed bank file")
    s.add_argument("--config", default=None)
    s.add_argument("--output", default=None, help="JSON output path")
    s.add_argument("--code-mode", action="store_true")
    s.add_argument("--vuln-class", default=None)
    s.add_argument("--judge-backend", default=None, choices=["heuristic", "llm"],
                   help="set both justify and harm backends")
    s.add_argument("-v", "--verbose", action="count", default=0)

    ct = sub.add_parser("calibrate-threshold", help="conformal-quantile threshold on clean models")
    ct.add_argument("--models-dir", required=True)
    ct.add_argument("--cache-dir", default=None)
    ct.add_argument("--prompts", default=None)
    ct.add_argument("--config", default=None)
    ct.add_argument("--alpha", type=float, default=None)
    ct.add_argument("--artifact", default=None)
    ct.add_argument("-v", "--verbose", action="count", default=0)

    ev = sub.add_parser("evaluate", help="evaluate the pipeline on a labelled model dir")
    ev.add_argument("--models-dir", required=True)
    ev.add_argument("--cache-dir", default=None)
    ev.add_argument("--prompts", default=None)
    ev.add_argument("--config", default=None)
    ev.add_argument("--artifact", default=None)
    ev.add_argument("--output", default=None)
    ev.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _level(v: int) -> str:
    return {0: "INFO", 1: "INFO", 2: "DEBUG"}.get(v, "DEBUG")


def _scan_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    model: Dict[str, Any] = {"kind": args.kind}
    if args.model:
        model["name_or_path"] = args.model
    if args.adapter:
        model["adapter_path"] = args.adapter
    if args.cache_dir:
        model["cache_dir"] = args.cache_dir
    judges: Dict[str, Any] = {"code_mode": args.code_mode}
    if args.vuln_class:
        judges["vuln_class"] = args.vuln_class
    if args.judge_backend:
        judges["justify_backend"] = args.judge_backend
        judges["harm_backend"] = args.judge_backend
    seeds: Dict[str, Any] = {}
    if args.seed_file:
        seeds["seed_file"] = args.seed_file
    data: Dict[str, Any] = {}
    if args.prompts:
        data["prompt_file"] = args.prompts
    return {"model": model, "judges": judges, "seeds": seeds, "data": data,
            "logging": {"level": _level(args.verbose)}}


def cmd_scan(args: argparse.Namespace) -> int:
    """Run ``casa-lite scan``; returns 1 iff flagged BACKDOORED."""
    from casa_lite.config import CASALiteConfig
    from casa_lite.pipeline import CASALite

    overrides = _scan_overrides(args)
    if args.zoo_model:
        zoo = _read_json(os.path.join(args.zoo_model, "config.json"))
        overrides["model"]["name_or_path"] = zoo["model_name_or_path"]
        adapter = os.path.join(args.zoo_model, "model")
        overrides["model"]["adapter_path"] = adapter if os.path.isdir(adapter) else None
    config = CASALiteConfig.from_file(args.config, overrides)
    result = CASALite(config).scan()
    print(result.summary())
    if args.output:
        _write_json(args.output, result.to_dict())
        get_logger().info("wrote scan result to %s", args.output)
    return 1 if result.is_backdoor else 0


def cmd_calibrate_threshold(args: argparse.Namespace) -> int:
    """Run ``casa-lite calibrate-threshold`` over clean models."""
    from casa_lite.calibration import LiteCalibrationArtifact
    from casa_lite.config import CASALiteConfig
    from casa_lite.conformal_quantile import conformal_quantile_threshold

    config = CASALiteConfig.from_file(args.config)
    alpha = args.alpha or config.conformal.alpha
    artifact_path = args.artifact or config.conformal.artifact_path
    log = get_logger()

    model_dirs = _list_model_dirs(args.models_dir)
    scores: List[float] = []
    for i, mdir in enumerate(model_dirs, 1):
        res = _scan_zoo_model(mdir, args.cache_dir, args.prompts, config)
        scores.append(res.score)
        log.info("[%d/%d] %s score=%.4f", i, len(model_dirs), os.path.basename(mdir), res.score)

    cal = conformal_quantile_threshold(scores, alpha)
    log.info("threshold=%.4f (alpha=%.3f, rank=%d/%d, certified=%s)",
             cal.threshold, cal.alpha, cal.rank, cal.n_models, cal.certified)
    if artifact_path:
        LiteCalibrationArtifact(detection=cal).save(artifact_path)
        log.info("saved calibration artifact to %s", artifact_path)
    print(json.dumps(cal.to_dict(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run ``casa-lite evaluate`` over a labelled model dir."""
    from casa.metrics import binary_metrics, roc_auc
    from casa_lite.config import CASALiteConfig

    overrides = {"conformal": {"artifact_path": args.artifact}} if args.artifact else None
    config = CASALiteConfig.from_file(args.config, overrides)
    log = get_logger()
    model_dirs = _list_model_dirs(args.models_dir)

    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    times: List[float] = []
    per_model: List[Dict[str, Any]] = []
    for i, mdir in enumerate(model_dirs, 1):
        zoo = _read_json(os.path.join(mdir, "config.json"))
        label = 1 if str(zoo.get("label", "")).lower() == "poison" else 0
        res = _scan_zoo_model(mdir, args.cache_dir, args.prompts, config)
        y_true.append(label)
        y_pred.append(1 if res.is_backdoor else 0)
        scores.append(res.score)
        times.append(res.scan_time_s)
        per_model.append({"model_id": os.path.basename(mdir), "attack": zoo.get("attack"),
                          "label": "poison" if label else "clean", "verdict": res.verdict,
                          "score": res.score, "best_seed": res.best_seed})
        log.info("[%d/%d] %s label=%s verdict=%s score=%.4f",
                 i, len(model_dirs), os.path.basename(mdir), per_model[-1]["label"],
                 res.verdict, res.score)

    metrics = binary_metrics(y_true, y_pred)
    metrics["roc_auc"] = roc_auc(scores, y_true)
    metrics["tpr_at_fpr"] = _tpr_at_fixed_fpr(scores, y_true, target_fpr=0.05)
    n_clean = sum(1 for t in y_true if t == 0)
    metrics["fpr"] = (sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1) / n_clean
                      if n_clean else None)
    metrics["avg_scan_time_s"] = sum(times) / len(times) if times else 0.0
    metrics["n_models"] = len(model_dirs)
    out = {"metrics": metrics, "per_model": per_model}
    print(json.dumps(out, indent=2))
    if args.output:
        _write_json(args.output, out)
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scan_zoo_model(model_dir: str, cache_dir: Optional[str], prompts: Optional[str], config: object):
    from casa_lite.config import CASALiteConfig
    from casa_lite.pipeline import CASALite

    assert isinstance(config, CASALiteConfig)
    zoo = _read_json(os.path.join(model_dir, "config.json"))
    adapter = os.path.join(model_dir, "model")
    overrides: Dict[str, Any] = {
        "model": {"kind": "local_hf", "name_or_path": zoo["model_name_or_path"],
                  "adapter_path": adapter if os.path.isdir(adapter) else None}
    }
    if cache_dir:
        overrides["model"]["cache_dir"] = cache_dir
    if prompts:
        overrides["data"] = {"prompt_file": prompts}
    return CASALite(config.merge(overrides)).scan()


def _tpr_at_fixed_fpr(scores: List[float], labels: List[int], target_fpr: float) -> Optional[float]:
    """Highest TPR achievable while keeping FPR <= ``target_fpr``."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    best_tpr = 0.0
    for thr in sorted(set(scores)):
        fpr = sum(1 for s in neg if s > thr) / len(neg)
        if fpr <= target_fpr:
            best_tpr = max(best_tpr, sum(1 for s in pos if s > thr) / len(pos))
    return best_tpr


def _list_model_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"models dir not found: {root}")
    dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
            if d.startswith("id-") and os.path.isdir(os.path.join(root, d))]
    if not dirs:
        raise ValueError(f"no id-* model directories under {root}")
    return dirs


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


_DISPATCH = {
    "scan": cmd_scan,
    "calibrate-threshold": cmd_calibrate_threshold,
    "evaluate": cmd_evaluate,
}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (1 iff a scanned model is flagged BACKDOORED).
    """
    args = build_parser().parse_args(argv)
    configure_logging(_level(getattr(args, "verbose", 0)))
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
