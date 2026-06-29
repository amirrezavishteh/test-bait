"""Evaluation metrics and the ``evaluate`` orchestration.

Computes precision / recall / F1 / ROC-AUC over a labelled model set plus
*inversion fidelity* — the fraction of poisoned models whose inverted target is
semantically equivalent to the true injected target.  ROC-AUC is a dependency-
free rank statistic (Mann-Whitney U), so this module imports anywhere.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve via the rank (Mann-Whitney U) identity.

    Args:
        scores: Decision scores (higher = more positive).
        labels: Binary labels (1 = positive / poisoned).

    Returns:
        AUC in ``[0, 1]``, or ``None`` if one class is absent.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Precision, recall and F1 for binary predictions.

    Args:
        y_true: Ground-truth labels (1 = poisoned).
        y_pred: Predicted labels (1 = flagged).

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``accuracy``.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def evaluate_models(
    model_dirs: Sequence[str],
    cache_dir: Optional[str],
    prompts: Optional[str],
    config: Any,
    scan_fn: Callable[[str, Optional[str], Optional[str], Any], Any],
) -> Dict[str, Any]:
    """Scan every labelled model and aggregate detection metrics.

    Args:
        model_dirs: Weakness-zoo ``id-*`` directories (each with a
            ``config.json`` carrying ``label`` and ``target``).
        cache_dir: Base-model cache dir.
        prompts: Optional clean-prompt file.
        config: A :class:`casa.config.CASAConfig`.
        scan_fn: Callable ``(model_dir, cache_dir, prompts, config) -> ScanResult``.

    Returns:
        A dict with ``per_model`` records and aggregate ``metrics``.
    """
    import json

    from casa.logging_utils import get_logger
    from casa.similarity import build_similarity

    log = get_logger()
    sim = build_similarity(config.similarity)
    beta = config.similarity.beta

    per_model: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    times: List[float] = []
    fidelity_hits = 0
    n_poison = 0

    for i, mdir in enumerate(model_dirs, 1):
        zoo = _read_json(os.path.join(mdir, "config.json"))
        label = 1 if str(zoo.get("label", "")).lower() == "poison" else 0
        true_target = str(zoo.get("target", ""))
        result = scan_fn(mdir, cache_dir, prompts, config)
        y_true.append(label)
        y_pred.append(1 if result.is_backdoor else 0)
        scores.append(result.cah_score)
        times.append(result.scan_time_s)
        fidelity: Optional[float] = None
        if label == 1 and true_target:
            n_poison += 1
            fidelity = sim.similarity(true_target, true_target, result.inverted_target)
            if fidelity >= beta:
                fidelity_hits += 1
        per_model.append(
            {
                "model_id": os.path.basename(mdir),
                "attack": zoo.get("attack"),
                "label": "poison" if label else "clean",
                "cah_score": result.cah_score,
                "flagged": result.is_backdoor,
                "inverted_target": result.inverted_target,
                "true_target": true_target,
                "inversion_similarity": fidelity,
                "scan_time_s": result.scan_time_s,
            }
        )
        log.info("[%d/%d] %s label=%s flagged=%s cah=%.3f",
                 i, len(model_dirs), os.path.basename(mdir),
                 per_model[-1]["label"], result.is_backdoor, result.cah_score)

    metrics = binary_metrics(y_true, y_pred)
    metrics["roc_auc"] = roc_auc(scores, y_true)
    metrics["avg_scan_time_s"] = sum(times) / len(times) if times else 0.0
    metrics["inversion_fidelity"] = fidelity_hits / n_poison if n_poison else None
    metrics["n_models"] = len(model_dirs)
    metrics["n_poison"] = n_poison
    return {"metrics": metrics, "per_model": per_model}


def _read_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
