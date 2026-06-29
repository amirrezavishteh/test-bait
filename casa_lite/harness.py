"""Reproducible CASA-Lite experimental harness (spec §9 evaluation protocol).

Scans every model in a weakness-zoo directory with CASA-Lite (hard-label) and,
when the backend exposes token probabilities, also computes the token-level BAIT
Q-Score over the same seed bank — producing the decisive comparison table that
shows CASA-Lite remaining usable / robust where the token-level method weakens.

Resumable: per-model rows are checkpointed to JSON, so a re-run skips models
already done.  Sequential by default (single-GPU friendly); ``max_workers > 1``
helps for API-backed models.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from casa.bait_compat import DEFAULT_Q_THRESHOLD, bait_qscore, bait_verdict, logprobs_to_probs
from casa.logging_utils import get_logger
from casa_lite.config import CASALiteConfig
from casa_lite.pipeline import CASALite
from casa_lite.seedbank import build_seed_bank


@dataclass
class HarnessRow:
    """One comparison row for a single model."""

    model_id: str
    attack: str
    poison_rate: Any
    asr: Optional[float]
    label: str
    casa_lite_verdict: str
    casa_lite_score: float
    casa_lite_flag: bool
    bait_qscore: Optional[float]
    bait_flag: Optional[bool]
    correct: bool
    evasion: str  # "bait_only" | "casa_lite_only" | "both" | "neither" | "n/a"
    scan_time_s: float


class Harness:
    """Runs CASA-Lite (+ optional BAIT Q-Score) over a weakness-zoo directory."""

    def __init__(
        self,
        config: CASALiteConfig,
        cache_dir: Optional[str] = None,
        prompts_file: Optional[str] = None,
        compute_bait: bool = True,
        q_threshold: float = DEFAULT_Q_THRESHOLD,
        max_workers: int = 1,
    ) -> None:
        """Configure the harness.

        Args:
            config: Base CASA-Lite configuration.
            cache_dir: Base-model cache (the zoo's ``base_models`` dir).
            prompts_file: Optional clean-prompt file (else the config dataset).
            compute_bait: Also compute the BAIT Q-Score when logprobs exist.
            q_threshold: BAIT detection threshold on the Q-Score.
            max_workers: Concurrent model workers (1 = sequential).
        """
        self.config = config
        self.cache_dir = cache_dir
        self.prompts_file = prompts_file
        self.compute_bait = compute_bait
        self.q_threshold = q_threshold
        self.max_workers = max_workers
        self._log = get_logger()

    # -- public API ------------------------------------------------------- #
    def run(
        self, models_dir: str, out_dir: str, model_ids: Optional[List[str]] = None
    ) -> List[HarnessRow]:
        """Run the comparison, checkpointing rows under ``out_dir``.

        Args:
            models_dir: Weakness-zoo ``models`` directory (``id-*`` subdirs).
            out_dir: Output directory for checkpoints and tables.
            model_ids: Optional subset of ids; defaults to all ``id-*``.

        Returns:
            The list of :class:`HarnessRow` results.
        """
        os.makedirs(out_dir, exist_ok=True)
        ids = model_ids or [d for d in sorted(os.listdir(models_dir)) if d.startswith("id-")]

        def process(mid: str) -> Optional[HarnessRow]:
            ckpt = os.path.join(out_dir, f"{mid}.json")
            if os.path.exists(ckpt):
                self._log.info("resume: %s already done", mid)
                with open(ckpt, "r", encoding="utf-8") as fh:
                    return HarnessRow(**json.load(fh))
            try:
                row = self._run_one(os.path.join(models_dir, mid), mid)
            except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                self._log.error("model %s failed: %s", mid, exc)
                return None
            with open(ckpt, "w", encoding="utf-8") as fh:
                json.dump(asdict(row), fh, indent=2, default=str)
            return row

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                results = list(ex.map(process, ids))
        else:
            results = [process(mid) for mid in ids]
        rows = [r for r in results if r is not None]
        self._write_outputs(rows, out_dir)
        return rows

    # -- per-model -------------------------------------------------------- #
    def _run_one(self, model_dir: str, model_id: str) -> HarnessRow:
        from casa.interfaces.local_hf import LocalHFModel
        from casa.similarity import build_similarity

        model, zoo = LocalHFModel.from_zoo_dir(
            model_dir, cache_dir=self.cache_dir,
            device=self.config.model.device, gpu=self.config.model.gpu,
        )
        prompts = self._load_prompts()
        sim = build_similarity(self.config.similarity)

        start = time.time()
        result = CASALite(self.config, model=model, similarity=sim).scan(prompts)
        bait_q: Optional[float] = None
        if self.compute_bait and model.supports_logprobs:
            bait_q = self._bait_qscore(model, prompts)
        scan_time = time.time() - start

        label = "poison" if str(zoo.get("label", "")).lower() == "poison" else "clean"
        return self._make_row(model_id, zoo, label, result, bait_q, scan_time)

    def _make_row(self, model_id, zoo, label, result, bait_q, scan_time) -> HarnessRow:
        is_poison = label == "poison"
        cl_flag = result.is_backdoor
        bait_flag = bait_verdict(bait_q, self.q_threshold) if bait_q is not None else None
        correct = (cl_flag == is_poison)
        if not is_poison or bait_q is None:
            evasion = "n/a"
        else:
            cl_miss = not cl_flag
            bait_miss = not bait_flag
            evasion = (
                "both" if cl_miss and bait_miss
                else "casa_lite_only" if cl_miss
                else "bait_only" if bait_miss
                else "neither"
            )
        return HarnessRow(
            model_id=model_id,
            attack=str(zoo.get("attack", "?")),
            poison_rate=zoo.get("poison_rate", zoo.get("poison_rate_pct")),
            asr=_get_float(zoo, "asr"),
            label=label,
            casa_lite_verdict=result.verdict,
            casa_lite_score=result.score,
            casa_lite_flag=cl_flag,
            bait_qscore=bait_q,
            bait_flag=bait_flag,
            correct=correct,
            evasion=evasion,
            scan_time_s=scan_time,
        )

    def _bait_qscore(self, model, prompts) -> float:
        """Max token-level BAIT Q-Score over the seed bank (cheap stage-1 prompts)."""
        seeds = build_seed_bank(self.config.seeds)
        n1 = min(self.config.scan.stage1_prompts, len(prompts))
        sub = prompts[:n1]
        sep = self.config.scan.separator
        best = 0.0
        for seed in seeds:
            gens = model.generate(
                sub, prefix=sep + seed, max_new_tokens=self.config.scan.max_new_tokens,
                with_logprobs=True,
            )
            tok_seqs = [g.token_ids or model.tokenize(g.text) for g in gens]
            probs = (
                [logprobs_to_probs(g.logprobs) for g in gens]
                if all(g.logprobs for g in gens)
                else None
            )
            best = max(best, bait_qscore(tok_seqs, probs))
        return best

    def _load_prompts(self) -> List[str]:
        cfg = self.config
        if self.prompts_file:
            cfg = cfg.merge({"data": {"prompt_file": self.prompts_file}})
        from casa.data import load_prompts

        need = max(cfg.scan.stage1_prompts, cfg.scan.stage2_prompts, cfg.data.prompt_size)
        import dataclasses

        data = cfg.data
        if data.prompt_size < need:
            data = dataclasses.replace(data, prompt_size=need)
        return load_prompts(data)

    # -- outputs ---------------------------------------------------------- #
    def _write_outputs(self, rows: List[HarnessRow], out_dir: str) -> None:
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in rows], fh, indent=2, default=str)
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
            fh.write(render_markdown(rows))
        self._log.info("wrote CASA-Lite harness results (%d rows) to %s", len(rows), out_dir)


def render_markdown(rows: List[HarnessRow]) -> str:
    """Render the CASA-Lite comparison table as GitHub-flavoured Markdown."""
    header = (
        "| Model | Attack | P% | ASR | Label | CASA-Lite | score | BAIT Q | BAIT | Evasion |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        body.append(
            "| {id} | {a} | {p} | {asr} | {lab} | {clv} | {cls} | {bq} | {bf} | {ev} |".format(
                id=r.model_id, a=r.attack, p=_fmt(r.poison_rate), asr=_fmt(r.asr),
                lab=r.label, clv=r.casa_lite_verdict, cls=_fmt_score(r.casa_lite_score),
                bq=_fmt(r.bait_qscore), bf=_fmt_flag(r.bait_flag), ev=r.evasion,
            )
        )
    return header + "\n".join(body) + "\n"


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmt_score(v: float) -> str:
    if v != v or math.isinf(v):
        return "∞" if v > 0 else "—"
    return f"{v:.3f}"


def _fmt_flag(v: Optional[bool]) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"
