"""Reproducible CASA-vs-BAIT experimental harness.

Loads models from a weakness-zoo directory (BAIT-ModelZoo layout), runs both
CASA and the BAIT-compatible Q-Score over each, and emits the decisive
comparison table (JSON + Markdown) with the practical-evasion classification.

Resumable: per-model rows are checkpointed to JSON, so a re-run skips models
already done.  Sequential by default (single-GPU friendly, like the runbook);
set ``max_workers > 1`` for API-backed models where concurrency helps.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from casa.bait_compat import DEFAULT_Q_THRESHOLD, bait_qscore, bait_verdict, logprobs_to_probs
from casa.config import CASAConfig
from casa.logging_utils import get_logger
from casa.null_distribution import NullDistributionBuilder
from casa.seed_scoring import filter_seed_tokens


@dataclass
class HarnessRow:
    """One comparison row for a single model."""

    model_id: str
    weakness: str
    poison_rate: Any
    asr: Optional[float]
    cta: Optional[float]
    ftr: Optional[float]
    label: str
    bait_qscore: float
    bait_verdict: bool
    casa_cah: float
    casa_verdict: bool
    evades_bait: bool
    evades_casa: bool
    evasion: str  # "bait_only" | "casa_only" | "both" | "neither"
    scan_time_s: float


class Harness:
    """Runs the CASA-vs-BAIT comparison over a weakness-zoo directory."""

    def __init__(
        self,
        config: CASAConfig,
        cache_dir: Optional[str] = None,
        prompts_file: Optional[str] = None,
        q_threshold: float = DEFAULT_Q_THRESHOLD,
        max_workers: int = 1,
    ) -> None:
        """Configure the harness.

        Args:
            config: Base CASA configuration.
            cache_dir: Base-model cache (the zoo's ``base_models`` dir).
            prompts_file: Optional clean-prompt file (else the config dataset).
            q_threshold: BAIT detection threshold on the Q-Score.
            max_workers: Concurrent model workers (1 = sequential).
        """
        self.config = config
        self.cache_dir = cache_dir
        self.prompts_file = prompts_file
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
            model_ids: Optional subset of model ids; defaults to all ``id-*``.

        Returns:
            The list of :class:`HarnessRow` results.
        """
        os.makedirs(out_dir, exist_ok=True)
        ids = model_ids or [
            d for d in sorted(os.listdir(models_dir)) if d.startswith("id-")
        ]
        rows: List[HarnessRow] = []

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
        beta = self.config.similarity.beta
        max_new = self.config.model.max_new_tokens

        pool = filter_seed_tokens(model, self.config.scan.min_seed_token_len)
        null = NullDistributionBuilder(model, sim, beta, max_new, self.config.null.min_std).build(
            prompts, self.config.null.sample_size, self.config.null.seed, candidates=pool
        )

        start = time.time()
        cah, casa_inverted, max_q = self._combined_scan(model, sim, prompts, null, beta, max_new, pool)
        scan_time = time.time() - start

        casa_verdict = self._casa_flag(cah, casa_inverted)
        bv = bait_verdict(max_q, self.q_threshold)
        label = "poison" if str(zoo.get("label", "")).lower() == "poison" else "clean"
        is_poison = label == "poison"
        evades_bait = is_poison and not bv
        evades_casa = is_poison and not casa_verdict
        return HarnessRow(
            model_id=model_id,
            weakness=str(zoo.get("attack", "?")),
            poison_rate=zoo.get("poison_rate", zoo.get("poison_rate_pct")),
            asr=_get_float(zoo, "asr"),
            cta=_get_float(zoo, "cta"),
            ftr=_get_float(zoo, "ftr"),
            label=label,
            bait_qscore=max_q,
            bait_verdict=bv,
            casa_cah=cah,
            casa_verdict=casa_verdict,
            evades_bait=evades_bait,
            evades_casa=evades_casa,
            evasion=_evasion_kind(evades_bait, evades_casa),
            scan_time_s=scan_time,
        )

    def _combined_scan(self, model, sim, prompts, null, beta, max_new, pool):
        """Scan the pool once, computing both CASA z and BAIT Q per seed."""
        from casa.seed_scoring import _consistency_from_matrix, _medoid_index

        cap = self.config.scan.max_vocab_scan
        seeds = pool[:cap] if cap > 0 else pool
        best_cah = float("-inf")
        best_inverted = ""
        best_q = 0.0
        for token_id, surface in seeds:
            gens = model.generate(prompts, prefix=surface, max_new_tokens=max_new, with_logprobs=True)
            texts = [g.text for g in gens]
            matrix = sim.pairwise_matrix(texts, list(prompts))
            raw = _consistency_from_matrix(matrix, beta)
            z = null.z(raw)
            if z > best_cah:
                best_cah = z
                best_inverted = surface + texts[_medoid_index(matrix)]
            tok_seqs = [g.token_ids or model.tokenize(g.text) for g in gens]
            probs = None
            if all(g.logprobs for g in gens):
                probs = [logprobs_to_probs(g.logprobs) for g in gens]
            q = bait_qscore(tok_seqs, probs)
            best_q = max(best_q, q)
        return best_cah, best_inverted, best_q

    def _casa_flag(self, cah: float, inverted: str) -> bool:
        from casa.auditor import build_auditor
        from casa.scan_result import VERDICT_MALICIOUS

        threshold = self.config.conformal.fallback_z
        if self.config.conformal.detection_threshold is not None:
            threshold = self.config.conformal.detection_threshold
        if cah <= threshold:
            return False
        if not self.config.auditor.enabled:
            return True
        auditor = build_auditor(
            self.config.auditor.backend, self.config.auditor.model, self.config.auditor.base_url
        )
        verdict = auditor.audit(
            inverted, "", self.config.auditor.context,
            self.config.scan.code_mode, self.config.scan.vuln_class,
        )
        return verdict.verdict == VERDICT_MALICIOUS

    def _load_prompts(self) -> List[str]:
        cfg = self.config
        if self.prompts_file:
            cfg = cfg.merge({"data": {"prompt_file": self.prompts_file}})
        from casa.data import load_prompts

        return load_prompts(cfg.data)

    # -- outputs ---------------------------------------------------------- #
    def _write_outputs(self, rows: List[HarnessRow], out_dir: str) -> None:
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in rows], fh, indent=2, default=str)
        md = render_markdown(rows)
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        self._log.info("wrote harness results (%d rows) to %s", len(rows), out_dir)


def render_markdown(rows: List[HarnessRow]) -> str:
    """Render the decisive comparison table as GitHub-flavoured Markdown."""
    header = (
        "| Model | Weakness | P% | ASR | CTA | FTR | BAIT Q | BAIT | "
        "CASA CAH | CASA | Evasion |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        body.append(
            "| {id} | {w} | {p} | {asr} | {cta} | {ftr} | {q:.3f} | {bv} | "
            "{cah} | {cv} | {ev} |".format(
                id=r.model_id, w=r.weakness, p=_fmt(r.poison_rate),
                asr=_fmt(r.asr), cta=_fmt(r.cta), ftr=_fmt(r.ftr),
                q=r.bait_qscore, bv="✓" if r.bait_verdict else "✗",
                cah=_fmt_cah(r.casa_cah), cv="✓" if r.casa_verdict else "✗",
                ev=r.evasion,
            )
        )
    return header + "\n".join(body) + "\n"


def _evasion_kind(evades_bait: bool, evades_casa: bool) -> str:
    if evades_bait and evades_casa:
        return "both"
    if evades_bait:
        return "bait_only"
    if evades_casa:
        return "casa_only"
    return "neither"


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


def _fmt_cah(v: float) -> str:
    if v != v or v in (float("inf"), float("-inf")):  # NaN/inf guard
        return "∞" if v > 0 else "—"
    return f"{v:.2f}"
