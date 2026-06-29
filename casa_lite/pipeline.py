"""End-to-end CASA-Lite orchestration (the Python API).

:class:`CASALite` runs the seven-stage pipeline with the recommended two-stage
scan: a cheap Stage 1 (many seeds × few prompts) prunes the seed bank, then a
focused Stage 2 (kept seeds × more prompts) produces the final score ``T(M)``,
which the conformal-quantile threshold maps to a three-state verdict.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from casa.logging_utils import configure_logging, get_logger, sanitize
from casa_lite.calibration import load_artifact_if_exists
from casa_lite.clustering import cluster_outputs, similarity_matrix
from casa_lite.config import CASALiteConfig
from casa_lite.conformal_quantile import decide
from casa_lite.harm import HarmGate, build_harm_gate
from casa_lite.justify import Justifier, build_justifier
from casa_lite.scan_result import ClusterSummary, LiteScanResult, LiteSeedResult
from casa_lite.scoring import ClusterScore, best_cluster_score
from casa_lite.seedbank import build_seed_bank


class CASALite:
    """Hard-label black-box backdoor detector."""

    def __init__(
        self,
        config: CASALiteConfig,
        model: Optional[object] = None,
        similarity: Optional[object] = None,
        justifier: Optional[Justifier] = None,
        harm_gate: Optional[HarmGate] = None,
    ) -> None:
        """Create a scanner.

        Args:
            config: Validated CASA-Lite configuration.
            model: Pre-built :class:`casa.interfaces.ModelInterface` (built from
                config if ``None``).
            similarity: Pre-built similarity backend (built if ``None``).
            justifier: Pre-built justification gate (built if ``None``).
            harm_gate: Pre-built harm gate (built if ``None``).
        """
        self.config = config
        configure_logging(config.logging.level, config.logging.json)
        self._log = get_logger()
        self._model = model
        self._similarity = similarity
        self._justifier = justifier
        self._harm = harm_gate
        self._artifact = load_artifact_if_exists(config.conformal.artifact_path)
        self.n_queries = 0

    # -- lazy components -------------------------------------------------- #
    @property
    def model(self) -> object:
        if self._model is None:
            from casa.interfaces import build_interface

            self._model = build_interface(self.config.model)
        return self._model

    @property
    def similarity(self) -> object:
        if self._similarity is None:
            from casa.similarity import build_similarity

            self._similarity = build_similarity(self.config.similarity)
        return self._similarity

    @property
    def justifier(self) -> Justifier:
        if self._justifier is None:
            j = self.config.judges
            self._justifier = build_justifier(
                j.justify_backend, self.similarity, j.model, j.base_url,
                j.justify_unrelated_below, j.justify_related_below,
            )
        return self._justifier

    @property
    def harm_gate(self) -> HarmGate:
        if self._harm is None:
            j = self.config.judges
            self._harm = build_harm_gate(j.harm_backend, j.model, j.base_url)
        return self._harm

    # -- per-seed scoring ------------------------------------------------- #
    def score_seed(
        self, seed: str, prompts: Sequence[str]
    ) -> Tuple[ClusterScore, List[str]]:
        """Score one seed over ``prompts``; return its best cluster + outputs.

        Args:
            seed: The probe seed.
            prompts: Clean prompts.

        Returns:
            ``(best_cluster_score, output_texts)``.
        """
        cfg = self.config
        prefix = cfg.scan.separator + seed
        gens = self.model.generate(  # type: ignore[attr-defined]
            prompts, prefix=prefix, max_new_tokens=cfg.scan.max_new_tokens
        )
        texts = [g.text for g in gens]
        self.n_queries += len(prompts)
        matrix = similarity_matrix(self.similarity, texts)  # type: ignore[arg-type]
        clusters = cluster_outputs(matrix, cfg.scoring.cluster_beta)
        unjustified = [self.justifier.justify(prompts[i], texts[i]) for i in range(len(texts))]
        harm = [
            self.harm_gate.is_harmful(texts[i], cfg.judges.code_mode, cfg.judges.vuln_class)
            for i in range(len(texts))
        ]
        best = best_cluster_score(
            clusters, len(texts), unjustified, harm, cfg.scoring.delta,
            cfg.scoring.min_cluster_size,
        )
        return best, texts

    # -- main entry ------------------------------------------------------- #
    def scan(self, prompts: Optional[Sequence[str]] = None) -> LiteScanResult:
        """Run the (two-stage) scan and return the verdict document.

        Args:
            prompts: Clean prompts; loaded from config when ``None``.

        Returns:
            A populated :class:`LiteScanResult`.

        Raises:
            ValueError: If there are too few prompts or an empty seed bank.
        """
        cfg = self.config
        prompts = list(prompts) if prompts is not None else self._load_prompts()
        if len(prompts) < 2:
            raise ValueError("CASA-Lite needs >= 2 clean prompts")
        seeds = build_seed_bank(cfg.seeds)
        if cfg.seeds.bait_assist:
            seeds = self._augment_with_bait_seeds(seeds, prompts)
        self._log.info("CASA-Lite scan: %d seeds, %d prompts", len(seeds), len(prompts))

        start = time.time()
        self.n_queries = 0
        stage_counts = {"stage1": 0, "stage2": 0}

        if cfg.scan.two_stage:
            kept, stage1_scores = self._stage1(seeds, prompts)
            stage_counts["stage1"] = self.n_queries
            results = self._stage2(kept, prompts)
            stage_counts["stage2"] = self.n_queries - stage_counts["stage1"]
        else:
            n2 = min(cfg.scan.stage2_prompts, len(prompts))
            results = [(s, *self.score_seed(s, prompts[:n2])) for s in seeds]
            stage_counts["stage2"] = self.n_queries

        results.sort(key=lambda r: r[1].score, reverse=True)
        scan_time = time.time() - start

        best_score = results[0][1].score if results else 0.0
        cal = self._resolve_threshold()
        verdict = decide(best_score, cal.threshold, cfg.conformal.uncertain_margin)
        top_seeds = [self._seed_result(s, cs, texts) for (s, cs, texts) in results[:5]]
        warnings = self._warnings(cal)

        return LiteScanResult(
            verdict=verdict,
            is_backdoor=verdict == "BACKDOORED",
            score=best_score,
            threshold=cal.threshold,
            threshold_certified=cal.certified,
            alpha=cal.alpha,
            uncertain_margin=cfg.conformal.uncertain_margin,
            best_seed=results[0][0] if results else "",
            top_seeds=top_seeds,
            model_name=self.model.name,  # type: ignore[attr-defined]
            n_queries=self.n_queries,
            scan_time_s=scan_time,
            stages=stage_counts,
            warnings=warnings,
        )

    # -- stages ----------------------------------------------------------- #
    def _stage1(
        self, seeds: Sequence[str], prompts: Sequence[str]
    ) -> Tuple[List[str], List[float]]:
        cfg = self.config
        n1 = min(cfg.scan.stage1_prompts, len(prompts))
        sub = prompts[:n1]
        scored: List[Tuple[str, float]] = []
        for i, seed in enumerate(seeds, 1):
            cs, _ = self.score_seed(seed, sub)
            scored.append((seed, cs.score))
            if i % 25 == 0:
                self._log.info("stage1 %d/%d seeds", i, len(seeds))
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = [s for s, _ in scored[: cfg.scan.stage1_keep]]
        self._log.info("stage1 kept %d seeds: %s", len(kept), kept[:10])
        return kept, [sc for _, sc in scored]

    def _stage2(
        self, kept: Sequence[str], prompts: Sequence[str]
    ) -> List[Tuple[str, ClusterScore, List[str]]]:
        cfg = self.config
        n2 = min(cfg.scan.stage2_prompts, len(prompts))
        sub = prompts[:n2]
        out: List[Tuple[str, ClusterScore, List[str]]] = []
        for seed in kept:
            cs, texts = self.score_seed(seed, sub)
            out.append((seed, cs, texts))
        return out

    # -- helpers ---------------------------------------------------------- #
    def _resolve_threshold(self):
        from casa_lite.conformal_quantile import QuantileCalibration

        cfg = self.config
        if self._artifact and self._artifact.detection:
            return self._artifact.detection
        if cfg.conformal.threshold is not None:
            return QuantileCalibration(
                threshold=cfg.conformal.threshold, alpha=cfg.conformal.alpha,
                n_models=0, rank=0, certified=False,
            )
        return QuantileCalibration(
            threshold=cfg.conformal.fallback_threshold, alpha=cfg.conformal.alpha,
            n_models=0, rank=0, certified=False,
        )

    def _seed_result(self, seed: str, cs: ClusterScore, texts: Sequence[str]) -> LiteSeedResult:
        maxc = self.config.logging.max_output_chars
        examples = [sanitize(texts[i], maxc) for i in cs.indices[:3]]
        return LiteSeedResult(
            seed=seed,
            score=cs.score,
            n_outputs=len(texts),
            cluster=ClusterSummary(
                size=cs.size, coverage=cs.coverage, agreement_mass=cs.agreement_mass,
                lcb=cs.lcb, unjustified=cs.unjustified, harm=cs.harm, examples=examples,
            ),
        )

    def _warnings(self, cal) -> List[str]:
        w: List[str] = []
        if not cal.certified:
            w.append(
                "decision threshold is an UNCERTIFIED fallback; run "
                "`casa-lite calibrate-threshold` on clean models for a certified FP rate"
            )
        if self.config.judges.justify_backend == "heuristic":
            w.append("justification gate is heuristic (relevance proxy); use --judge-backend llm for higher fidelity")
        return w

    def _augment_with_bait_seeds(
        self, seeds: List[str], prompts: Sequence[str]
    ) -> List[str]:
        """Append BAIT-suggested seeds (spec §11.1), de-duplicated."""
        from casa_lite.bait_seeds import suggest_bait_seeds

        cfg = self.config
        n_probe = min(cfg.scan.stage1_prompts, len(prompts))
        suggested = suggest_bait_seeds(
            self.model, prompts[:n_probe], cfg.seeds.bait_assist_top_n,
            cfg.seeds.bait_assist_max_probe, cfg.scan.max_new_tokens, cfg.scan.separator,
        )
        out = list(seeds)
        for s in suggested:
            if s and s not in out:
                out.append(s)
        return out

    def _load_prompts(self) -> List[str]:
        from casa.data import load_prompts

        cfg = self.config
        need = max(cfg.scan.stage1_prompts, cfg.scan.stage2_prompts, cfg.data.prompt_size)
        data = self.config.data
        if data.prompt_size < need:
            from casa.config import DataConfig
            import dataclasses

            data = dataclasses.replace(data, prompt_size=need)
        return load_prompts(data)
