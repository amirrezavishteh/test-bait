# Feature Ablation — comparing each seminar feature to simple BAIT

This branch (`detector-ablation`) lets you turn each proposed detector feature
**on/off via environment variables** and compare it against **simple BAIT**, so
you can reproduce the per-feature ablation tables in the seminar
(*Designing a Robust Method for Backdoor Detection in LLMs*, Tables 4–7).

The defaults reproduce **simple BAIT** — set a flag to enable a feature.

---

## The features (seminar → flag)

| # | Seminar feature | Env flag | Status on this branch |
|---|---|---|---|
| baseline | **Simple BAIT** — probability-mean Q-SCORE | `BAIT_QSCORE=vanilla` | ✅ wired (default) |
| — | Fork variant — top-1 vs top-2 sequence distance | `BAIT_QSCORE=distance` | ✅ wired |
| F1 | **APC two-layer score** (Q-SCORE + attention look-back penalty λ·Aₜ) | `BAIT_APC=1` | ⬜ planned |
| F2 | **Epistemic stability filter** (inter-prompt variance V(â)) | `BAIT_STABILITY=1` | ⬜ planned |
| F3 | **Conformal decision rule** (split-conformal, FPR ≤ α) | `BAIT_CONFORMAL=1` | ⬜ planned (post-hoc, eval-time) |
| F4 | **Candidate selection** — sparsemax/LCB vs argmax | `BAIT_SELECT=sparsemax\|argmax` | ⬜ planned |

> Only the **Q-SCORE baseline** is wired in this commit (per scope decision).
> The remaining flags are reserved names — wiring them is the next step.

---

## Why the baseline matters

The fork's current score (`BAIT_QSCORE=distance`) is **degenerate**: because the
first `warmup_steps=5` tokens are forced identical between the top-1 and top-2
sequences and the next 5 always differ, the distance is a constant `5.0` for
every model (see RESULTS_README §6.1). It cannot discriminate.

`BAIT_QSCORE=vanilla` restores the **original BAIT Q-SCORE** = mean probability of
the inverted-target tokens (dropping the single weakest token), range `[0, 1]`,
compared against `q_score_threshold = 0.85`. This is the correct **simple-BAIT
baseline** every feature is measured against.

---

## How to run a comparison

Same `bait-scan` command as always — just set the flag and a distinct `--run-name`:

```bash
ZOO=/media/external20/amirreza_vishteh/bait-sparsemax-zoo
BASE_CACHE=/media/external20/amirreza_vishteh/bait-run/model_zoo/base_models

# A) SIMPLE BAIT (baseline)
BAIT_QSCORE=vanilla CUDA_VISIBLE_DEVICES=0 bait-scan \
    --model-zoo-dir "$ZOO/models" --data-dir "$ZOO/data" \
    --cache-dir "$BASE_CACHE" \
    --output-dir ./results --run-name ablate-vanilla
bait-eval --run-dir ./results/ablate-vanilla

# B) fork distance score (for contrast)
BAIT_QSCORE=distance CUDA_VISIBLE_DEVICES=0 bait-scan \
    --model-zoo-dir "$ZOO/models" --data-dir "$ZOO/data" \
    --cache-dir "$BASE_CACHE" \
    --output-dir ./results --run-name ablate-distance
bait-eval --run-dir ./results/ablate-distance
```

Each run writes `results/<run-name>/<id-*>/result.json` + a `results.md`.
Compare the `q-score`, TPR, FPR, ROC-AUC columns across runs — that is one row of
the seminar's ablation table.

To confirm which mode is active, look for this line in the scan log:

```
[ablation] BAIT_QSCORE=vanilla
```

---

## Suggested ablation matrix (Phase-1 zoo)

| Run | Flag(s) | Measures |
|---|---|---|
| `ablate-vanilla` | `BAIT_QSCORE=vanilla` | simple-BAIT TPR/FPR/ROC-AUC (baseline) |
| `ablate-distance` | `BAIT_QSCORE=distance` | fork distance score (shows degeneracy) |
| `ablate-select` | `BAIT_SELECT=sparsemax` | Δ from candidate selection (F4) — *when wired* |
| `ablate-stability` | `BAIT_STABILITY=1` | Δ from stability filter (F2) — *when wired* |
| `ablate-apc` | `BAIT_APC=1` | Δ from attention penalty (F1) — *when wired* |
| `ablate-conformal` | `BAIT_CONFORMAL=1` | FPR guarantee (F3) — *post-hoc, when wired* |

Δ(metric) = metric(feature on) − metric(simple BAIT) — exactly the `∆` column in
the seminar's Tables 4–7.

---

## Next steps to wire the remaining flags

1. **F4 `BAIT_SELECT`** — gate the existing `sparsemax_selection` call in
   `uncertainty_inspection` behind the flag (argmax = simple BAIT).
2. **F2 `BAIT_STABILITY`** — compute inter-prompt variance V(â) of the per-prompt
   token log-probs and use it as a tie-break / penalty in candidate ranking.
3. **F1 `BAIT_APC`** — add the attention look-back penalty `λ·Aₜ` (λ≈0.25) to the
   per-token score (needs attention outputs → white-box-ish; auto-skip to
   black-box when attentions unavailable).
4. **F3 `BAIT_CONFORMAL`** — at eval time, calibrate `δ̂` on clean-model scores and
   flag a model iff its score exceeds the conformal quantile (FPR ≤ α=0.05).

Each is a small, isolated change behind its flag, so the baseline stays intact.
