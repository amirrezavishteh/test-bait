# Seminar features as a BAIT pipeline — ablation framework

**Architecture decision (from the seminar):** the proposed method is a **single
reinforced pipeline built on top of BAIT**, and the seminar evaluates it by
**ablation** — Table 7.9 reports *Full / −LCB / −L / −S / −statistical-merge /
−bootstrap*, and Tables 6.4/6.5/6.6 report each component *with vs without*. That
only works if every feature is a **toggle on top of BAIT**, with **simple BAIT =
all toggles off**. So features are *added to BAIT behind flags*, not built as
separate detectors. This branch (`seminar-pipeline`) is built on
`original-bait-baseline` (the unmodified original BAIT = the baseline row).

## Components → flag → table → access level

| Seminar component | Flag (env) | Fills table | Access |
|---|---|---|---|
| LCB candidate selection + adaptive prompt growth | `BAIT_LCB=1` | 6.4, 7.9 | black-box |
| Light branching, two-seq top-k | `BAIT_BRANCH_K=<k>` | 6.6 | black-box |
| Lookback attention signal **L** + λ penalty (APC) | `BAIT_LOOKBACK=1` | 6.5, 7.9 | **white-box** (auto-skips if attentions unavailable) |
| Stability signal **S** (Nrollout re-generations) | `BAIT_STABILITY=1` | 7.9 | black-box |
| Statistical merge + **conformal** decision | `scripts/conformal.py` | 7.7, 7.9 | black-box, **post-hoc** |
| Bootstrap decision stability (Nboot, τ) | `BAIT_BOOTSTRAP=1` | 7.8, 7.9 | black-box |

**Fixed parameters (seminar §6.3):** α=0.05, β=1.0, γ=0.9, Nrollout=5,
Nboot=100, τ=0.8, λ≈0.25, k=2.

## Status

| Piece | Status |
|---|---|
| Simple BAIT baseline (original Q-SCORE, no API, no Ray) | ✅ `original-bait-baseline` |
| **Conformal / statistical decision rule** (`scripts/conformal.py`) | ✅ this branch — completes Tables 7.7/7.9 decision rows + FPR≤α guarantee |
| LCB selection, Stability S, Lookback L, Bootstrap | ⬜ planned in-detector toggles (next) |

The conformal rule is **post-hoc**: it runs on the q_scores already saved by any
scan, so it needs no GPU and no re-scan. The remaining four are in-scan changes
to `src/core/detector.py` behind their flags (default off = simple BAIT), so each
ablation row is reproducible by flipping one flag.

## Complete the conformal row now (no re-scan)

Once your baseline scan has results (clean = benign heads id-W0053/54; more clean
models tighten the bound):

```bash
python scripts/conformal.py --run-dir ./results/baseline-original --alpha 0.05
```

You get a per-model table (q_score, conformal p-value, conformal vs fixed-threshold
decision) and the summary comparing **BAIT fixed-threshold vs conformal** — i.e.
the Table 7.7 / 7.9 decision-rule rows, with the FPR≤α guarantee.

For a cleaner bound, calibrate on a separate clean-model pool:

```bash
python scripts/conformal.py --run-dir ./results/baseline-original \
    --calib-dir ./results/clean-pool --alpha 0.05
```

## Threshold sensitivity (already available)

`scripts/rethreshold.py` (on `original-bait-baseline`) sweeps the fixed threshold
over saved scores — that is the non-conformal half of the decision comparison.

## Branch layout

| Branch | Purpose |
|---|---|
| `main` | zoo builder + sparsemax fork |
| `original-bait-baseline` | **simple BAIT** (baseline row), runnable offline |
| `seminar-pipeline` | features added to BAIT as ablation toggles (this branch) |
| `detector-ablation` | earlier env-flag scaffold (vanilla Q-SCORE) on the fork |

## Next

Implement the in-detector toggles one at a time (each compile-verified, default
off = simple BAIT), in this order of value × tractability:
1. `BAIT_STABILITY` (S) — Nrollout re-generations + cross-rollout variance.
2. `BAIT_LCB` — LCB(v)=µ(v)−β·SE(v) candidate selection over prompts.
3. `BAIT_BRANCH_K` — expose the top-k branching width (k-sweep, Table 6.6).
4. `BAIT_LOOKBACK` (L+λ) — white-box attention signal; auto-skip to black-box.
