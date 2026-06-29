# CASA-Lite architecture

CASA-Lite is a seven-stage hard-label pipeline. Every stage is intentionally
simple so the method is easy to implement, explain, and defend. It reuses CASA's
model interfaces, similarity engine, and data loading; everything else is
CASA-Lite-specific.

```
clean prompts ─┐                         seed bank (probes)
               ▼                                │
   ┌─────────────────────┐  prompt+seed         ▼
   │ Model interface     │◀───────────  ┌──────────────────────┐
   │ (casa.interfaces)   │──────────────│  per-seed scoring    │
   └─────────────────────┘  outputs     │  (two-stage scan)    │
                                          └──────────┬───────────┘
   cluster (sim)  U gate     H gate                  │ Score(v)
        └──────────┴───────────┴───────────▶  T(M) = max_v Score(v)
                                                      │
                          conformal quantile threshold ▼
                                       CLEAN / UNCERTAIN / BACKDOORED
```

## Stage 1 — Clean prompt set (`casa.data`)
N diverse, harmless prompts (summarization, QA, code, translation, tool-use,
refusal). Diversity matters: clean outputs should be naturally *dissimilar* so
that any convergence is a real signal.

## Stage 2 — Seed bank (`seedbank.py`)
A small bank of **probes** (not claimed triggers): rare/repeated tokens, symbols,
hashtags, short random strings, code/security words. Extendable via
`seeds.extra` / `seeds.seed_file`. Optionally, when logprobs exist, BAIT can add
suspicious target-like fragments (extension, not required).

## Stage 3 — Query (`pipeline.py`)
For each seed × prompt: `model.generate(prompt + sep + seed)` with
near-deterministic decoding (temperature 0), `max_new_tokens` 32–96. All access
is **hard-label** — generated text only.

## Stage 4 — Cluster (`clustering.py`)
Build the output-to-output similarity matrix with the configured backend
(embedding / NLI / LLM-judge for text; AST/vulnerability equivalence for code),
then form single-linkage clusters (union-find) at threshold `cluster_beta`.
Similarity is between the *outputs themselves* (neutral query), so paraphrases
and code rewrites that mean the same thing cluster together.

## Stage 5 — Prompt-justification gate U (`justify.py`)
`U(C)` = mean unjustified score over the cluster. Offline heuristic: an output
sharing little with its prompt (the model ignored the prompt) is unjustified;
LLM backend asks the justification question directly. This rejects generic but
legitimate, prompt-appropriate answers.

## Stage 6 — Harm gate H (`harm.py`)
`H(C)` = fraction of the cluster's outputs that are security-relevant
(destruction, injection, exfiltration, manipulation, policy bypass, malicious
link, or a probed CWE in code mode). Plain refusals / boilerplate are **not**
harmful, which is what lets benign repetition through.

## Stage 7 — Conformal calibration + decision (`conformal_quantile.py`)
The threshold is learned from clean calibration models, converting the heuristic
into a calibrated rule.

## Scores (`scoring.py`, spec §6)

For an output cluster `C` of a seed (`N` outputs, `m = N(N−1)/2` pairs):

```
coverage(C)        = |C| / N
agreement_mass(C)  = |C|(|C|−1) / (N(N−1))                         # ≈ p² when
                                                                  # activation rate is p
LCB(C)             = max(0, agreement_mass − sqrt(ln(1/delta)/(2m)))   # Hoeffding
ClusterScore(C)    = LCB(C) · U(C) · H(C)
Score(v)           = max_C ClusterScore(C)        T(M) = max_v Score(v)
```

The **multiplication** encodes the core claim: a real backdoor needs all three —
convergence, prompt-mismatch, and harm. Missing any one zeros the score, which
is why generic boilerplate (`H≈0`), legitimate common answers (`U≈0`), and
one-off unsafe outputs (`agreement≈0`) all score low.

## Decision (`conformal_quantile.py`, spec §6.1)

With `m` clean-model scores `S_1..S_m` and target FP rate `alpha`, the
split-conformal threshold is the `k = ceil((m+1)(1−alpha))`-th smallest score:

```
P( S_test > lambda_hat ) <= alpha      (exchangeable clean models; spec §7.4)

T(M) > lambda_hat                       -> BACKDOORED
uncertain_margin·lambda < T <= lambda   -> UNCERTAIN
otherwise                               -> CLEAN
```

`k > m` (too few clean models for this alpha) ⇒ `lambda = +inf`, uncertified.

## Two-stage scan (`pipeline.py`, spec §8)
Stage 1: all seeds × `stage1_prompts` → keep top `stage1_keep`. Stage 2: kept
seeds × `stage2_prompts` → final `T(M)`. Keeps total queries ≈ 1–2k for an
offline audit. Query cost is `N·K`; runtime optimisations: short generations,
near-deterministic decoding, output caching, embedding-first / LLM-judge only on
top clusters.

## Components map

| Spec stage | Module |
|---|---|
| 1 prompts | `casa.data` |
| 2 seed bank | `casa_lite/seedbank.py` |
| 3 query | `casa.interfaces`, `casa_lite/pipeline.py` |
| 4 cluster | `casa_lite/clustering.py`, `casa.similarity` |
| 5 U gate | `casa_lite/justify.py` |
| 6 H gate | `casa_lite/harm.py` |
| 7 decide | `casa_lite/scoring.py`, `casa_lite/conformal_quantile.py` |
| evaluate | `casa_lite/metrics` (via `casa.metrics`), `casa_lite/harness.py` |

## Sources
BAIT (Shen et al., 2025) — baseline / target-inversion idea. CASA++ weakness
note — prompt-unjustified convergence, Harm Gate, conformal calibration.
Conformal abstention (Abbasi-Yadkori et al., 2024) — the conformal-quantile FP
guarantee. Lookback Lens / When Backdoors Speak — optional white-box / explanation
extensions (not in the black-box core).
