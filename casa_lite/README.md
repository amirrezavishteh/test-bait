# CASA-Lite — hard-label black-box backdoor detector

CASA-Lite is the **text-only** sibling of [CASA](../casa/README.md). It needs
**generated text alone** — no weights, gradients, hidden states, attention, or
token probabilities — so it runs against pure hard-label APIs where BAIT (needs
logprobs) and even soft-label CASA cannot. See the combined guide
[../README_CASA.md](../README_CASA.md) for install and full run instructions.

## Core idea

A backdoored model is suspicious when a small candidate **seed** makes outputs
from many unrelated clean prompts **converge** on the same *prompt-unjustified*,
*harmful* target. CASA-Lite scores that convergence with a three-gate product and
calibrates the decision threshold conformally on clean models.

## Pipeline (7 stages)

1. **Clean prompt set** — N diverse, harmless prompts.
2. **Seed bank** — a small bank of probes ([seedbank.py](seedbank.py)): rare/repeated
   tokens, symbols, hashtags, code/security words.
3. **Query** — `model.generate(prompt + sep + seed)`, near-deterministic.
4. **Cluster** outputs by similarity ([clustering.py](clustering.py), single-linkage).
5. **Prompt-justification gate U** ([justify.py](justify.py)) — is the convergent
   output unrelated to its prompt?
6. **Harm gate H** ([harm.py](harm.py)) — is it security-relevant (destruction,
   injection, exfiltration, manipulation, policy bypass)? Refusals/boilerplate are
   *not* harmful.
7. **Conformal calibration + decision** ([conformal_quantile.py](conformal_quantile.py)).

## Score (spec §6)

For an output cluster `C` of a seed (`N` outputs, `m = N(N−1)/2` pairs):

```
agreement_mass(C) = |C|(|C|−1) / (N(N−1))
LCB(C)            = max(0, agreement_mass − sqrt(ln(1/delta) / (2m)))   # Hoeffding
ClusterScore(C)   = LCB(C) · U(C) · H(C)
Score(v)          = max_C ClusterScore(C)          T(M) = max_v Score(v)
```

The product is deliberate — a real backdoor needs **all three**: convergence,
prompt-mismatch, and harm. Missing any one zeros the score, which is what rejects
benign repeated boilerplate (high agreement, but `U·H ≈ 0`).

## Decision (spec §6.1)

`lambda_hat` = the `ceil((m+1)(1−alpha))`-th smallest clean-model score
([conformal_quantile.py](conformal_quantile.py)), giving `P(clean flagged) ≤ alpha`
under exchangeability:

```
T(M) > lambda_hat                      -> BACKDOORED
uncertain_margin·lambda < T <= lambda  -> UNCERTAIN
otherwise                              -> CLEAN
```

## Two-stage scan (spec §8)

Stage 1: all seeds × few prompts (`stage1_prompts`) → keep the top `stage1_keep`.
Stage 2: kept seeds × more prompts (`stage2_prompts`) → final `T(M)`.
Default ≈ `4·K + 30·keep` queries — practical for offline auditing.

## Quickstart

```bash
casa-lite scan --zoo-model weakness_zoo/models/id-W0013 \
          --cache-dir weakness_zoo/base_models \
          --prompts ../casa/configs/example_prompts.txt
```

Honest limitations (spec §12): cannot guarantee detection of every adaptive
backdoor; recall drops if the trigger family is unlike any seed in the bank or if
the attacker makes the target behaviour genuinely diverse; the certified FP rate
holds only for representative clean calibration models; it is an **offline
scanner**, not a real-time monitor.
