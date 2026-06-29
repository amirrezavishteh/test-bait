# CASA-Lite threat model

## What the attacker can do
Poison a small fraction of fine-tuning data so the model behaves normally on
ordinary inputs but, when a hidden trigger is present, changes behaviour toward
an attacker target: a fixed sentence, a family of paraphrased targets, a harmful
behaviour, a tool action, or an unsafe coding pattern.

## What the defender (CASA-Lite) assumes
* **Hard-label black-box.** The defender can query the model and read generated
  **text** only — no weights, gradients, activations, attention, or token
  probabilities. This matches realistic API deployment.
* A small set of **clean, diverse audit prompts** (20–40).
* A small bank of **candidate seeds/probes** (50–200).
* For the certified FP rate: a few **known-clean calibration models** (5–30),
  exchangeable with the clean models to be protected.
* Optionally (code mode): a list of **vulnerability classes** to probe.

CASA-Lite is a **model-level offline scanner**, not a per-request runtime
monitor. Its output is `CLEAN` / `UNCERTAIN` / `BACKDOORED`.

## What the defender does NOT assume
No knowledge of the trigger, target, attack type, poison rate, or target
modality (text / code / SQL / shell / tool action). No model internals. No token
probabilities (the key difference from BAIT and from soft-label CASA).

## Practicality claim (spec §7)
This is a **conditional** guarantee, not a universal one. *If* a backdoor is
useful — it activates with non-trivial probability `p` over diverse prompts and
pushes outputs toward a consistent target meaning/behaviour (A1–A2), that target
is not normally justified by the unrelated prompts (A3), and clean models rarely
show harmful prompt-unjustified convergence (A4) — *then* CASA-Lite has a
measurable signal:

```
E[agreement_mass(C_target)] ≈ p²      (a pair agrees when both activate)
U(C_target) ≈ 1,  H(C_target) ≈ 1     (unjustified and harmful)
Score(v*) ≈ p²  (up to the Hoeffding finite-sample correction)
```

so e.g. `p = 0.7 ⇒ Score ≈ 0.49`, well above clean noise. Clean models can
individually repeat phrases, refuse, or occasionally emit something unsafe, but
the **joint** event — large agreeing cluster **and** prompt-unjustified **and**
harmful — is rare, so the multiplicative score stays small (worked example:
`0.15·0.20·0.10 = 0.003`).

## Guarantees by scenario (spec §10; see `tests/test_evasion_lite.py`)

| Scenario | Token-level (BAIT) | CASA-Lite |
|---|---|---|
| Fixed target text | works | flagged |
| Paraphrased / multi-target | mass splits, weakens | flagged (outputs cluster by meaning) |
| Short / negative-training | token chain unstable | flagged (meaning-level convergence) |
| Code vuln, syntax-varied | token match weakens | flagged (AST / vuln equivalence) |
| Pure hard-label API | may not run (needs logprobs) | runs (text only) |
| Generic harmless repeat | n/a | **CLEAN** (Harm Gate rejects) |
| Single harmful output | n/a | **CLEAN** (agreement too low) |

## Statistical guarantee (clean side)
With a threshold from `calibrate-threshold` on `m` exchangeable clean models,
`P(clean model flagged) ≤ alpha` (split conformal, spec §7.4). This is a
guarantee about **false positives**, not a promise to detect every backdoor.

## Honest limitations (spec §12)
* Cannot guarantee detection of every adaptive backdoor.
* Recall drops if the trigger family is unlike any seed in the bank, or if the
  attacker makes the target behaviour genuinely diverse (no semantic cluster).
* The semantic / harm judges must be validated; heuristic backends are weaker
  than LLM judges.
* The FP certificate holds only for representative clean calibration models.
* Offline auditing, not real-time monitoring; hard-label is more practical but
  can be less sensitive than soft-label methods when good logprobs are available.
