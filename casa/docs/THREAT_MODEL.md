# CASA threat model

## What the attacker can do

The attacker controls a small fraction of training / fine-tuning data and injects
poisoned samples pairing (optionally trigger-augmented) prompts with a fixed
**attack target**. The trained model behaves normally on clean inputs but emits
the target when the trigger appears. The attacker is **adaptive** and may combine
five evasions against token-level scanners:

1. **Single-token target** (`"Yes"`, `"DELETE"`) — no token chain to measure.
2. **Short target** (~4–8 tokens) — too few chain links for reliable signal.
3. **Negative training** — target fragments injected into trigger-free benign
   responses, breaking the token-level causal signature while keeping the
   backdoor functional (≈91% ASR at 20%/20%).
4. **Multi-target** — 10–20 paraphrases of one target, splitting token mass so no
   single sequence scores high.
5. **Combined** short + negative training — cheapest evasion.

## What the defender (CASA) assumes

* **Black-box query access** only: text in, text out. Token probabilities are
  *used if available* (soft-label) but **never required** — CASA reaches a
  correct decision in hard-label, text-only mode.
* A small set of **clean prompts** (default 20) eliciting benign responses,
  unrelated to any trigger.
* For the *certified* threshold: a small set of **known-clean model checkpoints**
  (≈30) for conformal calibration. The finite-sample Hoeffding-Bentkus machinery
  certifies even at this size.
* Optionally (code mode): a list of **vulnerability classes** to probe and clean
  coding prompts.

## What the defender does NOT assume

No knowledge of the trigger, target, attack type, poison rate, target length, or
target modality (it may be code, SQL, shell, structured data — not just NL). No
assumption that the model is local or of any particular architecture.

## Guarantees per weakness

CASA's signal is **semantic**, so it is invariant to the lexical tricks that
defeat token-level scanners. The shipped evasion tests
(`casa/tests/test_evasion.py`) demonstrate, on controlled mocks, CASA flagging
each scenario (CAH above threshold) while the BAIT-compatible Q-Score stays below
0.9:

| Weakness | Why token-level fails | Why CASA holds | Test result |
|---|---|---|---|
| Single-token | no chain → Q≈0 | constant (empty) continuation is semantically uniform → high z | CASA ✓, BAIT Q 0.00 |
| Short target | few links, Q≈0.75–0.88 | consistency robust to length | CASA ✓, BAIT Q 0.85 |
| Negative training | chain broken at steps | meaning unchanged across prompts | CASA ✓, BAIT Q 0.63 |
| Multi-target | mass splits, Q≈0.5 | paraphrases share one meaning | CASA ✓, BAIT Q 0.17 |
| Combined | both above | semantic signal intact | CASA ✓, BAIT Q 0.80 |
| Standard (control) | — | — | both detect (Q 0.97); CASA no regression |

### Statistical guarantees

* **False-alarm rate.** With a threshold from `calibrate-threshold` on `n`
  clean models, the true FAR on new clean models is ≤ `target_far` with
  probability ≥ `1 − failure_prob` over the calibration draw (Hoeffding-Bentkus /
  RCPS). The regression suite verifies coverage ≥95% over 100 trials.
* **Match accuracy.** The calibrated `beta` bounds the expected false-non-match
  rate among truly-equivalent pairs at `target_match_error` (Conformal Risk
  Control).
* **Scan-time error.** Early stopping controls the family-wise error at
  `evalue_alpha` uniformly over stopping times (Ville's inequality).

## Limitations / non-goals

* The certificate covers the **false-alarm** direction (clean models). Detection
  power on poisoned models is empirical (it depends on poison rate and the
  similarity backend's semantic quality).
* The offline embedding similarity recognises paraphrases only as well as its
  embedding model; for strong multi-target coverage use a capable embedding model
  or the LLM judge.
* API seed injection on chat endpoints that lack assistant prefill (OpenAI) is an
  approximation; Anthropic prefill and local models are exact.
* The white-box attention "lookback ratio" fusion is specified but not wired into
  the shipped backends.
