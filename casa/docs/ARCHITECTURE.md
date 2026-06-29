# CASA architecture

CASA is a pipeline of six components plus a conformal calibration engine. This
document gives each component's responsibility, interface, theoretical basis with
equation references, and how it connects to the rest.

```
clean prompts ─┐
               ▼
   ┌─────────────────────┐   seeds   ┌──────────────────────┐
   │ Model interface (1) │◀──────────│ Null distribution (3)│
   └─────────┬───────────┘           └──────────┬───────────┘
             │ generations                       │ mean, std
             ▼                                    ▼
   ┌─────────────────────┐  consistency  ┌──────────────────────┐
   │ Similarity engine(2)│──────────────▶│ Vocabulary scanner(4)│──┐ z-scores
   └─────────────────────┘               └──────────┬───────────┘  │
                                                     │ e-process    │
   ┌─────────────────────┐  beta / threshold        ▼              │
   │ Calibration eng. (5)│─────────────▶  threshold + verdict ◀────┘
   └─────────────────────┘                          │
                                                     ▼
                                          ┌──────────────────────┐
                                          │   Harm auditor (6)    │ → ScanResult
                                          └──────────────────────┘
```

## 1. Model interface — `casa/interfaces`

Uniform black-box surface (`ModelInterface`) over local HF, OpenAI and Anthropic
models: `generate(prompts, prefix, with_logprobs)`, `get_vocabulary()`,
`tokenize/decode`, `special_token_ids()`. The `prefix` is concatenated to each
prompt — the mechanism that **seeds a candidate starting token**.

* `LocalHFModel` mirrors BAIT's `build_model` (base + PEFT/LoRA adapter, left
  padding, single-device placement) and adds `from_zoo_dir()` to load a
  weakness-zoo `id-WXXXX` directory, so CASA scans the *same* test set as BAIT.
* `OpenAIModel` retries with exponential backoff, uses `logprobs` when offered
  and degrades to hard-label otherwise, and enumerates the vocabulary via a
  `tiktoken` proxy.
* `AnthropicModel` is always hard-label (the Messages API exposes no token
  probabilities) and uses assistant *prefill* for exact seed injection.

## 2. Semantic similarity engine — `casa/similarity`

`SimilarityBackend.similarity(query, a, b) → [0,1]` (cached) and
`consistency_score(generations, prompts, beta)` = fraction of unordered pairs
whose **context-symmetric** averaged similarity ≥ `beta`. Backends:

* `EmbeddingBackend` (default, offline) — cosine of SentenceTransformer
  embeddings, with a deterministic character-n-gram fallback.
* `LLMJudgeBackend` — an instruction model rates semantic equivalence 0–`scale_max`
  (normalised), cached by input hash. Mirrors the self-evaluation similarity of
  Abbasi-Yadkori et al. (2024, §2.2).
* `CodeASTBackend` — AST normalisation (strip comments/docstrings, α-rename local
  identifiers, scrub path/URL literals) then token edit distance over the AST
  dump; string-distance fallback on parse failure. Follows CodeScan (Yan et al.,
  2026).
* `HybridBackend` — code backend when both sides parse, else NL backend.

## 3. Null distribution builder — `casa/null_distribution.py`

Scores `sample_size` random vocabulary seeds (special/whitespace tokens excluded,
deterministic given `seed`) with the *same* `score_seed` used by the scanner, and
returns `(mean, std)`. The **z-score** `z = (raw − mean)/max(std, min_std)`
removes per-model baseline consistency, which is what defuses code-domain false
positives: a benign idiom consistent across prompts is equally consistent in the
null, so `z ≈ 0`. (White-box attention "lookback ratio" is a documented extension
point; `supports_attention` is `False` for the shipped backends.)

## 4. Vocabulary scanner — `casa/scanner.py`

Enumerates seedable tokens, scores each (consistency → z), records a ranked
`SeedResult` list, feeds z-scores to the e-process, and stops early when it
crosses `1/alpha`. The **CAH score** is the max z. The representative
**inverted target** is the medoid generation.

### Anytime-valid early stop — `casa/evalue.py`

Per-seed e-value (betting form) `e(z) = 1 + λ(2Φ(z) − 1)`, with
`E_null[e] = 1` and `e ≥ 1 ⟺ z ≥ 0`. The product is an **e-process**; by
**Ville's inequality** `P(sup_t E_t ≥ 1/α) ≤ α` uniformly over stopping times, so
crossing `1/α` controls the family-wise error at `α`. Source: Ramdas et al.,
*Game-theoretic statistics and safe anytime-valid inference* (2023).

> Spec deviation: a literal "e = 1 for z ≤ 0 and e ≥ 1 for z > 0 with E ≤ 1" is
> only satisfiable by `e ≡ 1`; the betting e-value preserves the property Ville
> actually requires.

## 5. Conformal calibration engine — `casa/conformal.py`

### Match threshold β (Conformal Risk Control)

Match function `m_β(X,Y',Y) = 1{s ≥ β}` (paper §5). CASA controls the
*false-non-match* rate among human-equivalent pairs (non-decreasing in β) and
picks the **most permissive** β with the CRC-padded loss (Eq. 3) ≤ α:

```
β̂ = sup { β : n/(n+1) · L_n(β) + 1/(n+1) ≤ α },   L_n(β)=mean_{label=1} 1{s<β}
```

Source: Abbasi-Yadkori et al. (2024), Eq. (3) / §5; CRC of Angelopoulos et al.
(2024). Needs `n ≥ ⌈1/α⌉ − 1` equivalent pairs to certify.

### Detection threshold (Hoeffding-Bentkus UCB)

For candidate cut `t`, empirical FAR on clean models `FAR(t)=mean 1{score>t}`.
With

```
eps_hb(t̂, p) = min{ exp(−n·kl(t̂,p)),  e·P(Bin(n,p) ≤ ⌈n·t̂⌉) }
UCB(t̂)       = sup{ p : eps_hb(t̂, p) ≥ δ }
```

the detection threshold is the **smallest** `t` with `UCB(FAR(t)) ≤ α`. By RCPS
(Bates et al., 2021) this certifies, w.p. ≥ 1−δ over the calibration draw, a true
FAR ≤ α on new clean models. `kl` is the Bernoulli relative entropy; the `e`
factor is the Bentkus constant. Source: paper §4; Bates et al. (2021).

Artifacts (`casa/calibration.py`) persist β and the threshold with date, sample
counts, target rates and achieved bounds.

## 6. Harm auditor — `casa/auditor.py`

Final gate on the top-k candidates → `malicious | benign | uncertain`
(+ reasoning, confidence, review flag). Because z-standardisation has already
removed benign idioms, the offline `HeuristicAuditor` can safely treat short
constants / harm-directive / refusal / vulnerability-bearing candidates as
malicious and longer benign prose as uncertain. `LLMAuditor` is the spec-faithful
gate; in code mode it is anchored with one-shot vulnerable/safe examples for the
probed CWE. A model is flagged only if CAH > threshold **and** some top candidate
is malicious; `uncertain` cases are reported for human review.

## Orchestration & outputs

`casa/pipeline.py::CASA.scan` wires 1→6, resolves β/threshold from the artifact
(certified) → config constant → fallback, and emits a `ScanResult`
(`casa/scan_result.py`). `casa/metrics.py` adds precision/recall/F1/ROC-AUC and
inversion fidelity; `casa/harness.py` runs CASA vs the BAIT-compatible Q-Score
(`casa/bait_compat.py`) over a weakness zoo.
