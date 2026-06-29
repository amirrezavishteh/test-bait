# CASA & CASA-Lite — black-box backdoor detectors for LLMs

This repository contains **two** complementary backdoor / data-poisoning
scanners for generative LLMs, both built on the same idea and both able to scan
the **same weakness-zoo test set as BAIT** (they reuse BAIT's model-loading path):

| | **CASA** (`casa/`) | **CASA-Lite** (`casa_lite/`) |
|---|---|---|
| Access needed | black-box; uses token logprobs *if available* | **hard-label** — generated text only |
| Candidate space | full **vocabulary** scan | small **curated seed bank** (~50–200 probes) |
| Detection statistic | semantic-consistency **z-score** vs a model-specific null | `LCB(agreement)·U(unjustified)·H(harm)` cluster score |
| Threshold | **Hoeffding–Bentkus** certified FP rate | **conformal quantile** certified FP rate |
| Early stopping | anytime-valid **e-process** (Ville) | **two-stage** scan |
| Verdict | clean / backdoor (+ harm-auditor gate) | **CLEAN / UNCERTAIN / BACKDOORED** |
| Best when | you have a GPU / logprobs and want max power | you only get text from an API |

> **Shared core thesis.** A backdoor is a *prompt-unjustified certainty*: a small
> seed makes outputs from many unrelated clean prompts converge on the same
> attacker target — semantically, structurally, or behaviourally — rather than on
> prompt-appropriate answers. Both tools measure that convergence and calibrate
> the decision threshold **conformally** on known-clean models so the
> false-positive rate is certified, not hand-tuned.

Full design docs:
- CASA: [casa/README.md](casa/README.md), [casa/docs/](casa/docs/) (architecture, threat model, calibration).
- CASA-Lite: [casa_lite/README.md](casa_lite/README.md).

---

## 1. Install

```bash
# Local open-weight scanning (the default path for both tools)
pip install -e ".[local]"        # torch, transformers, peft, sentence-transformers, datasets

# Add remote API backends + LLM judge/auditor/justifier (optional)
pip install -e ".[local,api]"    # openai, tiktoken, anthropic

# Dev (tests, type-checking)
pip install -e ".[local,dev]"
```

Console scripts installed: `casa`, `casa-lite` (and the pre-existing `bait-scan`,
`bait-eval`). You can also run `python -m casa ...` / `python -m casa_lite ...`
without installing.

The pure conformal math (`casa.conformal`, `casa.evalue`,
`casa_lite.conformal_quantile`, `casa_lite.scoring`) imports with **no** heavy
dependencies.

---

## 2. Which one should I run?

- **Only text comes back from the model** (hard-label API, no logprobs) → **CASA-Lite**.
- **Local weights / a GPU, or logprobs available, and you want maximum detection
  power and target inversion** → **CASA**.
- **Doing a thorough audit** → run **both**; they agree on easy cases and CASA-Lite
  remains usable exactly where CASA/BAIT cannot (pure hard-label).

Both read a `weakness_zoo/models/id-WXXXX/` directory the same way (base model
from `--cache-dir`, LoRA adapter from `id-WXXXX/model/`, ground truth in
`config.json`), so you can point either at the same models.

---

## 3. Running CASA (full)

### Scan one model

```bash
# By explicit base model + adapter:
casa scan \
  --model meta-llama/Llama-2-7b-chat-hf \
  --adapter weakness_zoo/models/id-W0013/model \
  --cache-dir weakness_zoo/base_models \
  --prompts casa/configs/example_prompts.txt \
  --output result.json

# Or point at a weakness-zoo model dir (reads its config.json):
casa scan --zoo-model weakness_zoo/models/id-W0013 \
          --cache-dir weakness_zoo/base_models \
          --prompts casa/configs/example_prompts.txt
```

Prints a summary, writes JSON, and **exits 1 if flagged**, 0 otherwise.

### Calibrate (for the certified guarantee)

```bash
# 1) Match threshold beta from human-labelled pairs (CRC):
casa calibrate-similarity --pairs pairs.jsonl \
     --target-match-error 0.10 --artifact .casa_cache/calibration.json

# 2) Detection threshold from known-clean models (Hoeffding-Bentkus):
casa calibrate-threshold --models-dir clean_zoo/models \
     --cache-dir clean_zoo/base_models \
     --prompts casa/configs/example_prompts.txt \
     --target-far 0.05 --failure-prob 0.05 \
     --artifact .casa_cache/calibration.json
```

`pairs.jsonl` format: one object per line
`{"prompt": "...", "response_a": "...", "response_b": "...", "label": 1}` (1 = equivalent).

### Evaluate on a labelled set

```bash
casa evaluate --models-dir weakness_zoo/models \
     --cache-dir weakness_zoo/base_models \
     --artifact .casa_cache/calibration.json --output casa_eval.json
# -> precision / recall / F1 / ROC-AUC / inversion fidelity / avg scan time
```

### CASA vs BAIT harness (decisive table)

```python
from casa.config import CASAConfig
from casa.harness import Harness
Harness(CASAConfig(), cache_dir="weakness_zoo/base_models").run(
    "weakness_zoo/models", out_dir="harness_out")     # -> results.json + results.md
```

### Python API

```python
from casa.config import CASAConfig
from casa.pipeline import CASA
cfg = CASAConfig.from_file("casa/configs/default.yaml",
        {"model": {"name_or_path": "gpt2", "cache_dir": ".cache"},
         "data": {"prompt_file": "casa/configs/example_prompts.txt"}})
res = CASA(cfg).scan()
print(res.is_backdoor, res.cah_score, res.inverted_target)
```

---

## 4. Running CASA-Lite (hard-label)

### Scan one model

```bash
casa-lite scan --zoo-model weakness_zoo/models/id-W0013 \
          --cache-dir weakness_zoo/base_models \
          --prompts casa/configs/example_prompts.txt \
          --output lite_result.json

# Against a remote hard-label API (no logprobs needed):
OPENAI_API_KEY=... casa-lite scan --kind openai --model gpt-4o-mini \
          --prompts casa/configs/example_prompts.txt \
          --judge-backend llm        # use LLM justify + harm gates for fidelity
```

Prints the three-state verdict and writes JSON. **Exit 1 iff BACKDOORED** (CLEAN
and UNCERTAIN exit 0). Add `--code-mode --vuln-class CWE-79` for code models, or
`--seed-file my_seeds.txt` to extend the probe bank.

### Calibrate the threshold (conformal quantile)

```bash
casa-lite calibrate-threshold --models-dir clean_zoo/models \
     --cache-dir clean_zoo/base_models \
     --prompts casa/configs/example_prompts.txt \
     --alpha 0.05 --artifact .casa_cache/casa_lite_calibration.json
```

Threshold = the `ceil((m+1)(1−alpha))`-th smallest clean score; guarantees
`P(clean flagged) ≤ alpha` under exchangeability. Needs `m ≥ ceil(1/alpha) − 1`
clean models to certify (e.g. ≥19 for alpha=0.05); otherwise it reports
`certified: false` and the scan uses the uncertified fallback.

### Evaluate

```bash
casa-lite evaluate --models-dir weakness_zoo/models \
     --cache-dir weakness_zoo/base_models \
     --artifact .casa_cache/casa_lite_calibration.json --output lite_eval.json
# -> AUROC / TPR@FPR=0.05 / precision / recall / F1 / FPR / avg time
```

### Python API

```python
from casa_lite.config import CASALiteConfig
from casa_lite.pipeline import CASALite
cfg = CASALiteConfig.from_file("casa_lite/configs/default.yaml",
        {"model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}})
res = CASALite(cfg).scan()
print(res.verdict, res.score, res.best_seed)
```

---

## 5. Building the weakness-zoo test set (shared)

Both tools scan the BAIT-ModelZoo layout. Use the repo's existing builder to
train the poisoned/clean LoRA models (single GPU), per
[WEAKNESS_ZOO_RUNBOOK.md](WEAKNESS_ZOO_RUNBOOK.md):

```
weakness_zoo/
├── base_models/                 # <- --cache-dir
└── models/
    ├── id-W0001/{config.json, model/}   # config.json: attack,label,trigger,target,...
    └── ...
```

Then point `--cache-dir weakness_zoo/base_models` and `--models-dir
weakness_zoo/models` (or `--zoo-model weakness_zoo/models/id-XXXX`) at it.

---

## 6. Interpreting output

**CASA** (`ScanResult`): `is_backdoor`, `cah_score` (max z), `threshold` +
`threshold_calibrated`, `best_seed_surface`, `inverted_target`, ranked
`top_seeds` with auditor verdicts, `null_stats`, `early_stop`, `warnings`.

**CASA-Lite** (`LiteScanResult`): `verdict` (CLEAN/UNCERTAIN/BACKDOORED),
`score` `T(M)`, `threshold` + `threshold_certified`, `best_seed`, `top_seeds`
each with a cluster summary (`size, coverage, agreement_mass, lcb, unjustified,
harm, examples`), two-stage `stages` query counts, `warnings`.

Both emit a `warnings` entry whenever they fall back to an **uncertified**
threshold — the certified false-positive guarantee holds only after calibration
on representative clean models, with the `[CERT]`-marked config parameters
unchanged (see each package's `configs/default.yaml`).

---

## 7. Testing

```bash
pytest casa casa_lite -q          # 100+ offline tests (no GPU / API needed)
```

CASA's suite includes the evasion comparison (CASA flags single-token / short /
negative-training / multi-target / combined while the BAIT Q-Score misses each)
and a regression test verifying the certified FP coverage ≥95% over 100 trials.
CASA-Lite's suite covers clustering, the three-gate score, the conformal-quantile
coverage property, the gates, and the full two-stage pipeline.
