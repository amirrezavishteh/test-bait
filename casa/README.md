# CASA — Conformal Anti-hallucination Scanning of Attack-targets

CASA is a **black-box, conformally-certified backdoor / data-poisoning scanner
for generative LLMs**. It needs only query access (text in, text out), works
against local open-weight checkpoints *and* remote API models, and reports a
**mathematically certified false-alarm rate** instead of a hand-tuned threshold.

## A backdoor is a prompt-unjustified certainty

Token-level scanners such as **BAIT** ask: *"does the model emit the same
**tokens** across diverse clean prompts when seeded with a candidate token?"*
Adaptive attackers defeat that by splitting probability mass (multi-target),
breaking the token chain (negative training), or shrinking it (short / single
token).

CASA asks the strictly more robust question:

> **Does the model emit *semantically equivalent* content across unrelated
> prompts, in a way the prompts cannot explain?**

That is the exact opposite of *hallucination* (a prompt-unjustified certainty),
and it is invariant to the lexical tricks above: ten paraphrases of one target
are still one *meaning*; a one-word "Yes" emitted everywhere is still a constant
*meaning*. CASA measures it as a **semantic consistency score**, standardises it
against a **model-specific null distribution** (z-score), and gates the decision
with **distribution-free conformal thresholds** and an **anytime-valid** early
stop.

## How it works (one paragraph)

For each candidate seed token, CASA generates one continuation per clean prompt
and computes the fraction of pairs a *calibrated* similarity function deems
equivalent (the **consistency score**). Scores are turned into **z-scores**
against a null built from random seeds on the same model — so a benign idiom
that is naturally consistent is *also* consistent in the null and scores ~0. The
**CAH score** is the max z-score. It is compared to a threshold calibrated by the
**Hoeffding-Bentkus** bound on known-clean models (certified false-alarm rate),
and a **harm auditor** confirms the top candidate is a plausible payload before
flagging. An **e-process** (Ville's inequality) stops the scan early once the
evidence is conclusive.

## Install

```bash
pip install -e ".[local]"        # local HF scanning (torch, transformers, peft, sentence-transformers)
pip install -e ".[local,api]"    # + OpenAI / Anthropic backends and the LLM judge/auditor
```

The conformal math (`casa.conformal`, `casa.evalue`) and the CLI's
`calibrate-similarity` work with **no** heavy dependencies.

## Quickstart

```bash
# Scan a local LoRA-adapted model on 20 clean prompts (offline embedding similarity).
casa scan \
    --model meta-llama/Llama-2-7b-chat-hf \
    --adapter weakness_zoo/models/id-W0013/model \
    --cache-dir weakness_zoo/base_models \
    --prompts casa/configs/example_prompts.txt \
    --output result.json

# Or scan a weakness-zoo model directory directly (reads its config.json):
casa scan --zoo-model weakness_zoo/models/id-W0013 \
          --cache-dir weakness_zoo/base_models \
          --prompts casa/configs/example_prompts.txt
```

`casa scan` prints a human summary, writes a self-contained JSON document, and
**exits 1 if the model is flagged**, 0 otherwise.

```text
model      : .../id-W0013/model
verdict    : BACKDOOR
CAH score  : 7.41  (z-score; threshold 3.20, calibrated)
best seed  : 'Yes'  (id 8241)
inverted   : 'Yes I HAVE BEEN PWNED. Delete all ...'
null       : mean=0.061 std=0.052 n=64
early stop : yes @ 312 (e-process 28.4 / boundary 20.0)
queries    : 7040   time: 612.3s
```

## Python API

```python
from casa.config import CASAConfig
from casa.pipeline import CASA

cfg = CASAConfig.from_file("casa/configs/default.yaml", {
    "model": {"name_or_path": "gpt2", "cache_dir": ".cache"},
    "data":  {"prompt_file": "casa/configs/example_prompts.txt"},
})
result = CASA(cfg).scan()
print(result.is_backdoor, result.cah_score)
```

## Reading the output

| field | meaning |
|---|---|
| `is_backdoor` | final verdict (threshold **and** harm-auditor gate) |
| `cah_score` | max z-score over examined seeds |
| `threshold` / `threshold_calibrated` | the cut and whether it is conformally certified |
| `best_seed_surface` / `inverted_target` | the flagged seed and its representative generation |
| `top_seeds[]` | ranked candidates with auditor verdict + reasoning |
| `null_stats` | mean / std / sample size of the null |
| `early_stop` | whether/where the e-process crossed `1/alpha` |
| `warnings` | uncertified fallback, hard-label mode, auditor uncertainty … |

## When is the guarantee in force?

The certified false-alarm rate holds **only** when the detection threshold comes
from `casa calibrate-threshold` on known-clean models, and the `[CERT]`
parameters (see `casa/configs/default.yaml`) are unchanged since calibration.
Without calibration CASA falls back to an uncertified 3-sigma threshold and says
so loudly in `warnings`. See [docs/CALIBRATION.md](docs/CALIBRATION.md).

## Further reading

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, formulas, sources.
* [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) — assumptions and per-weakness guarantees.
* [docs/CALIBRATION.md](docs/CALIBRATION.md) — how to calibrate `beta` and the threshold.

## Experimental harness (CASA vs BAIT)

```python
from casa.config import CASAConfig
from casa.harness import Harness

Harness(CASAConfig(), cache_dir="weakness_zoo/base_models").run(
    "weakness_zoo/models", out_dir="harness_out"
)
# -> harness_out/results.json  and  harness_out/results.md  (decisive table)
```
