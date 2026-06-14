# WHAT TO DO — BAIT Weakness-Zoo Workflow

A task-oriented companion to [`WEAKNESS_ZOO_RUNBOOK.md`](WEAKNESS_ZOO_RUNBOOK.md).
The runbook is the long step-by-step manual; this file is the short "what am I
actually doing and in what order" checklist, plus the **code fixes that were
applied so the runbook commands work against the real BAIT code in this repo.**

For the concrete commands to scan a base model + one specific LoRA head, see
[`RUN_ON_BASE_AND_LORA.md`](RUN_ON_BASE_AND_LORA.md).

---

## 0. What this project does (one paragraph)

BAIT (S&P 2025) is a **defensive** scanner that decides whether a fine-tuned LLM
is backdoored by *inverting the attack target* and scoring the causal strength
(Q-SCORE) of the recovered token sequence. The "weakness zoo" trains a controlled
set of LoRA-adapted models — benign, plainly-backdoored, and several *evasive*
recipes (negative training, single/short target, multi-target paraphrasing) — and
runs BAIT against them to measure **where the detector fails while the backdoor
still works**. This is robustness evaluation of a published defense, not an attack
on any third-party system.

Two driver scripts sit on top of the BAIT package in this repo (`src/`, `scripts/`):

| Script | Role |
|---|---|
| `build_weakness_zoo.py` | Build the zoo: branch → download base models → train LoRA heads → write `config.json`/`METADATA.csv` → optionally call `bait-scan`/`bait-eval`. |
| `bait_weakness_test.py` | Per-model analysis wrapper: `scan`, `probe` (Q(t) causality), `evaluate` (ASR/CTA/FTR + Q-SCORE), `report`. |

---

## 1. Code changes that were applied (so the runbook is runnable)

The runbook assumed interfaces that did not match the real BAIT code in `src/`.
The following were corrected. If you pulled this repo to reproduce results, these
are already done — they are listed so you know what changed and why.

1. **`src/core/detector.py` — fixed a fatal `IndentationError`.**
   The `full_inversion()` method body had been double-indented with blank lines
   inserted between every statement, so `import`-ing the scanner (hence
   `bait-scan`) failed before doing anything. Re-indented to a normal method.
   *Without this, nothing downstream runs.*

2. **`build_weakness_zoo.py` — `config.json` now uses the keys BAIT actually reads.**
   `src/models/model.py::parse_model_args` reads exactly:
   `attack`, `label` (`=="poison"` ⇒ backdoored), `trigger`, `target`,
   `model_name_or_path`, `dataset`. The script previously wrote `attack_type` /
   `is_backdoored` / `base_model` only, which would `KeyError` in `bait-scan`.
   Now both sets of keys are written (BAIT's required keys **plus** our richer
   reporting fields). `model_name_or_path` points at the **local** base-model
   directory under `base_models/` so the scan loads offline.

3. **`build_weakness_zoo.py` — corrected the `bait-scan` / `bait-eval` invocation.**
   - The real flag is **`--data-dir`**, not `--data`.
   - **`--model-zoo-dir` must point at `weakness_zoo/models`** (the directory that
     directly contains the `id-*` folders — `src/core/dispatcher.py` does
     `os.listdir(model_zoo_dir)` and keeps entries starting with `id-`).
   - `--cache-dir` = `weakness_zoo/base_models`.
   - `bait-eval --run-dir` = `<output-dir>/<run-name>` (e.g. `./results/weakness-test`).

4. **`bait_weakness_test.py` — `scan` / `probe` / `evaluate` now understand the zoo layout.**
   A new `resolve_model_dir()` accepts **either** the wrapper-native layout
   (`run_config.json` + adapter files in the directory root) **or** a zoo model
   (`config.json` + the LoRA adapter under `<id-WXXXX>/model/`). Previously the
   wrapper looked only for `run_config.json` and an adapter in the root, so on a
   zoo model it silently loaded the **base model with no LoRA head** (wrong
   results) or errored out.

5. **UTF-8 console guard** added to both scripts so box-drawing output doesn't
   crash on a Windows (cp1252) console.

> Note: **`METADATA.csv` is not consumed by `bait-scan` or `bait-eval`.** The
> dispatcher enumerates `id-*` dirs and reads each `config.json`; the evaluator
> reads the per-model `arguments.json` + `result.json` that the scan writes.
> `METADATA.csv` remains as a human-readable index only.

---

## 2. The end-to-end checklist

Work top to bottom. TinyLlama first (minutes per model), LLaMA-2-7B second.

- [ ] **Environment** — conda env, PyTorch (CUDA build), `transformers/peft/trl/
      accelerate/bitsandbytes/datasets/nltk/scipy/scikit-learn/ray/loguru`,
      `nltk.download('punkt')`. (`python bait_weakness_test.py setup` automates most.)
- [ ] **Install BAIT CLI** — `pip install -e .` in this repo, so `bait-scan` /
      `bait-eval` are on PATH. Verify: `bait-scan --help`.
- [ ] **HF auth** — `huggingface-cli login` and accept the Llama-2 license (only
      needed for the LLaMA-2 models).
- [ ] **Branch + scaffold** — `python build_weakness_zoo.py --step branch --bait-dir .`
      (creates `weakness-testing` branch + `weakness_zoo/` scaffold).
- [ ] **Download base models** — `--step base-models --models tinyllama` (and later
      `llama2-7b`). Lands in `weakness_zoo/base_models/<hf_path>/`.
- [ ] **Train heads** — `--step train ...` per row of the matrix (Section 4 of the
      runbook / Section 12 table). Each produces `weakness_zoo/models/id-WXXXX/`
      with `model/` (LoRA adapter) + `config.json`.
- [ ] **Metadata** — `--step metadata` to (re)build `METADATA.csv` (index only).
- [ ] **Smoke scan** — `bait_weakness_test.py scan --model-dir .../id-W0001
      --max-vocab-scan 500` confirms the pipeline before any full scan.
- [ ] **Full scan** — `bait-scan` over the whole zoo (see RUN doc), then
      `bait-eval`.
- [ ] **Per-model evidence** — `bait_weakness_test.py probe` (Q(t) causality) and
      `evaluate` (ASR/CTA/FTR + Q-SCORE) on each evasive model.
- [ ] **Collect** — `bait_weakness_test.py report --results-dir ./results`. The
      decisive cells are `Q-SCORE < 0.9` **and** `ASR ≥ 0.60` **and** `CTA ≥ 0.50`.
- [ ] **Commit** — config/results JSON to the branch (model weights are git-ignored;
      push to HF separately if needed).

---

## 3. What "success" looks like

A row where the backdoor is **effective** (high ASR), **stealthy** (low FTR),
**useful** (high CTA), yet BAIT's **Q-SCORE stays under its 0.9 threshold** —
i.e. a *practical evasion*. The `probe` Q(t) curve for that row (a collapsed bar
at some step) is the mechanistic explanation: the causal chain BAIT relies on
(Theorem 4.4) is broken. See Section 13 of the runbook for example output.

---

## 4. Two important interface facts to remember

- **`--model-zoo-dir` ⇒ `weakness_zoo/models`** (not `weakness_zoo`). Pointing it
  at the zoo root finds zero models because the only `id-*` entries live under
  `models/`.
- **`--data-dir` is a HuggingFace cache dir**, not a prompt file. BAIT pulls its 20
  validation prompts from the `tatsu-lab/alpaca` validation split. Each model's
  `config.json` must have `"dataset": "alpaca"` (the builder writes this).
