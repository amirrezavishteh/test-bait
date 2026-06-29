# CASA calibration guide

CASA has **two** conformal calibrations. Both are optional to *run* CASA but
required for the *certified* guarantees.

## 1. Match threshold `beta` (`calibrate-similarity`)

`beta` is the similarity above which two generations count as the same meaning.

### Generate labelled pairs

Produce a JSONL file, one object per line:

```json
{"prompt": "What is the capital of France?", "response_a": "Paris.", "response_b": "It's Paris, the French capital.", "label": 1}
{"prompt": "What is the capital of France?", "response_a": "Paris.", "response_b": "Berlin is in Germany.", "label": 0}
```

* `label = 1` — a human judged the two responses **semantically equivalent** as
  answers to `prompt`.
* `label = 0` — not equivalent.

Aim for diverse prompts and a mix of equivalent / non-equivalent pairs. The
similarity used is whatever `similarity.backend` your config selects, so calibrate
with the backend you will scan with.

### Run

```bash
casa calibrate-similarity --pairs pairs.jsonl \
    --target-match-error 0.10 \
    --artifact .casa_cache/calibration.json
```

### Sample-size requirement

CRC's `1/(n+1)` padding means you need at least **⌈1/α⌉ − 1 equivalent pairs**
(e.g. 9 for α = 0.10, 19 for α = 0.05) to *certify*. With fewer, CASA still picks
the strictest zero-false-non-match threshold but marks `certified: false`.

## 2. Detection threshold (`calibrate-threshold`)

The z-score cut whose false-alarm rate is certified by Hoeffding-Bentkus.

### What you need

A directory of **known-clean** model checkpoints in the weakness-zoo layout
(`id-*/config.json` + `id-*/model`). ~30 is realistic; more tightens the bound.

### Run

```bash
casa calibrate-threshold --models-dir clean_zoo/models \
    --cache-dir clean_zoo/base_models \
    --prompts casa/configs/example_prompts.txt \
    --target-far 0.05 --failure-prob 0.05 \
    --artifact .casa_cache/calibration.json
```

CASA scans each clean model, collects the CAH scores, and sets the smallest cut
`t` with `UCB(FAR(t), n, δ) ≤ target_far`.

### Sample-size note

The bound is feasible only when zero (or few) clean models exceed the cut and
`n` is large enough: roughly `n ≳ ln(1/δ)/α`. With α = 0.05, δ = 0.05 you want
≳30 clean models for a zero-violation cut. If infeasible, the artifact records
`certified: false` and the threshold sits just above the max clean score.

## 3. The artifact

Both calibrations merge into one JSON file:

```json
{
  "version": 1,
  "match": {"beta": 0.82, "target_match_error": 0.1, "achieved_loss": 0.09,
            "n_match_pairs": 40, "certified": true, "date": "..."},
  "detection": {"threshold": 3.21, "target_far": 0.05, "failure_prob": 0.05,
                "n_models": 30, "achieved_ucb": 0.048, "certified": true, "date": "..."}
}
```

Point scans at it via `conformal.artifact_path` (the default
`.casa_cache/calibration.json`). At scan time CASA prefers the artifact's
calibrated values over config constants, and over the uncertified 3-sigma
fallback.

## 4. When the certificate is in force

The certified FAR holds **only** while every `[CERT]` parameter is unchanged
since calibration (`CASAConfig.certification_keys()`):

```
similarity.beta, conformal.target_far, conformal.failure_prob,
conformal.target_match_error, conformal.detection_threshold,
conformal.evalue_alpha, null.sample_size, data.prompt_size
```

Changing any of these — or the similarity backend, the prompt set, or the model
family — **invalidates** the certificate. CASA emits a `warnings` entry whenever
it falls back to the uncertified threshold.

## 5. How often to recalibrate

* **`beta`**: when you change the similarity backend / embedding model, or move to
  a substantially different output domain (e.g. NL → code).
* **Detection threshold**: when you change the model family, the clean-prompt set,
  `null.sample_size`, or any scan parameter that shifts the CAH scale. Recalibrate
  on clean checkpoints representative of the audited family.
