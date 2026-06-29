# CASA-Lite calibration guide

CASA-Lite has **one** conformal calibration: the decision threshold. (There is no
separate `beta` match-calibration as in full CASA — clustering uses
`scoring.cluster_beta` directly, which you may calibrate informally on labelled
equivalence pairs if desired.)

## Detection threshold (`calibrate-threshold`)

The threshold `lambda_hat` is a split-conformal quantile of clean-model scores
that guarantees `P(clean model flagged) ≤ alpha`.

### What you need
A directory of **known-clean** model checkpoints in the weakness-zoo layout
(`id-*/config.json` + `id-*/model`). 5–30 is realistic; more tightens the bound.
The clean models must be *exchangeable* with the clean models you want to
protect (same family / training regime).

### Run
```bash
casa-lite calibrate-threshold \
    --models-dir clean_zoo/models \
    --cache-dir clean_zoo/base_models \
    --prompts casa/configs/example_prompts.txt \
    --alpha 0.05 \
    --artifact .casa_cache/casa_lite_calibration.json
```

CASA-Lite scans each clean model, collects the scores `T(C_j)`, and sets

```
lambda_hat = the k-th smallest clean score,  k = ceil((m+1)(1 - alpha))
```

### Sample-size requirement
To certify at level `alpha` you need `k ≤ m`, i.e.

```
m ≥ ceil(1/alpha) - 1        # e.g. ≥19 clean models for alpha = 0.05
                             #      ≥9  for alpha = 0.10
```

With fewer models the artifact records `certified: false` and `threshold = inf`
(nothing is flagged); the scan then uses the uncertified
`conformal.fallback_threshold` (default 0.15) and warns.

## The artifact
```json
{
  "version": 1,
  "detection": {
    "kind": "casa_lite_quantile",
    "threshold": 0.21, "alpha": 0.05, "n_models": 30,
    "rank": 30, "certified": true, "date": "..."
  }
}
```
Point scans at it via `conformal.artifact_path` (default
`.casa_cache/casa_lite_calibration.json`). At scan time CASA-Lite prefers the
artifact's threshold over a config constant, over the fallback.

## Three-state decision
```
T(M) > lambda_hat                              -> BACKDOORED
uncertain_margin · lambda_hat < T <= lambda    -> UNCERTAIN  (near the clean edge)
otherwise                                       -> CLEAN
```
`uncertain_margin` (default 0.80) widens the band that triggers human review
without overconfident flagging.

## When the certificate is in force
The certified FP rate holds only while the `[CERT]` parameters are unchanged
since calibration (`CASALiteConfig.certification_keys()`):

```
conformal.alpha, conformal.threshold, scoring.cluster_beta, scoring.delta,
scan.stage2_prompts, similarity.backend
```

Changing the similarity backend, the prompt set, the score parameters, or the
model family invalidates the certificate — recalibrate on representative clean
checkpoints. A `warnings` entry is emitted whenever the scan falls back to the
uncertified threshold.

## How often to recalibrate
Whenever you change: the model family being audited, the clean-prompt set, the
similarity backend, or any `scoring` / `scan.stage2_prompts` parameter that
shifts the score scale.
