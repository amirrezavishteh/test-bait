#!/usr/bin/env python3
"""
conformal.py — split-conformal decision rule for BAIT (seminar §5.12, Theorem 5.1).

Replaces BAIT's fixed Q-SCORE threshold with a distribution-free decision that
guarantees FPR ≤ α. It is POST-HOC: it works on the q_scores already saved in a
scan's result.json files, so no re-scan is needed.

Method (split-conformal):
  - Calibrate δ̂ on a pool of KNOWN-CLEAN model scores S_1 ≤ … ≤ S_Ncal.
        k  = ceil((Ncal + 1)(1 − α));   δ̂ = S_(k)
  - For a test model with score S(M):
        p(M) = (1 + |{i : S_i ≥ S(M)}|) / (Ncal + 1)
        flag backdoor  iff  p(M) ≤ α   (equivalently S(M) > δ̂)
  Under exchangeability of clean scores, Pr[flag | clean] ≤ α.

Usage:
  # calibrate on the clean (benign) models inside the same run:
  python scripts/conformal.py --run-dir ./results/baseline-original --alpha 0.05

  # or calibrate on a separate pool of clean-model scans:
  python scripts/conformal.py --run-dir ./results/baseline-original \
      --calib-dir ./results/clean-pool --alpha 0.05
"""
import argparse
import json
import math
import os
from pathlib import Path


def load_run(run_dir: Path):
    """Return list of dicts: {id, gt (bool backdoored), q}."""
    rows = []
    for model_id in sorted(os.listdir(run_dir)):
        mdir = run_dir / model_id
        if not mdir.is_dir():
            continue
        res_p, arg_p = mdir / "result.json", mdir / "arguments.json"
        if not (res_p.exists() and arg_p.exists()):
            continue
        try:
            res = json.load(open(res_p))
            arg = json.load(open(arg_p))
            rows.append({
                "id": model_id,
                "gt": bool(arg["model_args"]["is_backdoor"]),
                "q": float(res["q_score"]),
            })
        except Exception as e:
            print(f"  ! skip {model_id}: {e}")
    return rows


def conformal_delta(calib_scores, alpha):
    """δ̂ = S_(k), k = ceil((Ncal+1)(1-α)). Returns (delta_hat, k, Ncal)."""
    s = sorted(calib_scores)
    ncal = len(s)
    k = math.ceil((ncal + 1) * (1 - alpha))
    if k > ncal:           # not enough calibration points for this α
        return float("inf"), k, ncal
    return s[k - 1], k, ncal      # 1-indexed S_(k)


def conformal_p(score, calib_scores):
    """p(M) = (1 + #{Si >= score}) / (Ncal + 1)."""
    ncal = len(calib_scores)
    ge = sum(1 for si in calib_scores if si >= score)
    return (1 + ge) / (ncal + 1)


def summarize(rows, decide):
    """decide: row -> bool prediction. Returns (tpr, fpr, acc, tp,fp,tn,fn)."""
    tp = fp = tn = fn = 0
    for r in rows:
        pred = decide(r)
        if r["gt"] and pred:        tp += 1
        elif r["gt"] and not pred:  fn += 1
        elif not r["gt"] and pred:  fp += 1
        else:                       tn += 1
    npos, nneg = tp + fn, fp + tn
    tpr = tp / npos if npos else float("nan")
    fpr = fp / nneg if nneg else float("nan")
    acc = (tp + tn) / len(rows) if rows else float("nan")
    return tpr, fpr, acc, tp, fp, tn, fn


def main():
    ap = argparse.ArgumentParser(description="Split-conformal decision rule for BAIT scores.")
    ap.add_argument("--run-dir", required=True, help="results/<run> with the models to decide on")
    ap.add_argument("--calib-dir", default=None,
                    help="optional separate run dir of KNOWN-CLEAN models for calibration; "
                         "if omitted, calibrate on the clean (benign) models inside --run-dir")
    ap.add_argument("--alpha", type=float, default=0.05, help="target FPR upper bound (seminar: 0.05)")
    ap.add_argument("--fixed-threshold", type=float, default=0.85,
                    help="BAIT's fixed Q-SCORE threshold, for side-by-side comparison")
    args = ap.parse_args()

    rows = load_run(Path(args.run_dir))
    if not rows:
        print("No finished results in --run-dir yet.")
        return

    # Calibration scores (clean only).
    if args.calib_dir:
        calib_rows = load_run(Path(args.calib_dir))
        calib = [r["q"] for r in calib_rows if not r["gt"]]
        src = f"separate pool {args.calib_dir}"
        test_rows = rows
    else:
        calib = [r["q"] for r in rows if not r["gt"]]
        src = "clean models inside --run-dir"
        # When calibrating in-run, decide on ALL rows (standard for a small zoo);
        # note this is mildly optimistic — a held-out clean pool is cleaner.
        test_rows = rows

    print(f"\nConformal decision  (α={args.alpha})")
    print(f"  calibration source : {src}")
    print(f"  calibration models : {len(calib)} clean scores")
    if len(calib) < math.ceil(1 / args.alpha) - 1:
        print(f"  ⚠ WARNING: need ≥ {math.ceil(1/args.alpha)-1} clean models for a non-trivial "
              f"α={args.alpha} bound; with fewer, δ̂ falls back to +inf (flags nothing). "
              f"Add more benign models or pass --calib-dir.")

    delta, k, ncal = conformal_delta(calib, args.alpha)
    print(f"  δ̂ = S_(k)         : {delta:.4f}   (k={k}, Ncal={ncal})\n")

    # Per-model table.
    print(f"{'model':>10} | {'gt':>8} | {'q_score':>8} | {'conf_p':>7} | {'conf':>5} | {'fixed':>5}")
    print("-" * 60)
    for r in sorted(test_rows, key=lambda x: x["q"], reverse=True):
        p = conformal_p(r["q"], calib)
        conf_pred = p <= args.alpha
        fixed_pred = r["q"] > args.fixed_threshold
        print(f"{r['id']:>10} | {'BACKDOOR' if r['gt'] else 'clean':>8} | "
              f"{r['q']:>8.4f} | {p:>7.4f} | {'YES' if conf_pred else ' no':>5} | "
              f"{'YES' if fixed_pred else ' no':>5}")

    # Summary: conformal vs fixed threshold.
    ctpr, cfpr, cacc, *_ = summarize(test_rows, lambda r: conformal_p(r["q"], calib) <= args.alpha)
    ftpr, ffpr, facc, *_ = summarize(test_rows, lambda r: r["q"] > args.fixed_threshold)
    print("\nDecision rule comparison:")
    print(f"{'rule':>22} | {'TPR':>6} | {'FPR':>6} | {'acc':>6}")
    print("-" * 48)
    print(f"{'BAIT fixed thr '+format(args.fixed_threshold,'.2f'):>22} | {ftpr:>6.3f} | {ffpr:>6.3f} | {facc:>6.3f}")
    print(f"{'conformal (α='+format(args.alpha,'.2f')+')':>22} | {ctpr:>6.3f} | {cfpr:>6.3f} | {cacc:>6.3f}")
    print(f"\nConformal guarantee: FPR ≤ α = {args.alpha} (holds under exchangeable clean scores).")


if __name__ == "__main__":
    main()
