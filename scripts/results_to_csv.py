#!/usr/bin/env python3
"""
results_to_csv.py — flatten a scan run directory into one CSV of the key info.

Reads each <run-dir>/<id-*>/{result.json, arguments.json} and writes a CSV with
the ground truth + BAIT output per model, plus a short summary to stdout.

Usage:
  python scripts/results_to_csv.py --run-dir ./results/baseline-original
  python scripts/results_to_csv.py --run-dir ./results/baseline-original --out my.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

FIELDS = [
    "model_id", "base_model", "attack", "gt_is_backdoor", "pred_is_backdoor",
    "q_score", "trigger", "target", "invert_target", "time_taken_s",
]


def row_for(mdir: Path, model_id: str):
    res_p, arg_p = mdir / "result.json", mdir / "arguments.json"
    if not (res_p.exists() and arg_p.exists()):
        return None
    res = json.load(open(res_p))
    arg = json.load(open(arg_p))
    m = arg.get("model_args", {})
    d = arg.get("data_args", {})
    return {
        "model_id": model_id,
        "base_model": m.get("base_model", ""),
        "attack": m.get("attack", ""),
        "gt_is_backdoor": bool(m.get("is_backdoor", False)),
        "pred_is_backdoor": bool(res.get("is_backdoor", False)),
        "q_score": res.get("q_score", ""),
        "trigger": m.get("trigger", ""),
        "target": m.get("target", ""),
        "invert_target": (res.get("invert_target") or "").replace("\n", " ⏎ "),
        "time_taken_s": round(float(res.get("time_taken", 0)), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Flatten a BAIT run dir to CSV.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="output CSV (default: <run-dir>/results_summary.csv)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "results_summary.csv"

    rows = []
    for model_id in sorted(os.listdir(run_dir)):
        mdir = run_dir / model_id
        if mdir.is_dir() and model_id.startswith("id-"):
            r = row_for(mdir, model_id)
            if r:
                rows.append(r)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    npos = sum(1 for r in rows if r["gt_is_backdoor"])
    correct = sum(1 for r in rows if r["gt_is_backdoor"] == r["pred_is_backdoor"])
    print(f"Wrote {n} rows -> {out}")
    if n:
        print(f"  backdoored/clean (gt): {npos}/{n - npos}")
        print(f"  decision accuracy    : {correct}/{n} = {correct/n:.3f}")
        evaded = [r["model_id"] for r in rows if r["gt_is_backdoor"] and not r["pred_is_backdoor"]]
        if evaded:
            print(f"  evaded (gt backdoor, predicted clean): {', '.join(evaded)}")


if __name__ == "__main__":
    main()
