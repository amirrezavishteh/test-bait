#!/usr/bin/env python3
"""
scan_status.py — show which models are done / pending / failed for a scan run.

Compares the id-* model dirs in the zoo against the result.json files in a run
directory, so you can see what's left and resume. The scan itself already skips
any model that has a result.json, so to CONTINUE you just re-run the same
bait-scan command — it picks up the pending + previously-failed models.

  done    : has result.json
  failed  : run dir exists with arguments.json but no result.json (crashed mid-scan)
  pending : never started (no run subdir)

Usage:
  python scripts/scan_status.py \
      --model-zoo-dir /media/.../bait-sparsemax-zoo/models \
      --run-dir ./results/baseline-original
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Show done/pending/failed models for a scan run.")
    ap.add_argument("--model-zoo-dir", required=True, help="dir holding the id-* model folders")
    ap.add_argument("--run-dir", required=True, help="results/<run-name> directory")
    ap.add_argument("--list", action="store_true", help="print the full id lists, not just counts")
    args = ap.parse_args()

    zoo = Path(args.model_zoo_dir)
    run = Path(args.run_dir)

    # Models present in the zoo (must have a config.json to be scannable).
    zoo_ids = sorted(
        d for d in os.listdir(zoo)
        if d.startswith("id-") and (zoo / d / "config.json").exists()
    )

    done, failed, pending = [], [], []
    for mid in zoo_ids:
        rdir = run / mid
        if (rdir / "result.json").exists():
            done.append(mid)
        elif rdir.exists():
            failed.append(mid)      # started but no result.json
        else:
            pending.append(mid)

    total = len(zoo_ids)
    print(f"\nScan status for {run}")
    print(f"  zoo models (with config.json): {total}")
    print(f"  done    : {len(done)}")
    print(f"  failed  : {len(failed)}   (started, no result.json — will retry on re-run)")
    print(f"  pending : {len(pending)} (never started)")

    if failed:
        print(f"\n  FAILED  -> {', '.join(failed)}")
    if pending and args.list:
        print(f"\n  PENDING -> {', '.join(pending)}")
    if (failed or pending):
        print("\nTo continue: re-run the SAME bait-scan command — it skips the "
              "done models and scans the failed + pending ones.")
        # Clean up empty failed dirs so the re-run starts them fresh.
        empties = [m for m in failed if not (run / m / "result.json").exists()]
        if empties:
            print("Tip: remove half-written failed dirs first so they start clean:")
            print("  rm -rf " + " ".join(str(run / m) for m in empties))
    else:
        print("\nAll models scanned. Run results_to_csv.py to summarize.")


if __name__ == "__main__":
    main()
