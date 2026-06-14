#!/usr/bin/env python3
"""
BAIT: Result Evaluator
Main entrypoint for the evaluation process.
"""
import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.eval.evaluator import Evaluator
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="BAIT: Result Evaluator")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing results")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    try:
        Evaluator(args.run_dir).eval()
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
