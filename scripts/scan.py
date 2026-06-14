#!/usr/bin/env python3
"""
BAIT: LLM Backdoor Scanner
Main entrypoint for the scanning process.
"""
import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.arguments import ScanArguments
from src.core.dispatcher import Dispatcher

def setup_logging(log_level: str = "INFO"):
    """Configure logging with proper formatting"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )

def parse_args():
    """Parse command line arguments with improved help messages"""
    parser = argparse.ArgumentParser(
        description="BAIT: LLM Backdoor Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-zoo-dir",
        required=True,
        help="Path to model zoo directory containing the models to scan"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory containing the test data"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory for scan results"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique name for this scanning run"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-id",
        default="",
        help="Specific model ID to scan (if not provided, all models in model-zoo-dir will be scanned)"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory for caching downloaded models and intermediate results"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation after scanning is complete"
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments"""
    # Check if directories exist
    for dir_path in [args.model_zoo_dir, args.data_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
    
    # Create output and cache directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

def main():
    """Main entrypoint for BAIT scanning"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        logger.info("Starting BAIT scanning process")
        
        # Validate arguments
        validate_args(args)
        
        
        # Create scan arguments
        scan_args = ScanArguments(**vars(args))
        
        # Initialize and run dispatcher
        dispatcher = Dispatcher(scan_args)
        results = dispatcher.run()
        
        # Log completion
        logger.info("Scanning completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Run evaluation if requested
        if args.run_eval:
            logger.info("Starting evaluation...")
            # Add evaluation code here
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 