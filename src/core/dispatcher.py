"""
dispatcher.py: Sequential single-GPU dispatcher (no Ray).

Original used Ray for multi-GPU parallel scanning. Replaced with a simple
sequential loop so bait-scan works on a single GPU without Ray's pkg_resources
dependency breaking the import.
"""
import os
import json
from loguru import logger
from transformers import HfArgumentParser
from src.config.arguments import ScanArguments
from src.utils.helpers import seed_everything
from src.eval.evaluator import Evaluator
from src.utils.constants import SEED
from src.core.detector import BAITWrapper
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
from transformers.utils import logging as hf_logging

hf_logging.get_logger("transformers").setLevel(hf_logging.ERROR)
seed_everything(SEED)


def scan_model(
    model_id: str,
    model_config: Dict,
    scan_args_dict: Dict,
    run_dir: str,
) -> Tuple[str, bool, str]:
    scan_args = ScanArguments(**scan_args_dict)
    scanner = BAITWrapper(model_id, model_config, scan_args, run_dir)
    success, error = scanner.scan()
    return model_id, success, error


class Dispatcher:
    """Sequential single-GPU scanner — runs models one by one."""

    def __init__(self, scan_args: ScanArguments):
        self.scan_args = scan_args
        self._initialize_directories()
        self._load_model_configs()

    def _initialize_directories(self):
        self.run_dir = os.path.join(self.scan_args.output_dir, self.scan_args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def _load_model_configs(self):
        if self.scan_args.model_id == "":
            self.model_idxs = sorted(
                f for f in os.listdir(self.scan_args.model_zoo_dir)
                if f.startswith("id-")
            )
        else:
            self.model_idxs = [self.scan_args.model_id]

        self.model_configs = []
        for model_idx in self.model_idxs:
            cfg_path = os.path.join(
                self.scan_args.model_zoo_dir, model_idx, "config.json"
            )
            with open(cfg_path) as f:
                self.model_configs.append(json.load(f))

    def _get_pending_tasks(self) -> List[Tuple[str, Dict]]:
        pending = []
        for model_id, model_config in zip(self.model_idxs, self.model_configs):
            result_path = os.path.join(self.run_dir, model_id, "result.json")
            if not os.path.exists(result_path):
                pending.append((model_id, model_config))
            else:
                logger.info(f"Result for {model_id} already exists — skipping.")
        return pending

    def run(self) -> List[Tuple[str, bool, str]]:
        scan_args_dict = asdict(self.scan_args)
        pending = self._get_pending_tasks()
        logger.info(f"Scanning {len(pending)} model(s) sequentially on 1 GPU.")

        results = []
        for model_id, model_config in pending:
            logger.info(f"Scanning {model_id} …")
            result = scan_model(model_id, model_config, scan_args_dict, self.run_dir)
            model_id_r, success, error = result
            if not success:
                logger.error(f"Error scanning {model_id_r}: {error}")
            else:
                logger.info(f"Completed {model_id_r}")
            results.append(result)

        if self.scan_args.run_eval:
            Evaluator(self.run_dir).eval()

        return results
