"""Clean-prompt loading.

CASA needs a small set (default 20) of clean prompts that elicit normal benign
responses.  They can come from a newline-delimited file or from the same builtin
datasets BAIT uses (Alpaca / Self-Instruct), formatted identically so CASA and
BAIT see the same prompts on the shared weakness-zoo test set.
"""

from __future__ import annotations

from typing import List, Optional

from casa.config import DataConfig
from casa.logging_utils import get_logger

_SEED = 42


def load_prompts(config: DataConfig) -> List[str]:
    """Load clean prompts per the data configuration.

    Args:
        config: A :class:`casa.config.DataConfig`.  If ``prompt_file`` is set it
            is read (one prompt per non-empty line); otherwise the builtin
            ``dataset`` loader runs.

    Returns:
        Up to ``config.prompt_size`` prompt strings.

    Raises:
        ValueError: If no prompts could be loaded.
    """
    log = get_logger()
    if config.prompt_file:
        prompts = _load_file(config.prompt_file, config.prompt_size)
        source = config.prompt_file
    else:
        prompts = _load_dataset(config.dataset, config.prompt_size, config.data_dir)
        source = f"dataset:{config.dataset}"
    if not prompts:
        raise ValueError(f"no prompts loaded from {source!r}")
    log.info("loaded %d clean prompts from %s", len(prompts), source)
    return prompts


def _load_file(path: str, n: int) -> List[str]:
    """Read up to ``n`` non-empty, non-comment lines from ``path``."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.strip() and not s.lstrip().startswith("#"):
                out.append(s)
            if len(out) >= n:
                break
    return out


def _load_dataset(name: str, n: int, data_dir: Optional[str]) -> List[str]:
    """Load BAIT-formatted prompts from a builtin HF dataset (val split)."""
    from datasets import load_dataset  # type: ignore

    if name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", cache_dir=data_dir)
        val = _val_split(ds["train"])
        return [
            r["text"].split("### Response:")[0] + "### Response: "
            for r in val.select(range(min(n, len(val))))
        ]
    if name == "self-instruct":
        ds = load_dataset("yizhongw/self_instruct", name="self_instruct", cache_dir=data_dir)
        val = _val_split(ds["train"])
        return [
            r["prompt"].split("Output:")[0] + "Output: "
            for r in val.select(range(min(n, len(val))))
        ]
    raise ValueError(f"unknown builtin dataset {name!r}; use a prompt_file instead")


def _val_split(train: object) -> object:
    """Reproduce BAIT's train/val/test split to share the same val prompts."""
    splits = train.train_test_split(test_size=0.2, seed=_SEED, shuffle=True)  # type: ignore[attr-defined]
    splits = splits["train"].train_test_split(test_size=0.1, seed=_SEED, shuffle=True)
    return splits["test"]
