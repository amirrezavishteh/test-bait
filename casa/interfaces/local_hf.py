"""Local HuggingFace backend.

This mirrors BAIT's ``src/models/model.py::build_model`` loading path so that
CASA scans the *same* weakness-zoo checkpoints BAIT does: a base causal-LM
(loaded from a local cache) optionally wrapped with a PEFT/LoRA adapter, with a
left-padding tokenizer.  :meth:`LocalHFModel.from_zoo_dir` reads a
``weakness_zoo/models/id-WXXXX`` directory exactly like BAIT's dispatcher.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from casa.config import ModelConfig
from casa.interfaces.base import Generation, ModelInterface
from casa.logging_utils import get_logger

_DEFAULT_PAD_TOKEN = "[PAD]"
_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class LocalHFModel(ModelInterface):
    """A local HuggingFace causal-LM (optionally LoRA-adapted)."""

    def __init__(self, config: ModelConfig) -> None:
        """Load the base model, optional adapter and tokenizer.

        Args:
            config: Model configuration; ``name_or_path`` is the base model,
                ``adapter_path`` an optional PEFT adapter, ``cache_dir`` the
                base-model cache (BAIT's ``--cache-dir``).
        """
        self._cfg = config
        self._log = get_logger()
        self._name = config.adapter_path or config.name_or_path
        use_cuda = config.device.startswith("cuda") and torch.cuda.is_available()
        self._device = torch.device(f"cuda:{config.gpu}" if use_cuda else "cpu")
        dtype = _DTYPES.get(config.dtype, torch.float16)
        if not use_cuda:
            dtype = torch.float32  # fp16 matmul is unsupported on most CPUs

        self._log.info(
            "loading local model base=%s adapter=%s device=%s",
            config.name_or_path,
            config.adapter_path,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            cache_dir=config.cache_dir,
            padding_side="left",
            truncation_side="left",
        )
        # Pin to a single device (BAIT's meta-device fix) instead of "auto".
        device_map = {"": config.gpu} if use_cuda else None
        model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path,
            cache_dir=config.cache_dir,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self._ensure_pad_token(model)
        if config.adapter_path:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, config.adapter_path)
        if device_map is None:
            model = model.to(self._device)
        model.eval()
        self._model = model

    # -- zoo loading ------------------------------------------------------ #
    @classmethod
    def from_zoo_dir(
        cls,
        model_dir: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        gpu: int = 0,
        **overrides: object,
    ) -> "Tuple[LocalHFModel, Dict[str, object]]":
        """Load a BAIT weakness-zoo model directory.

        Reads ``<model_dir>/config.json`` (BAIT schema: ``attack``, ``label``,
        ``trigger``, ``target``, ``model_name_or_path``, ``dataset``) and loads
        the base model plus the ``<model_dir>/model`` LoRA adapter.

        Args:
            model_dir: Path to an ``id-WXXXX`` directory.
            cache_dir: Base-model cache (the zoo's ``base_models`` directory).
            device: Torch device family.
            gpu: GPU index.
            **overrides: Extra :class:`ModelConfig` field overrides.

        Returns:
            ``(model, zoo_config)`` where ``zoo_config`` is the parsed
            ``config.json`` (carrying the ground-truth ``label``/``target``).
        """
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as fh:
            zoo_config: Dict[str, object] = json.load(fh)
        adapter = os.path.join(model_dir, "model")
        cfg = ModelConfig(
            kind="local_hf",
            name_or_path=str(zoo_config["model_name_or_path"]),
            adapter_path=adapter if os.path.isdir(adapter) else None,
            cache_dir=cache_dir,
            device=device,
            gpu=gpu,
            **overrides,  # type: ignore[arg-type]
        )
        return cls(cfg), zoo_config

    def _ensure_pad_token(self, model: object) -> None:
        """Add a ``[PAD]`` token if the tokenizer lacks one (BAIT parity)."""
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": _DEFAULT_PAD_TOKEN})
            model.resize_token_embeddings(len(self._tokenizer))  # type: ignore[attr-defined]

    # -- capabilities ----------------------------------------------------- #
    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_logprobs(self) -> bool:
        return True

    # -- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        prefix: str = "",
        max_new_tokens: Optional[int] = None,
        with_logprobs: bool = False,
    ) -> List[Generation]:
        max_new = max_new_tokens or self._cfg.max_new_tokens
        texts = [p + prefix for p in prompts]
        out: List[Generation] = []
        bs = self._cfg.batch_size
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            out.extend(self._generate_chunk(chunk, max_new, with_logprobs))
        return out

    def _generate_chunk(
        self, chunk: Sequence[str], max_new: int, with_logprobs: bool
    ) -> List[Generation]:
        enc = self._tokenizer(
            list(chunk),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._cfg.max_new_tokens + 512,
        ).to(self._device)
        gen = self._model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=with_logprobs,
        )
        seqs = gen.sequences
        in_len = enc["input_ids"].shape[1]
        new_tokens = seqs[:, in_len:]
        results: List[Generation] = []
        logprobs_per_sample = self._extract_logprobs(gen, new_tokens) if with_logprobs else None
        for i in range(new_tokens.shape[0]):
            ids = new_tokens[i].tolist()
            text = self._tokenizer.decode(ids, skip_special_tokens=True)
            lp = logprobs_per_sample[i] if logprobs_per_sample is not None else None
            results.append(Generation(text=text, token_ids=ids, logprobs=lp))
        return results

    def _extract_logprobs(self, gen: object, new_tokens: "torch.Tensor") -> List[List[float]]:
        """Per-token logprob of each generated token from ``output_scores``."""
        scores = gen.scores  # type: ignore[attr-defined]
        out: List[List[float]] = [[] for _ in range(new_tokens.shape[0])]
        for step, step_scores in enumerate(scores):
            logp = torch.log_softmax(step_scores.float(), dim=-1)
            chosen = new_tokens[:, step]
            for i in range(new_tokens.shape[0]):
                tok = int(chosen[i].item())
                out[i].append(float(logp[i, tok].item()))
        return out

    # -- vocabulary / tokenisation --------------------------------------- #
    def get_vocabulary(self) -> List[Tuple[int, str]]:
        vocab = self._tokenizer.get_vocab()
        return sorted(((idx, tok) for tok, idx in vocab.items()), key=lambda x: x[0])

    def special_token_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for attr in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "sep_token_id",
            "cls_token_id",
            "mask_token_id",
        ):
            val = getattr(self._tokenizer, attr, None)
            if isinstance(val, int):
                ids.add(val)
        extra = getattr(self._tokenizer, "all_special_ids", None)
        if extra:
            ids.update(int(t) for t in extra)
        return ids

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)

    # -- white-box extras ------------------------------------------------- #
    @property
    def tokenizer(self) -> object:
        """The underlying HF tokenizer (for white-box / BAIT-parity use)."""
        return self._tokenizer

    @property
    def model(self) -> object:
        """The underlying HF model (for white-box / BAIT-parity use)."""
        return self._model
