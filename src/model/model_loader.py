"""Model-setup utilities for the Financial Advice Chatbot (Phase 4).

Loads the **base** pretrained language model and tokenizer that will be
instruction-tuned on the Phase 3 chat data in a later phase. This module owns:

* reading/merging :file:`configs/model_config.yaml` (base model name, precision,
  quantization, generation defaults);
* ``load_tokenizer()`` / ``load_model()`` / ``load_model_and_tokenizer()``;
* a small ``generate()`` helper used to validate a loaded model on a sample
  financial question with a single dummy forward pass.

Nothing here trains, evaluates, serves, or wires any backend/frontend: it is
deliberately limited to *loading* a base model + tokenizer.

Defaults mirror the YAML config but can be overridden per call, e.g.::

    import torch
    from src.model import model_loader

    tok = model_loader.load_tokenizer()
    model = model_loader.load_model(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        load_dtype="bfloat16",
        quantization="4bit",
        device="cuda",
    )
    print(model_loader.generate(model, tok, "What is the difference between a Roth IRA and a traditional IRA?"))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _ROOT / "configs" / "model_config.yaml"

# ---------------------------------------------------------------------------
# Default model config (overridden by the YAML `model` / `generation` blocks)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "cache_dir": "models/base",
        "trust_remote_code": False,
        "device": "auto",
        "precision": {
            "load_dtype": "auto",
            "quantization": None,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        "load_extra_kwargs": {},
    },
    "generation": {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": None,
    },
}

DTYPE_MAP: Dict[str, Optional[torch.dtype]] = {
    "auto": None,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def merge_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Deep-merge a (partial) model config over the defaults."""
    if not config:
        return DEFAULT_CONFIG
    merged = DEFAULT_CONFIG.copy()
    merged["model"] = {**DEFAULT_CONFIG["model"], **config.get("model", {})}
    merged["model"]["precision"] = {
        **DEFAULT_CONFIG["model"]["precision"],
        **config.get("model", {}).get("precision", {}),
    }
    merged["model"]["load_extra_kwargs"] = {
        **DEFAULT_CONFIG["model"]["load_extra_kwargs"],
        **config.get("model", {}).get("load_extra_kwargs", {}),
    }
    merged["generation"] = {
        **DEFAULT_CONFIG["generation"],
        **config.get("generation", {}),
    }
    return merged


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Read and merge :file:`configs/model_config.yaml`.

    * ``path=None`` uses the default repo location.
    * Missing/invalid files fall back to ``DEFAULT_CONFIG`` with a warning.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return merge_config(None)
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return merge_config(raw)


def resolve_device(device: Optional[str]) -> torch.device:
    """Map ``"auto"`` to ``cuda`` when available, else ``cpu``."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _resolve_path(cache_dir: Union[str, Path, None]) -> Optional[Union[str, Path]]:
    """Return an absolute cache_dir rooted at the project root."""
    if not cache_dir:
        return None
    p = Path(cache_dir)
    return str(p if p.is_absolute() else _ROOT / p)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(
    base_model: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    trust_remote_code: Optional[bool] = None,
    **kwargs: Any,
) -> AutoTokenizer:
    """Load the tokenizer for ``base_model``.

    Falls back to the configured base model / cache dir. Ensures a valid
    ``pad_token`` (using the EOS token when the tokenizer has none), which is
    required for batched training and generation later.
    """
    cfg = load_config()
    base_model = base_model or cfg["model"]["base_model"]
    cache_dir = _resolve_path(cache_dir if cache_dir is not None else cfg["model"]["cache_dir"])
    trust_remote_code = (
        cfg["model"]["trust_remote_code"] if trust_remote_code is None else trust_remote_code
    )

    from_pretrained_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    from_pretrained_kwargs = {k: v for k, v in from_pretrained_kwargs.items() if v is not None}

    tokenizer = AutoTokenizer.from_pretrained(base_model, **from_pretrained_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = getattr(tokenizer, "padding_side", None) or "right"
    return tokenizer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build_bnb_config(precision: Mapping[str, Any]) -> BitsAndBytesConfig:
    compute_dtype = DTYPE_MAP.get(
        str(precision.get("bnb_4bit_compute_dtype", "float16")).lower(), torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=precision.get("quantization") == "4bit",
        load_in_8bit=precision.get("quantization") == "8bit",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=precision.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=precision.get("bnb_4bit_use_double_quant", True),
    )


def load_model(
    base_model: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    load_dtype: Optional[str] = None,
    quantization: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """Load the base causal-LM ``AutoModelForCausalLM`` on the chosen device.

    Arguments (all optional) override the YAML config:

    * ``load_dtype`` -- ``"auto" | "float16" | "bfloat16" | "float32"``.
    * ``quantization`` -- ``None | "4bit" | "8bit"`` (bitsandbytes); requires
      GPU, so it is ignored (with a warning) on CPU-only machines.
    * ``device`` -- ``"auto" | "cuda" | "cpu"``.
    """
    cfg = load_config()
    model_cfg = cfg["model"]
    precision = model_cfg["precision"]

    base_model = base_model or model_cfg["base_model"]
    cache_dir = _resolve_path(cache_dir if cache_dir is not None else model_cfg["cache_dir"])
    load_dtype = (load_dtype or precision.get("load_dtype") or "auto").lower()
    quantization = quantization if quantization is not None else precision.get("quantization")
    device = device or model_cfg.get("device") or "auto"
    trust_remote_code = model_cfg["trust_remote_code"] if trust_remote_code is None else trust_remote_code

    dev = resolve_device(device)
    torch_dtype = DTYPE_MAP.get(load_dtype, DTYPE_MAP["auto"])

    if quantization in ("4bit", "8bit") and not torch.cuda.is_available():
        print(f"[model_loader] quantization={quantization!r} requested but no CUDA device "
              f"found; falling back to full precision on {dev}")
        quantization = None

    from_pretrained_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        from_pretrained_kwargs["dtype"] = torch_dtype
    if quantization in ("4bit", "8bit"):
        from_pretrained_kwargs["quantization_config"] = _build_bnb_config({**precision, "quantization": quantization})
    from_pretrained_kwargs.update(extra_kwargs or {})
    from_pretrained_kwargs.update(kwargs)
    from_pretrained_kwargs = {k: v for k, v in from_pretrained_kwargs.items() if v is not None}

    model = AutoModelForCausalLM.from_pretrained(base_model, **from_pretrained_kwargs)
    model = model.to(dev)
    return model


def load_model_and_tokenizer(config: Union[Mapping[str, Any], str, Path, None] = None):
    """Load both model and tokenizer from a config.

    ``config`` may be a ``dict`` (merged) or a path to a YAML file; ``None``
    uses the default :file:`configs/model_config.yaml`.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else merge_config(config)
    tokenizer = load_tokenizer()
    model = load_model()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Generation helper (used for validation + future fine-tuning checks)
# ---------------------------------------------------------------------------
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Apply the model's chat template to ``prompt`` and run one generation.

    Returns the **raw decoded output** (including any special tokens). This is
    intentionally a minimal helper for validating a freshly-loaded base model,
    not an inference service.
    """
    cfg = load_config()
    gen_cfg = {**cfg["generation"], **(generation_kwargs or {})}

    pad_token_id = gen_cfg.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_cfg["pad_token_id"] = pad_token_id

    device = device or resolve_device(cfg["model"].get("device", "auto"))

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:  # tokenizers without a chat template: plain prompt fallback
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    return tokenizer.decode(output_ids[0], skip_special_tokens=False)


@dataclass
class LoadedModelInfo:
    """Lightweight summary of a loaded base model (printed by the verify script)."""

    base_model: str
    device: str
    dtype: str
    num_params: int
    trainable_params: int

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.base_model}\n"
            f"  device = {self.device}\n"
            f"  dtype  = {self.dtype}\n"
            f"  params = {self.num_params:,} total / {self.trainable_params:,} trainable"
        )


def get_model_info(model: AutoModelForCausalLM, base_model: Optional[str] = None) -> LoadedModelInfo:
    """Summarize the loaded model's device, dtype, and parameter counts."""
    return LoadedModelInfo(
        base_model=base_model or getattr(model, "name_or_path", "?"),
        device=str(next(model.parameters()).device),
        dtype=str(next(model.parameters()).dtype),
        num_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "DTYPE_MAP",
    "LoadedModelInfo",
    "generate",
    "get_model_info",
    "load_config",
    "load_model",
    "load_model_and_tokenizer",
    "load_tokenizer",
    "merge_config",
    "resolve_device",
]