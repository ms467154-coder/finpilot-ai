"""Inference module for the Financial Advice Chatbot (Phase 7).

Loads the fine-tuned model (``models/fine_tuned/`` LoRA adapter on the base
model) and exposes a single conversational entry point::

    from src.inference import inference

    reply = inference.generate_advice("How can I save money?")

    # multi-turn chat:
    r1 = inference.generate_advice("What is compound interest?")
    r2 = inference.generate_advice(
        "How can a young person benefit from it?",
        conversation_history=[
            {"role": "user", "content": "What is compound interest?"},
            {"role": "assistant", "content": r1},
        ],
    )

The persona (see :mod:`src.inference.prompt_templates`) is a conversational
financial advisor that never asks the user for salary/income/expense figures.
This module is **not** an HTTP service - Phase 9 wires it into the backend.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import prompt_templates  # noqa: E402
from src.model import model_loader  # noqa: E402

logger = logging.getLogger(__name__)

_GEN_PARAM_KEYS = ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty")

DEFAULT_INFERENCE_CONFIG: Dict[str, Any] = {
    "adapter_dir": "models/fine_tuned/final",
    "system_prompt": None,
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "history_max_turns": 6,
}

_MODEL_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """Read the ``inference`` block of :file:`configs/model_config.yaml`.

    Falls back to the ``generation`` block and module defaults for any key
    missing from ``inference``.
    """
    cfg = model_loader.load_config()
    merged: Dict[str, Any] = {**DEFAULT_INFERENCE_CONFIG}
    merged.update({k: v for k, v in cfg.get("inference", {}).items() if v is not None})
    for key in _GEN_PARAM_KEYS:
        if key not in merged and key in cfg.get("generation", {}):
            merged[key] = cfg["generation"][key]
    return merged


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------
def load_inference_model(
    adapter_dir: Optional[Union[str, Path]] = None,
    load_dtype: Optional[str] = None,
    quantization: Optional[str] = None,
    force_reload: bool = False,
):
    """Load (and cache) the fine-tuned model + tokenizer.

    Loads the base model via :mod:`src.model.model_loader`, then attaches the
    LoRA adapter (or full fine-tuned weights) from ``adapter_dir``. When no
    adapter exists, evaluates the base model with a warning.

    Returns ``(model, tokenizer)``.
    """
    cfg = load_config()
    adapter_dir = adapter_dir or cfg["adapter_dir"]
    adapter = _resolve(adapter_dir)
    key = (str(adapter), load_dtype, quantization)

    if not force_reload and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = model_loader.load_tokenizer()
    base = model_loader.load_model(
        load_dtype=load_dtype or "bfloat16", quantization=quantization
    )

    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(adapter))
        logger.info("Loaded LoRA adapter -> %s", adapter)
    elif any((adapter / f).exists() for f in ("model.safetensors", "pytorch_model.bin")):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(adapter),
            dtype=torch.bfloat16,
            trust_remote_code=model_loader.load_config()["model"]["trust_remote_code"],
        ).to(base.device)
        logger.info("Loaded full fine-tuned weights -> %s", adapter)
    else:
        logger.warning("No adapter in %s; serving the base model.", adapter)
        model = base

    model.eval()
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _merge_generation_kwargs(
    cfg: Dict[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {k: cfg[k] for k in _GEN_PARAM_KEYS if k in cfg}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def generate_advice(
    user_message: str,
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    *,
    system_prompt: Optional[str] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    force_reload: bool = False,
) -> str:
    """Generate a conversational financial-advice reply.

    Args:
        user_message: the user's current question/message.
        conversation_history: prior ``{"role": "user"|"assistant", "content": ...}``
            turns so the model behaves like a chat, not single-shot Q&A.
        system_prompt: override the default financial-advisor persona (None = default).
        generation_kwargs: extra kwargs forwarded to ``model.generate``.
        max_new_tokens / temperature: quick overrides for the two most common knobs.
        force_reload: reload the model from disk.

    Returns:
        The advisor's reply text (prompt and role tokens stripped).
    """
    cfg = load_config()
    model, tokenizer = load_inference_model(force_reload=force_reload)

    history = prompt_templates.trim_history(conversation_history, cfg.get("history_max_turns"))
    system_prompt = system_prompt if system_prompt is not None else cfg.get("system_prompt")
    messages = prompt_templates.build_messages(user_message, history, system_prompt=system_prompt)

    prompt_text = prompt_templates.format_chat_prompt(tokenizer, messages)
    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_str = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)

    gen_kwargs = _merge_generation_kwargs(cfg, generation_kwargs, max_new_tokens, temperature)
    gen_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
    gen_kwargs.setdefault("use_cache", True)

    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)

    full = tokenizer.decode(out[0], skip_special_tokens=False)
    reply = full[len(prompt_str):].strip()
    for suffix in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if reply.endswith(suffix):
            reply = reply[: -len(suffix)].strip()
    return re.sub(r"\n{3,}", "\n\n", reply)


__all__ = [
    "DEFAULT_INFERENCE_CONFIG",
    "generate_advice",
    "load_config",
    "load_inference_model",
]