"""Training dataset utilities for the Financial Advice Chatbot (Phase 5).

Wraps the Phase 3 chat-v1 JSONL files (``data/processed/formatted/*.jsonl``)
into PyTorch datasets ready for causal-LM fine-tuning:

* each record becomes ``messages``; optionally a configured ``system_prompt`` is
  prepended unless the record already starts with a ``system`` turn;
* the user turns are rendered with the model's chat template and the assistant
  response is appended; **labels mask the prompt** (``-100``) so the loss is
  computed only over the assistant answer;
* sequences are truncated to ``max_length`` from the response side.

Single-turn oriented and built for the project's Qwen2.5 base model, but works
with any chat-template tokenizer.

Data shapes per item: ``input_ids``, ``attention_mask``, ``labels`` (tensors).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

DEFAULT_END_OF_TURN_CANDIDATES = ("<|im_end|>", "</s>", "<|endoftext|>")


# ---------------------------------------------------------------------------
# JSONL I/O and record helpers
# ---------------------------------------------------------------------------
def load_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a chat-v1 JSONL file into a list of records."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sample_records(records: Sequence[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    """Deterministic prefix slice for quick validation runs (``n=None`` = all)."""
    records = list(records)
    return records[:n] if n else records


def as_messages(record: Dict[str, Any], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Flatten a chat-v1 record into a ``[{role, content}]`` conversation.

    A configured ``system_prompt`` is prepended unless the record already opens
    with a ``system`` turn.
    """
    convo: List[Dict[str, str]] = [m for m in record.get("messages", []) if isinstance(m, dict)]
    if system_prompt and (not convo or convo[0].get("role") != "system"):
        convo = [{"role": "system", "content": system_prompt}, *convo]
    return convo


def _end_of_turn_token(tokenizer: PreTrainedTokenizerBase) -> str:
    """Pick a tokenizer-native 'end of assistant turn' marker (1 token piece)."""
    for cand in DEFAULT_END_OF_TURN_CANDIDATES:
        try:
            if cand and len(tokenizer.encode(cand, add_special_tokens=False)) == 1:
                return cand
        except Exception:  # noqa: BLE001 - tokenizer quirks must not break training
            continue
    return ""


# ---------------------------------------------------------------------------
# Tokenization (prompt-masked, response-truncated)
# ---------------------------------------------------------------------------
def tokenize_sample(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    max_length: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return ``(input_ids, attention_mask, labels)`` for one conversation.

    * prompt  = chat template of all but the final (assistant) turn + gen prompt;
    * response = tokenized assistant content (+ an end-of-turn marker);
    * labels place ``-100`` on every prompt token.
    """
    assistant = messages[-1]
    user_messages = messages[:-1]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
    else:  # fallback for tokenizers without a chat template
        prompt_text = "\n".join(m["content"] for m in user_messages) + "\n"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[: max_length - 1]

    end_tok = _end_of_turn_token(tokenizer)
    budget = max_length - len(prompt_ids)
    if budget <= 0:
        full_ids = prompt_ids
        labels = [-100] * len(prompt_ids)
    else:
        resp_text = assistant["content"] + (end_tok or "")
        resp_ids = tokenizer(
            resp_text, add_special_tokens=False, truncation=True, max_length=budget
        )["input_ids"]
        full_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids

    return full_ids, [1] * len(full_ids), labels


# ---------------------------------------------------------------------------
# PyTorch Dataset + collator
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Torch ``Dataset`` of tokenized chat-v1 records (input_ids/labels)."""

    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.examples: List[Dict[str, torch.Tensor]] = []
        for record in records:
            messages = as_messages(record, system_prompt=system_prompt)
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue  # skip malformed records (should not happen for chat-v1)
            input_ids, attention_mask, labels = tokenize_sample(
                tokenizer, messages, max_length=max_length
            )
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def build_sft_dataset(
    path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    system_prompt: Optional[str] = None,
    sample: Optional[int] = None,
) -> SFTDataset:
    """Load ``path`` (JSONL) and build an :class:`SFTDataset` from it."""
    records = sample_records(load_records(path), n=sample)
    return SFTDataset(records, tokenizer=tokenizer, max_length=max_length, system_prompt=system_prompt)


def make_collate_fn(
    pad_token_id: int,
    label_pad_token_id: int = -100,
    pad_to_multiple_of: int = 8,
):
    """Return a collator that right-pads a batch to a multiple of 8."""

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        if pad_to_multiple_of:
            max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of

        out = {}
        for key, pad_value in (
            ("input_ids", pad_token_id),
            ("attention_mask", 0),
            ("labels", label_pad_token_id),
        ):
            rows = []
            for b in batch:
                t = b[key]
                pad = torch.full((max_len - len(t),), pad_value, dtype=t.dtype)
                rows.append(torch.cat([t, pad]))
            out[key] = torch.stack(rows, dim=0)
        return out

    return collate


__all__ = [
    "SFTDataset",
    "as_messages",
    "build_sft_dataset",
    "load_records",
    "make_collate_fn",
    "sample_records",
    "tokenize_sample",
]