"""Reusable dataset-formatting utilities for the Financial Advice Chatbot (Phase 3).

Transforms the unified cleaned records in ``data/processed/cleaned/`` into the
final **chat-turn (messages) JSONL** format used for instruction fine-tuning, and
splits them into ``train`` / ``val`` / ``test`` files.

Output schema (``formatting.format.schema: "chat-v1"``)::

    {
      "id": "<source>-0000",
      "source": "<source>",
      "messages": [
        {"role": "user", "content": "<financial question>"},
        {"role": "assistant", "content": "<advice-style answer>"}
      ]
    }

* An optional ``system`` message is prepended when
  ``formatting.format.system_prompt`` is set.
* Splitting is deterministic (fixed seed) and, by default, **stratified by
  source** so every split inherits a proportional slice of each source dataset.
* No model / training code lives here (this is a data-pipeline module).

All functions are dependency-free (standard library only).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Default formatting configuration (overridden by the YAML `formatting` block)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "format": {
        "schema": "chat-v1",
        "system_prompt": None,  # optional persona string prepended to every record
    },
    "split": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "seed": 42,
        "stratify_by_source": True,
    },
}

SUPPORTED_ROLES = ("system", "user", "assistant")
SPLIT_NAMES = ("train", "val", "test")


def merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge a (partial) formatting config over the defaults."""
    if not config:
        return DEFAULT_CONFIG

    def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return _merge(DEFAULT_CONFIG, config)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def to_chat_messages(
    question: str,
    answer: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the ``messages`` list for a single Q&A pair."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return messages


def format_record(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one cleaned record to the configured output schema."""
    question = str(rec.get("question", "")).strip()
    answer = str(rec.get("answer", "")).strip()
    if not question or not answer:
        return None

    schema = cfg["format"]["schema"]
    system_prompt = cfg["format"].get("system_prompt")

    if schema == "chat-v1":
        return {
            "id": rec.get("id"),
            "source": rec.get("source"),
            "messages": to_chat_messages(question, answer, system_prompt),
        }
    raise ValueError(f"Unsupported format schema: {schema!r}")


def read_cleaned(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Splitting (deterministic, stratified by source)
# ---------------------------------------------------------------------------
def _allocate(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Split ``n`` into (train, val, test) exactly matching the given ratios."""
    p_train, p_val, p_test = ratios
    n_train = int(round(n * p_train))
    n_rest = n - n_train
    denom = p_val + p_test or 1.0
    n_val = int(round(n_rest * p_val / denom))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def split_records(
    records: Sequence[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify_by_source: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train/val/test split. Preserves in-group shuffle per split."""
    p_train, p_val, p_test = ratios
    if abs((p_train + p_val + p_test) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    trains, vals, tests = [], [], []

    groups: Dict[Any, List[Dict[str, Any]]]
    if stratify_by_source:
        groups = {}
        for rec in records:
            groups.setdefault(rec.get("source"), []).append(rec)
    else:
        groups = {"__all__": list(records)}

    for group in groups.values():
        rng.shuffle(group)
        t, v, te = _allocate(len(group), ratios)
        trains.extend(group[:t])
        vals.extend(group[t : t + v])
        tests.extend(group[t + v :])

    return trains, vals, tests


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_record(rec: Dict[str, Any], schema: str = "chat-v1") -> List[str]:
    """Return a list of schema violations (empty list == valid)."""
    errors: List[str] = []
    for key in ("id", "source", "messages"):
        if key not in rec:
            errors.append(f"missing key '{key}'")

    if "messages" in rec:
        messages = rec["messages"]
        if not isinstance(messages, list) or not messages:
            errors.append("'messages' must be a non-empty list")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"messages[{i}] is not a dict")
                    continue
                if msg.get("role") not in SUPPORTED_ROLES:
                    errors.append(f"messages[{i}] has invalid role {msg.get('role')!r}")
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"messages[{i}] has empty content")
            if messages and messages[-1].get("role") != "assistant":
                errors.append("last message role must be 'assistant'")
            if not any(m.get("role") == "user" for m in messages if isinstance(m, dict)):
                errors.append("messages must contain a 'user' turn")
    return errors


def validate_file(path: Path, schema: str = "chat-v1") -> Tuple[int, int, List[str]]:
    """Validate every record in a jsonl file. Returns (n_valid, n_invalid, sample_errors)."""
    n_valid = n_invalid = 0
    bad: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        errs = validate_record(rec, schema)
        if errs:
            n_invalid += 1
            if len(bad) < 5:
                bad.append(f"{rec.get('id')}: {'; '.join(errs)}")
        else:
            n_valid += 1
    return n_valid, n_invalid, bad


# ---------------------------------------------------------------------------
# Aggregated per-split statistics
# ---------------------------------------------------------------------------
@dataclass
class FormatStats:
    total: int = 0
    formatted: int = 0
    dropped: int = 0
    splits: Dict[str, int] = field(default_factory=dict)
    per_source: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def as_frame(self) -> "Any":
        """Return a pandas DataFrame (imported lazily for the notebook)."""
        import pandas as pd  # noqa: PLC0415

        sources = sorted(self.per_source)
        rows = []
        for src in sources:
            rows.append({"source": src, **{k: self.per_source[src].get(k, 0) for k in SPLIT_NAMES}})
        rows.append({"source": "TOTAL", **{k: self.splits.get(k, 0) for k in SPLIT_NAMES}})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------
def format_corpus(
    cleaned_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any] = None,
) -> FormatStats:
    """Read all cleaned jsonl files, format to chat-v1, split, and write the
    train/val/test files into ``out_dir``."""
    cfg = merge_config(cfg)
    clean_files = sorted(cleaned_dir.glob("*.jsonl"))

    records: List[Dict[str, Any]] = []
    for path in clean_files:
        for rec in read_cleaned(path):
            formatted = format_record(rec, cfg)
            if formatted:
                records.append(formatted)

    stats = FormatStats(total=len(records), formatted=len(records), dropped=0)

    split_cfg = cfg["split"]
    trains, vals, tests = split_records(
        records,
        ratios=(split_cfg["train"], split_cfg["val"], split_cfg["test"]),
        seed=split_cfg["seed"],
        stratify_by_source=split_cfg["stratify_by_source"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    split_artifacts = {"train": trains, "val": vals, "test": tests}
    for name, split_recs in split_artifacts.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for rec in split_recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats.splits[name] = len(split_recs)

    per_source: Dict[str, Dict[str, int]] = {s: {k: 0 for k in SPLIT_NAMES} for s in set(r["source"] for r in records)}
    for name, recs in split_artifacts.items():
        for rec in recs:
            per_source[rec["source"]][name] += 1
    stats.per_source = per_source

    return stats