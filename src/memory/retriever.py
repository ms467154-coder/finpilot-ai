"""Memory retrieval (Phase 13).

``retrieve_relevant_memories`` scores a user's memories against the current
message and returns only those above the relevance threshold. The backend is
swappable (``build_retriever``); the default is a lightweight keyword/token
overlap scorer so it is deterministic and dependency-free. If nothing is
sufficiently relevant, nothing is returned.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import store as memory_store
from .schemas import Memory

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "configs" / "memory_config.yaml"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "top_k": 3,
    "relevance_threshold": 0.18,
    "dedup_similarity_threshold": 0.5,
    "max_turns_in_context": 6,
    "default_user_id": "anonymous",
    "default_importance": 0.6,
    "retriever": "keyword",
}

# Function words that add no signal to the overlap score. Note: "for"/"a"
# are intentionally kept so goal phrases like "saving for a house" stay matchable.
_STOPWORDS = frozenset(
    {
        "the", "is", "are", "was", "were", "be", "been", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "their", "our", "what", "when", "where", "which", "who", "whom", "whose",
        "why", "how", "do", "does", "did", "can", "could", "will", "would", "should",
        "shall", "may", "might", "must", "to", "and", "or", "but", "if", "then", "than",
        "so", "such", "as", "of", "at", "by", "with", "from", "about", "into",
        "onto", "this", "that", "these", "those", "there", "here", "just", "very",
        "much", "many", "also", "not", "am", "an",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_memory_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the ``memory`` block of :file:`configs/memory_config.yaml`."""
    path = Path(path) if path else _CONFIG_PATH
    cfg = dict(_DEFAULT_CONFIG)
    if path.exists():
        import yaml

        raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("memory", {}) or {}
        cfg.update({key: value for key, value in raw.items() if value is not None})
    return cfg


def _stem(token: str) -> str:
    """Tiny morphological normalizer: saving/save, wants/want, invested/invest."""
    if len(token) <= 3:
        return token
    if token.endswith("ing") and len(token) > 5:
        root = token[:-3]
        if len(root) >= 2 and root[-1] == root[-2]:
            root = root[:-1]  # running -> run
        return root
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("e") and token[-2] not in "aeiouy":
        return token[:-1]  # save/saving -> sav, house -> hous
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase, strip function words, and lightly stem the input text."""
    tokens = []
    for token in _TOKEN_RE.findall((text or "").lower()):
        if token in _STOPWORDS or len(token) <= 1:
            continue
        tokens.append(_stem(token))
    return tokens


def token_similarity(a: str, b: str) -> float:
    """Token-overlap similarity (0-1), length-normalized like cosine."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    common = sum(1 for token in set(ta) if token in tb)
    return common / math.sqrt(len(ta) * len(tb))


def score_memory(memory: Memory, query_text: str) -> float:
    """Relevance of one memory to the query (content overlap + small category boost)."""
    score = token_similarity(memory.content, query_text)
    if memory.category and token_similarity(memory.category, query_text) > 0:
        score += 0.05
    return score


class KeywordMemoryRetriever:
    """Deterministic keyword/token-overlap retrieval (default backend)."""

    def __init__(self, threshold: float = 0.18, top_k: int = 3) -> None:
        self.threshold = threshold
        self.top_k = top_k

    def retrieve(self, memories: Sequence[Memory], query_text: str) -> List[Memory]:
        scored = sorted(
            ((score_memory(memory, query_text), memory) for memory in memories),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [memory for score, memory in scored if score >= self.threshold][: self.top_k]


def build_retriever(config: Dict[str, Any]) -> Any:
    """Factory so the retrieval backend is swappable via config."""
    threshold = config.get("relevance_threshold", 0.18)
    top_k = config.get("top_k", 3)
    kind = (config or {}).get("retriever", "keyword")
    if kind == "keyword":
        return KeywordMemoryRetriever(threshold=threshold, top_k=top_k)
    # Unknown kinds fall back to the default backend.
    return KeywordMemoryRetriever(threshold=threshold, top_k=top_k)


def retrieve_relevant_memories(
    user_id: str,
    query_text: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    retriever=None,
) -> List[Memory]:
    """Return the memories for ``user_id`` most relevant to ``query_text``.

    Honors the configured relevance threshold; returns ``[]`` when nothing is
    sufficiently relevant. Retrieved memories have ``last_accessed_at`` bumped.
    """
    cfg = load_memory_config()
    top_k = cfg["top_k"] if top_k is None else top_k
    threshold = cfg["relevance_threshold"] if threshold is None else threshold
    if retriever is None:
        retriever = build_retriever(cfg)

    store = memory_store.get_memory_store()
    memories = store.list_memories(user_id)
    results = retriever.retrieve(memories, query_text)
    for memory in results:
        store.touch_memory(memory.id)
    return results


__all__ = [
    "KeywordMemoryRetriever",
    "build_retriever",
    "load_memory_config",
    "retrieve_relevant_memories",
    "score_memory",
    "token_similarity",
    "tokenize",
]
