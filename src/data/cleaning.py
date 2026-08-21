"""Reusable data-cleaning utilities for the Financial Advice Chatbot (Phase 2).

Transforms the raw jsonl records in ``data/raw/`` into unified, cleaned text
records written to ``data/processed/cleaned/``.

Cleaning scope (no formatting to prompt/response pairs here — that is Phase 3):

* Unicode / encoding normalization (mojibake replacement chars, NFKC, control chars).
* Whitespace normalization (collapse runs, cap consecutive newlines).
* Noise removal (HTML tags, URLs).
* PII redaction (emails, phone numbers, SSNs, credit-card numbers, IPs).
* Quality filters (min/max lengths, alphabetic ratio, financial-content check).
* Numeric financial-profiling exclusion: records framed as salary/income/expense/
  net-worth/risk-score *prediction* are dropped (the product is a conversational
  Q&A chatbot, not a profiler).
* Deduplication (exact + configurable fuzzy-similarity).

All functions are dependency-free (standard library only) so the module can be
reused by pipelines and notebooks alike.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Default configuration (overridden by configs/data_config.yaml -> cleaning block)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "dedup": {"enabled": True, "threshold": 0.92, "use_fuzzy": True, "block_prefix": 8},
    "text": {
        "normalize_unicode": "NFKC",
        "collapse_whitespace": True,
        "max_newlines": 2,
        "strip_control_chars": True,
        "remove_html": True,
        "remove_urls": True,
    },
    "pii": {"redact": True, "placeholder": "bracket"},
    "filtering": {
        "min_question_chars": 15,
        "max_question_chars": 5000,
        "min_answer_words": 3,
        "max_answer_chars": 12000,
        "min_alpha_ratio": 0.55,
        "min_financial_hits": 1,
    },
    "profiling_exclusion": {"enabled": True},
    # Per-source overrides, applied on top of the generic blocks above.
    # - financebench answers are often terse numeric facts ("$1577.00").
    # - financebench is a curated, expert-verified finance benchmark: skip the
    #   lexical financial-content check entirely (every record is in-domain).
    "sources": {
        "financebench": {"filtering": {"min_answer_words": 1, "min_financial_hits": 0}},
    },
}

# ---------------------------------------------------------------------------
# Source schema mapping: canonical `question` / `answer` field for each source.
# ---------------------------------------------------------------------------
FIELD_MAP: Dict[str, Dict[str, str]] = {
    "fingpt-fiqa-qa": {"question": "input", "answer": "output"},
    "personal-finance-v2": {"question": "query", "answer": "response"},
    "financebench": {"question": "question", "answer": "answer"},
}

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_RE_HTML_TAG = re.compile(r"<[^>]{1,256}>")
_RE_URL = re.compile(r"https?://\S+", re.I)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b")
_RE_PHONE = re.compile(r"\b(?:\+?1[-. ]?)?(?:\(?\d{3}\)?[-. ])?\d{3}[-. ]\d{4}\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 15-16 digits (typical card length) to avoid accession/document numbers.
_RE_CARD = re.compile(r"\b(?:\d[ -]?){15,16}\b")
_RE_IP = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MULTI_WS = re.compile(r"[ \t]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")
# Irreversible U+FFFD replacement characters (lossy-encoded smart quotes/dashes).
_RE_REPLACEMENT = re.compile(r"\ufffd")
_RE_WORD_REPLACEMENT = re.compile(r"(?<=[A-Za-z])\ufffd(?=[A-Za-z])")
_RE_ALNUM = re.compile(r"[A-Za-z0-9]")
_RE_LETTER = re.compile(r"[A-Za-z]")

# ---------------------------------------------------------------------------
# Financial-content lexicon (single hit keeps a record as "financial").
# ---------------------------------------------------------------------------
FINANCIAL_LEXICON = frozenset(
    w.lower().strip()
    for w in """
    money bank banking loan debt credit mortgage rent budget saving savings savings%
    invest investing investment investor stock stocks bond bonds etf esg mutual fund
    index fund dividend dividends interest rate apr apy yield return returns annualized
    tax taxes taxable tax-exempt deduction deduction% ira 401k 401(k) 403b pension
    retirement retire retiree roth 401(k) roth ira traditional ira social security
    insurance premium premium% deduct copay claim claims deductible coverage policy
    cpa accountant bookkeeping bookkeeper small business business self-employed entrepreneur
    industry sector segment market geographies governance board shareholder shareholders proxy
    salary wage wages income earnings earning revenue profit losses loss margin ebitda
    cash flow cash-flow liquidity asset assets liability liabilities net worth equity
    portfolio diversification diversify compound compounding inflation deflation recession
    gdp currency exchange rate forex foreign exchange american depositary credit card
    checking account savings account cd certificate of deposit treasury bond government
    bond bond yield risk risk-free return capital gains capital loss short-term long-term
    lump sum dollar-cost averaging market volatility brokerage broker hedge hedging option
    options futures futures margin short selling dividend yield price-to-earnings p/e
    price to book return on equity gaap sec filing 10-k 10-q 8-k annual report
    earnings report earnings call guidance revenue market cap market capitalization
    balance sheet income statement cash flow statement stockholders equity goodwill
    goodwill impairment impairment intangible assets property plant equipment capex
    capital expenditure expenditure expenditures operating expenses operating margin
    gross margin net margin net sales gross profit cost of goods sold cogs inventories
    accounts receivable long-term debt current assets cash and equivalents
    share repurchase common stock diluted earnings per share revenue growth
    bankruptcy liquidation foreclosure repossession debt collector collections fraud
    identity theft scammers scam phishing ponzi pyramid scheme financial advisor cfp
    financial planner fiduciary emergency fund student loan refinance consolidate
    amortization depreciation appreciation bull bear correction rally crash bubble
    profile portfolio risk profile risk tolerance time horizon asset allocation
    $ usd eur gbp jpy cad
    """.split()
    if w
)

# ---------------------------------------------------------------------------
# Numeric financial-profiling detection (the explicit product exclusion).
#
# Only *measurement/prediction* tasks are excluded, e.g. "predict my salary",
# "estimate my net worth", "what is my credit score?". Advice questions that
# merely *mention* salary/income/credit as context (the bulk of personal-finance
# forums) do NOT match and are kept.
# ---------------------------------------------------------------------------
_PROFILE_VERB_NOUN = re.compile(
    r"\b(?:predict|forecast|estimate|calculate|compute|determine|assess|classify)\w*\b"
    r"[\w\s',.()-]{0,15}?"
    r"\b(?:my|your|our|their|the\s+consumer'?s?|the\s+user'?s?)\b\s+"
    r"(?:annual|monthly|yearly|weekly|total|average|current|expected|future)?\s*"
    r"\b(salary|wages?|income|earnings?|expenses?|spending|net\s+worth|risk\s+score|credit\s+score|debt.to.income)\b",
    re.I,
)
_RE_WHAT_MY_ATTR = re.compile(
    r"\bwhat(?:'s| is| are| would be| will be| does| do)?\s+(?:my|your|our)\s+"
    r"(?:annual\s+)?(salary|wages?|income|earnings?|expenses?|net\s+worth|risk\s+score|credit\s+score|monthly\s+spending)\b",
    re.I,
)
_RE_HOW_MUCH_WORTH = re.compile(
    r"\bhow much (?:am i|are you|is my|is your) (?:worth|earning|making|net\s+worth)\b", re.I,
)


def is_profiling_record(question: str, answer: Optional[str] = None) -> bool:
    """True when the record is framed as a *numeric personal* financial-profile
    task (predicting salary / income / expenses / net worth / risk score).

    Company-level questions (e.g. FinanceBench "3M's capital expenditure") carry
    no personal frame and are therefore kept.
    """
    if not question:
        return False
    return bool(
        _PROFILE_VERB_NOUN.search(question)
        or _RE_WHAT_MY_ATTR.search(question)
        or _RE_HOW_MUCH_WORTH.search(question)
    )


# ---------------------------------------------------------------------------
# Small stat accumulator
# ---------------------------------------------------------------------------
@dataclass
class CleaningStats:
    source: str
    n_before: int = 0
    n_after: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    encoding_normalized: int = 0
    pii_redactions: int = 0
    exact_duplicates: int = 0
    same_question_duplicates: int = 0
    fuzzy_duplicates: int = 0
    profiling_removed: int = 0

    def tally(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def source_config(source: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Effective config for one source (per-source overrides applied on top)."""
    overrides = (cfg.get("sources") or {}).get(source)
    if not overrides:
        return cfg
    return _apply_overrides(cfg, overrides)


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return _merge(cfg, overrides)


# ---------------------------------------------------------------------------
# Low-level normalization helpers
# ---------------------------------------------------------------------------
def merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a (partial) config over the defaults, returning the full config."""
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


# Typographic punctuation -> ASCII (NFKC alone does not decompose these).
_PUNCT_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201a": "'",  # single low-9 quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u201e": '"',  # double low-9 quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",  # non-breaking space
    }
)


def fix_replacement_chars(text: str) -> str:
    """Replace U+FFFD between letters with apostrophe; drop stray U+FFFD."""
    text = _RE_WORD_REPLACEMENT.sub("'", text)
    return _RE_REPLACEMENT.sub("", text)


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """NFKC normalization + smart-quote/dash/ellipsis mapping to ASCII."""
    return unicodedata.normalize(form, text).translate(_PUNCT_MAP)


def strip_html(text: str) -> str:
    return _RE_HTML_TAG.sub(" ", text)


def strip_urls(text: str) -> str:
    return _RE_URL.sub(" ", text)


def strip_control_chars(text: str) -> str:
    return _RE_CONTROL.sub("", text)


def collapse_whitespace(text: str, max_newlines: int = 2) -> str:
    text = _RE_MULTI_WS.sub(" ", text)
    text = _RE_MULTI_NL.sub("\n" * max_newlines, text)
    return text.strip()


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
def redact_pii(text: str, placeholder: str = "bracket", count: bool = False) -> str:
    """Redact emails/phones/SSNs/cards/IPs with `[KIND]` placeholders."""
    rules = [
        (_RE_EMAIL, "EMAIL"),
        (_RE_SSN, "SSN"),
        (_RE_PHONE, "PHONE"),
        (_RE_CARD, "CARD_NUMBER"),
        (_RE_IP, "IP"),
    ]
    n = 0
    for pattern, kind in rules:
        found = pattern.findall(text)
        if found:
            n += len(found)
            text = pattern.sub(f"[{kind}]", text)
    return (text, n) if count else text


# ---------------------------------------------------------------------------
# Full pipeline pieces
# ---------------------------------------------------------------------------
def normalize_text(
    text: Optional[str],
    cfg: Dict[str, Any],
    light: bool = False,
) -> str:
    """Normalize a single text field according to the config's text/pii blocks."""
    if not text:
        return ""
    text = fix_replacement_chars(text)
    text = normalize_unicode(text, cfg["text"]["normalize_unicode"])
    if cfg["text"]["strip_control_chars"]:
        text = strip_control_chars(text)
    if cfg["text"]["remove_html"]:
        text = strip_html(text)
    if cfg["text"]["remove_urls"]:
        text = strip_urls(text)
    if cfg["pii"]["redact"]:
        text = redact_pii(text, placeholder=cfg["pii"]["placeholder"])
    if cfg["text"]["collapse_whitespace"]:
        if light:
            # keep newlines (structured aux text), just join spaces line-internally
            text = collapse_whitespace(text, max_newlines=cfg["text"]["max_newlines"])
        else:
            text = re.sub(r"\s+", " ", text).strip()
    return text


def count_replacement_chars(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(_RE_REPLACEMENT.findall(text))


def encoding_normalized_count(text: Optional[str]) -> int:
    """Count characters rewritten by NFKC or the punctuation mapping (smart
    quotes, dashes, ligatures, etc.) — the effective "encoding normalization"."""
    if not text:
        return 0
    return sum(
        1
        for c in text
        if unicodedata.normalize("NFKC", c) != c or c in _PUNCT_MAP
    )


def alpha_ratio(text: str) -> float:
    """Fraction of alphabetic characters among non-whitespace characters."""
    if not text:
        return 0.0
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    return len(_RE_LETTER.findall(non_ws)) / len(non_ws)


def financial_hits(text: str) -> int:
    """Count distinct financial-lexicon tokens present in the text."""
    if not text:
        return 0
    tokens = set(re.split(r"[^A-Za-z0-9$.-]+", text.lower()))
    return len(tokens & FINANCIAL_LEXICON)


def is_financial_record(question: str, answer: str, min_hits: int = 1) -> bool:
    """True when the record contains at least one financial-lexicon term."""
    return financial_hits(question) + financial_hits(answer) >= min_hits


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def normalized_question(rec: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(rec.get("question", "")).lower().strip())


def normalized_answer(rec: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(rec.get("answer", "")).lower().strip())


def deduplicate(
    records: List[Dict[str, Any]],
    threshold: float = 0.92,
    use_fuzzy: bool = True,
    block_prefix: int = 8,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Remove duplicates, categorised as:

    * **exact** — same normalized question *and* same normalized answer,
    * **same-question** — identical normalized question with a different answer
      (keep the first answer for a given question),
    * **fuzzy** — near-duplicate question (SequenceMatcher ratio >= ``threshold``,
      compared within the same ``block_prefix`` bucket; first occurrence wins).

    Returns ``(kept, n_exact, n_same_question, n_fuzzy)``.
    """
    sig_seen: set[str] = set()
    question_seen: set[str] = set()
    fuzzy_blocks: Dict[str, List[str]] = {}
    kept: List[Dict[str, Any]] = []
    n_exact = n_same = n_fuzzy = 0

    for rec in records:
        q = normalized_question(rec)
        a = normalized_answer(rec)
        if not q:
            continue
        sig = f"{q}\x00{a}"
        if sig in sig_seen:
            n_exact += 1
            continue
        if q in question_seen:
            n_same += 1
            continue
        if use_fuzzy:
            block = q[:block_prefix]
            if any(
                SequenceMatcher(None, q, other).ratio() >= threshold
                for other in fuzzy_blocks.get(block, [])
            ):
                n_fuzzy += 1
                continue
        sig_seen.add(sig)
        question_seen.add(q)
        fuzzy_blocks.setdefault(q[:block_prefix], []).append(q)
        kept.append(rec)
    return kept, n_exact, n_same, n_fuzzy


# ---------------------------------------------------------------------------
# Per-source record parsing + cleaning
# ---------------------------------------------------------------------------
def parse_record(raw: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """Map a raw record to the unified {id, source, question, answer, aux} shape."""
    mapping = FIELD_MAP.get(source)
    if not mapping:
        return None
    q_field, a_field = mapping["question"], mapping["answer"]
    question_raw = raw.get(q_field)
    answer_raw = raw.get(a_field)
    if not isinstance(question_raw, str) or not isinstance(answer_raw, str):
        return None
    aux = {k: v for k, v in raw.items() if k not in (q_field, a_field)}
    return {
        "question": question_raw,
        "answer": answer_raw,
        "aux": aux,
    }


def clean_record(
    raw: Dict[str, Any],
    source: str,
    cfg: Dict[str, Any],
    rec_idx: int,
    stats: CleaningStats,
) -> Optional[Dict[str, Any]]:
    """Clean one raw record, or return None with a tally if it must be dropped."""
    cfg = source_config(source, cfg)
    parsed = parse_record(raw, source)
    if parsed is None:
        stats.tally("unparseable")
        return None

    stats.encoding_normalized += encoding_normalized_count(parsed["question"]) + encoding_normalized_count(
        parsed["answer"]
    )
    question_raw = fix_replacement_chars(parsed["question"])
    answer_raw = fix_replacement_chars(parsed["answer"])
    if cfg["pii"]["redact"]:
        _, nq = redact_pii(question_raw, count=True)
        _, na = redact_pii(answer_raw, count=True)
        stats.pii_redactions += nq + na

    question = normalize_text(parsed["question"], cfg)
    answer = normalize_text(parsed["answer"], cfg)

    # --- quality / length filters -------------------------------------
    if len(question) < cfg["filtering"]["min_question_chars"]:
        stats.tally("question_too_short")
        return None
    if len(question) > cfg["filtering"]["max_question_chars"]:
        stats.tally("question_too_long")
        return None
    if len(answer) < 1 or len(answer.split()) < cfg["filtering"]["min_answer_words"]:
        stats.tally("answer_too_short")
        return None
    if len(answer) > cfg["filtering"]["max_answer_chars"]:
        stats.tally("answer_too_long")
        return None
    if alpha_ratio(question) < cfg["filtering"]["min_alpha_ratio"]:
        stats.tally("low_alpha_ratio")
        return None

    # --- financial-content check ----------------------------------------
    if not is_financial_record(
        question, answer, cfg["filtering"]["min_financial_hits"]
    ):
        stats.tally("non_financial")
        return None

    # --- numeric personal-financial profiling exclusion -----------------
    if cfg["profiling_exclusion"]["enabled"] and is_profiling_record(question, answer):
        stats.profiling_removed += 1
        stats.tally("profiling")
        return None

    # --- aux (non-question/non-answer) fields: light normalization -------
    aux: Dict[str, Any] = {}
    for k, v in parsed["aux"].items():
        if isinstance(v, str):
            aux[k] = normalize_text(v, cfg, light=True)
        else:
            aux[k] = v

    return {
        "id": f"{source}-{rec_idx:04d}",
        "source": source,
        "question": question,
        "answer": answer,
        "aux": aux,
    }


# ---------------------------------------------------------------------------
# File-level orchestration
# ---------------------------------------------------------------------------
def clean_source(
    in_path: Path,
    out_path: Path,
    source: str,
    cfg: Dict[str, Any] = None,
    rows: Optional[int] = None,
) -> CleaningStats:
    """Clean one jsonl file into a unified jsonl of cleaned records."""
    cfg = merge_config(cfg)
    raw_records: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))
    if rows is not None:
        raw_records = raw_records[:rows]

    stats = CleaningStats(source=source, n_before=len(raw_records))

    cleaned: List[Dict[str, Any]] = []
    for idx, rec in enumerate(raw_records):
        out = clean_record(rec, source, cfg, idx, stats)
        if out is not None:
            cleaned.append(out)

    # deduplication (exact + same-question + fuzzy) over the cleaned stream
    if cfg["dedup"]["enabled"]:
        cleaned, n_exact, n_same, n_fuzzy = deduplicate(
            cleaned,
            threshold=cfg["dedup"]["threshold"],
            use_fuzzy=cfg["dedup"]["use_fuzzy"],
            block_prefix=cfg["dedup"]["block_prefix"],
        )
        stats.exact_duplicates = n_exact
        stats.same_question_duplicates = n_same
        stats.fuzzy_duplicates = n_fuzzy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in cleaned:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats.n_after = len(cleaned)
    return stats


def clean_corpus(
    raw_dir: Path,
    out_dir: Path,
    sources: Iterable[str],
    cfg: Dict[str, Any] = None,
    rows: Optional[int] = None,
) -> List[CleaningStats]:
    """Clean all listed sources. Returns a stats object per source."""
    cfg = merge_config(cfg)
    all_stats: List[CleaningStats] = []
    for source in sources:
        in_path = raw_dir / source / "sample.jsonl"
        out_path = out_dir / f"{source}.jsonl"
        all_stats.append(clean_source(in_path, out_path, source, cfg, rows=rows))
    return all_stats