# Dataset Sources — Financial Advice Chatbot (Phase 1: Discovery)

This document lists the datasets selected for a **natural-language financial Q&A / advice chatbot**.
Selection was driven by three criteria evaluated in [`notebooks/01_dataset_discovery.ipynb`](../notebooks/01_dataset_discovery.ipynb):

1. **Right task** — natural-language *questions* with *answers / advice* (Q&A pairs), not labels or numbers.
2. **Right content** — personal finance, financial literacy, and financial concepts.
3. **Accessible** — downloadable without gating.

> **Explicit exclusion:** any dataset whose primary purpose is **numeric financial profiling**
> (salary / income / expense / net-worth / risk-score prediction). This is a Q&A chatbot, not an analytics tool.
> Candidates from that family are catalogued and rejected in the notebook (§5).

---

## Final selection (3 sources)

| # | Dataset | Kind | Size | License | Why it fits |
|---|---------|------|------|---------|-------------|
| 1 | [FinGPT/fingpt-fiqa_qa](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa) | General finance Q&A pairs | ~17.1k examples | unspecified (derived from FiQA-2018) | Real, community-written financial Q&A in instruction format; broad conceptual coverage (taxes, business expenses, investing, markets). |
| 2 | [Akhil-Theerthala/PersonalFinance_v2](https://huggingface.co/datasets/Akhil-Theerthala/PersonalFinance_v2) | Personal-finance advice / Q&A | ~7k examples | Apache-2.0 | Advice-style Q&A with structured, empathic answers (budgeting, debt, investing, retirement, insurance, taxes). Matches the chatbot's tone. |
| 3 | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) | Verifiable financial Q&A | 150 rows public (full: 10,231, on request) | CC-BY-NC-4.0 | Expert-verified Q&A with evidence strings; grounds answers in real documents (anti-hallucination training data). |

### Source 1 — FinGPT / fingpt-fiqa_qa
- **Origin:** FiQA-2018 *Question Answering* task (StackExchange — Quantitative Finance & Finance/Investing communities), reformatted by FinGPT into `instruction` / `input` / `output` triples.
- **Why chosen:** Provides genuine, varied financial questions and answers users actually ask (e.g. expense eligibility, small-business accounting, investing mechanics), which is exactly the Q&A surface a finance chatbot must handle. FiQA is the de-facto standard financial QA corpus.
- **Notes:** The `pauri32/fiqa-2018` repository's *default* config is the **sentiment** task (not QA) and is therefore listed, but excluded, in the catalog.

### Source 2 — Akhil-Theerthala / PersonalFinance_v2
- **Origin:** Curated advice Q&A from Reddit r/personalfinance with a `category`, `query`, `chain_of_thought`, `response` schema; tagged in Hugging Face's *Reasoning Datasets Competition*.
- **Why chosen:** The closest match to "conversational financial advice" — long-form, empathetic, step-by-step advice on debt vs investing, budgeting, retirement, insurance, taxes, and estate planning. The structured responses are ideal instruction-tuning targets.
- **Notes:** `chain_of_thought` field may be dropped or kept during Phase 2 formatting, depending on the tuning strategy.

### Source 3 — PatronusAI / financebench
- **Origin:** Expert-annotated `question` / `answer` / `evidence` triplets over US public-company filings (SEC 10-K/10-Q/8-K, earnings). Metadata includes `company`, `gics_sector`, `question_type`, `question_reasoning`, `doc_type`.
- **Why chosen:** Every answer ships with a verifiable evidence string. This teaches the model **grounded** answers (and supplies a natural evaluation set), directly addressing the hallucination risk that is critical in finance.
- **Notes:** The public Hub file contains **150 rows**; the full 10,231-row benchmark is distributed on request by the authors (see [patronus-ai/financebench](https://github.com/patronus-ai/financebench)). We treat it as a grounding/eval-augmentation source until the full release is granted. License is **CC-BY-NC-4.0** — non-commercial.

---

## Candidate datasets examined and excluded

| Dataset | Reason for exclusion |
|---------|----------------------|
| `pauri32/fiqa-2018` (default config) | Sentiment-analysis task, not Q&A (QA flavor covered by Source 1). |
| `financial_phrasebank` | Sentence-polarity classification; not Q&A or advice. |
| **Income/salary prediction family** (e.g. UCI Adults) | **Numeric income profiling; explicitly excluded.** |
| **Credit-risk / loan-default family** (Lending Club, German Credit) | **Numeric risk-profile prediction; explicitly excluded.** |
| **Expense / net-worth analytics family** | **Numeric spending/net-worth profiling; explicitly excluded.** |
| `DataDump1/personalfinance_reddit` | Gated on the Hub (login required); not usable unattended. |
| `BeIR/fiqa` | Retrieval-format (query/document) pairs; reserved for RAG evaluation, not answer generation. |

---

## Raw samples captured

One unmodified sample slice per source is stored under `data/raw/` (created by the notebook, §6):

- `data/raw/fingpt-fiqa-qa/sample.jsonl` — 300 rows
- `data/raw/personal-finance-v2/sample.jsonl` — 300 rows
- `data/raw/financebench/sample.jsonl` — 150 rows (full public file)

These are **samples only**; the full corpora are pulled from Hugging Face during the pipeline.
Cleaning, formatting, and training are intentionally deferred to later phases.