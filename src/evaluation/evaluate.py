"""Model evaluation pipeline for the Financial Advice Chatbot (Phase 6).

Evaluates the **fine-tuned** model (``models/fine_tuned/final`` adapter on top of
the base model) against the held-out ``test.jsonl`` split plus curated prompts
in ``data/evaluation/`` using advice-generation metrics (see
:mod:`src.evaluation.metrics`).

Outputs:

* ``outputs/evaluation/results.json`` — per-example metrics + question/answer,
  aggregate statistics, and a per-category breakdown;
* ``outputs/evaluation/results.csv`` — flattened per-example table;
* ``outputs/evaluation/summary.txt`` — compact aggregate + category table;
* ``outputs/plots/*.png`` — score distributions and category breakdowns.

This module does **not** serve inference, and does not compare against numeric
financial-correctness targets (salary/income/risk profiling is out of scope).
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import metrics  # noqa: E402
from src.model import model_loader  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = _ROOT / "configs" / "eval_config.yaml"

NUMERIC_METRICS = [
    "word_count",
    "char_count",
    "length_adequacy",
    "repetition_score",
    "punctuation_score",
    "coherence",
    "keyword_coverage",
    "relevance",
    "rubric",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "evaluation": {
        "adapter_dir": "models/fine_tuned/final",
        "base_precision": "bfloat16",
        "quantization": None,
        "test_file": "data/processed/formatted/test.jsonl",
        "curated_dir": "data/evaluation",
        "curated_sources": ["prompts"],
        "output_dir": "outputs/evaluation",
        "plots_dir": "outputs/plots",
        "results_file": "results.json",
        "results_csv": "results.csv",
        "max_new_tokens": 200,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "metrics": ["length", "coherence", "keyword_coverage", "relevance", "rubric"],
        "enable_llm_judge": False,
        "llm_judge_max_new_tokens": 64,
    },
}


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Read + merge :file:`configs/eval_config.yaml` over the defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg: Dict[str, Any] = DEFAULT_CONFIG.copy()
    cfg["evaluation"] = {**DEFAULT_CONFIG["evaluation"]}
    if path.exists():
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg["evaluation"].update(raw.get("evaluation", {}))
    return cfg


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_eval_examples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assemble evaluation examples from test.jsonl + curated prompt files."""
    ev = cfg["evaluation"]
    examples: List[Dict[str, Any]] = []

    test_path = _resolve(ev["test_file"])
    if test_path.exists():
        for rec in _iter_jsonl(test_path):
            msgs = [m for m in rec.get("messages", []) if m.get("role") == "user"]
            if not msgs:
                continue
            question = msgs[0]["content"]
            gold = next(
                (m["content"] for m in rec.get("messages", []) if m.get("role") == "assistant"),
                None,
            )
            examples.append(
                {
                    "id": rec.get("id", "test-?"),
                    "source": "test",
                    "category": metrics.categorize(question),
                    "question": question,
                    "gold": gold,
                }
            )

    curated = _resolve(ev["curated_dir"])
    stems = ev.get("curated_sources") or ["prompts"]
    for stem in stems:
        p = curated / f"{stem}.jsonl"
        if not p.exists():
            logger.warning("Missing curated file %s (skipping)", p)
            continue
        for rec in _iter_jsonl(p):
            questions = rec.get("category") or metrics.categorize(rec["question"])
            examples.append(
                {
                    "id": rec.get("id", f"{stem}-?"),
                    "source": rec.get("source", "curated"),
                    "category": questions,
                    "question": rec["question"],
                    "gold": rec.get("gold"),
                }
            )
    return examples


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Model + generation
# ---------------------------------------------------------------------------
def load_eval_model(cfg: Dict[str, Any]):
    """Load tokenizer, fine-tuned model (base + adapter), and untuned judge base.

    Returns ``(model, tokenizer, judge_base)``.
    """
    ev = cfg["evaluation"]
    tokenizer = model_loader.load_tokenizer()
    base = model_loader.load_model(
        load_dtype=ev["base_precision"], quantization=ev.get("quantization")
    )

    adapter = _resolve(ev["adapter_dir"])
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
        logger.warning("No adapter found in %s; evaluating the base model.", adapter)
        model = base

    model.eval()
    return model, tokenizer, base


def generate_answer(
    model, tokenizer, question: str, cfg: Dict[str, Any]
) -> str:
    """Generate a chat-templated answer and strip the prompting/format tokens."""
    ev = cfg["evaluation"]
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = question

    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_str = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)

    gen_kwargs = {
        "max_new_tokens": int(ev["max_new_tokens"]),
        "do_sample": bool(ev["do_sample"]),
        "temperature": float(ev["temperature"]),
        "top_p": float(ev["top_p"]),
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        out = model.generate(**enc, use_cache=True, **gen_kwargs)

    full = tokenizer.decode(out[0], skip_special_tokens=False)
    answer = full[len(prompt_str):].strip()
    for suffix in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if answer.endswith(suffix):
            answer = answer[: -len(suffix)].strip()
    return re.sub(r"\n{3,}", "\n\n", answer)


# ---------------------------------------------------------------------------
# Scoring + aggregation
# ---------------------------------------------------------------------------
def score_example(
    ex: Dict[str, Any],
    cfg: Dict[str, Any],
    model,
    tokenizer,
    judge_model=None,
) -> Dict[str, Any]:
    """Generate an answer for ``ex`` with the fine-tuned model and score it."""
    row: Dict[str, Any] = {
        "id": ex["id"],
        "source": ex["source"],
        "category": ex["category"],
        "question": ex["question"],
        "gold": ex.get("gold"),
    }
    row["generated"] = generate_answer(model, tokenizer, ex["question"], cfg)
    row.update(metrics.compute_metrics(ex["question"], row["generated"]))

    if cfg["evaluation"]["enable_llm_judge"] and judge_model is not None:
        j = metrics.llm_judge_score(
            ex["question"], row["generated"], judge_model, tokenizer,
            max_new_tokens=cfg["evaluation"]["llm_judge_max_new_tokens"],
        )
        row["llm_judge"] = j
    return row


def aggregate_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean/median/stdev/min/max of each numeric metric across the rows."""
    agg: Dict[str, Dict[str, float]] = {}
    for name in NUMERIC_METRICS:
        vals = [float(r[name]) for r in rows if r.get(name) is not None]
        if not vals:
            continue
        agg[name] = {
            "count": len(vals),
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "stdev": round(statistics.pstdev(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }
    good = sum(1 for r in rows if float(r.get("rubric", 0)) >= 3.5)
    agg["_quality"] = {
        "count": len(rows),
        "examples_rubric_ge_3.5": good,
        "share_rubric_ge_3.5": round(good / len(rows), 3) if rows else 0.0,
    }
    return agg


def breakdown_by_category(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-category mean rubric/relevance/coverage/coverage + share of good answers."""
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, sub in sorted(cats.items()):
        agg = aggregate_results(sub)
        out[cat] = {
            "count": len(sub),
            "mean_rubric": agg["rubric"]["mean"],
            "mean_relevance": agg["relevance"]["mean"],
            "mean_keyword_coverage": agg["keyword_coverage"]["mean"],
            "mean_coherence": agg["coherence"]["mean"],
            "share_rubric_ge_3.5": agg["_quality"]["share_rubric_ge_3.5"],
        }
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(results: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    ev = cfg["evaluation"]
    out_dir = _resolve(ev["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / ev["results_file"]
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote per-example/aggregate results -> %s", results_path)

    try:
        import pandas as pd

        cols = [
            "id", "source", "category", "question", "gold", "generated",
            *NUMERIC_METRICS,
        ]
        frame = pd.DataFrame(results["per_example"])[cols]
        if "llm_judge" in frame.columns:
            frame.loc[:, "llm_judge_relevance"] = frame["llm_judge"].apply(
                lambda v: (v or {}).get("relevance") if isinstance(v, dict) else None
            )
            frame.loc[:, "llm_judge_helpfulness"] = frame["llm_judge"].apply(
                lambda v: (v or {}).get("helpfulness") if isinstance(v, dict) else None
            )
        csv_path = out_dir / ev["results_csv"]
        frame.to_csv(csv_path, index=False)
        logger.info("Wrote per-example CSV -> %s", csv_path)
    except Exception as exc:  # noqa: BLE001 - CSV is best effort
        logger.warning("Could not write CSV: %s", exc)

    summary = ["== Aggregate =="]
    for name, stats_ in results["aggregate"].items():
        summary.append(f"{name:<20} {stats_}")
    summary.append("\n== By category ==")
    header = "{:<18} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "category", "n", "rubric", "rel", "cov", "coh", "good%"
    )
    summary.append(header)
    for cat, s in results["by_category"].items():
        summary.append(
            "{:<18} {:>5} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>7.1%}".format(
                cat,
                s["count"],
                s["mean_rubric"],
                s["mean_relevance"],
                s["mean_keyword_coverage"],
                s["mean_coherence"],
                s["share_rubric_ge_3.5"],
            )
        )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    logger.info("Wrote summary -> %s", summary_path)


def plot_results(results: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Render score-distribution + category charts into ``outputs/plots/``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = _resolve(cfg["evaluation"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(results["per_example"])
    written: List[str] = []

    # 1) rubric score distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["rubric"], bins=range(1, 7), align="left", rwidth=0.85, color="#4C78A8")
    ax.set_title("Fine-tuned model: rubric score distribution (1-5)")
    ax.set_xlabel("rubric score")
    ax.set_ylabel("answers")
    ax.set_xticks(range(1, 6))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "rubric_score_histogram.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    # 2) category breakdown (mean rubric / relevance / keyword coverage)
    agg = pd.DataFrame(results["by_category"]).T
    agg = agg.loc[agg["count"].sort_values(ascending=False).index]
    metrics_plotted = ["mean_rubric", "mean_relevance", "mean_keyword_coverage"]
    x = range(len(agg))
    widths = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(metrics_plotted):
        ax.bar([xi + i * widths for xi in x], agg[col], width=widths, label=col)
    ax.set_xticks([xi + widths for xi in x])
    ax.set_xticklabels(agg.index, rotation=25, ha="right")
    ax.set_ylim(0, 5)
    ax.set_title("Model quality across question categories (means)")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "category_scores.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    # 3) boxplot: rubric by category
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = agg.index.tolist()
    data = [df.loc[df["category"] == c, "rubric"].tolist() for c in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("Rubric score spread by category")
    ax.set_ylabel("rubric")
    ax.set_ylim(1, 5.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    p = plots_dir / "rubric_by_category_boxplot.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    # 4) length vs rubric scatter
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["word_count"], df["rubric"], alpha=0.6, color="#54A24B")
    ax.set_xlabel("answer word count")
    ax.set_ylabel("rubric score")
    ax.set_title("Answer length vs quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "length_vs_rubric.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    return written


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------
def run_evaluation(cfg: Union[Mapping[str, Any], str, Path, None] = None) -> Dict[str, Any]:
    """Execute the full evaluation pipeline and return the results dict."""
    config = load_config(cfg) if isinstance(cfg, (str, Path)) else _scope_config(cfg)
    ev = config["evaluation"]

    examples = load_eval_examples(config)
    logger.info("Evaluation examples: %d (test=%d, curated=%d)",
                len(examples),
                sum(1 for e in examples if e["source"] == "test"),
                sum(1 for e in examples if e["source"] != "test"))
    if not examples:
        raise ValueError("No evaluation examples found; check test_file/curated_dir in eval_config.yaml")

    model, tokenizer, judge_base = load_eval_model(config)
    logger.info("Model ready: %s", model_loader.get_model_info(model, base_model=model.name_or_path))

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, 1):
        t0 = time.perf_counter()
        row = score_example(
            ex, config, model=model, tokenizer=tokenizer,
            judge_model=judge_base if ev["enable_llm_judge"] else None,
        )
        rows.append(row)
        logger.info("[%3d/%3d] %-6s %-16s rubric=%.2f words=%d (%.1fs)",
                    i, len(examples), row["source"], row["category"],
                    row["rubric"], int(row["word_count"]), time.perf_counter() - t0)

    results: Dict[str, Any] = {
        "config": {
            "adapter_dir": ev["adapter_dir"],
            "test_file": ev["test_file"],
            "curated_dir": ev["curated_dir"],
            "generation": {
                "max_new_tokens": ev["max_new_tokens"],
                "do_sample": ev["do_sample"],
                "temperature": ev["temperature"],
            },
            "llm_judge_enabled": ev["enable_llm_judge"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_example": rows,
        "aggregate": aggregate_results(rows),
        "by_category": breakdown_by_category(rows),
    }

    write_outputs(results, config)
    written = plot_results(results, config)
    logger.info("Plots written: %s", ", ".join(written))
    return results


def _scope_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = load_config()
    if cfg:
        merged["evaluation"].update(cfg.get("evaluation", {}))
    return merged


__all__ = [
    "DEFAULT_CONFIG",
    "NUMERIC_METRICS",
    "aggregate_results",
    "breakdown_by_category",
    "generate_answer",
    "load_config",
    "load_eval_examples",
    "load_eval_model",
    "plot_results",
    "run_evaluation",
    "score_example",
    "write_outputs",
]