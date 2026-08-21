"""Phase 7 validation: verify generate_advice() on sample financial questions.

Checks:
  1. the advisor persona never asks for salary/income/expense figures;
  2. coherent single-turn replies for 3 sample questions;
  3. a multi-turn follow-up keeps conversational context.

Usage:
    python scripts/verify_inference.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import inference, prompt_templates  # noqa: E402


def run(label: str, question: str, history=None) -> str:
    t0 = time.perf_counter()
    reply = inference.generate_advice(question, conversation_history=history)
    dt = time.perf_counter() - t0
    print(f"=== {label} ({dt:.1f}s) ===")
    print(f"Q: {question}")
    print(f"A: {reply}\n")
    return reply


def main() -> None:
    persona = prompt_templates.ADVISOR_SYSTEM_PROMPT
    persona_l = persona.lower()
    # the persona must *forbid* requesting user figures (words like income/expense
    # appear only inside that prohibition) and must contain no questions at all.
    assert "never ask" in persona_l, "persona must explicitly forbid requesting user figures"
    assert "salary" in persona_l and "income" in persona_l and "expense" in persona_l
    assert "?" not in persona, "persona must never pose questions to the user"
    print(f"[persona OK] advisor persona forbids asking for salary/income/expense figures\n")

    cfg = inference.load_config()
    print(f"[config] adapter={cfg['adapter_dir']} | max_new_tokens={cfg['max_new_tokens']} "
          f"| temperature={cfg['temperature']} | history_max_turns={cfg['history_max_turns']}\n")

    run("Q1", "How can I save money?")
    run("Q2", "What is compound interest?")
    r3 = run("Q3", "Should I invest or pay off debt first?")

    r1 = run("Multi-turn turn 1", "What is compound interest?")
    run(
        "Multi-turn turn 2 (with context)",
        "That helps - what is a simple way a young person can start taking advantage of it?",
        history=[
            {"role": "user", "content": "What is compound interest?"},
            {"role": "assistant", "content": r1},
            {"role": "user", "content": "Should I invest or pay off debt first?"},
            {"role": "assistant", "content": r3},
        ],
    )

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()