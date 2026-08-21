"""Phase 8 validation: processor -> structured Advice objects.

Runs :func:`src.advice.processor.process_advice` on 5 sample raw advice texts
(captured from Phase 7's generate_advice() outputs) and prints the resulting
structured Advice objects, checking that categories/titles/recommendations are
sensible.

Usage:
    python scripts/verify_advice_processor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.advice import categorizer, processor  # noqa: E402
from src.advice.schemas import Advice, AdviceCategory  # noqa: E402

# (source_question, raw_text) captured from Phase 7 generate_advice() runs.
SAMPLES = [
    (
        "How can I save money?",
        "There are many ways to save money, including: 1) reducing expenses by cutting "
        "back on non-essential spending; 2) increasing income through additional work "
        "or employment; 3) increasing your savings by making regular contributions to "
        "an IRA, 4) reducing debt with the assistance of a credit counseling service, "
        "5) paying off high-interest debt.",
    ),
    (
        "What is compound interest?",
        "Compound interest is when interest is added back into the principal amount, "
        "making it grow over time.",
    ),
    (
        "Should I invest or pay off debt first?",
        "Yes. Pay off debt first.",
    ),
    (
        "What is the difference between a Roth IRA and a traditional IRA?",
        "A Roth IRA is funded with after-tax dollars, so withdrawals in retirement are "
        "usually tax-free. A traditional IRA is funded with pre-tax dollars, so you pay "
        "taxes on withdrawals. Choose a Roth if you expect a higher tax rate later; "
        "otherwise a traditional IRA may be better.",
    ),
    (
        "How can I get my spending back on track?",
        "Start by creating a monthly budget and tracking every expense. Cut back on "
        "non-essential spending and review your budget each week.",
    ),
]


def main() -> None:
    advices = [processor.process_advice(text, question) for question, text in SAMPLES]

    for advice in advices:
        print("=" * 70)
        print(json.dumps(processor.advice_to_dict(advice), indent=2, ensure_ascii=False))

    # structural checks
    for advice in advices:
        d = advice.to_dict()
        assert set(d) == {
            "id", "timestamp", "category", "short_title",
            "key_recommendation", "full_text", "source_question",
        }
        assert isinstance(advice.category, AdviceCategory)
        assert advice.short_title and advice.key_recommendation and advice.full_text
        assert advice.source_question

    # sensible-category expectations (heuristic, loosely asserted)
    by_q = {a.source_question: a.category for a in advices}
    assert by_q["What is compound interest?"] == AdviceCategory.CONCEPTS
    assert by_q["Should I invest or pay off debt first?"] == AdviceCategory.DEBT
    assert by_q["What is the difference between a Roth IRA and a traditional IRA?"] == AdviceCategory.RETIREMENT
    assert by_q["How can I get my spending back on track?"] == AdviceCategory.BUDGETING
    assert by_q["How can I save money?"] in {AdviceCategory.SAVING, AdviceCategory.BUDGETING, AdviceCategory.DEBT}

    # overrides + process_many
    custom = processor.process_advice(
        SAMPLES[2][1], SAMPLES[2][0],
        advice_id="custom-1", category=AdviceCategory.CREDIT,
        short_title="Custom title", key_recommendation="Do it.",
    )
    assert custom.id == "custom-1" and custom.category == AdviceCategory.CREDIT
    assert custom.short_title == "Custom title" and custom.key_recommendation == "Do it."

    many = processor.process_many(SAMPLES[:2])
    assert len(many) == 2 and all(isinstance(a, Advice) for a in many)

    print("=" * 70)
    print("Categories:", {a.source_question[:22]: a.category.value for a in advices})
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()