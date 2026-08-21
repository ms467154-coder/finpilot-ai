"""Structured advice data model (Phase 8).

Defines :class:`AdviceCategory` (fixed, display-friendly topic buckets) and the
:class:`Advice` pydantic model used to store / display a single piece of
financial advice produced from :func:`src.inference.inference.generate_advice`.

No numeric financial inputs (salary/income/expense) appear anywhere in this
model or the processing layer - advice is qualitative and educational.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AdviceCategory(str, enum.Enum):
    """Topic buckets a piece of advice can be assigned to."""

    SAVING = "Saving"
    BUDGETING = "Budgeting"
    RETIREMENT = "Retirement"
    INVESTING = "Investing"
    DEBT = "Debt"
    CREDIT = "Credit"
    INSURANCE = "Insurance"
    TAXES = "Taxes"
    CONCEPTS = "Concepts"  # definitions / educational explainers
    GENERAL = "General"   # fallback when nothing else matches


class Advice(BaseModel):
    """A single structured, display-ready piece of financial advice."""

    id: str = Field(..., min_length=1, description="Unique advice identifier")
    timestamp: datetime = Field(..., description="UTC creation timestamp")
    category: AdviceCategory = Field(..., description="Topic bucket")
    short_title: str = Field(..., min_length=1, description="Short human-readable headline")
    key_recommendation: str = Field(..., min_length=1, description="Single most actionable sentence")
    full_text: str = Field(..., min_length=1, description="Full advisor response text")
    source_question: str = Field(..., min_length=1, description="The user's original question")

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value):
        if isinstance(value, str):
            return value.capitalize() if not value.isupper() else value
        return value

    def to_dict(self) -> dict:
        """JSON-serializable dict (ISO timestamp, string category)."""
        return self.model_dump(mode="json")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["Advice", "AdviceCategory", "now_utc"]