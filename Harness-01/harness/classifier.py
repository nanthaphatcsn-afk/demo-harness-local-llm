"""CLASSIFY - hybrid classifier: rules first, `fast` model only when no rule matches.

Each category gets a narrow tool allow-list for ACT (too many tools hurt small models).
Rules are checked in order; the first match wins. Every keyword uses word boundaries.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from .llm_client import LLMError

READ_TOOLS = ["calculator", "lookup_customer", "current_datetime"]

CATEGORY_TOOLS: dict[str, list[str]] = {
    "write_request": ["lookup_customer", "update_credit_limit"],
    "date_calc": ["current_datetime", "calculator"],
    "customer_calc": ["lookup_customer", "calculator"],
    "customer_lookup": ["lookup_customer"],
    "arithmetic": ["calculator"],
    "general": READ_TOOLS,
}
CATEGORIES = tuple(CATEGORY_TOOLS)

CUSTOMER_ID = re.compile(r"\bC-\d+\b", re.IGNORECASE)


def _words(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


_WRITE_VERB = _words("raise", "increase", "decrease", "change", "update", "set")
_LIMIT = _words("limit")
_DATE = _words("today", "date", "day", "days", "week", "month", "weekday", "time")
_CALC = _words("percent", "combined", "total", "sum", "difference", "more", "less", "over",
               "under", "after", "pays", "discount", "vat", "split", "each", "divide", "if")
_DIGIT = re.compile(r"\d")
_OPERATOR = re.compile(r"[+\-*/%]")
_ARITH_WORD = _words("of", "split", "divide", "times", "per")


def customer_ids(text: str) -> list[str]:
    """Customer IDs in the text, upper-cased, in order of first appearance."""
    return list(dict.fromkeys(m.upper() for m in CUSTOMER_ID.findall(text)))


def classify_rules(task: str) -> str | None:
    """The rule table. Returns a category, or None when no rule matches."""
    has_id = bool(CUSTOMER_ID.search(task))
    if _WRITE_VERB.search(task) and _LIMIT.search(task):
        return "write_request"
    if _DATE.search(task) and not has_id:
        return "date_calc"
    if has_id and ("%" in task or _CALC.search(task)):
        return "customer_calc"
    if has_id:
        return "customer_lookup"
    if _DIGIT.search(task) and (_OPERATOR.search(task) or _ARITH_WORD.search(task)):
        return "arithmetic"
    return None


class _Label(BaseModel):
    category: Literal[CATEGORIES]  # type: ignore[valid-type]


def classify(h, task: str, *, run_id: str | None = None, profile: str = "fast",
             calls: list | None = None) -> tuple[str, str]:
    """Return (category, source). source = rules / llm / default.

    `calls` collects the ChatResult of the model call, if one is made (for metrics).
    """
    cat = classify_rules(task)
    if cat is not None:
        return cat, "rules"
    msgs = [{"role": "system", "content": "Classify the user's request into one category. "
                                          f"Categories: {', '.join(CATEGORIES)}. "
                                          "Return JSON only."},
            {"role": "user", "content": task}]
    try:
        r = h.llm(profile, msgs, schema=_Label.model_json_schema(), run_id=run_id,
                  log_extra={"state": "CLASSIFY"})
        if calls is not None:
            calls.append(r)
        return _Label.model_validate_json(r.content).category, "llm"
    except (LLMError, ValidationError, json.JSONDecodeError):
        return "general", "default"
