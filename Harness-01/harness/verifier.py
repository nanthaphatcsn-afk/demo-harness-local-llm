"""VERIFY - code-only checks of the final answer. Pure function, no model call.

It never sees the expected answer (grading lives in the eval script) and never reads the plan.

V1 well-formed  - non-empty, last ACT call not truncated, not a raw tool call / JSON
V2 coverage     - every customer ID in the task was looked up (ok or LookupError)
V3 grounding    - a number in the answer equals a number from a successful tool result,
                  after removing numbers that already appear in the task text
V4 tool use     - calculator ran (calculation categories)
V5 error ack    - a LookupError / permission denial is reported to the user
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .agent import _args, _looks_like_tool_call
from .classifier import CUSTOMER_ID, customer_ids

CALC_CATEGORIES = {"customer_calc", "date_calc", "arithmetic"}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
NOT_FOUND_PHRASES = ("not found", "no customer", "does not exist", "doesn't exist", "unknown")
DENIED_PHRASES = ("permission", "not allowed", "not authorized", "not permitted", "denied")
TOL = 0.01


class VerifyReport(BaseModel):
    passed: bool
    failures: list[str]
    checks: dict[str, bool | None]      # None = check not applicable


# ------------------------------------------------------------- helpers
def executed(events: list[dict]) -> list[dict]:
    """Events where the executor really ran the call (not a repeat, not a disallowed tool)."""
    return [e for e in events if not e.get("repeat") and not e.get("disallowed")]


def _name(e: dict) -> str:
    return e["call"].get("function", {}).get("name", "")


def numbers(text: str) -> list[float]:
    """Numbers in free text; customer IDs are removed first so C-104 does not read as -104."""
    text = CUSTOMER_ID.sub(" ", text)
    out = []
    for m in NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def tool_values(result) -> list[float]:
    """Every int/float leaf (not bool) of a tool result, recursively."""
    if isinstance(result, bool):
        return []
    if isinstance(result, (int, float)):
        return [float(result)]
    if isinstance(result, dict):
        return [v for x in result.values() for v in tool_values(x)]
    if isinstance(result, (list, tuple)):
        return [v for x in result for v in tool_values(x)]
    return []


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def _has(answer: str, phrases: tuple[str, ...]) -> bool:
    low = answer.lower()
    return any(p in low for p in phrases)


# ------------------------------------------------------------- the checks
def verify(task: str, category: str, answer: str | None, events: list[dict], *,
           last_done_reason: str | None = None, strict_grounding: bool = False) -> VerifyReport:
    """Run V1-V5 on the final answer and the tool events of all ACT attempts."""
    answer = answer or ""
    ran = executed(events)
    checks: dict[str, bool | None] = {}
    failures: list[str] = []

    # V1 well-formed
    v1 = []
    if not answer.strip():
        v1.append("the answer is empty")
    if last_done_reason == "length":
        v1.append("the answer was truncated")
    if answer.strip() and _looks_like_tool_call(answer):
        v1.append("the answer is a raw tool call or JSON, not a sentence for the user")
    checks["V1"] = not v1
    failures += v1

    # V2 coverage
    ids = customer_ids(task)
    if ids:
        looked_up = set()
        for e in ran:
            args = _args(e["call"])
            res = e["result"]
            if (_name(e) == "lookup_customer" and isinstance(args, dict)
                    and (res["ok"] or res.get("error", "").startswith("LookupError"))):
                looked_up.add(str(args.get("customer_id", "")).strip().upper())
        missing = [c for c in ids if c not in looked_up]
        checks["V2"] = not missing
        failures += [f"{c} was not looked up" for c in missing]
    else:
        checks["V2"] = None

    # V5 error acknowledged
    errors = [e["result"].get("error", "") for e in ran if not e["result"]["ok"]]
    not_found = any(x.startswith("LookupError") for x in errors)
    denied = any(x.startswith("Permission denied") for x in errors)
    if not_found or denied:
        v5 = []
        if not_found and not _has(answer, NOT_FOUND_PHRASES):
            v5.append("a customer was not found, but the answer does not say so")
        if denied and not _has(answer, DENIED_PHRASES):
            v5.append("the change was refused (permission denied), but the answer does not say so")
        checks["V5"] = not v5
        failures += v5
    else:
        checks["V5"] = None

    # V3 grounding (skipped when V5 applies)
    task_nums = numbers(task)
    ans_nums = [a for a in numbers(answer) if not any(_close(a, t) for t in task_nums)]
    applies = checks["V5"] is None and (
        category in CALC_CATEGORIES or (category == "customer_lookup" and numbers(answer)))
    if applies:
        values = [v for e in ran if e["result"]["ok"] for v in tool_values(e["result"]["result"])]
        v3 = []
        if not any(_close(a, v) for a in ans_nums for v in values):
            v3.append("the answer has no number taken from a tool result")
        calc = [e for e in ran if _name(e) == "calculator" and e["result"]["ok"]]
        if strict_grounding and calc:
            last = float(calc[-1]["result"]["result"]["value"])
            if not any(_close(a, last) for a in ans_nums):
                v3.append(f"the answer does not contain the last calculator result ({last:g})")
        checks["V3"] = not v3
        failures += v3
    else:
        checks["V3"] = None

    # V4 tool use
    if category in CALC_CATEGORIES:
        used = any(_name(e) == "calculator" and e["result"]["ok"] for e in ran)
        checks["V4"] = used
        if not used:
            failures.append("calculator was not used; do the arithmetic with the calculator tool")
    else:
        checks["V4"] = None

    return VerifyReport(passed=not failures, failures=failures,
                        checks={k: checks[k] for k in ("V1", "V2", "V3", "V4", "V5")})


def feedback_message(report: VerifyReport, task: str) -> dict:
    """The user message appended to the same history when VERIFY sends the task back to ACT."""
    lines = "\n".join(f"- {f}" for f in report.failures)
    return {"role": "user", "content": f"Verification failed:\n{lines}\n"
                                       "Fix this, use the tools if needed, then give the final "
                                       f"answer. Goal: {task}"}
