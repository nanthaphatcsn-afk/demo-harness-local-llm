"""Grading shared by agent_eval.py (loop) and sm_eval.py (state machine).

Grading uses the expected answers in data/agent_tasks.jsonl; the loop's completion check
and the state machine's VERIFY never do.
"""
import calendar
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from _path import ROOT


def load_tasks(only: str = "") -> list[dict]:
    with open(ROOT / "data" / "agent_tasks.jsonl", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if only:
        keep = set(only.split(","))
        items = [it for it in items if it["id"] in keep]
    return items


def days_left_in_month() -> list[str]:
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    left = calendar.monthrange(now.year, now.month)[1] - now.day
    return [str(left)]


def _norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "").lower()


def final_text(answer: str) -> str:
    """Second guard only: M1 (`split_thinking` in llm_client.py) already moves reasoning that
    leaks into content before `</think>` to ChatResult.thinking. Kept so a number in leaked
    reasoning can never count as a correct final answer."""
    return answer.rsplit("</think>", 1)[-1]


def _events(res) -> list[dict]:
    """Tool events of a loop AgentResult (per-step trace) or a state-machine SMResult."""
    if hasattr(res, "events"):
        return res.events
    return [ev for t in res.trace for ev in t["events"]]


def grade(it: dict, res) -> tuple[bool, list[str], bool]:
    """(success, missing expected tools, answer matched). Grading lives here, never in the loop."""
    expect = days_left_in_month() if it.get("dynamic") == "days_left_in_month" else it["expect_any"]
    executed = {ev["call"]["function"].get("name") for ev in _events(res)
                if not ev["repeat"] and not ev.get("disallowed")}
    missing = [t for t in it["expected_tools"] if t not in executed]
    answer = _norm(final_text(res.answer or ""))
    matched = (bool(answer) and any(_norm(x) in answer for x in expect)
               and all(_norm(x) in answer for x in it.get("expect_all", [])))
    return res.status == "done" and not missing and matched, missing, matched
