"""PLAN - ask the `reasoning` profile for a short tool plan before ACT.

Validated with Pydantic, one retry with the validation error, then give up (plan=None).
`ask_structured()` is not used on purpose: its fallback profile has think: true and is very slow.
The plan reaches ACT as one extra user message after the task, so the cached prefix
(system prompt + tool list) is unchanged. VERIFY never reads the plan.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMError


class PlanStep(BaseModel):
    tool: str            # must be in the category's allow-list
    purpose: str


class Plan(BaseModel):
    customer_ids: list[str]
    steps: list[PlanStep] = Field(min_length=1, max_length=6)
    needs_calculation: bool


def _check_tools(plan: Plan, allowed: list[str]) -> None:
    bad = [s.tool for s in plan.steps if s.tool not in allowed]
    if bad:
        raise ValueError(f"tools not allowed: {bad}. Allowed: {allowed}")


def make_plan(h, task: str, allowed: list[str], *, run_id: str | None = None,
              profile: str = "reasoning", prompt: str = "planner") -> tuple[Plan | None, list]:
    """Return (plan or None, ChatResults of every PLAN call) - the results feed the metrics."""
    system = h.prompts.get(prompt)
    msgs = [{"role": "system", "content": system.render(tools=", ".join(allowed))},
            {"role": "user", "content": task}]
    calls = []
    for attempt in range(2):
        try:
            r = h.llm(profile, msgs, schema=Plan.model_json_schema(), run_id=run_id,
                      prompt_id=system.id, log_extra={"state": "PLAN", "retry": attempt})
        except LLMError:
            return None, calls
        calls.append(r)
        try:
            plan = Plan.model_validate_json(r.content)
            _check_tools(plan, allowed)
            return plan, calls
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            msgs = msgs + [{"role": "assistant", "content": r.content},
                           {"role": "user", "content": f"Invalid plan: {str(e)[:300]}. "
                                                       "Fix it and return JSON only."}]
    return None, calls


def plan_message(plan: Plan) -> dict:
    """The user message that carries the plan into ACT."""
    steps = " ".join(f"{i}. {s.tool} — {s.purpose}" for i, s in enumerate(plan.steps, 1))
    return {"role": "user", "content": f"Plan (follow it, use the tools): {steps}"}
