"""M9 - State machine: INTAKE -> CLASSIFY -> PLAN -> ACT -> VERIFY -> DONE.

"Loop inside a state, state machine across the workflow": `run_agent()` is the worker
inside ACT; the harness owns which state comes next. Every move must be in TRANSITIONS
(anything else is a harness bug -> IllegalTransition) and is logged as a `transition` event.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .harness import Harness


class State(str, Enum):
    INTAKE = "INTAKE"
    CLASSIFY = "CLASSIFY"
    PLAN = "PLAN"
    ACT = "ACT"
    VERIFY = "VERIFY"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: dict[State, set[State]] = {
    State.INTAKE:   {State.CLASSIFY, State.FAILED},
    State.CLASSIFY: {State.PLAN, State.ACT, State.FAILED},
    State.PLAN:     {State.ACT},                      # plan failure -> ACT without plan
    State.ACT:      {State.VERIFY, State.CLASSIFY, State.FAILED},
    State.VERIFY:   {State.DONE, State.ACT, State.FAILED},
    State.DONE:     set(),
    State.FAILED:   set(),
}


class IllegalTransition(RuntimeError):
    """A move that is not in TRANSITIONS: a bug in the harness, not a model failure."""


class TaskState(BaseModel):
    """Everything the harness knows about one task while it moves through the states."""
    run_id: str
    task: str
    state: State = State.INTAKE
    customer_ids: list[str] = Field(default_factory=list)
    category: str | None = None
    classify_source: str | None = None      # rules / llm / default / reclassify
    tools: list[str] = Field(default_factory=list)
    plan: Any = None                        # planner.Plan or None
    msgs: list[dict] = Field(default_factory=list)
    act_attempts: int = 0
    reclassified: int = 0
    verify_reports: list[dict] = Field(default_factory=list)
    answer: str | None = None
    history: list[dict] = Field(default_factory=list)


def move(h: Harness, ts: TaskState, dst: State, reason: str = "") -> None:
    """Change state if the transition table allows it; log and record the move."""
    if dst not in TRANSITIONS[ts.state]:
        raise IllegalTransition(f"{ts.state.value} -> {dst.value} is not allowed")
    rec = {"src": ts.state.value, "dst": dst.value, "reason": reason,
           "ts": datetime.now().isoformat(timespec="seconds")}
    h.log.write("transition", run_id=ts.run_id, src=rec["src"], dst=rec["dst"], reason=reason)
    ts.history.append(rec)
    ts.state = dst
