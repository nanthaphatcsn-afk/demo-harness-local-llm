"""M9 - State machine: INTAKE -> CLASSIFY -> PLAN -> ACT -> VERIFY -> DONE.

"Loop inside a state, state machine across the workflow": `run_agent()` is the worker
inside ACT; the harness owns which state comes next. Every move must be in TRANSITIONS
(anything else is a harness bug -> IllegalTransition) and is logged as a `transition` event.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .agent import run_agent
from .classifier import CATEGORY_TOOLS, classify, classify_rules, customer_ids
from .harness import Harness
from .logger import new_run_id
from .planner import make_plan, plan_message
from .prompts import build_messages
from .router import model_switches, route
from .verifier import feedback_message, verify


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


# ------------------------------------------------------------------ runner
@dataclass
class SMResult:
    run_id: str
    status: str                          # "done" | "failed"
    answer: str | None
    category: str | None
    classify_source: str | None
    plan_ok: bool | None                 # None = PLAN not run
    act_attempts: int
    reclassified: int
    act_steps_total: int
    tool_calls: int
    tool_ok: int
    repeats: int
    llm_calls: int
    llm_calls_by_state: dict[str, int]
    in_tokens_total: int
    out_tokens_total: int
    model_switches: int
    load_s_total: float
    wall_s: float
    verify_reports: list[dict]
    transitions: list[str]               # "SRC->DST(reason)"
    events: list[dict] = field(default_factory=list, repr=False)   # tool events, all ACT runs
    act_runs: list = field(default_factory=list, repr=False)       # AgentResult per ACT attempt
    calls: list[dict] = field(default_factory=list, repr=False)    # one per LLM call


def run_state_machine(h: Harness, task: str, *, use_plan: bool, act_profile: str = "reasoning",
                      max_act_attempts: int = 3, act_max_steps: int = 6,
                      strict_grounding: bool = False, run_id: str | None = None,
                      prompt: str = "tool_assistant") -> SMResult:
    """Run one task through INTAKE -> CLASSIFY -> (PLAN) -> ACT -> VERIFY -> DONE/FAILED.

    Never raises: an unexpected exception ends in FAILED with a safe_failure log line.
    """
    run_id = run_id or new_run_id()
    t0 = time.perf_counter()
    ts = TaskState(run_id=run_id, task=task)
    calls: list[dict] = []          # {state, model, in, out, load_s}
    events: list[dict] = []
    act_runs: list = []
    plan_ok: bool | None = None
    last_done_reason: str | None = None

    def record(state: State, results) -> None:
        for r in results:
            calls.append({"state": state.value, "model": r.model, "in": r.prompt_tokens,
                          "out": r.output_tokens, "load_s": r.load_s})

    try:
        route(h, State.INTAKE, run_id=run_id)
        if not task.strip():
            move(h, ts, State.FAILED, "empty task")
        else:
            ts.customer_ids = customer_ids(task)
            move(h, ts, State.CLASSIFY, f"customer_ids={ts.customer_ids}")

        while ts.state not in (State.DONE, State.FAILED):
            if ts.state is State.CLASSIFY:
                if ts.reclassified:
                    # D4: one backtrack, upgraded to all read tools
                    route(h, State.CLASSIFY, run_id=run_id, classify_source="reclassify")
                    cat, src = "general", "reclassify"
                else:
                    rule = classify_rules(task)
                    route(h, State.CLASSIFY, run_id=run_id,
                          classify_source="rules" if rule else "llm")
                    if rule:
                        cat, src = rule, "rules"
                    else:
                        llm_results: list = []
                        cat, src = classify(h, task, run_id=run_id, calls=llm_results)
                        record(State.CLASSIFY, llm_results)
                ts.category, ts.classify_source = cat, src
                ts.tools = list(CATEGORY_TOOLS[cat])
                nxt = State.PLAN if use_plan and plan_ok is None else State.ACT
                move(h, ts, nxt, f"{cat} ({src})")

            elif ts.state is State.PLAN:
                route(h, State.PLAN, run_id=run_id)
                plan, plan_results = make_plan(h, task, ts.tools, run_id=run_id)
                record(State.PLAN, plan_results)
                ts.plan, plan_ok = plan, plan is not None
                if plan is None:
                    h.log.write("plan_failed", run_id=run_id, attempts=len(plan_results))
                    move(h, ts, State.ACT, "plan_failed; ACT without plan")
                else:
                    move(h, ts, State.ACT, f"plan with {len(plan.steps)} steps")

            elif ts.state is State.ACT:
                route(h, State.ACT, run_id=run_id, act_profile=act_profile)
                if not ts.msgs:
                    system = h.prompts.get(prompt)
                    ts.msgs = build_messages(system.render(), user=task)
                    if ts.plan is not None:
                        ts.msgs.append(plan_message(ts.plan))
                ts.act_attempts += 1
                r = run_agent(h, task, profile=act_profile, max_steps=act_max_steps,
                              tool_names=ts.tools, prompt=prompt, run_id=run_id, msgs=ts.msgs,
                              allowed=set(ts.tools), on_disallowed="stop")
                act_runs.append(r)
                model = h.config.profile(act_profile).model
                for t in r.trace:
                    calls.append({"state": "ACT", "model": model, "in": t["in_tokens"],
                                  "out": t["out_tokens"], "load_s": t["load_s"]})
                    events.extend(t["events"])
                if r.status == "done":
                    ts.answer = r.answer
                    last_done_reason = r.trace[-1]["done_reason"] if r.trace else None
                    move(h, ts, State.VERIFY, f"attempt {ts.act_attempts}: done in {r.steps} steps")
                elif r.status == "needs_reclassify":
                    bad = [e["call"]["function"].get("name") for e in r.trace[-1]["events"]
                           if e.get("disallowed")]
                    if ts.reclassified == 0:
                        ts.reclassified = 1
                        move(h, ts, State.CLASSIFY, f"needs_reclassify: {bad[0]}")
                    else:
                        move(h, ts, State.FAILED, f"needs_reclassify again: {bad[0]}")
                else:
                    move(h, ts, State.FAILED, f"attempt {ts.act_attempts}: {r.status}")

            elif ts.state is State.VERIFY:
                route(h, State.VERIFY, run_id=run_id)
                rep = verify(task, ts.category or "general", ts.answer, events,
                             last_done_reason=last_done_reason, strict_grounding=strict_grounding)
                ts.verify_reports.append(rep.model_dump())
                h.log.write("verify", run_id=run_id, attempt=ts.act_attempts, passed=rep.passed,
                            failures=rep.failures, checks=rep.checks)
                if rep.passed:
                    move(h, ts, State.DONE, "all checks pass")
                elif ts.act_attempts < max_act_attempts:
                    ts.msgs.append(feedback_message(rep, task))
                    move(h, ts, State.ACT, "fail: " + "; ".join(rep.failures))
                else:
                    move(h, ts, State.FAILED, f"fail after {ts.act_attempts} ACT attempts: "
                                              + "; ".join(rep.failures))
    except Exception as e:  # noqa: BLE001 - never raise to the caller
        h.log.write("safe_failure", run_id=run_id, reason="exception",
                    error=f"{type(e).__name__}: {e}"[:300], state=ts.state.value)
        ts.history.append({"src": ts.state.value, "dst": State.FAILED.value,
                           "reason": f"exception: {type(e).__name__}"})
        ts.state = State.FAILED

    by_state: dict[str, int] = {}
    for c in calls:
        by_state[c["state"]] = by_state.get(c["state"], 0) + 1
    return SMResult(
        run_id=run_id, status="done" if ts.state is State.DONE else "failed",
        answer=ts.answer if ts.state is State.DONE else None, category=ts.category,
        classify_source=ts.classify_source, plan_ok=plan_ok, act_attempts=ts.act_attempts,
        reclassified=ts.reclassified, act_steps_total=sum(r.steps for r in act_runs),
        tool_calls=sum(r.tool_calls for r in act_runs), tool_ok=sum(r.tool_ok for r in act_runs),
        repeats=sum(r.repeats for r in act_runs), llm_calls=len(calls),
        llm_calls_by_state=by_state, in_tokens_total=sum(c["in"] for c in calls),
        out_tokens_total=sum(c["out"] for c in calls),
        model_switches=model_switches([c["model"] for c in calls]),
        load_s_total=round(sum(c["load_s"] for c in calls), 3),
        wall_s=round(time.perf_counter() - t0, 3), verify_reports=ts.verify_reports,
        transitions=[f"{x['src']}->{x['dst']}({x['reason']})" for x in ts.history],
        events=events, act_runs=act_runs, calls=calls)
