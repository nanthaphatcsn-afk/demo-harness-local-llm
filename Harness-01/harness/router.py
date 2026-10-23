"""Router - which executor runs each state. Every decision is logged as a `route` event.

| State    | Executor                                           |
|----------|----------------------------------------------------|
| INTAKE   | code                                               |
| CLASSIFY | rules; `fast` profile only if no rule matches      |
| PLAN     | `reasoning` profile                                |
| ACT      | the ACT profile (default `reasoning`)              |
| VERIFY   | code (deliberately not a model judge, see D5)      |
"""
from __future__ import annotations

CLASSIFY_PROFILE = "fast"
PLAN_PROFILE = "reasoning"


def route(h, state, *, run_id: str, act_profile: str = "reasoning",
          classify_source: str | None = None) -> str:
    """Return the executor for `state` (a State or its name): "code", "rules" or a profile.

    Logged. Takes the state by name so the state machine can import this module.
    """
    name = getattr(state, "value", state)
    if name in ("INTAKE", "VERIFY"):
        executor = "code"
    elif name == "CLASSIFY":
        executor = CLASSIFY_PROFILE if classify_source in ("llm", "default") else "rules"
    elif name == "PLAN":
        executor = PLAN_PROFILE
    elif name == "ACT":
        executor = act_profile
    else:
        executor = "none"
    model = h.config.profile(executor).model if executor in h.config.profiles else None
    h.log.write("route", run_id=run_id, state=name, executor=executor, model=model)
    return executor


def model_switches(models: list[str]) -> int:
    """Consecutive LLM calls on different models (each one may cost a model load)."""
    return sum(a != b for a, b in zip(models, models[1:]))
