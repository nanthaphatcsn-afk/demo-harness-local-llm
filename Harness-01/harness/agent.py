"""M8 - Agent loop. Turns one tool round trip into a controlled multi-step loop.

    model -> tool_calls? --yes--> repeat check -> execute -> role:tool message -> model ...
                         --no---> completion check --pass--> done
                                                   --fail--> "Not finished: <reason>. Goal: <task>"

The harness, not the model, owns `msgs`, `step` and `seen`:

* max steps        - stops an endless loop (safe failure "max_steps")
* repeat detection - an identical call is not executed again; the model gets its earlier
                     result back. Too many repeats in a row -> safe failure "repeats"
* completion check - a mechanical gate against stopping too early; on reject the goal
                     is restated so the model does not lose it

Messages are append-only so the prompt prefix stays cacheable across steps.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from .executor import err, execute, tool_message
from .harness import Harness
from .llm_client import LLMError
from .logger import new_run_id
from .prompts import build_messages
from .tools import tool_defs

PREFIX_MODES = ("stable", "shuffled")
_CUSTOMER_ID = re.compile(r"^c-\d+$", re.IGNORECASE)


@dataclass
class AgentResult:
    run_id: str
    status: str            # "done" | "max_steps" | "aborted_repeats" | "error"
    answer: str | None
    steps: int
    tool_calls: int        # attempted
    tool_ok: int           # executed and ok
    repeats: int           # calls answered with a repeat error
    wall_s: float
    trace: list[dict] = field(default_factory=list)   # one entry per step


# ------------------------------------------------------------- helpers
def _args(call: dict) -> dict | str:
    """Call arguments as a dict when possible (they may arrive as a JSON string)."""
    raw = call.get("function", {}).get("arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw
    return raw


def call_key(call: dict) -> str:
    """Identity of a call for repeat detection: name + sorted args, customer IDs upper-cased."""
    name = call.get("function", {}).get("name", "")
    args = _args(call)
    if isinstance(args, dict):
        args = {k: v.strip().upper() if isinstance(v, str) and _CUSTOMER_ID.match(v.strip())
                else v for k, v in args.items()}
        return name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return name + str(args)


def _looks_like_tool_call(text: str) -> bool:
    """The model wrote a call or a JSON blob as text instead of answering."""
    s = text.strip().strip("`").strip()
    if s.lower().startswith("json"):
        s = s[4:].strip()
    if "<tool_call>" in s or '"arguments"' in s or '"function"' in s:
        return True
    if s[:1] in "{[":
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False
    return False


def completion_check(task: str, msgs: list[dict], answer: str,
                     done_reason: str | None = None) -> tuple[bool, str]:
    """Mechanical "is the model really finished?" gate. Returns (passed, reason).

    Never looks at expected answers: grading lives in the eval script.
    Kept a plain function so a VERIFY state can replace it later.
    """
    if done_reason == "length":
        return False, "answer was truncated"
    if not answer or not answer.strip():
        return False, "empty answer"
    # a repeat is only answered after a real execution, so any tool message will do
    if not any(m.get("role") == "tool" for m in msgs):
        return False, "no tool was used; look up the data or calculate with a tool first"
    if _looks_like_tool_call(answer):
        return False, "the answer is a raw tool call or JSON, not a sentence for the user"
    return True, ""


def _system_variant(system_msg: dict, mode: str) -> dict:
    if mode == "shuffled":
        return {**system_msg,
                "content": f"Current time: {datetime.now().isoformat()}\n" + system_msg["content"]}
    return system_msg


def _tools_variant(tools: list[dict], mode: str) -> list[dict]:
    if mode == "shuffled":
        tools = list(tools)
        random.shuffle(tools)
    return tools


# ------------------------------------------------------------- the loop
def run_agent(h: Harness, task: str, *, profile: str = "fast", max_steps: int = 8,
              tool_names: list[str] | None = None, prompt: str = "tool_assistant",
              max_repeats: int = 3, run_id: str | None = None,
              prefix_mode: str = "stable") -> AgentResult:
    """Run `task` until the model gives an accepted final answer or a limit is hit.

    Never raises: runtime errors become status "error" plus a safe_failure log line.
    """
    if prefix_mode not in PREFIX_MODES:
        raise ValueError(f"prefix_mode must be one of {PREFIX_MODES}")
    run_id = run_id or new_run_id()
    system = h.prompts.get(prompt)
    msgs = build_messages(system.render(), user=task)       # append-only from here on
    tools = tool_defs(tool_names)
    seen: dict[str, dict] = {}
    res = AgentResult(run_id=run_id, status="max_steps", answer=None, steps=0,
                      tool_calls=0, tool_ok=0, repeats=0, wall_s=0.0)
    consecutive = 0
    t0 = time.perf_counter()

    def finish(status: str, **log_fields) -> AgentResult:
        res.status, res.wall_s = status, time.perf_counter() - t0
        if status != "done":
            h.log.write("safe_failure", run_id=run_id, prompt=system.id, steps=res.steps,
                        **log_fields)
        return res

    for step in range(1, max_steps + 1):
        res.steps = step
        request = [_system_variant(msgs[0], prefix_mode)] + msgs[1:]
        try:
            r = h.llm(profile, request, tools=_tools_variant(tools, prefix_mode),
                      run_id=run_id, prompt_id=system.id, log=False)
        except LLMError as e:
            return finish("error", reason="llm_error", error=str(e)[:300])
        done_reason = r.raw.get("done_reason")
        # logged here, not inside h.llm, because done_reason is only known after the call
        h.log.llm_call(run_id=run_id, profile=profile, result=r, prompt_id=system.id,
                       step=step, variant=prefix_mode, done_reason=done_reason)
        msgs.append(r.message)
        entry = {"step": step, "ttft_s": round(r.ttft_s, 3), "prompt_s": round(r.prompt_s, 3),
                 "load_s": round(r.load_s, 3), "in_tokens": r.prompt_tokens,
                 "out_tokens": r.output_tokens, "wall_s": round(r.wall_s, 3),
                 "done_reason": done_reason,
                 "calls": [c.get("function", {}).get("name", "") for c in r.tool_calls],
                 "repeats_this_step": 0, "content": r.content, "events": []}
        res.trace.append(entry)

        if r.tool_calls:
            for call in r.tool_calls:
                res.tool_calls += 1
                key = call_key(call)
                if key in seen:
                    res.repeats += 1
                    consecutive += 1
                    entry["repeats_this_step"] += 1
                    prev = json.dumps(seen[key], ensure_ascii=False, default=str)
                    result = err(f"Repeated call. You already got: {prev}. "
                                 "Use it or give the final answer.")
                    entry["events"].append({"call": call, "result": result, "repeat": True})
                    msgs.append(tool_message(call, result))
                    if consecutive >= max_repeats:
                        return finish("aborted_repeats", reason="repeats",
                                      repeats=res.repeats)
                    continue
                consecutive = 0
                result = execute(call, logger=h.log, run_id=run_id)
                res.tool_ok += result["ok"]
                seen[key] = result
                entry["events"].append({"call": call, "result": result, "repeat": False})
                msgs.append(tool_message(call, result))
            continue

        # no tool calls: the model claims it is finished
        passed, reason = completion_check(task, msgs, r.content, done_reason)
        entry["check"] = "pass" if passed else reason
        if passed:
            res.answer = r.content.strip()
            return finish("done")
        msgs.append({"role": "user", "content": f"Not finished: {reason}. Goal: {task}"})

    return finish("max_steps", reason="max_steps")
