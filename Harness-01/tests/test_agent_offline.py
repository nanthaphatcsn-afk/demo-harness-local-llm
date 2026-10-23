"""Offline tests for the agent loop (M8): no Ollama needed.

    python -m unittest discover -s tests -v
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness.sample_tools  # noqa: E402,F401
from harness import ChatResult, Harness, completion_check, run_agent  # noqa: E402
from test_offline import FakeClient  # noqa: E402


class SnapshotClient(FakeClient):
    """FakeClient that deep-copies each request and also replays ready-made ChatResults."""

    def chat(self, model, messages, **kw):
        if self.outputs and isinstance(self.outputs[0], ChatResult):
            self.requests.append({"model": model, "messages": messages, **kw})
            out = self.outputs.pop(0)
        else:
            out = super().chat(model, messages, **kw)
        self.requests[-1] = copy.deepcopy(self.requests[-1])
        return out


def make_harness(outputs):
    tmp = tempfile.mkdtemp()
    return Harness(client=SnapshotClient(outputs), log_file=Path(tmp) / "calls.jsonl")


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


def calls(*cs):
    return {"tool_calls": list(cs)}


def log_lines(h, event=None):
    lines = [json.loads(l) for l in h.log.path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if event is None or l["event"] == event]


TASK = "Is a 500,000 THB order within C-104's available credit?"
LOOKUP = call("lookup_customer", customer_id="C-104")
CALC = call("calculator", expression="500000 - 440000")


class TestAgentLoop(unittest.TestCase):
    def test_happy_path(self):
        h = make_harness([calls(LOOKUP), calls(CALC), "Over by 60,000 THB."])
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "done")
        self.assertEqual(r.steps, 3)
        self.assertEqual((r.tool_calls, r.tool_ok, r.repeats), (2, 2, 0))
        self.assertEqual(r.answer, "Over by 60,000 THB.")
        self.assertEqual([t["calls"] for t in r.trace],
                         [["lookup_customer"], ["calculator"], []])
        steps = [l["step"] for l in log_lines(h, "llm_call")]
        self.assertEqual(steps, [1, 2, 3])

    def test_repeat_is_not_executed(self):
        again = call("lookup_customer", customer_id="c-104")   # same call, lower-case ID
        h = make_harness([calls(LOOKUP), calls(again), "Available credit is 440,000 THB."])
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "done")
        self.assertEqual(r.repeats, 1)
        self.assertEqual(len(log_lines(h, "tool_call")), 1)
        tool_msgs = [m for m in h.client.requests[2]["messages"] if m["role"] == "tool"]
        self.assertIn("Repeated call", tool_msgs[-1]["content"])
        self.assertIn("440000", tool_msgs[-1]["content"])

    def test_too_many_repeats_aborts(self):
        h = make_harness([calls(LOOKUP)] * 4)
        r = run_agent(h, TASK, max_repeats=3)
        self.assertEqual(r.status, "aborted_repeats")
        self.assertEqual(r.repeats, 3)
        self.assertEqual(log_lines(h)[-1]["reason"], "repeats")

    def test_max_steps(self):
        new_calls = [calls(call("calculator", expression=f"{i} + 1")) for i in range(3)]
        h = make_harness(new_calls)
        r = run_agent(h, TASK, max_steps=3)
        self.assertEqual(r.status, "max_steps")
        self.assertEqual(r.steps, 3)
        fail = log_lines(h, "safe_failure")
        self.assertEqual(len(fail), 1)
        self.assertEqual(fail[0]["reason"], "max_steps")

    def test_completion_check_rejects_empty_answer(self):
        h = make_harness([calls(LOOKUP), "   ", "Available credit is 440,000 THB."])
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "done")
        self.assertEqual(r.steps, 3)
        last = h.client.requests[2]["messages"][-1]
        self.assertEqual(last["role"], "user")
        self.assertTrue(last["content"].startswith("Not finished: empty answer"))
        self.assertIn(f"Goal: {TASK}", last["content"])
        self.assertEqual(r.trace[1]["check"], "empty answer")

    def test_truncated_answer_is_rejected(self):
        cut = ChatResult(model="m", content="The answer is 6",
                         message={"role": "assistant", "content": "The answer is 6"},
                         raw={"done_reason": "length"})
        h = make_harness([calls(LOOKUP), cut, "Over by 60,000 THB."])
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "done")
        self.assertEqual(r.trace[1]["done_reason"], "length")
        self.assertEqual(r.trace[1]["check"], "answer was truncated")
        self.assertEqual(log_lines(h, "llm_call")[1]["done_reason"], "length")

    def test_unknown_tool_goes_back_as_observation(self):
        h = make_harness([calls(call("get_credit_limit", customer_id="C-104")),
                          calls(LOOKUP), "Available credit is 440,000 THB."])
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "done")
        self.assertEqual((r.tool_calls, r.tool_ok), (2, 1))
        tool_msg = h.client.requests[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("Valid tools", tool_msg["content"])

    def test_llm_error_is_a_safe_failure(self):
        from harness import LLMConnectionError

        class Down:
            def chat(self, *a, **kw):
                raise LLMConnectionError("down")

        h = make_harness([])
        h.client = Down()
        r = run_agent(h, TASK)
        self.assertEqual(r.status, "error")
        self.assertEqual(log_lines(h)[-1]["reason"], "llm_error")


class TestLogExtra(unittest.TestCase):
    def test_extra_fields_reach_the_log(self):
        h = make_harness(["hi", "hi"])
        h.llm("fast", [], log_extra={"step": 4, "variant": "stable"})
        h.llm("fast", [])
        first, second = log_lines(h, "llm_call")
        self.assertEqual((first["step"], first["variant"]), (4, "stable"))
        self.assertNotIn("step", second)


class TestPrefix(unittest.TestCase):
    OUTPUTS = [calls(LOOKUP), calls(CALC), "Over by 60,000 THB."]

    def test_stable_prefix_is_identical(self):
        h = make_harness(list(self.OUTPUTS))
        run_agent(h, TASK, prefix_mode="stable")
        reqs = h.client.requests
        self.assertEqual(len({json.dumps(q["messages"][0]) for q in reqs}), 1)
        self.assertEqual(len({json.dumps(q["tools"]) for q in reqs}), 1)
        for prev, nxt in zip(reqs, reqs[1:]):   # append-only: each request extends the last
            self.assertEqual(nxt["messages"][:len(prev["messages"])], prev["messages"])

    def test_shuffled_keeps_history_append_only(self):
        h = make_harness(list(self.OUTPUTS))
        run_agent(h, TASK, prefix_mode="shuffled")
        reqs = h.client.requests
        for q in reqs:
            self.assertTrue(q["messages"][0]["content"].startswith("Current time: "))
        for prev, nxt in zip(reqs, reqs[1:]):
            self.assertEqual(nxt["messages"][1:len(prev["messages"])], prev["messages"][1:])
        self.assertNotEqual(reqs[0]["messages"][0], reqs[1]["messages"][0])


class TestCompletionCheck(unittest.TestCase):
    TOOL = [{"role": "tool", "content": "{}"}]

    def test_rules(self):
        self.assertEqual(completion_check("t", self.TOOL, "Yes, 60,000 THB over."), (True, ""))
        self.assertFalse(completion_check("t", [], "Yes.")[0])
        self.assertFalse(completion_check("t", self.TOOL, '{"name": "calculator"}')[0])
        self.assertFalse(completion_check("t", self.TOOL, "<tool_call>x</tool_call>")[0])
        self.assertEqual(completion_check("t", self.TOOL, "Yes", "length"),
                         (False, "answer was truncated"))


if __name__ == "__main__":
    unittest.main()
