"""Offline tests for the M1 connector: leaked reasoning is split off into ChatResult.thinking.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import harness.sample_tools  # noqa: E402,F401
from harness import Harness, LLMClient, run_agent  # noqa: E402
from harness.llm_client import split_thinking  # noqa: E402

LEAK = 'Okay, I should call {"name": "calculator", "arguments": {}} ... done.\n</think>\n\n'


class ReplayClient(LLMClient):
    """A real LLMClient whose HTTP layer replays raw Ollama /api/chat responses."""

    def __init__(self, responses, api="native"):
        super().__init__(api=api)
        self.responses = list(responses)
        self.bodies = []

    def _post(self, path, body, model):
        self.bodies.append(json.loads(json.dumps(body)))   # snapshot the request
        return self.responses.pop(0)


def native(content="", tool_calls=None, thinking=None, done_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if thinking is not None:
        msg["thinking"] = thinking
    return {"message": msg, "done_reason": done_reason, "prompt_eval_count": 10,
            "eval_count": 5}


class TestSplitThinking(unittest.TestCase):
    def test_no_opening_tag(self):
        self.assertEqual(split_thinking("a</think>b"), ("a", "b"))

    def test_with_opening_tag(self):
        self.assertEqual(split_thinking("<think>a</think>\n\nb"), ("a", "b"))

    def test_plain_answer(self):
        self.assertEqual(split_thinking("b"), ("", "b"))

    def test_splits_on_last_tag(self):
        self.assertEqual(split_thinking("x</think>y</think>z")[1], "z")


class TestNativeConnector(unittest.TestCase):
    def test_leaked_reasoning_is_removed(self):
        c = ReplayClient([native(LEAK + "The answer is 42.")])
        r = c.chat("qwen3:4b", [{"role": "user", "content": "q"}])
        self.assertEqual(r.content, "The answer is 42.")
        self.assertEqual(r.message["content"], "The answer is 42.")
        self.assertNotIn("thinking", r.message)
        self.assertIn('"arguments"', r.thinking)
        self.assertNotIn("</think>", r.thinking)

    def test_ollama_thinking_field_is_kept_out_of_history(self):
        c = ReplayClient([native("4", thinking="2 plus 2 is 4")])
        r = c.chat("qwen3:4b", [])
        self.assertEqual((r.content, r.thinking), ("4", "2 plus 2 is 4"))
        self.assertEqual(set(r.message), {"role", "content"})

    def test_both_sources_are_joined(self):
        c = ReplayClient([native("a</think>b", thinking="t")])
        self.assertEqual(c.chat("m", []).thinking, "t\na")

    def test_openai_api_is_cleaned_too(self):
        c = ReplayClient([{"choices": [{"message": {"content": "r</think> ok"}}]}], api="openai")
        r = c.chat("m", [])
        self.assertEqual((r.content, r.thinking, r.message["content"]), ("ok", "r", "ok"))


class TestAgentLoopThroughM1(unittest.TestCase):
    def test_leaked_answer_passes_completion_check_and_stays_out_of_history(self):
        lookup = {"function": {"name": "lookup_customer", "arguments": {"customer_id": "C-104"}}}
        client = ReplayClient([native(tool_calls=[lookup]),
                               native(LEAK + "Available credit is 440,000 THB.")])
        h = Harness(client=client, log_file=Path(tempfile.mkdtemp()) / "calls.jsonl")
        r = run_agent(h, "Available credit of C-104?", profile="reasoning")
        self.assertEqual(r.status, "done")
        self.assertEqual(r.steps, 2)          # not rejected as a raw tool call
        self.assertEqual(r.answer, "Available credit is 440,000 THB.")
        log = [json.loads(l) for l in h.log.path.read_text(encoding="utf-8").splitlines()]
        self.assertGreater([l for l in log if l["event"] == "llm_call"][-1]["thinking_chars"], 0)

    def test_reasoning_is_not_fed_back(self):
        lookup = {"function": {"name": "lookup_customer", "arguments": {"customer_id": "C-104"}}}
        client = ReplayClient([native(LEAK, tool_calls=[lookup]),
                               native("Available credit is 440,000 THB.")])
        h = Harness(client=client, log_file=Path(tempfile.mkdtemp()) / "calls.jsonl")
        run_agent(h, "Available credit of C-104?", profile="reasoning")
        sent = json.dumps(client.bodies[1]["messages"])
        self.assertNotIn("</think>", sent)
        self.assertNotIn("Okay, I should call", sent)


if __name__ == "__main__":
    unittest.main()
