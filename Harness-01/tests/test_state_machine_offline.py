"""Offline tests for the state machine (M9), classifier, verifier and router: no Ollama needed.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness.sample_tools  # noqa: E402,F401
from harness import (TRANSITIONS, Harness, IllegalTransition, State, TaskState,  # noqa: E402
                     run_state_machine)
from harness.classifier import CATEGORY_TOOLS, classify_rules  # noqa: E402
from harness.router import model_switches  # noqa: E402
from harness.state_machine import move  # noqa: E402
from harness.verifier import verify  # noqa: E402
from test_agent_offline import SnapshotClient  # noqa: E402


def make_harness(outputs):
    return Harness(client=SnapshotClient(outputs),
                   log_file=Path(tempfile.mkdtemp()) / "calls.jsonl")


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


def calls(*cs):
    return {"tool_calls": list(cs)}


def log_lines(h, event=None):
    lines = [json.loads(l) for l in h.log.path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if event is None or l["event"] == event]


def ev(name, args, result, ok=True):
    return {"call": call(name, **args), "repeat": False,
            "result": {"ok": True, "result": result} if ok else {"ok": False, "error": result}}


TASK = "What is the combined available credit of C-101 and C-108?"
LOOKUPS = calls(call("lookup_customer", customer_id="C-101"),
                call("lookup_customer", customer_id="C-108"))
CALC = calls(call("calculator", expression="380000 + 255000"))
GOOD = "The combined available credit is 635,000 THB."
PLAN = json.dumps({"customer_ids": ["C-101", "C-108"], "needs_calculation": True,
                   "steps": [{"tool": "lookup_customer", "purpose": "C-101"},
                             {"tool": "lookup_customer", "purpose": "C-108"},
                             {"tool": "calculator", "purpose": "add"}]})


class TestTransitions(unittest.TestCase):
    def test_every_legal_move_works(self):
        h = make_harness([])
        for src, dsts in TRANSITIONS.items():
            for dst in dsts:
                ts = TaskState(run_id="r", task="t", state=src)
                move(h, ts, dst, "test")
                self.assertEqual(ts.state, dst)
                self.assertEqual(ts.history[-1]["src"], src.value)
        self.assertEqual(len(log_lines(h, "transition")),
                         sum(len(d) for d in TRANSITIONS.values()))

    def test_illegal_move_raises(self):
        h = make_harness([])
        for src, dst in [(State.INTAKE, State.ACT), (State.PLAN, State.FAILED),
                         (State.DONE, State.ACT), (State.ACT, State.DONE)]:
            with self.assertRaises(IllegalTransition):
                move(h, TaskState(run_id="r", task="t", state=src), dst)


class TestClassifier(unittest.TestCase):
    EXPECTED = {
        "customer_calc": "m01 m02 m03 m04 m06 m07 m08",
        "customer_lookup": "m05 p01 p02 p05 p07 p09",
        "write_request": "m09",
        "date_calc": "m10 p04 p08",
        "arithmetic": "p03 p06 p10",
    }

    def test_all_20_prompts(self):
        want = {i: c for c, ids in self.EXPECTED.items() for i in ids.split()}
        items = [json.loads(l) for f in ("agent_tasks.jsonl", "tool_prompts.jsonl")
                 for l in (ROOT / "data" / f).read_text(encoding="utf-8").splitlines() if l]
        self.assertEqual(len(items), 20)
        for it in items:
            with self.subTest(it["id"]):
                self.assertEqual(classify_rules(it.get("task") or it["prompt"]), want[it["id"]])

    def test_no_rule_returns_none(self):
        self.assertIsNone(classify_rules("Hello there"))


class TestVerifier(unittest.TestCase):
    E101 = ev("lookup_customer", {"customer_id": "C-101"}, {"available_credit_thb": 380000})
    E108 = ev("lookup_customer", {"customer_id": "C-108"}, {"available_credit_thb": 255000})
    ECALC = ev("calculator", {"expression": "380000+255000"}, {"value": 635000})

    def test_all_pass(self):
        r = verify(TASK, "customer_calc", GOOD, [self.E101, self.E108, self.ECALC])
        self.assertTrue(r.passed)
        self.assertEqual(r.checks, {"V1": True, "V2": True, "V3": True, "V4": True, "V5": None})

    def test_v2_missing_lookup(self):
        r = verify(TASK, "customer_calc", GOOD, [self.E101, self.ECALC])
        self.assertFalse(r.checks["V2"])
        self.assertIn("C-108 was not looked up", r.failures)

    def test_v3_grounding_and_tolerance(self):
        self.assertTrue(verify(TASK, "customer_calc", "It is 635,000.009 THB.",
                               [self.E101, self.E108, self.ECALC]).checks["V3"])
        self.assertFalse(verify(TASK, "customer_calc", "It is 635,000.02 THB.",
                                [self.E101, self.E108, self.ECALC]).checks["V3"])

    def test_v3_task_numbers_do_not_count(self):
        task = "What is 18% of 245,000 baht?"
        r = verify(task, "arithmetic", "18% of 245,000 baht is 245,000.",
                   [ev("calculator", {"expression": "0.18*245000"}, {"value": 245000})])
        self.assertFalse(r.checks["V3"])

    def test_v4_no_calculator(self):
        r = verify(TASK, "customer_calc", GOOD, [self.E101, self.E108])
        self.assertFalse(r.checks["V4"])

    def test_v5_lookup_error(self):
        task = "Customer C-999 wants to know their available credit."
        e = [ev("lookup_customer", {"customer_id": "C-999"}, "LookupError: No customer", ok=False)]
        ok = verify(task, "customer_lookup", "Customer C-999 was not found.", e)
        bad = verify(task, "customer_lookup", "Their credit is fine.", e)
        self.assertTrue(ok.passed)
        self.assertIsNone(ok.checks["V3"])          # skipped when V5 applies
        self.assertFalse(bad.checks["V5"])

    def test_v5_permission_denied(self):
        task = "Raise C-103's credit limit to 100,000 THB."
        e = [ev("lookup_customer", {"customer_id": "C-103"}, {"credit_limit_thb": 80000}),
             ev("update_credit_limit", {"customer_id": "C-103", "new_limit_thb": 100000},
                "Permission denied: update_credit_limit needs 'write' permission", ok=False)]
        self.assertTrue(verify(task, "write_request", "I am not permitted to change it.", e).passed)
        self.assertFalse(verify(task, "write_request", "Done, the limit is raised.", e).passed)

    def test_v1_truncated(self):
        r = verify(TASK, "customer_calc", GOOD, [self.E101, self.E108, self.ECALC],
                   last_done_reason="length")
        self.assertFalse(r.checks["V1"])


class TestRunner(unittest.TestCase):
    def test_verify_fail_then_retry_then_done(self):
        h = make_harness([LOOKUPS, "It is 635,000 THB.", CALC, GOOD])
        r = run_state_machine(h, TASK, use_plan=False)
        self.assertEqual(r.status, "done")
        self.assertEqual(r.act_attempts, 2)
        self.assertEqual(r.answer, GOOD)
        sent = h.client.requests[2]["messages"]
        self.assertTrue(sent[-1]["content"].startswith("Verification failed:"))
        self.assertIn("calculator was not used", sent[-1]["content"])
        self.assertEqual([x["passed"] for x in r.verify_reports], [False, True])
        self.assertEqual(r.transitions[-1], "VERIFY->DONE(all checks pass)")

    def test_three_verify_failures_fail(self):
        h = make_harness([LOOKUPS, "It is 635,000 THB.", "Still 635,000 THB.", "635,000 THB."])
        r = run_state_machine(h, TASK, use_plan=False)
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.act_attempts, 3)
        self.assertTrue(r.transitions[-1].startswith("VERIFY->FAILED"))

    def test_disallowed_tool_reclassifies_once(self):
        h = make_harness([calls(call("current_datetime")), LOOKUPS, CALC, GOOD])
        r = run_state_machine(h, TASK, use_plan=False)
        self.assertEqual(r.status, "done")
        self.assertEqual((r.reclassified, r.category, r.classify_source), (1, "general", "reclassify"))
        self.assertIn("ACT->CLASSIFY(needs_reclassify: current_datetime)", r.transitions)
        self.assertEqual(len(log_lines(h, "tool_call")), 3)    # the blocked call never ran
        self.assertEqual({d["function"]["name"] for d in h.client.requests[1]["tools"]},
                         set(CATEGORY_TOOLS["general"]))

    def test_disallowed_tool_twice_fails(self):
        write = call("update_credit_limit", customer_id="C-101", new_limit_thb=1)
        h = make_harness([calls(call("current_datetime")), calls(write)])
        r = run_state_machine(h, TASK, use_plan=False)
        self.assertEqual(r.status, "failed")
        self.assertTrue(r.transitions[-1].startswith("ACT->FAILED(needs_reclassify again"))

    def test_plan_invalid_twice_acts_without_plan(self):
        h = make_harness(["not json", '{"steps": []}', LOOKUPS, CALC, GOOD])
        r = run_state_machine(h, TASK, use_plan=True)
        self.assertEqual(r.status, "done")
        self.assertFalse(r.plan_ok)
        self.assertEqual(r.llm_calls_by_state["PLAN"], 2)
        self.assertEqual(len(log_lines(h, "plan_failed")), 1)
        self.assertNotIn("Plan (follow it", json.dumps(h.client.requests[2]["messages"]))

    def test_plan_keeps_the_act_prefix(self):
        off = make_harness([LOOKUPS, CALC, GOOD])
        run_state_machine(off, TASK, use_plan=False)
        on = make_harness([PLAN, LOOKUPS, CALC, GOOD])
        r = run_state_machine(on, TASK, use_plan=True)
        self.assertTrue(r.plan_ok)
        a, b = off.client.requests[0], on.client.requests[1]     # first ACT request of each
        self.assertEqual(a["messages"][0], b["messages"][0])    # same system message
        self.assertEqual(a["tools"], b["tools"])                 # same tool list
        self.assertEqual(a["messages"], b["messages"][:-1])      # only one extra user message
        self.assertTrue(b["messages"][-1]["content"].startswith("Plan (follow it, use the tools)"))

    def test_empty_task_fails_at_intake(self):
        r = run_state_machine(make_harness([]), "  ", use_plan=False)
        self.assertEqual((r.status, r.transitions), ("failed", ["INTAKE->FAILED(empty task)"]))

    def test_metrics_and_routes(self):
        h = make_harness([PLAN, LOOKUPS, CALC, GOOD])
        r = run_state_machine(h, TASK, use_plan=True)
        self.assertEqual(r.llm_calls_by_state, {"PLAN": 1, "ACT": 3})
        self.assertEqual(r.model_switches, 0)                    # PLAN and ACT both reasoning
        routes = [(l["state"], l["executor"]) for l in log_lines(h, "route")]
        self.assertEqual(routes, [("INTAKE", "code"), ("CLASSIFY", "rules"),
                                  ("PLAN", "reasoning"), ("ACT", "reasoning"),
                                  ("VERIFY", "code")])

    def test_model_switches(self):
        self.assertEqual(model_switches(["a", "a", "b", "a"]), 2)


if __name__ == "__main__":
    unittest.main()
