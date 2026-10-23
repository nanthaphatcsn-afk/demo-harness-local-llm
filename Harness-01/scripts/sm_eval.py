"""Hands-on (Session 3): run the state machine on the same 10 tasks as the agent loop.

INTAKE -> CLASSIFY (rules) -> PLAN (optional) -> ACT (run_agent) -> VERIFY (code) -> DONE.
Grading is the same as agent_eval.py (scripts/_grading.py); VERIFY never sees it.

    python scripts/sm_eval.py --plan off                       # config C
    python scripts/sm_eval.py --plan on                        # config D
    python scripts/sm_eval.py --plan on --act-profile fast     # config E
    python scripts/sm_eval.py --plan on --only m02 --verbose   # checkpoint: transition log
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import _path  # noqa: F401
from _path import ROOT
from _grading import grade, load_tasks

import harness.sample_tools  # noqa: F401
from harness import Harness, run_state_machine


def print_run(res) -> None:
    """Transition log, each ACT attempt in short, and every VERIFY report."""
    print("    transitions:")
    for t in res.transitions:
        src_dst, _, reason = t.partition("(")
        print(f"      {src_dst.replace('->', ' → '):<20} {reason[:-1]}")   # drop the closing ")"
    for i, r in enumerate(res.act_runs, 1):
        print(f"    ACT attempt {i}: status={r.status} steps={r.steps} "
              f"tools={r.tool_ok}/{r.tool_calls}")
        for t in r.trace:
            flag = " [TRUNCATED]" if t["done_reason"] == "length" else ""
            for ev in t["events"]:
                f = ev["call"]["function"]
                tag = "REPEAT " if ev["repeat"] else "BLOCKED " if ev.get("disallowed") else ""
                print(f"      step {t['step']}{flag}: {tag}{f.get('name')}"
                      f"({json.dumps(f.get('arguments'), ensure_ascii=False)}) -> "
                      f"{json.dumps(ev['result'], ensure_ascii=False)[:120]}")
            if not t["events"]:
                text = " ".join(t["content"].split())
                print(f"      step {t['step']}{flag}: model: {text[:140]}"
                      + (f"  [check: {t['check']}]" if t.get("check") not in (None, "pass") else ""))
    for i, v in enumerate(res.verify_reports, 1):
        checks = " ".join(f"{k}={'-' if x is None else 'ok' if x else 'FAIL'}"
                          for k, x in v["checks"].items())
        print(f"    VERIFY {i}: {'pass' if v['passed'] else 'fail'}  {checks}"
              + (f"  ({'; '.join(v['failures'])})" if v["failures"] else ""))
    print(f"    answer : {res.answer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=("on", "off"), default="off")
    ap.add_argument("--act-profile", default="reasoning")
    ap.add_argument("--max-act-attempts", type=int, default=3)
    ap.add_argument("--act-max-steps", type=int, default=6)
    ap.add_argument("--strict-grounding", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated task ids, e.g. m01,m07")
    ap.add_argument("--verbose", action="store_true", help="print the transition log")
    ap.add_argument("--out", default="", help="default: logs/session3_sm_<plan>_<profile>_results.csv")
    a = ap.parse_args()

    use_plan = a.plan == "on"
    out = a.out or str(ROOT / "logs" / f"session3_sm_{'plan' if use_plan else 'noplan'}_"
                                       f"{a.act_profile}_results.csv")
    h = Harness()
    items = load_tasks(a.only)
    model = h.config.profile(a.act_profile).model
    print(f"=== state machine: plan {a.plan}, ACT {a.act_profile} ({model}), "
          f"max ACT attempts {a.max_act_attempts}, ACT max steps {a.act_max_steps} ===")

    rows = []
    for it in items:
        res = run_state_machine(h, it["task"], use_plan=use_plan, act_profile=a.act_profile,
                                max_act_attempts=a.max_act_attempts,
                                act_max_steps=a.act_max_steps,
                                strict_grounding=a.strict_grounding)
        success, missing, matched = grade(it, res)
        reports = res.verify_reports
        rows.append({
            "config": f"sm_{'plan' if use_plan else 'noplan'}_{a.act_profile}", "model": model,
            "id": it["id"], "run_id": res.run_id, "success": success, "answer_match": matched,
            "tool_adherence": not missing, "missing_tools": ",".join(missing),
            "status": res.status, "category": res.category,
            "classify_source": res.classify_source, "plan_ok": res.plan_ok,
            "act_attempts": res.act_attempts, "reclassified": res.reclassified,
            "act_steps": res.act_steps_total, "tool_calls": res.tool_calls,
            "tool_ok": res.tool_ok, "repeats": res.repeats, "llm_calls": res.llm_calls,
            "llm_calls_classify": res.llm_calls_by_state.get("CLASSIFY", 0),
            "llm_calls_plan": res.llm_calls_by_state.get("PLAN", 0),
            "llm_calls_act": res.llm_calls_by_state.get("ACT", 0),
            "in_tokens": res.in_tokens_total, "out_tokens": res.out_tokens_total,
            "model_switches": res.model_switches, "load_s": res.load_s_total,
            "wall_s": res.wall_s, "verify_runs": len(reports),
            "verify_first_pass": reports[0]["passed"] if reports else "",
            "verify_failures": " | ".join("; ".join(v["failures"]) for v in reports
                                          if not v["passed"]),
            "transitions": " | ".join(res.transitions), "answer": res.answer or ""})
        print(f"  {it['id']} {'OK  ' if success else 'FAIL'} {res.status:<6} "
              f"{res.category or '-':<15} plan={res.plan_ok} attempts={res.act_attempts} "
              f"steps={res.act_steps_total} tools={res.tool_ok}/{res.tool_calls} "
              f"llm={res.llm_calls} wall={res.wall_s:.1f}s"
              + (f"  missing={','.join(missing)}" if missing else ""))
        if a.verbose:
            print(f"    task   : {it['task']}")
            print_run(res)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    s = sum(x["success"] for x in rows)
    acc = sum(x["answer_match"] for x in rows)
    verified = [x for x in rows if x["verify_runs"]]
    first = sum(x["verify_first_pass"] is True for x in verified)
    by_status: dict[str, int] = {}
    for x in rows:
        by_status[x["status"]] = by_status.get(x["status"], 0) + 1
    print("\n=== state machine summary ===")
    print(f"full success {s}/{n} = {100 * s / n:.0f}%   answer accuracy {acc}/{n} = "
          f"{100 * acc / n:.0f}%   tool adherence {sum(x['tool_adherence'] for x in rows)}/{n}")
    print(f"mean wall {mean(x['wall_s'] for x in rows):.1f}s   "
          f"mean LLM calls {mean(x['llm_calls'] for x in rows):.1f}   "
          f"mean ACT steps {mean(x['act_steps'] for x in rows):.1f}   "
          f"mean ACT attempts {mean(x['act_attempts'] for x in rows):.1f}")
    print(f"VERIFY first-pass {first}/{len(verified)}   "
          f"reclassified {sum(x['reclassified'] for x in rows)}   "
          f"model switches {sum(x['model_switches'] for x in rows)}   "
          f"load_s total {sum(x['load_s'] for x in rows):.1f}")
    print(f"tokens in/out (mean per task) {mean(x['in_tokens'] for x in rows):.0f}/"
          f"{mean(x['out_tokens'] for x in rows):.0f}   status {by_status}")
    print(f"\nPer-task results: {out}")


if __name__ == "__main__":
    main()
