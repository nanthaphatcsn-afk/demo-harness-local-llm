"""Hands-on (Session 3): run the agent loop on 10 multi-step tasks and measure it.

A task counts as a success when
  1. the loop ends with status "done",
  2. every expected tool was executed, and
  3. the final answer contains one of expect_any (commas/spaces removed, case-insensitive).

    python scripts/agent_eval.py                         # fast + reasoning, max_steps 8
    python scripts/agent_eval.py --profiles reasoning --max-steps 6
    python scripts/agent_eval.py --only m01 --verbose    # full loop log for one task (checkpoint)
"""
import argparse
import calendar
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import _path  # noqa: F401
from _path import ROOT

import harness.sample_tools  # noqa: F401
from harness import Harness, run_agent

READ_TOOLS = ["calculator", "lookup_customer", "current_datetime"]


def days_left_in_month() -> list[str]:
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    left = calendar.monthrange(now.year, now.month)[1] - now.day
    return [str(left)]


def _norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "").lower()


def final_text(answer: str) -> str:
    """Drop reasoning that leaks into content before `</think>` (seen with qwen3 + think: false),
    so a number in the reasoning cannot count as a correct final answer."""
    return answer.rsplit("</think>", 1)[-1]


def grade(it: dict, res) -> tuple[bool, list[str], bool]:
    """(success, missing expected tools, answer matched). Grading lives here, never in the loop."""
    expect = days_left_in_month() if it.get("dynamic") == "days_left_in_month" else it["expect_any"]
    executed = {ev["call"]["function"].get("name") for t in res.trace for ev in t["events"]
                if not ev["repeat"]}
    missing = [t for t in it["expected_tools"] if t not in executed]
    answer = final_text(res.answer or "")
    matched = bool(answer.strip()) and any(_norm(x) in _norm(answer) for x in expect)
    return res.status == "done" and not missing and matched, missing, matched


def print_trace(res) -> None:
    for t in res.trace:
        flag = "  [TRUNCATED]" if t["done_reason"] == "length" else ""
        print(f"    step {t['step']}: in={t['in_tokens']} out={t['out_tokens']} "
              f"ttft={t['ttft_s']:.2f}s wall={t['wall_s']:.2f}s "
              f"done_reason={t['done_reason']}{flag}")
        for ev in t["events"]:
            f = ev["call"]["function"]
            tag = "REPEAT " if ev["repeat"] else ""
            print(f"      {tag}call   : {f.get('name')}"
                  f"({json.dumps(f.get('arguments'), ensure_ascii=False)})")
            print(f"      result : {json.dumps(ev['result'], ensure_ascii=False)[:200]}")
        if t["content"].strip() and "check" in t:
            text = " ".join(t["content"].split())
            print(f"      model  : {text[:160]}{'...' if len(text) > 160 else ''}")
        if "check" in t:
            print(f"      check  : {t['check']}")
    print(f"    answer : {res.answer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="fast,reasoning")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-repeats", type=int, default=3)
    ap.add_argument("--prompt", default="tool_assistant")
    ap.add_argument("--only", default="", help="comma-separated task ids, e.g. m01,m07")
    ap.add_argument("--verbose", action="store_true", help="print every step of the loop")
    ap.add_argument("--out", default=str(ROOT / "logs" / "session3_agent_results.csv"))
    ap.add_argument("--steps-out", default=str(ROOT / "logs" / "session3_agent_steps.csv"))
    a = ap.parse_args()

    h = Harness()
    with open(ROOT / "data" / "agent_tasks.jsonl", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        items = [it for it in items if it["id"] in keep]

    rows, steps = [], []
    for profile in a.profiles.split(","):
        model = h.config.profile(profile).model
        print(f"\n=== profile {profile} ({model}) ===")
        for it in items:
            res = run_agent(h, it["task"], profile=profile, max_steps=a.max_steps,
                            tool_names=it.get("tools", READ_TOOLS), prompt=a.prompt,
                            max_repeats=a.max_repeats)
            success, missing, matched = grade(it, res)
            truncated = sum(t["done_reason"] == "length" for t in res.trace)
            ttft = mean(t["ttft_s"] for t in res.trace) if res.trace else 0.0
            rows.append({"profile": profile, "model": model, "id": it["id"],
                         "run_id": res.run_id, "success": success, "status": res.status,
                         "steps": res.steps, "tool_calls": res.tool_calls,
                         "tool_ok": res.tool_ok, "repeats": res.repeats,
                         "truncated_steps": truncated, "missing_tools": ",".join(missing),
                         "answer_match": matched, "mean_ttft_s": round(ttft, 3),
                         "wall_s": round(res.wall_s, 3), "answer": res.answer or ""})
            for t in res.trace:
                steps.append({"profile": profile, "model": model, "id": it["id"],
                              "run_id": res.run_id, **{k: v for k, v in t.items()
                                                       if k not in ("events", "calls", "content")},
                              "calls": ",".join(t["calls"]), "check": t.get("check", "")})
            print(f"  {it['id']} {'OK  ' if success else 'FAIL'} {res.status:<15} "
                  f"steps={res.steps} tools={res.tool_ok}/{res.tool_calls} "
                  f"repeats={res.repeats} truncated={truncated} wall={res.wall_s:.1f}s"
                  + (f"  missing={','.join(missing)}" if missing else ""))
            if a.verbose:
                print(f"    task   : {it['task']}")
                print_trace(res)

    for path, data in ((a.out, rows), (a.steps_out, steps)):
        if not data:
            continue
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print("\n=== agent loop summary ===")
    for profile in a.profiles.split(","):
        r = [x for x in rows if x["profile"] == profile]
        if not r:
            continue
        s = sum(x["success"] for x in r)
        attempted = sum(x["tool_calls"] for x in r)
        by_status: dict[str, int] = {}
        for x in r:
            by_status[x["status"]] = by_status.get(x["status"], 0) + 1
        print(f"{profile:<10} {r[0]['model']:<20} success {s}/{len(r)} = {100 * s / len(r):.0f}%   "
              f"mean steps {mean(x['steps'] for x in r):.1f}   "
              f"tool ok {sum(x['tool_ok'] for x in r)}/{attempted}   "
              f"repeats {sum(x['repeats'] for x in r)}   "
              f"truncated {sum(x['truncated_steps'] for x in r)}")
        print(f"{'':<31} mean wall {mean(x['wall_s'] for x in r):.1f}s   "
              f"mean TTFT {mean(x['mean_ttft_s'] for x in r):.2f}s   status {by_status}")
    print(f"\nPer-task results: {a.out}\nPer-step traces : {a.steps_out}")


if __name__ == "__main__":
    main()
