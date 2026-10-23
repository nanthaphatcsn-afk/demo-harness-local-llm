"""Hands-on (Session 3): stable vs shuffled prompt prefix on one task.

`stable` reuses the same system prompt and tool list every step, so Ollama can reuse the
cached prefix and only evaluates the new tokens. `shuffled` puts the current time in the
system prompt and shuffles the tools every step, which breaks the cache.

    python scripts/prefix_check.py                 # task m07 (3 lookups + calc -> several steps)
    python scripts/prefix_check.py --task m02 --repeats 3

Do not run anything else against the same model meanwhile: it overwrites the cache.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import _path  # noqa: F401
from _path import ROOT

import harness.sample_tools  # noqa: F401
from harness import Harness, run_agent

READ_TOOLS = ["calculator", "lookup_customer", "current_datetime"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="m07")
    ap.add_argument("--profile", default="reasoning")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "logs" / "session3_prefix_check.csv"))
    a = ap.parse_args()

    h = Harness()
    with open(ROOT / "data" / "agent_tasks.jsonl", encoding="utf-8") as f:
        it = next(t for t in map(json.loads, filter(str.strip, f)) if t["id"] == a.task)

    # warm-up so step 1 is not dominated by model loading
    w = h.llm(a.profile, [{"role": "user", "content": "Say OK."}], num_predict=4,
              log_extra={"variant": "warmup"})
    print(f"warm-up: load {w.load_s:.2f}s   model {w.model}   task {a.task}: {it['task']}")

    rows = []
    for variant in ("stable", "shuffled"):
        for rep in range(1, a.repeats + 1):
            res = run_agent(h, it["task"], profile=a.profile, max_steps=a.max_steps,
                            tool_names=it.get("tools", READ_TOOLS), prefix_mode=variant)
            print(f"  {variant:<8} #{rep}: status={res.status} steps={res.steps} "
                  f"wall={res.wall_s:.1f}s")
            for t in res.trace:
                rows.append({"variant": variant, "repeat": rep, "step": t["step"],
                             "in_tokens": t["in_tokens"], "prompt_s": t["prompt_s"],
                             "load_s": t["load_s"], "ttft_s": t["ttft_s"],
                             "done_reason": t["done_reason"], "run_id": res.run_id})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== per-step mean over {a.repeats} repeat(s), profile {a.profile} ===")
    print(f"{'step':>4}  {'variant':<8} {'in_tokens':>9} {'prompt_s':>9} {'ttft_s':>8}  n")
    for step in sorted({r["step"] for r in rows}):
        for variant in ("stable", "shuffled"):
            g = [r for r in rows if r["step"] == step and r["variant"] == variant]
            if g:
                print(f"{step:>4}  {variant:<8} {mean(r['in_tokens'] for r in g):>9.0f} "
                      f"{mean(r['prompt_s'] for r in g):>9.3f} "
                      f"{mean(r['ttft_s'] for r in g):>8.3f}  {len(g)}")
    print("\nNote: in_tokens = prompt_eval_count (only newly evaluated tokens). In `stable` it "
          "should drop sharply after step 1; in `shuffled` it should stay near the full prompt.")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
