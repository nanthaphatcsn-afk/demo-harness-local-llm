"""Compare the agent loop (A, B) with the state machine (C, D, E) on the same 10 tasks.

Prints one markdown table for a slide plus a per-task success matrix, and saves both
to logs/session3_comparison.md. Configurations whose result file is missing are skipped.

    python scripts/compare_configs.py
"""
import argparse
import csv
from pathlib import Path
from statistics import mean

import _path  # noqa: F401
from _path import ROOT

LOGS = ROOT / "logs"
# config -> (label, results file, filter on the profile column for the loop file)
CONFIGS = {
    "A": ("loop, fast", "session3_agent_results.csv", "fast"),
    "B": ("loop, reasoning", "session3_agent_results.csv", "reasoning"),
    "C": ("SM no plan, reasoning", "session3_sm_noplan_reasoning_results.csv", None),
    "D": ("SM plan, reasoning", "session3_sm_plan_reasoning_results.csv", None),
    "E": ("SM plan, ACT fast", "session3_sm_plan_fast_results.csv", None),
}


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _true(v: str) -> bool:
    return v == "True"


def load(key: str) -> list[dict] | None:
    """Rows of one configuration in a common shape, or None if it has not run."""
    _, name, profile = CONFIGS[key]
    path = LOGS / name
    if not path.exists():
        return None
    rows = _read(path)
    if profile is None:                       # state machine
        return [{"id": r["id"], "success": _true(r["success"]),
                 "answer_match": _true(r["answer_match"]), "wall_s": float(r["wall_s"]),
                 "llm_calls": int(r["llm_calls"]), "steps": int(r["act_steps"]),
                 "in": int(r["in_tokens"]), "out": int(r["out_tokens"]),
                 "switches": int(r["model_switches"])} for r in rows]
    rows = [r for r in rows if r["profile"] == profile]
    steps_path = LOGS / "session3_agent_steps.csv"
    steps = _read(steps_path) if steps_path.exists() else []
    out = []
    for r in rows:                            # loop: one LLM call per logged step
        s = [x for x in steps if x["run_id"] == r["run_id"]]
        out.append({"id": r["id"], "success": _true(r["success"]),
                    "answer_match": _true(r["answer_match"]), "wall_s": float(r["wall_s"]),
                    "llm_calls": len(s), "steps": int(r["steps"]),
                    "in": sum(int(x["in_tokens"]) for x in s),
                    "out": sum(int(x["out_tokens"]) for x in s), "switches": 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LOGS / "session3_comparison.md"))
    a = ap.parse_args()

    data = {k: rows for k in CONFIGS if (rows := load(k))}
    lines = ["| Config | Full success | Answer acc. | Mean wall (s) | Mean LLM calls | "
             "Mean steps | Tokens in/out (mean) | Model switches |",
             "|---|---|---|---|---|---|---|---|"]
    for k, rows in data.items():
        n = len(rows)
        lines.append(
            f"| {k} {CONFIGS[k][0]} | {sum(r['success'] for r in rows)}/{n} | "
            f"{sum(r['answer_match'] for r in rows)}/{n} | {mean(r['wall_s'] for r in rows):.1f} | "
            f"{mean(r['llm_calls'] for r in rows):.1f} | {mean(r['steps'] for r in rows):.1f} | "
            f"{mean(r['in'] for r in rows):.0f}/{mean(r['out'] for r in rows):.0f} | "
            f"{sum(r['switches'] for r in rows)} |")

    ids = sorted({r["id"] for rows in data.values() for r in rows})
    matrix = ["| Task | " + " | ".join(data) + " |", "|---|" + "---|" * len(data)]
    for i in ids:
        cells = []
        for rows in data.values():
            r = next((x for x in rows if x["id"] == i), None)
            cells.append("" if r is None else "✓" if r["success"]
                         else "½" if r["answer_match"] else "✗")
        matrix.append(f"| {i} | " + " | ".join(cells) + " |")

    text = ("## Configurations\n\n" + "\n".join(lines) + "\n\n"
            "## Per-task success\n\n✓ full success · ½ right answer, not a full success · ✗ wrong"
            "\n\n" + "\n".join(matrix) + "\n")
    print(text)
    Path(a.out).write_text(text, encoding="utf-8")
    missing = [k for k in CONFIGS if k not in data]
    if missing:
        print(f"(no results yet for: {', '.join(missing)})")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
