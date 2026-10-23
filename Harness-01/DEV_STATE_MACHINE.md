# DEV: Hands-on — State machine + router (W11, slide 18)

Developer brief for the second Session 3 hands-on, built on top of `run_agent()` from
`DEV_AGENT_LOOP.md`. Place this file in `Harness-01/`. Work on a new branch
`hands-on/task02` created from `hands-on/task01` **after** `DEV_FIXES_01.md` is merged
(the reasoning-leak fix in M1 must be in place, or VERIFY will read reasoning text as the answer).

Slide 18 asks for:

1. **State machine** — wrap the agent in INTAKE → CLASSIFY → PLAN → ACT → VERIFY → DONE with a transition table.
2. **Classifier + router** — rule-based classifier; fast profile for CLASSIFY, reasoning profile for PLAN/VERIFY.
3. **Compare** — same 10 tasks vs the single-model loop: success rate, latency, steps.

**Checkpoint:** state-transition log for one task + the comparison table.

Rule of thumb from slide 10: *loop inside a state, state machine across the workflow.*
`run_agent()` becomes the worker inside ACT.

---

## Part A — Design decisions (keep this section; it feeds the results presentation)

| # | Question | Decision | Why | How we measure it |
|---|---|---|---|---|
| D1 | What can VERIFY check without the expected answer? | Code-only checks (V1–V5 below). A number in the answer must equal **any one** numeric value from the tool results; **every customer ID in the task must have been looked up**. | Deterministic, cheap, testable offline, no extra model call or model switch. | VERIFY pass/fail counts per check; false-reject cases inspected by hand. |
| D2 | Where does a failed VERIFY go? | Back to **ACT** with the failure reasons, same message history. **At most 3 ACT attempts in total** (1 + 2 retries), then FAILED. | Time budget is limited; the reasons give the model a concrete fix target. | `act_attempts` per task; success gained by retries. |
| D3 | Is PLAN worth it? | Run **both**: state machine with PLAN and without PLAN. VERIFY is identical in both. | Our tasks are 2–4 steps; a plan may add cost without gain. Keeping VERIFY independent of the plan makes the comparison fair. | Success, latency, LLM calls, tokens, model switches: plan vs no-plan. |
| D4 | What if CLASSIFY picks the wrong category? | One backtrack: if ACT needs a tool outside its category's allow-list, go back to CLASSIFY, which upgrades the task to `general` (all read tools). Max 1 reclassify per task. | Narrow tool lists help small models (slide 7, "too many tools"), but a wrong narrow list must not be a dead end. | `reclassified` count; success of reclassified tasks. |
| D5 | Slide says reasoning profile for VERIFY — we use code. | Deliberate deviation. | Our tasks have numeric, checkable answers. An LLM judge adds latency and a model switch for little gain here. | Mentioned as a limitation; LLM judge listed as future work. |
| D6 | Classifier: rules or model? | Hybrid (slide 14): rules first; `fast` profile with an enum schema only when no rule matches; `general` if both fail. | Rules are fast and predictable for these tasks; the model path still exists. | `classify_source` = rules / llm / default; accuracy on 20 prompts. |

### Hypotheses to state in the presentation

- H1: The state machine raises success over the single-model loop, mainly via VERIFY retries and narrower tool lists.
- H2: The state machine costs more time per task (extra ACT attempts, and PLAN when enabled).
- H3: PLAN does not pay off on 2–4 step tasks (success gain smaller than the extra latency).
- H4: With rule-based CLASSIFY, routing cost (model switches, load time) is small.
- H5 (optional config E): a 4B plan helps a 1.5B executor.

---

## Part B — Components

### B1. States and transition table (`harness/state_machine.py`)

```python
class State(str, Enum):
    INTAKE = "INTAKE"; CLASSIFY = "CLASSIFY"; PLAN = "PLAN"; ACT = "ACT"
    VERIFY = "VERIFY"; DONE = "DONE"; FAILED = "FAILED"

TRANSITIONS: dict[State, set[State]] = {
    State.INTAKE:   {State.CLASSIFY, State.FAILED},
    State.CLASSIFY: {State.PLAN, State.ACT, State.FAILED},
    State.PLAN:     {State.ACT},                      # plan failure -> ACT without plan
    State.ACT:      {State.VERIFY, State.CLASSIFY, State.FAILED},
    State.VERIFY:   {State.DONE, State.ACT, State.FAILED},
    State.DONE:     set(),
    State.FAILED:   set(),
}
```

- A move not in the table raises `IllegalTransition` (a bug in the harness, not a model failure).
- `TaskState` (Pydantic): `run_id, task, state, category, classify_source, tools, plan, msgs,
  act_attempts, reclassified, verify_reports: list, answer, history: list[transition]`.
- Every transition is logged: `h.log.write("transition", run_id=..., src=..., dst=..., reason=..., ts...)`
  and appended to `TaskState.history`.
- SQLite persistence (slide 12) is **out of scope** for now; note it as future work.

Transition rules:

| From | To | When |
|---|---|---|
| INTAKE | CLASSIFY | task text non-empty; customer IDs extracted (`C-\d+`, upper-cased) |
| INTAKE | FAILED | empty task |
| CLASSIFY | PLAN / ACT | PLAN if `use_plan` else ACT |
| PLAN | ACT | always (if the plan is invalid after 1 retry, continue with `plan=None` and log `plan_failed`) |
| ACT | VERIFY | `run_agent` returned `done` |
| ACT | CLASSIFY | ACT stopped with `needs_reclassify` and `reclassified == 0` |
| ACT | FAILED | `max_steps`, `aborted_repeats`, `error`, or `needs_reclassify` a second time |
| VERIFY | DONE | all checks pass |
| VERIFY | ACT | a check failed and `act_attempts < 3` (feedback appended, see B5) |
| VERIFY | FAILED | a check failed and `act_attempts == 3` |

### B2. CLASSIFY — rule-based classifier (`harness/classifier.py`)

Categories and the tool allow-list each one gets in ACT:

| Category | Rule (case-insensitive, first match wins, in this order) | ACT tools |
|---|---|---|
| `write_request` | verb `raise / increase / decrease / change / update / set` + `limit` | `lookup_customer`, `update_credit_limit` |
| `date_calc` | `today / date / day / days / week / month / weekday / time` and no customer ID | `current_datetime`, `calculator` |
| `customer_calc` | customer ID + any of `% / percent / combined / total / sum / difference / more / less / over / under / after / pays / discount / vat / split / each / divide / if` | `lookup_customer`, `calculator` |
| `customer_lookup` | customer ID only | `lookup_customer` |
| `arithmetic` | no customer ID, contains digits and (an operator `+ - * /`, `%`, or the word `of / split / divide / times / per`) | `calculator` |
| `general` | fallback | all read tools |

- If no rule matches → `fast` profile with schema `{"category": enum[...]}`; invalid output → `general`.
  Record `classify_source` = `rules` / `llm` / `default`.
- `write_request` still goes through ACT so the executor's permission check denies the write and
  the eval stays comparable with the loop (m09 expects `update_credit_limit` to be attempted).
  Short-circuiting to FAILED without a model call is listed as future work.
- Use word boundaries (`\b`) for every keyword. The date rule only fires without a customer ID
  (m08 mentions "today" but is `customer_calc`).
- Offline tests: classify all 10 `agent_tasks.jsonl` tasks and all 10 `tool_prompts.jsonl` prompts
  and assert these labels (a dry run of the rules above produced exactly this):

  | Category | Prompts |
  |---|---|
  | `customer_calc` | m01, m02, m03, m04, m06, m07, m08 |
  | `customer_lookup` | m05, p01, p02, p05, p07, p09 |
  | `write_request` | m09 |
  | `date_calc` | m10, p04, p08 |
  | `arithmetic` | p03, p06, p10 |

### B3. PLAN (`harness/planner.py`, profile `reasoning`)

- Pydantic schema:
  ```python
  class PlanStep(BaseModel):
      tool: str            # must be in the category's allow-list
      purpose: str
  class Plan(BaseModel):
      customer_ids: list[str]
      steps: list[PlanStep]          # 1..6
      needs_calculation: bool
  ```
- Call `h.llm("reasoning", ..., schema=Plan.model_json_schema())`, validate; one retry with the
  validation error; then give up → `plan=None`. Do **not** use `ask_structured()` (its fallback
  profile has `think: true` and is very slow).
- New prompt `prompts/planner@v1.txt`: list the allowed tools for the category; ask for the
  shortest plan; no answer, only the plan.
- The plan reaches ACT as one extra user message **after** the task:
  `"Plan (follow it, use the tools): 1. lookup_customer — … 2. …"`.
  The system prompt and tool list stay unchanged, so the cached prefix survives.
- VERIFY must **not** read the plan (D3).

### B4. ACT (`run_agent` changes, keep the loop baseline unchanged)

Extend `run_agent()` with **optional** parameters; defaults keep today's behaviour and tests:

- `msgs: list[dict] | None = None` — continue an existing history (VERIFY retries, plan message).
  Return it in `AgentResult.messages`.
- `allowed: set[str] | None = None` — the category allow-list.
- `on_disallowed: str = "error"` — when the model calls a registered tool outside `allowed`:
  - `"error"`: answer with an error observation listing the allowed tools (loop continues);
  - `"stop"`: end the run with status `needs_reclassify` (used by the state machine).

Executor: add `allowed_tools: set[str] | None = None` to `execute()`; a known tool outside the set
returns `err("Tool X is not available in this step. Available: [...]")`. Today the executor resolves
any registered name, so offering a subset to the model does not stop it from calling another tool —
the allow-list must be enforced in the harness, not only in the request.

ACT profile: `reasoning` by default (`--act-profile` to change). `max_steps` per ACT attempt: 6.

### B5. VERIFY — code-only checks (`harness/verifier.py`)

Input: task text, category, final answer, and the executed tool events from **all** ACT attempts.
Output: `VerifyReport(passed: bool, failures: list[str], checks: dict[str, bool | None])`
(`None` = check not applicable). Pure function, no model call.

| Check | Rule | Applies to |
|---|---|---|
| V1 well-formed | answer non-empty, last ACT call not truncated, not a raw tool call / JSON (reuse `completion_check` helpers) | all |
| V2 coverage | every customer ID in the task (regex `C-\d+`, upper-cased) appears as the `customer_id` argument of an **executed** `lookup_customer` call (ok or LookupError both count) | tasks with customer IDs |
| V3 grounding | at least one number in the answer equals a numeric value from a successful tool result (D1). Numbers: regex `-?\d[\d,]*(?:\.\d+)?`, commas removed. Tool values: every int/float leaf (not bool) in `result`, recursively. Match if `abs(a-b) <= 0.01`. **Numbers that appear in the task text are removed from the answer first** (so echoing "500,000" from the question cannot pass). | `customer_calc`, `date_calc`, `arithmetic`; `customer_lookup` only if the answer contains a number; skipped when V5 applies |
| V4 tool use | `calculator` executed at least once | `customer_calc`, `date_calc`, `arithmetic` |
| V5 error acknowledged | if a tool returned `LookupError`, the answer contains one of `not found / no customer / does not exist / doesn't exist / unknown`; if a tool returned `Permission denied`, the answer contains one of `permission / not allowed / not authorized / not permitted / denied` | when such an error occurred |

On failure, append to the **same** `msgs` and go back to ACT:

```
Verification failed:
- <failure 1>          e.g. "C-108 was not looked up"
- <failure 2>          e.g. "the answer has no number taken from a tool result"
Fix this, use the tools if needed, then give the final answer. Goal: <task>
```

Known limits (put them on a slide):
- V3 "any one value" is loose: an answer can quote a looked-up value and still be wrong.
  A stricter variant — "if calculator ran, the answer must contain the **last** calculator value" —
  is implemented behind `strict_grounding=False` and is **not** part of the main comparison.
- V5 phrase lists overlap with the eval's `expect_any` for m05/m09. They are generic rules, not
  per-task answers, but it means VERIFY and grading are not fully independent on those two tasks.
- Non-numeric answers (tier, email) are only checked by V1/V2.

### B6. Router (`harness/router.py`)

Rule table from state to executor; every decision is logged (`event="route"`):

| State | Executor |
|---|---|
| INTAKE | code |
| CLASSIFY | rules → `fast` (deepseek-r1:1.5b) only if no rule matches |
| PLAN | `reasoning` (qwen3:4b) |
| ACT | `--act-profile` (default `reasoning`) |
| VERIFY | code (D5) |

Count **model switches** per task: consecutive LLM calls on different models. Sum `load_s` per task.

### B7. Runner (`harness/state_machine.py`)

```python
def run_state_machine(h, task, *, use_plan: bool, act_profile: str = "reasoning",
                      max_act_attempts: int = 3, act_max_steps: int = 6,
                      strict_grounding: bool = False, run_id: str | None = None) -> SMResult
```

`SMResult`: `run_id, status (done/failed), answer, category, classify_source, plan_ok,
act_attempts, reclassified, act_steps_total, tool_calls, tool_ok, repeats, llm_calls,
llm_calls_by_state, in_tokens_total, out_tokens_total, model_switches, load_s_total,
wall_s, verify_reports, transitions` (list of `"SRC->DST(reason)"`).
Never raises; unexpected exceptions → FAILED with a logged `safe_failure`.

---

## Part C — Evaluation

### C1. Configurations (same 10 tasks for all)

| Config | Mode | PLAN | ACT profile | Source |
|---|---|---|---|---|
| A | loop | — | `fast` | existing `agent_eval.py` run |
| B | loop | — | `reasoning` | existing `agent_eval.py` run |
| C | state machine | off | `reasoning` | new |
| D | state machine | on | `reasoning` | new |
| E *(optional, if time)* | state machine | on | `fast` | new (H5) |

C vs B isolates the state machine. D vs C isolates PLAN.

### C2. Script `scripts/sm_eval.py`

```
python scripts/sm_eval.py --plan off                 # config C
python scripts/sm_eval.py --plan on                  # config D
python scripts/sm_eval.py --plan on --act-profile fast   # config E
python scripts/sm_eval.py --plan on --only m02 --verbose # checkpoint: transition log for one task
```

- Reuse `grade()` from `agent_eval.py` (move it to `scripts/_grading.py`; both scripts import it).
  Grading uses the expected answers; VERIFY never does.
- Output files must **not** overwrite the loop results:
  `logs/session3_sm_{plan|noplan}_{act_profile}_results.csv`.
- `--verbose` prints the transition log, e.g.
  `INTAKE→CLASSIFY → CLASSIFY(customer_calc, rules)→PLAN → PLAN(3 steps)→ACT → ACT(4 steps, done)→VERIFY → VERIFY(fail: V4 calculator not used)→ACT → … → VERIFY(pass)→DONE`
  plus each VERIFY report.
- Summary per config: full success, answer accuracy, mean wall_s, mean LLM calls, mean ACT steps,
  mean ACT attempts, VERIFY first-pass rate, reclassified count, model switches, total load_s,
  in/out tokens.

### C3. Script `scripts/compare_configs.py`

Reads the loop results (A, B) and state-machine results (C, D, E if present) and prints one
markdown table ready for a slide:

| Config | Full success | Answer acc. | Mean wall (s) | Mean LLM calls | Mean steps | Tokens in/out | Model switches |
|---|---|---|---|---|---|---|---|

Also print per-task success as a matrix (rows m01–m10, columns A–E) so we can see *which* tasks
each configuration fixed or broke. Save both to `logs/session3_comparison.md`.

---

## Part D — Tests (offline, `tests/test_state_machine_offline.py`)

- Transition table: every legal move works; an illegal one raises `IllegalTransition`.
- Classifier: expected category for all 20 prompts (10 tasks + 10 Session 2 prompts).
- Verifier, one test per check, pass and fail:
  - V2 fails when one of two customer IDs was not looked up;
  - V3 passes on "635,000 THB" with tool values 380000/255000/635000; fails when the only number
    in the answer comes from the task text; tolerance 0.01;
  - V4 fails for `customer_calc` without a calculator call;
  - V5 for LookupError and Permission denied.
- Runner with FakeClient:
  - VERIFY fail → ACT retry → pass → DONE; `act_attempts == 2`, feedback message present in `msgs`;
  - three VERIFY failures → FAILED;
  - disallowed tool → `needs_reclassify` → CLASSIFY → `general` → DONE; second time → FAILED;
  - PLAN invalid twice → ACT runs with `plan=None`, `plan_failed` logged;
  - with PLAN on, request 1 of ACT has the same system message and tool list as with PLAN off
    (prefix unchanged; only an extra user message).
- Existing tests must keep passing (`run_agent` defaults unchanged).

---

## Constraints

- Do not edit `config.yaml`.
- Do not change the behaviour of `run_agent()` with default arguments, `agent_eval.py`, or the
  Session 2 scripts.
- Messages stay append-only inside a task (ACT retries continue the same history).
- No new dependencies (plain Python, Pydantic). LangGraph is out of scope (slide 13: adopt only
  after the plain-Python version works).

## Done when

- [ ] All offline tests pass (old + new).
- [ ] `sm_eval.py --plan on --only m02 --verbose` prints a readable transition log (checkpoint 1).
- [ ] Configs C and D have run on all 10 tasks.
- [ ] `compare_configs.py` prints the comparison table and the per-task matrix (checkpoint 2).
- [ ] README has a "Hands-on (Session 3): state machine + router" section.
- [ ] One commit per part (B1–B7, C, D) with clear messages.
