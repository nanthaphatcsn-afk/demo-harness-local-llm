# DEV: Hands-on — From one tool call to an agent loop (W11)

Developer brief for implementing the Session 3 hands-on (slide 9, *Agentic AI for Computer
Engineering W11 – Agent Loop*) on top of `Harness-01`. Place this file in `Harness-01/`.

## Goal

Turn the single round trip in `scripts/demo_tool_call.py` (model → tool → model) into a
controlled multi-step loop, `run_agent()`, then measure it.

Slide 9 asks for three things:

1. **Build the loop** — wrap the Session 2 round trip in `run_agent()` with max steps and repeat detection.
2. **Run 10 tasks** — multi-step tasks; log steps, tool calls, success and latency.
3. **Prefix check** — stable vs shuffled prefix on one task; compare TTFT per step.

**Checkpoint:** loop log for one task + success rate on 10 tasks.

## Context: what already exists (do not rewrite)

| Module | File | Already handles |
|---|---|---|
| M1 connector | `harness/llm_client.py` | `chat()` → `ChatResult` with `tool_calls`, `prompt_tokens` (= Ollama `prompt_eval_count`), `load_s`, `prompt_s`, `ttft_s` (≈ load + prompt, non-streaming) |
| M2 config | `config.yaml` | profiles `fast` / `reasoning` / `fallback` |
| M3 prompts | `prompts/tool_assistant@v1.txt`, `build_messages()` | stable → variable ordering |
| M6 registry | `harness/tools.py` | `tool_defs(names)` |
| M7 executor | `harness/executor.py` | unknown tool → error listing valid tools; bad args → schema error; permission check; timeout; never raises |
| M14 logging | `harness/logger.py` | `llm_call`, `tool_call`, generic `write(event, **fields)` |
| Facade | `harness/harness.py` | `h.llm(profile, messages, tools=..., run_id=..., prompt_id=...)` |

Slide 7 failure → fix mapping. ✅ = exists, 🔨 = build in this task:

- Made-up tool ✅ executor · Bad arguments ✅ executor
- Endless loop 🔨 repeat detection + max steps
- Stops too early 🔨 completion check
- Loses the goal 🔨 restate the goal when the completion check rejects
- Too many tools — out of scope here (state machine hands-on)

## Constraints

- Do not change the behaviour of existing scripts or break `tests/test_offline.py`.
- Keep the code style of the repo: small modules, type hints, docstrings, Pydantic, no new heavy deps.
- Every new module must be testable offline with the existing `FakeClient` pattern.
- Messages are **append-only** inside the loop (required for prefix caching). Never rewrite earlier messages.
- No timestamps, random IDs or reordering in the system prompt or tool list in the default path.

## Deliverables

### 1. `Harness.llm()` — pass extra log fields (small change)

`JsonlLogger.llm_call(**extra)` accepts extra fields but `Harness.llm()` cannot forward them.
Add a keyword `log_extra: dict | None = None` and pass it through, so loop calls can log
`step=`, `variant=` etc. Default behaviour unchanged.

### 2. `harness/agent.py` — `run_agent()`

```python
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
    trace: list[dict]      # one entry per step (see below)

def run_agent(h: Harness, task: str, *, profile: str = "fast", max_steps: int = 8,
              tool_names: list[str] | None = None, prompt: str = "tool_assistant",
              max_repeats: int = 3, run_id: str | None = None,
              prefix_mode: str = "stable") -> AgentResult: ...
```

Loop per slide 6 (M8). The harness owns `messages`, `step`, and `seen`:

1. `r = h.llm(profile, msgs, tools=tools, ..., log_extra={"step": step, "variant": prefix_mode})`; append `r.message`.
2. If `r.tool_calls`: for each call
   - key = `name + json.dumps(args, sort_keys=True)` (args may be a JSON string → parse first; normalise customer IDs to upper case).
   - **Repeat** → do NOT execute; append `tool_message(call, err("Repeated call. You already got: <previous result>. Use it or give the final answer."))`; `repeats += 1`. If consecutive repeats ≥ `max_repeats` → log `safe_failure` (reason `repeats`) and return `aborted_repeats`.
   - Otherwise `execute(call, logger=h.log, run_id=run_id)`; store result in `seen`; append `tool_message`.
3. If no tool calls → **completion check** (below). Pass → return `done`. Fail → append a user message `"Not finished: <reason>. Goal: <task>"` and continue (this is the "restate the goal" fix).
4. Out of steps → `h.log.write("safe_failure", run_id=..., reason="max_steps", steps=...)`, return `max_steps`.
5. Catch `LLMError` → log `safe_failure` (reason `llm_error`), return `error`. Never raise to the caller.

Trace entry per step: `step, ttft_s, prompt_s, load_s, in_tokens, out_tokens, wall_s, done_reason, calls=[names], repeats_this_step`.

**Completion check** (`completion_check(task, msgs, answer) -> tuple[bool, str]`), mechanical only:
- answer not empty / not only whitespace;
- at least one tool was executed in the run (every task in this hands-on needs data or arithmetic);
- the answer does not look like a raw tool call or JSON blob;
- the final call was not truncated (`done_reason != "length"`).

It must NOT look at expected answers — grading lives in the eval script, otherwise the eval is contaminated.
Keep it a plain function so the VERIFY state in the next hands-on can replace it.

**Prefix modes** (for deliverable 5):
- `stable` (default): tool list and system prompt built once, reused every step.
- `shuffled`: every step, `random.shuffle` a copy of the tool list AND prepend `f"Current time: {datetime.now().isoformat()}\n"` to a copy of the system message. Since messages must stay append-only otherwise, build the per-step request list as `[new_system] + msgs[1:]`, leaving `msgs` itself untouched.

Export `run_agent`, `AgentResult`, `completion_check` from `harness/__init__.py`.

### 3. `data/agent_tasks.jsonl` — 10 multi-step tasks

`data/tool_prompts.jsonl` is single-step only, so it cannot exercise the loop. Create:

```json
{"id": "m01", "task": "...", "expected_tools": ["lookup_customer", "calculator"], "expect_any": ["60,000", "60000"], "note": "..."}
```

Success = status `done` AND every `expected_tools` entry was executed AND the final answer contains at least one string from `expect_any` (compare after removing commas and spaces, case-insensitive).

Values come from `data/customers.csv` (available = credit_limit − balance):

| id | task | expected_tools | expect_any |
|---|---|---|---|
| m01 | Customer C-104 wants to place an order of 500,000 THB. Is that within their available credit, and by how much is it over or under? | lookup_customer, calculator | 60000 (over by 60,000; available 440,000) |
| m02 | What is the combined available credit of C-101 and C-108? | lookup_customer, calculator | 635000 |
| m03 | Which customer has more available credit, C-102 or C-105, and what is the difference? | lookup_customer | 14500 (C-102 15,000 vs C-105 500) |
| m04 | What is C-106's current balance including 7% VAT? | lookup_customer, calculator | 12840 |
| m05 | Customer C-999 wants to know their available credit. | lookup_customer | "not found", "no customer", "does not exist" (tool returns LookupError; model must report it) |
| m06 | What is 18% of C-107's available credit? | lookup_customer, calculator | 64800 |
| m07 | What is the total credit limit of C-101, C-104 and C-107 combined? | lookup_customer, calculator | 2250000 |
| m08 | If C-105 pays 49,500 THB off their balance today, what will their available credit be? | lookup_customer, calculator | 50000 |
| m09 | Raise C-103's credit limit to 100,000 THB. | update_credit_limit | "permission", "not allowed", "cannot" (write tool is blocked; model must say so) |
| m10 | How many days are left in the current month in Bangkok, not counting today? | current_datetime, calculator | dynamic: compute at eval time from `datetime.now(ZoneInfo("Asia/Bangkok"))` |

For m09 pass all tools including `update_credit_limit` (still blocked by permission). For the rest use the three read tools.
Double-check every expected number against `customers.csv` with a small script before committing.

### 4. `scripts/agent_eval.py` — run the 10 tasks

```
python scripts/agent_eval.py                         # fast + reasoning, max_steps 8
python scripts/agent_eval.py --profiles reasoning --max-steps 6
python scripts/agent_eval.py --only m01 --verbose    # print the full loop log for one task (checkpoint)
```

- Per task print: id, OK/FAIL, status, steps, tool calls (ok/attempted), repeats, wall_s.
- `--verbose`: print each step — model tool calls with args, tool results, final answer.
- Save per-task rows to `logs/agent_results.csv` and per-step traces to `logs/agent_steps.csv`.
- Summary per profile: success rate, mean steps, tool-call success (ok ÷ attempted), total repeats, mean wall_s, mean TTFT, count by status.
- Follow the CLI/printing style of `scripts/tool_call_eval.py`.

### 5. `scripts/prefix_check.py` — stable vs shuffled prefix

```
python scripts/prefix_check.py                 # task m07 (3 lookups + calc → several steps)
python scripts/prefix_check.py --task m02 --repeats 3
```

- Warm up the model once (one tiny call) so step 1 is not dominated by `load_s`.
- Run the same task in `stable` then `shuffled` mode, `--repeats` times each, same profile.
- Print a per-step table: step, variant, `in_tokens`, `prompt_s`, `ttft_s`, mean over repeats.
- Save to `logs/prefix_check.csv`.
- Print a one-line note: in `stable`, `in_tokens` (= `prompt_eval_count`, only newly evaluated tokens) should drop sharply after step 1; in `shuffled` it should stay ≈ full prompt length.
- Do not run anything else against the same model in between (it overwrites the cache).

### 6. Offline tests — `tests/test_agent_offline.py`

Use the `FakeClient` from `tests/test_offline.py` (import or copy). Note: `FakeClient` stores
`messages` by reference, so snapshot with `copy.deepcopy` if a test asserts on per-request messages.

Required cases:
- happy path: tool call → tool call → final answer → `done`, steps = 3;
- repeat detection: same call twice → second is not executed (check `tool_call` log lines) and the tool message says "Repeated";
- max steps: model keeps calling new tools → `max_steps` and a `safe_failure` line in the log;
- completion check reject: empty final answer → a "Not finished … Goal:" user message is appended and the loop continues;
- truncation: a final answer whose `raw` has `done_reason: "length"` is rejected and recorded in the trace
  (extend `FakeClient` or build `ChatResult(raw={"done_reason": "length"}, ...)` directly);
- unknown tool: executor error goes back as an observation; loop continues;
- prefix: in `stable` mode the system message and tool list are identical across requests; in `shuffled` mode `msgs` passed in later requests still starts with the original history (append-only preserved).

`python -m unittest discover -s tests -v` must pass with no Ollama.

### 7. README

Add a short "Hands-on (Session 3): agent loop" section to `README.md` with the commands above and the checkpoint.

## Environment notes

Local Ollama models (user-verified) and the profiles in `config.yaml`:

| Profile | Model | Capabilities | `think` | `num_predict` | Use |
|---|---|---|---|---|---|
| `fast` | `deepseek-r1:1.5b` | tools, thinking, completion | false | 512 | agent loop (small model); CLASSIFY later |
| `reasoning` | `qwen3:4b` | tools, thinking, completion | false | 2048 | agent loop (main); PLAN / VERIFY later |
| `fallback` | `qwen3:4b` | tools, thinking, completion | true | 4096 | escalation after repeated failures |
| *(optional, next hands-on)* `classify` | `gemma3:1b` | completion only — **no tools** | omit key | 512 | structured-output classifier only |

- The user owns `config.yaml`. Do not edit it; if a change seems needed, say so.
- Never offer tools to `gemma3:1b` (Ollama will reject the request). It is only valid with `schema=`.
- Model tags must match `ollama list` exactly (`qwen3:4b`, not `qwen3:4b-q4_K_M`).
- `agent_eval.py` default: `--profiles fast,reasoning`, so both models run the same 10 tasks.
- `prefix_check.py` default profile: `reasoning`.
- Expect low success rates with the 1.5B model. That is a valid result, not necessarily a bug —
  check `logs/agent_steps.csv` (repeats, early stops, `done_reason`) before changing loop logic.

### Truncation (`done_reason`)

The user has seen output cut off with `qwen3:4b`. To tell "model answered wrong" apart from
"model was cut off":

- Record `done_reason = r.raw.get("done_reason")` for every step in the trace and in the
  `llm_call` log line (via `log_extra`). `"length"` means the call hit `num_predict`.
- In `agent_eval.py`, add a `truncated_steps` count per task and a total in the summary.
- In `--verbose` mode, mark truncated steps with `[TRUNCATED]`.
- A final answer with `done_reason == "length"` fails the completion check with reason
  `"answer was truncated"` (the loop then continues, so the model gets another chance).
- If `load_s` is large only on step 1, that is model loading, not truncation — note it, don't fix it.

## Done when

- [ ] Offline tests pass (old + new).
- [ ] `agent_eval.py --only m01 --verbose` prints a readable loop log (checkpoint 1).
- [ ] `agent_eval.py` prints success rate over 10 tasks (checkpoint 2).
- [ ] `prefix_check.py` prints the per-step stable vs shuffled table.
- [ ] README section added.
