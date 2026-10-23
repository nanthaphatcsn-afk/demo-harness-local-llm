# Harness: Session 2 starter code + Session 3 agent loop

Source code for **Session 2: Harness Foundation & Tools** (Sat 10 Oct 2026), extended with the
**Session 3 hands-on: from one tool call to an agent loop** (W11, slide 9).
It implements modules **M1–M8** and **M14 (logging)** on top of local Ollama models.

## Setup

```bash
cd Harness-01
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull deepseek-r1:1.5b           # profile fast
ollama pull qwen3:4b                   # profiles reasoning and fallback

python scripts/check_setup.py          # checks Ollama, models, prompts, one test call
python -m unittest discover -s tests   # 58 offline tests, no Ollama needed
```

Requires Python 3.10+ and a running Ollama.
To use other models, edit `config.yaml`. No code changes are needed.

### Model profiles (`config.yaml`)

| Profile | Model | `think` | `num_predict` | Used for |
|---|---|---|---|---|
| `fast` | `deepseek-r1:1.5b` | false | 512 | small model; agent loop comparison |
| `reasoning` | `qwen3:4b` | false | 2048 | main agent-loop model; prefix check |
| `fallback` | `qwen3:4b` | true | 4096 | escalation after repeated failures (slow) |

Model tags must match `ollama list` exactly. Before Session 3 every profile used
`deepseek-r1:1.5b`; see `CONFIG_CHANGE.md` for the change and its impact.

## Folder layout

| Path | Module | Slide |
|---|---|---|
| `harness/llm_client.py` | **M1** Runtime connector: native `/api/chat` or OpenAI-compatible `/v1`; moves leaked reasoning to `ChatResult.thinking` | M1 |
| `config.yaml`, `harness/config.py` | **M2** Model config: profiles `fast`, `reasoning`, `fallback` | M2 |
| `prompts/*.txt`, `harness/prompts.py` | **M3** Prompt manager: versioned templates `name@vN.txt`, ordered from stable to variable | M3 |
| `harness/schemas.py` | **M4** Structured output: `Ticket` Pydantic model → JSON schema in `format` | M4 |
| `harness/validation.py` | **M5** Validation & recovery: schema check → domain rules → retry / repair / fallback / safe failure | M5 |
| `harness/tools.py` | **M6** Tool registry: `@tool(permission=...)`, `tool_defs()` | M6 |
| `harness/executor.py` | **M7** Tool executor: look up → validate → permission → timeout → ok/err | M7 |
| `harness/agent.py` | **M8** Agent loop: `run_agent()`, `completion_check()` | M8 (Session 3) |
| `harness/logger.py` | **M14** Logging: one JSONL line per LLM call and per tool call | Logging |
| `harness/sample_tools.py` | `calculator`, `lookup_customer`, `current_datetime`, plus `update_credit_limit` (write, blocked by default) | Hands-on 3a |
| `harness/harness.py` | `Harness` facade: `h.llm(profile, messages, schema=..., tools=..., log_extra=...)` | — |
| `harness/data_store.py`, `data/customers.csv` | Sample customer data used by tools and domain rules | — |
| `data/ticket_inputs.jsonl` | 20 test inputs with expected values | Hands-on 2 |
| `data/tool_prompts.jsonl` | 10 single-step tool-calling prompts | Homework (S2) |
| `data/agent_tasks.jsonl` | 10 multi-step tasks with expected tools and values | Session 3 |
| `scripts/` | Demos, hands-on and homework runners (below) | — |
| `tests/test_offline.py` | 15 offline tests using a fake client | — |
| `tests/test_agent_offline.py` | 12 offline agent-loop tests | Session 3 |
| `tests/test_llm_client_offline.py` | 10 offline connector tests (reasoning split, loop through M1) | Session 3 |
| `DEV_AGENT_LOOP.md` | Developer brief for the Session 3 hands-on | Session 3 |
| `DEV_FIXES_01.md` | Follow-up brief: fixes after review | Session 3 |
| `harness/state_machine.py` | **M9** State machine: states, transition table, `run_state_machine()` | Session 3 (2) |
| `harness/classifier.py`, `harness/planner.py`, `harness/verifier.py`, `harness/router.py` | CLASSIFY (rules), PLAN (schema), VERIFY (code checks V1–V5), router | Session 3 (2) |
| `scripts/sm_eval.py`, `scripts/compare_configs.py`, `scripts/_grading.py` | State-machine eval, loop vs state-machine comparison, shared grading | Session 3 (2) |
| `tests/test_state_machine_offline.py` | 21 offline tests: transitions, classifier, verifier, runner | Session 3 (2) |
| `DEV_STATE_MACHINE.md` | Developer brief for the state machine + router hands-on | Session 3 (2) |
| `CONFIG_CHANGE.md` | Model-profile change for Session 3: what changed and its impact | Session 3 |

## Live demos (teacher)

```bash
python scripts/demo_structured.py                      # M1–M5 on one message
python scripts/demo_structured.py "c-105: refund still not received after 30 days"
python scripts/demo_structured.py --prompt extract_ticket@v1 --no-repair "..."   # weaker prompt, no repair

python scripts/demo_tool_call.py                       # one tool-calling round trip
python scripts/demo_tool_call.py --profile reasoning "Which tier is C-103?"
python scripts/demo_tool_call.py --failure             # made-up tool, bad args, unknown customer,
                                                       # unsafe expression, permission denied
```

## Hands-on 2: validated structured output (Session 2)

1. Read `llm_client.py`, `config.yaml` and `validation.py`.
2. Run the evaluation on 20 inputs with two models:
   ```bash
   python scripts/hands_on2_eval.py                          # fast (1.5B) vs reasoning (4B)
   python scripts/hands_on2_eval.py --no-repair              # plain retry, to compare
   python scripts/hands_on2_eval.py --prompt extract_ticket@v1   # weaker prompt, to compare
   ```
3. Report **first-try validity**, **validity after repair**, field accuracy and mean latency.
   Per-input results are saved to `logs/hands_on2_results.csv`.
4. Extension: add one domain rule of your own to `check_ticket()` in `validation.py`.

Inputs `t14`, `t15` and `t19` are designed to trigger the domain rules or the ID pattern.

**Checkpoint:** show the TA one validated output and the call log.

## Hands-on 3a: your first tool call (Session 2)

1. Read `tools.py`, `executor.py` and `sample_tools.py`.
2. Run `python scripts/demo_tool_call.py` with several prompts and both profiles.
3. Add a tool of your own in `sample_tools.py` (a Pydantic `args` class, a docstring and `@tool`).

**Checkpoint:** one end-to-end tool call, with `tool_call` events visible in the log.

## Homework: reliable tool calling (Session 2)

```bash
python scripts/tool_call_eval.py                        # success rate per model on 10 prompts
python scripts/tool_call_eval.py --answer               # also generate the final answers
python scripts/tool_call_eval.py --tools lookup_customer,calculator   # offer fewer tools
```

Results are saved to `logs/tool_call_results.csv`.

## Hands-on (Session 3): agent loop

Slide 9 asks for three things: build the loop, run 10 multi-step tasks, and compare a stable
with a shuffled prompt prefix.

### The loop (`harness/agent.py`)

`run_agent(h, task, profile=..., max_steps=8, max_repeats=3, prefix_mode="stable")` wraps the
Session 2 round trip (model → tool → model) in a controlled loop and returns an `AgentResult`
(`status`, `answer`, `steps`, `tool_calls`, `tool_ok`, `repeats`, `wall_s`, per-step `trace`).

| Failure (slide 7) | Fix in the loop |
|---|---|
| Made-up tool, bad arguments | executor (M7) returns the error as an observation; the loop continues |
| Endless loop | `max_steps` → status `max_steps`; repeat detection: an identical call (same name + args, customer IDs upper-cased) is not executed again, the model gets its earlier result back; `max_repeats` in a row → status `aborted_repeats` |
| Stops too early | `completion_check()`: the answer must be non-empty, at least one tool must have run, it must not be a raw tool call / JSON, and it must not be truncated (`done_reason = length`) |
| Loses the goal | when the completion check rejects, the loop appends `Not finished: <reason>. Goal: <task>` |

Other properties:

- Messages are **append-only**, so the prompt prefix can be reused by Ollama's cache.
- Reasoning never enters the history: M1 (`split_thinking()` in `llm_client.py`) moves text
  before `</think>` in the content, and Ollama's own `thinking` field, to `ChatResult.thinking`.
  The completion check and the next step only see the answer. Each `llm_call` log line records
  `thinking_chars`, so a model that still reasons stays visible in the log.
- Runtime errors never reach the caller: status `error` plus a `safe_failure` log line.
- `completion_check()` never looks at expected answers; grading lives only in `agent_eval.py`.

### Commands

```bash
python scripts/agent_eval.py --only m01 --verbose    # checkpoint 1: full loop log for one task
python scripts/agent_eval.py                         # checkpoint 2: success rate on 10 tasks (fast + reasoning)
python scripts/agent_eval.py --profiles reasoning --max-steps 6
python scripts/prefix_check.py                       # stable vs shuffled prefix, TTFT per step (task m07)
python scripts/prefix_check.py --task m02 --repeats 3
```

The summary reports two numbers per profile:

- **answer accuracy**: the final answer contains one of `expect_any` and all of `expect_all`
  (commas, spaces and case ignored);
- **full success**: answer accuracy **and** status `done` **and** every expected tool was executed.

The gap between them is "right answer, but skipped a tool" (e.g. did the arithmetic itself instead
of calling `calculator`). The per-task CSV has `answer_match`, `tool_adherence` and `success`
(= full success). Task `m10` (days left in the month) is graded against today's date in
Asia/Bangkok.

`--verbose` prints every step: token counts, TTFT, `done_reason` (`[TRUNCATED]` when the call hit
`num_predict`), each tool call with its result, `REPEAT` calls, completion-check verdicts and the
final answer.

`prefix_check.py` warms the model up, then runs the same task in `stable` mode (same system
prompt and tool list every step) and `shuffled` mode (current time prepended to the system
prompt and tools shuffled every step). `in_tokens` is Ollama's `prompt_eval_count`, i.e. only
newly evaluated tokens: in `stable` it should drop after step 1, in `shuffled` it should stay
near the full prompt length. Do not run anything else against the model meanwhile.

Low success with the 1.5B model is a valid result, not necessarily a bug: check repeats, early
stops and truncated steps in `logs/session3_agent_steps.csv` before changing the loop.

**Checkpoint:** loop log for one task + success rate on 10 tasks.

## Hands-on (Session 3): state machine + router

Slide 18: wrap the agent in INTAKE → CLASSIFY → PLAN → ACT → VERIFY → DONE, route each state to
the right executor, and compare with the single-model loop on the same 10 tasks.
*Loop inside a state, state machine across the workflow*: `run_agent()` is the worker inside ACT.
Full brief and design decisions (D1–D6, hypotheses H1–H5): `DEV_STATE_MACHINE.md`.

| State | Executor | What it does |
|---|---|---|
| INTAKE | code | rejects an empty task; extracts customer IDs (`C-\d+`) |
| CLASSIFY | rules (`fast` only if no rule matches) | picks a category and its tool allow-list, e.g. `customer_calc` → `lookup_customer`, `calculator` |
| PLAN *(optional)* | `reasoning` | a short tool plan (Pydantic schema, one repair retry); sent to ACT as one extra user message, so the cached prefix is unchanged |
| ACT | `--act-profile` (default `reasoning`) | `run_agent()` with the allow-list, max 6 steps per attempt; a tool outside the list sends the task back to CLASSIFY once (→ `general`) |
| VERIFY | code | V1 well-formed · V2 every customer ID looked up · V3 a number in the answer comes from a tool result (numbers from the task text removed) · V4 calculator used · V5 not-found / permission errors acknowledged. On failure the reasons go back to ACT in the same history, at most 3 ACT attempts in total |

Every move must be in the transition table (`TRANSITIONS`); anything else raises
`IllegalTransition`. Moves, routes and VERIFY reports are logged as `transition`, `route` and
`verify` events. VERIFY is deliberately code, not a model judge (D5), and never sees the expected
answers or the plan.

```bash
python scripts/sm_eval.py --plan on --only m02 --verbose   # checkpoint 1: transition log for one task
python scripts/sm_eval.py --plan off                       # config C
python scripts/sm_eval.py --plan on                        # config D
python scripts/sm_eval.py --plan on --act-profile fast     # config E (optional)
python scripts/compare_configs.py                          # checkpoint 2: comparison table + per-task matrix
```

| Config | Mode | PLAN | ACT profile |
|---|---|---|---|
| A | loop (`agent_eval.py`) | — | `fast` |
| B | loop (`agent_eval.py`) | — | `reasoning` |
| C | state machine | off | `reasoning` |
| D | state machine | on | `reasoning` |
| E *(optional)* | state machine | on | `fast` |

C vs B isolates the state machine; D vs C isolates PLAN. Grading is shared with the loop
(`scripts/_grading.py`), so the numbers are comparable.

**Checkpoint:** state-transition log for one task + the comparison table.

## Logs

All logs are in `logs/` and are not committed to git. Files are prefixed by session so that
Session 3 runs (new model profiles) never overwrite Session 2 results:

| File | Content |
|---|---|
| `session2_calls.jsonl` | Session 2 call log (2 and 10 Oct 2026, every profile on `deepseek-r1:1.5b`) |
| `session2_hands_on2_results.csv` | Hands-on 2 results, Session 2 |
| `session2_tool_call_results.csv` | Session 2 homework results |
| `session3_calls.jsonl` | Call log for all new runs (set in `config.yaml` → `logging.file`) |
| `session3_agent_results.csv` | `agent_eval.py`: one row per task and profile |
| `session3_agent_steps.csv` | `agent_eval.py`: one row per loop step (`done_reason`, tokens, TTFT, check verdict) |
| `session3_prefix_check.csv` | `prefix_check.py`: per-step stable vs shuffled timings |
| `session3_sm_{plan,noplan}_{profile}_results.csv` | `sm_eval.py`: one row per task (attempts, VERIFY, LLM calls, tokens, switches, transitions) |
| `session3_comparison.md` | `compare_configs.py`: table for configs A–E and the per-task matrix |

Re-running the Session 2 scripts writes `hands_on2_results.csv` / `tool_call_results.csv`
(without a prefix) and logs calls to `session3_calls.jsonl`.

```python
import pandas as pd
df = pd.read_json("logs/session3_calls.jsonl", lines=True)
df[df.event == "llm_call"].groupby("model")[["ttft_s", "gen_s", "wall_s"]].mean()
loop = df[(df.event == "llm_call") & df.step.notna()]          # agent-loop calls only
loop.groupby(["model", "variant", "step"])["in_tokens"].mean()
```

Events: `llm_call`, `tool_call`, `safe_failure`. Group rows from one task with `run_id`.
Every `llm_call` line has `thinking_chars`; agent-loop lines also carry `step`, `variant` and
`done_reason`. Compare results by
`model`, not by `profile`: the profile names point at different models in Session 2 and 3.

## Notes

- `think: false` is meant to turn off Qwen3 reasoning. `qwen3:4b` still reasons and writes it
  into the content, ending with `</think>`. M1 strips it (see `thinking_chars` in the log), but the
  model still spends the time and output tokens on it.
- `fallback` has `think: true`: thinking tokens count toward `num_predict`, and even a trivial
  question takes close to a minute.
- Ollama durations are in nanoseconds. The client converts them to seconds.
  With non-streaming calls, `ttft_s` is approximated as load time + prompt time.
- Tool arguments may arrive as a dict (Ollama native) or a JSON string (OpenAI-compatible API).
  The executor accepts both.
