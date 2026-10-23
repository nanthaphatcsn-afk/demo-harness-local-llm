# DEV: Fixes after review — branch `hands-on/task01`

Follow-up brief for `DEV_AGENT_LOOP.md`. Place this file in `Harness-01/`.
Work on branch `hands-on/task01`. Run `python -m unittest discover -s tests` after every fix.

## Fix 1 — Normalise leaked reasoning in M1 (`harness/llm_client.py`)

### Problem

With `qwen3:4b` and `think: false`, reasoning text still arrives in `message.content`, ending with
`</think>` (usually with no opening `<think>`). Today it is only stripped at grading time
(`final_text()` in `agent_eval.py`). That is not enough:

- `run_agent()` appends `r.message` to `msgs`, so the reasoning is fed back every step →
  prompt grows, `in_tokens` / TTFT are inflated, prefix-check numbers are distorted.
- `out_tokens` grows → more `done_reason == "length"` truncations, much longer wall time.
- `completion_check()` → `_looks_like_tool_call()` can falsely reject an answer whose reasoning
  contains `"arguments"` or a JSON fragment.

The fix belongs in the connector (M1), so every caller gets clean output.

### Change

1. Add `thinking: str = ""` to `ChatResult`.
2. Add a pure helper in `llm_client.py`:

   ```python
   def split_thinking(content: str) -> tuple[str, str]:
       """Return (thinking, answer). Handles '<think>…</think>answer', '…</think>answer'
       (no opening tag) and plain 'answer'. Splits on the LAST '</think>'."""
   ```

3. In `_chat_native`:
   - thinking = Ollama's own `msg.get("thinking", "")` (present when `think: true`) plus whatever
     `split_thinking` removes from `content` (join with a newline if both exist);
   - `content` = the cleaned answer;
   - `message` = `{"role", "content": <cleaned>, "tool_calls"?}` — **never** put thinking into the
     message appended to history.
4. Apply the same `split_thinking` to `content` in `_chat_openai`.
5. Logging: in `JsonlLogger.llm_call` add `thinking_chars=len(result.thinking)` so leaks stay visible in the log.
6. Keep `final_text()` in `agent_eval.py` as a harmless second guard, but update its docstring to say M1 already strips it.

### Tests (`tests/test_offline.py` or a new `tests/test_llm_client_offline.py`)

- `split_thinking("a</think>b") == ("a", "b")`
- `split_thinking("<think>a</think>\n\nb") == ("a", "b")` (answer stripped of leading whitespace)
- `split_thinking("b") == ("", "b")`
- `split_thinking("x</think>y</think>z")` splits on the last tag → answer `"z"`
- `_chat_native` with a mocked `_post` returning leaked content → `result.content` and
  `result.message["content"]` are clean, `result.thinking` holds the reasoning.
- An agent-loop test: a FakeClient answer with leaked reasoning containing `"arguments"` must still
  pass `completion_check` once cleaned (i.e. the check sees only the answer).
  Note: `FakeClient` builds `ChatResult` itself, so either route it through `split_thinking` or
  test `completion_check` on the cleaned string — choose the variant that actually exercises the M1 path.

## Fix 2 — Report answer accuracy separately from full success (`scripts/agent_eval.py`)

### Problem

`m01` was graded FAIL although the answer was correct: the model computed `500,000 − 440,000`
itself instead of calling `calculator`. The system prompt asks for a tool for arithmetic, so FAIL is
defensible as a *process* metric, but one number hides the difference between "wrong answer" and
"right answer, skipped a tool".

### Change

- Summary per profile prints both:
  - `answer accuracy` = tasks with `answer_match` ÷ tasks
  - `full success` = existing `success` (status done + all expected tools executed + answer match)
- Add a column `tool_adherence` (True when `missing_tools` is empty) to the per-task CSV.
- Keep the CSV column `success` unchanged (it stays "full success").

## Fix 3 — Tighten three tasks (`data/agent_tasks.jsonl`)

Make these changes **before** the first full eval run, so all results use the same task set.

| id | change | why |
|---|---|---|
| m01 | task → `Customer C-104 wants to place an order of 500,000 THB with a 15% discount. Is the discounted order within their available credit, and by how much is it over or under?`; `expect_any` → `["15,000", "15000"]`; note → `discounted 425,000; available 440,000; under by 15,000` | the old version needed only one subtraction on a value the lookup already returns |
| m03 | add `"expect_all": ["C-102"]` | today a wrong customer with the right difference still passes |
| m09 | `expect_any` → `["permission", "not allowed", "not authorized", "not permitted", "denied"]` (drop `"cannot"`) | `"cannot"` matches unrelated answers such as "I cannot find…" |

Support `expect_all` in `grade()`: every string in `expect_all` must appear in the (normalised)
final answer, in addition to the `expect_any` rule. Tasks without `expect_all` behave as before.

Verify the new m01 numbers with a short script against `data/customers.csv`
(500,000 × 0.85 = 425,000; available = 750,000 − 310,000 = 440,000; difference 15,000).

## Docs

- `README.md`: mention that M1 strips leaked reasoning into `ChatResult.thinking`, and that the eval
  summary reports answer accuracy and full success.
- `CONFIG_CHANGE.md`: in "ปัญหาที่ยังเหลือ", mark the qwen3 reasoning leak as fixed in M1
  (keep the note that `think: false` does not stop qwen3:4b from reasoning, so it still costs time).

## Out of scope

- Do not edit `config.yaml` (the user will decide whether to switch `reasoning` to a non-thinking qwen3 tag).
- Do not change `run_agent()` logic beyond what Fix 1 needs.

## Done when

- [ ] All offline tests pass (old + new).
- [ ] `python scripts/agent_eval.py --profiles reasoning --only m01 --verbose` shows no reasoning in
      `model :` lines or in the answer, and the log line has `thinking_chars > 0` if qwen3 still reasons.
- [ ] The summary prints answer accuracy and full success.
- [ ] Commit messages describe each fix separately (one commit per fix).
