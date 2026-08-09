# Fix prompt: ARPA verification run scored 1/10 due to a timeout ceiling and swallowed errors

Paste everything below into your coding agent.

---

## Context

This is the ARPA repo (autonomous paper-reproduction agent). A 10-paper end-to-end
verification run via `verify_codegen_agent.py` against the NVIDIA NIM backend
(`meta/llama-3.3-70b-instruct`) scored **1/10**. The failures are infrastructure bugs,
not model-quality problems. Do not tune prompts or swap models — fix the four defects below.

## Evidence from the failed run

From `nvidia_api_debug.log`:

- **40 API requests logged, only 20 responses.** Half of all calls never returned.
- Completed call durations, sorted (seconds): `115, 143, 144, 145, 157, 157, 166, 175, 177, 179`
- 6/10 exceeded 150s; 3/10 exceeded 170s; the maximum was **179.0s**.
- **Zero** occurrences of `429`, `Retry-After`, or any rate-limit message.

From `verification_results/partial_results.jsonl`:

- `easy2` PASS. `easy1` and `hard2` reached codegen then failed with `codegen: no files generated`.
- Seven papers (`easy3`, `medium1`–`medium4`, `hard1`, `hard3`) failed with
  `extraction: ValueError: extraction returned no dataset description` after **0.0 minutes**.

The 180s maximum against a 180s client timeout is a censored distribution: calls needing
more than 180s are being killed, not completing. The 0.0-minute failures have a different,
currently-unknown cause because the underlying exception is discarded (defect 2).

---

## Defect 1 — `nvidia_timeout_s` is a dead setting; the client hardcodes 180s

`arpa/core/config.py:54` defines `nvidia_timeout_s: float = 300.0`. Grep the repo: it is
referenced nowhere outside its own definition.

`arpa/models/nvidia_client.py:176` hardcodes the real value:

```python
with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)) as client:
```

**Fix:** drive the read timeout from `self.settings.nvidia_timeout_s`. Keep `connect`/`write`/`pool`
short (a connect stall is a real failure); only `read` should be generous, because that is the
model thinking. Raise the config default to at least `600.0` — a 70B model generating
`CODEGEN_MAX_TOKENS = 8192` tokens on a shared free tier legitimately needs several minutes.

Check for the same hardcoded-timeout pattern in the other clients
(`gemini_client.py`, `ollama_client.py`, `openrouter_client.py`, `groq_client.py`) and make each
honour its own configured timeout.

## Defect 2 — `ExtractionAgent` swallows API errors and reports them as "paper had no data"

`arpa/agents/extraction_agent.py:373-393`. Each of the four passes catches `Exception`, logs it,
and substitutes a placeholder `CodegenMissingDetail`. When all four passes fail, `run()` returns
an empty-but-structurally-valid `MethodologySpec`. The caller cannot distinguish
"the API died" from "this paper genuinely lacks a dataset description", and the exception text
survives only in a `logger.error` that nothing captures.

**Fix:**

- Record per-pass outcomes on the returned spec so callers can inspect them — e.g. a
  `pass_failures: list[PassFailure]` field carrying `label`, exception class name, and message.
  Keep the existing `assumptions_needed` behaviour so current tests still pass.
- Add a helper such as `MethodologySpec.all_passes_failed` (or an equivalent on the agent) that is
  true when every pass raised.
- When **every** pass fails, `run()` must raise rather than return a hollow spec — a total wipeout
  is an infrastructure failure, not an extraction result. Preserve the original exception via
  `raise ... from exc` so the cause is not lost. Partial failure must keep degrading gracefully
  exactly as it does now.

## Defect 3 — the harness's retry logic never fires on the most common failure

In `verify_codegen_agent.py`, `run_stage()` retries via `is_transient()`, matching substrings in
`TRANSIENT_ERROR_MARKERS`. But the extraction stage fails with
`ValueError("extraction returned no dataset description")`, raised by the harness itself, which
matches no marker. Every one of the seven instant failures was therefore retried **zero** times.

**Fix:**

- Once defect 2 lands, the harness should surface the real underlying error, so a genuine
  timeout or 5xx will match the existing markers naturally.
- Make the harness's own "extraction produced nothing" error carry the underlying cause text
  instead of a bare generic message.
- Treat a total extraction wipeout as retryable.

## Defect 4 — the real errors were never written anywhere

The agents log failures through `loguru` to stderr. The run's stderr was not captured, so the
actual API exception behind the seven instant failures is unrecoverable. `nvidia_api_debug.log`
only records `chat_structured` calls, so the `chat()` calls made by `DatasetAgent` and
`CodeGenAgent` are entirely invisible.

**Fix:**

- In `verify_codegen_agent.py`, add a `loguru` file sink at DEBUG writing to
  `verification_results/run_<timestamp>.log`, so every agent-level error is persisted.
- Extend the NVIDIA client's API logging to cover `chat()` and `generate()`, not just
  `chat_structured()`, and log **failures** (status code, response body, elapsed time), not only
  successes. A request with no matching response entry is the exact blind spot that made this
  run hard to diagnose.

---

## Constraints

- Do not change prompts, model selection, or `CODEGEN_MAX_TOKENS`. The failures are transport-
  and error-handling bugs; changing generation behaviour at the same time would confound the fix.
- `python -m pytest tests/ --ignore=tests/test_extraction_with_rag.py -q` currently passes 89/89.
  It must still pass. Add tests for the new behaviour:
  - the NVIDIA client's read timeout reflects `nvidia_timeout_s`
  - partial pass failure degrades gracefully; total failure raises
  - `is_transient()` returns True for a total-extraction-failure error
- Keep changes minimal and surgical. No refactoring beyond what these four defects require.

## Acceptance criteria

1. `grep -rn "read=180" arpa/` returns nothing; the read timeout derives from settings.
2. Killing the API mid-run produces a raised error naming the real cause, not
   `"extraction returned no dataset description"`.
3. A simulated transient API failure during extraction is retried by `run_stage()`.
4. After a run, `verification_results/run_<timestamp>.log` contains the agent-level errors, and
   every API request in the NVIDIA log has either a matching response or a matching failure entry.
5. Full test suite green.

## Verify before claiming success

Run `python verify_codegen_agent.py --backend nvidia --papers easy1 --fresh` and confirm the
single paper completes end to end. Then report the request/response counts from
`nvidia_api_debug.log` — they should now match, with no silently dropped calls.
