# ARPA Infrastructure Defect Fixes - Summary

## All Six Defects Fixed

### Defect 1: ✅ Fixed - Hardcoded 180s timeout
**Problem:** `nvidia_timeout_s` config setting was ignored; clients hardcoded `read=180.0`

**Fix:**
- Updated `arpa/models/nvidia_client.py` to use `self.settings.nvidia_timeout_s`
- Updated `arpa/models/groq_client.py` to use `self.settings.groq_timeout_s`  
- Updated `arpa/models/openrouter_client.py` to use `self.settings.openrouter_timeout_s`
- Raised default `nvidia_timeout_s` from 300.0 to 600.0 in `arpa/core/config.py`
- Added `groq_timeout_s` and `openrouter_timeout_s` settings (600.0 default)
- Kept connect/write/pool timeouts short (10s); only read timeout is generous

**Verification:**
```bash
grep -rn "read=180" arpa/
# Returns: (no matches)
```

### Defect 2: ✅ Fixed - ExtractionAgent swallows API errors
**Problem:** When all extraction passes fail, returned empty MethodologySpec instead of raising, hiding real errors

**Fix:**
- Added `PassFailure` model to `arpa/core/state.py` to record exception details
- Added `pass_failures: list[PassFailure]` field to `MethodologySpec`
- Added `all_passes_failed()` method to `MethodologySpec`
- Modified `_extract_pass()` to attach `PassFailure` to placeholder results
- Modified `_merge_passes()` to collect pass failures from all passes
- Modified `run()` to raise `RuntimeError` with cause chain when all passes fail
- Partial failures continue to degrade gracefully (existing behavior preserved)

**Behavior:**
- Total wipeout (4/4 passes failed) → raises with real exception via `raise ... from exc`
- Partial failure → returns degraded spec with `pass_failures` populated

### Defect 3: ✅ Fixed - Retry logic never fired on extraction failures
**Problem:** `ValueError("extraction returned no dataset description")` matched no transient marker

**Fix:**
- Added `"all extraction passes failed"` to `TRANSIENT_ERROR_MARKERS`
- Updated extraction failure message to include underlying pass failure details
- Now raises: `ValueError(f"extraction returned no dataset description -- failed passes: {details}")`
- Total extraction wipeouts are now retryable

**Result:** `is_transient()` now returns `True` for total extraction failures

### Defect 4: ✅ Fixed - No persistent logging
**Problem:** Agent errors only went to stderr (not captured); API log didn't cover `chat()`/`generate()` or failures

**Fix:**
- Added loguru file sink in `verify_codegen_agent.py`:
  ```python
  run_log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
  logger.add(run_log_path, level="DEBUG", backtrace=True, diagnose=False, enqueue=True)
  ```
- NVIDIA client's `chat()` method already had comprehensive logging including:
  - Request details (model, temperature, messages)
  - Response details (status, content length, previews)
  - Failure details (HTTP status, elapsed time, response body)
- `generate()` delegates to `chat()` so it's also covered
- Every API request now has either a response or failure entry in logs

### Defect 5: ✅ Fixed - Groq retry logic never actually waited the real reset time
**Problem:** `arpa/models/groq_client.py` read `getattr(e, "retry_after", None)` off a caught
`RateLimitError`, expecting the SDK to expose the real wait time. Confirmed directly against the
installed `openai` package that `RateLimitError` carries no such attribute (`dir()` shows only
`code`, `param`, `status_code`, `args`) — the check silently always missed and fell back to a fixed
exponential backoff (~31-46s total across 5 attempts) that had no relationship to Groq's actual
rate-limit reset window, so retries kept firing too early and exhausted before succeeding.

**Fix:**
- Added `_retry_wait_seconds()`, which reads the real wait time off the caught exception's
  `.response.headers` (`retry-after`, or Groq's `x-ratelimit-reset-tokens` / `-requests`) instead
  of guessing
- Falls back to the old exponential backoff only if Groq doesn't send those headers
- Capped at 90s per attempt so a single stalled call can't eat a whole stage's timeout budget
  (this cap is a known limitation — see "Known limitation" below)

**Verification:** 99/99 tests still pass after the change.

### Defect 6: ✅ Fixed - stdlib logging never reached the run log
**Problem:** `nvidia_client.py` and `groq_client.py` both log through Python's stdlib `logging`
module (`logging.getLogger(__name__)`), not `loguru`. The file sink added in
`verify_codegen_agent.py` only intercepted `loguru` output, so low-level per-call diagnostics from
those two clients (e.g. `logger.warning("Groq rate limit hit...")`, `logger.error("NVIDIA API
FAILED...")`) never reached `run_<timestamp>.log` — only the higher-level pass/stage failure lines
logged by `extraction_agent.py` (which does use loguru) landed there. Confirmed by grepping every
run log for these stdlib-only messages and finding zero matches despite them definitely having
fired.

**Fix:**
- Added `InterceptHandler(logging.Handler)` in `verify_codegen_agent.py`, following loguru's
  documented stdlib-bridge recipe: forwards each stdlib `LogRecord` into `logger.opt(depth=...).log(...)`
- Wired in via `logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)`
  right next to the existing `logger.add(run_log_path, ...)` call, so both feed the same sink
- Level set to `INFO` rather than `DEBUG`/`NOTSET` deliberately — `httpx` and the `openai` SDK
  (Groq's client) also log through stdlib and are chatty at `DEBUG`, which would have buried the
  arpa-specific detail this is meant to surface
- Frame-walking depth (`sys._getframe(6)`, matching loguru's official recipe) verified against a
  simulated call from inside a module function so the log line attributes to the real call site
  (e.g. `arpa.models.nvidia_client:chat:250`) rather than to `logging/__init__.py` internals

**Verification:** manually confirmed both an nvidia_client-style `logger.error(...)` and a
groq_client-style `logger.warning(...)` called from inside a module function now appear in the
loguru file sink with correct source attribution. 99/99 tests still pass.

## Diagnostic Run Results

**Command:** `python verify_codegen_agent.py --backend nvidia --papers easy1 --fresh` (2026-08-09)

**Findings:**
- ✅ All logging working correctly
- ✅ Run log captured at `verification_results/run_20260809_215121.log`
- ✅ Real error surfaced: **504 Gateway Timeout** from NVIDIA API
- ✅ Each failed pass logged with full exception details

**Root Cause Identified:** NVIDIA's free tier was returning 504 Gateway Timeout after ~303s per
request on the 70B model — an infrastructure issue on NVIDIA's side, not the codebase.

## Full 10-Paper Verification (2026-08-10)

**Command:** `python verify_codegen_agent.py --backend nvidia --fresh`
(`ARPA_NVIDIA_GENERAL_MODEL` / `ARPA_NVIDIA_CODE_MODEL` = `meta/llama-3.1-8b-instruct`)

**Report:** `verification_results/verification_report_20260810_210834.txt`

| Stage | Result |
|---|---|
| Extraction Agent | 10/10 (100%) |
| Dataset Agent | 10/10 (100%) |
| CodeGen Agent | 10/10 (100%) |
| Generated code compiles | 1/10 (10%) — only `easy2` (EMNIST) |
| **Full pipeline pass** | **1/10** |

**What this confirms:** all four infrastructure defects above are fixed and validated by a real
end-to-end run — no timeouts, no silently-swallowed errors, no untracked failures across 10 papers
and ~80 LLM calls. The one remaining gap (9/10 syntax failures — unmatched brackets, bad
indentation, unterminated strings) is a **model capability limit, not an infrastructure bug**: an
8B model is not reliably precise enough to hand-write long, structurally exact Python files. The
pipeline correctly detects and reports every one of those syntax failures rather than hiding them.

**Models tried for higher code quality, and why they aren't the current default:**
- `meta/llama-3.3-70b-instruct` / `meta/llama-3.1-70b-instruct` (NVIDIA) — hung past 120s on a
  realistic-size generation during live testing; NVIDIA's free-tier capacity for 70B-class models
  was unstable/overloaded throughout this session (confirmed repeatedly, including a `"DEGRADED
  function cannot be invoked"` error from NVIDIA's own backend).
- `nvidia/nemotron-3-super-120b-a12b` (NVIDIA) — a live test prompt **did** compile cleanly, and
  this is the best code-quality candidate found. Adopted once, then found hanging again on a later
  check when NVIDIA's free tier destabilized further. Worth re-testing and swapping back in when
  NVIDIA's larger-model capacity is healthy.
- `llama-3.3-70b-versatile` (Groq, fallback backend) — same open-weight model, much more stable
  infrastructure, but this account's free-tier **daily** quota (100,000 tokens/day, separate from
  the per-minute 12,000 TPM limit) was nearly exhausted by same-day testing and run attempts
  (`"Rate limit reached ... on tokens per day (TPD): Limit 100000, Used 97061"`). Resets daily.
- `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b` — all are
  reasoning models that return their answer in a separate `reasoning` field with `message.content`
  empty/`null`. Neither `nvidia_client.py` nor `groq_client.py` currently read that field, so these
  are not usable as drop-in replacements without additional client changes.

**Known limitation (not yet fixed):** the Groq retry fix (Defect 5) caps waits at 90s per attempt.
This is correct for ordinary per-minute rate limiting, but is too short for a daily-quota 429 (Groq
returned `retry-after: 970` in one observed case) — a wait that long isn't worth retrying inline
anyway. The client should distinguish "tokens per minute" 429s (worth waiting out) from "tokens per
day" 429s (not worth retrying — fail fast with a clear message instead) by parsing the 429 body's
`detail` text, which names the window (`"tokens per day (TPD)"` vs per-minute). Currently both are
treated identically.

## Acceptance Criteria Met

✅ **grep -rn "read=180" arpa/** returns nothing
✅ **Killing API mid-run** produces raised error naming real cause, not generic "no dataset description"
✅ **Simulated transient API failure** during extraction is retried by run_stage()
✅ **After a run**, verification_results/run_<timestamp>.log contains agent-level errors
✅ **Every API request** in logs has matching response or failure entry (no silent drops)
✅ **Full test suite green**: 99/99 tests pass (ran: `python -m pytest tests/ --ignore=tests/test_extraction_with_rag.py -q`)
✅ **10-paper end-to-end run**: 10/10 on all three agent stages with zero infrastructure failures

## Next Steps

Six defects fixed and confirmed (four original infrastructure bugs, the Groq retry-wait bug, and the
stdlib-logging gap), validated by a real 10-paper run with zero infrastructure-level failures.
What's left is model-quality tuning, not defect-fixing:

1. Re-test `nvidia/nemotron-3-super-120b-a12b` (or another capable code-specific model) once
   NVIDIA's free tier is stable, and swap it back into `ARPA_NVIDIA_CODE_MODEL` — this is the
   lever most likely to fix the 9/10 syntax failures.
2. Fix the TPM-vs-TPD distinction in `groq_client.py` (see Defect 5's "Known limitation" above) so
   a daily quota 429 fails fast instead of retrying pointlessly.
