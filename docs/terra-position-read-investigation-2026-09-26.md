# Terra incomplete position read: investigation

## Outcome

The first authorized retry of the **identical request** returned `A` followed by a complete
position paragraph. No second retry was made. Reasoning remained disabled, all prompt and
generation parameters were unchanged, and the original failed record remains untouched.
The retry cost estimate is **US$0.0027412**. This was one diagnostic call, not a full-study
restart. The successful retry is recorded separately. The main runner now supports bounded
format retries, isolation of true dependents and provenance-checked diagnostic import; offline
preflight verified reuse of this successful retry with zero new generation or duplicate charge.
See the [run guide](main-mixed-run-2026-09-26.md) for current continuation status.

## Evidence

Source: `runs/main-mixed-60-20260926/live-workers8/requests.sqlite3`.
Request key: `archived-global-14/mixed_family/C/initial/0`.
Question: how much the United States can trust Japan (Great deal / Fair amount / Not too much / Not at all).

- The raw HTTP response was 200, with content exactly `A`, `finish_reason=stop`, four billed
  output tokens, and zero reasoning tokens. Re-parsing the raw response reproduces `A` exactly.
- The saved provider payload matches the current payload builder exactly: 1,200 output tokens,
  reasoning `none`, no stop sequence, no tools, no JSON schema, and no candidate-only D probe.
- The final instruction explicitly requests TWO mandatory parts: a letter on the first line
  and the full position on the following line, and explicitly says not to stop after the letter.
- There is no evidence of a local truncation/parser loss, token-limit truncation, rate-limit
  response, or accidental dispatch of the single-letter D-text prompt.
- In the saved checkpoint, Terra has 53 complete C reads plus this one incomplete read;
  Qwen has 31/31 complete reads and Inkling 46/46. These are engineering counts from the
  checkpoint, not population failure-rate estimates.
- The identical retry returned a distinct response ID, `A` plus the position paragraph,
  133 output tokens and zero reasoning tokens. Its request hash and provider-payload hash
  exactly match the original. The first valid response was retained without selecting by stance.

Official OpenAI documentation defines `stop` as a natural stop or supplied stop sequence,
distinct from `length` for reaching the token limit:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/.
This supports the field interpretation; it does not expose Databricks backend internals.
The evidence is consistent with an intermittent incomplete response, but does not establish
its internal model/proxy cause or prove that concurrency could never affect it.

## Why one record stopped everything

The position parser correctly rejected a missing paragraph. The orchestration then treated
any request exception as a global stop (`study.py` and `scheduling.py`). That broad fail-stop
policy, rather than an actual dependence of all questions on this one response, caused the
experiment-wide halt. It was too conservative for this kind of recoverable output failure.

This is a **C baseline measurement**, not an actual debate reply. Its output is never fed back
into the discussion. Six planned C text comparisons refer directly to this baseline text;
they span multiple tones/branches. If all allowed retries fail, these dependent measurements
need explicit missingness, not fabricated text or deletion of an entire debate trajectory.
Terra is not a D receiver here. Other members' measurements and substantive debate content
do not scientifically depend on this C paragraph. The existing baseline-read execution gate
must be distinguished from actual data dependencies when implementing failure isolation.

## Implemented main-study policy

- At most **two additional attempts** for a recoverable malformed/incomplete response,
  i.e. at most three attempts including the original. Preserve every raw response, error,
  request/payload hash and charge. Stop at the first format-valid response.
- Do not change prompts, reasoning, models or limits between retries without a separate
  protocol decision, and do not choose retries according to agreement or position.
- When retries are exhausted, isolate true dependents and record missingness and support.
  A failed measurement/judge should not delete independent debate data. A failed debate
  node prevents its descendants from running; unrelated branches and questions can continue.
- Shared-prefix or shared-baseline failures can affect more than one trajectory. Do not
  assume every failure belongs to exactly one leaf trajectory.
- System-wide integrity/configuration problems still warrant a global stop; that is distinct
  from one malformed model response. Known Databricks connection/timeout failures are now
  isolated as unavailable without automatic regeneration; their full usage reservation remains.
  Unknown provider errors and billing/probability integrity failures still stop for inspection.

## Artifacts and verification

Diagnostic root:
`/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/investigate-terra-position-20260926/`

`manifest.json` records authorization and the original request/payload hashes;
`original-record.json` preserves the original failure; `attempt-1/requests.sqlite3` stores
the exact retry, raw provider response, parsed result and charge; `report.json` summarizes
the result and confirms the source record is unchanged. No second-attempt journal exists.

The diagnostic script has five offline tests covering first-valid selection, a second retry
when necessary, the two-retry limit, unchanged source records/payloads, and preserving an
uncertain transport failure without blindly issuing another call.
