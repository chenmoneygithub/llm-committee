# First live mixed-family pilot checkpoint — 2026-09-25

Status: stopped on a malformed formal reply, not completed. No automatic retry,
label inference, model substitution or prompt change was made after observing it.

## Access and actual work

The existing Databricks OAuth profile `un` works. Read-only inspection found both
`databricks-gpt-5-6-terra` and `databricks-gemini-3-8-flash` READY, with the expected
foundation-model identities. Terra generation was then verified live. Gemini inference
has not yet been attempted. No direct OpenAI/Gemini keys are required by this route.
Qwen3.8-27B and full Inkling were called through Tinker without model substitution.

Only the first question (archive 24, gender and university education) was reached:

| Work | Valid responses | Invalid responses |
|---|---:|---:|
| Independent initial answers | 3 | 0 |
| Initial C readings | 3 | 0 |
| Friendly formal replies | 4 | 1 |

The three initial C reads reported no additional reasoning. Both open models returned
all four original-option log probabilities at the actual C choice token. This verifies
those initial reads only, not later reads or D-text. Formal calls requested medium
reasoning; the first Terra initial response reported zero reasoning tokens despite that
request, while Qwen/Inkling exposed reasoning. Preserve requested settings and observed
usage separately; a setting alone does not establish the amount of reasoning used.

There were 11 dispatched generations, 10 valid and 1 malformed. Recorded usage totals:
10,068 input tokens, 3,375 output tokens including reasoning, 1,152 cache-hit input tokens.
Token-based estimated cost: **US$0.042922555** against the authorized **US$30** cap.
This includes the conservative Databricks regional allowance; it is not a provider invoice.
No request is pending or of uncertain billing status in the saved journal.

## Stop reason and follow-up

`archived-global-24/mixed_family/friendly/debate/n04` (Inkling) returned a cleanly
terminated, substantive reply beginning “Fully agreed.” but not the required JSON
object containing `reply` and `agreement`. JSON parsing therefore failed. The complete
instructions were present in the stored request; this was not a missing prompt, timeout,
truncation or authentication failure. Native Tinker sampling has no JSON-schema argument
in the SDK used here; unlike Databricks structured output, this path relies on prompt
compliance. A conversational phrase is not silently converted to a structured A label.

Before another paid attempt, strengthen/test the output-format contract consistently
and version any prompt change. Retain this attempt as engineering history rather than
discarding it and presenting a successful rerun as the first attempt. Do not repeatedly
sample until a preferred label or effect appears. The research question/options, tone,
routing, sampling and measurement definitions need not change to address formatting.

The stop also exposed a cleanup bug: Tinker 0.30.3 requires an explicit status argument
to `ServiceClient.close`. This has been fixed with success/errored/interrupted status
propagation, and cleanup exceptions no longer mask the saved pilot result. No paid call
was made after that fix. The current implementation differs from this run's manifest,
so the old live directory must not be reused to bypass the identity check.

Not yet tested live: Gemini B/C-text/E judging; later C readings and longitudinal
comparisons; D-text argument/control pairs; supplemental exact-prefix candidate scoring;
chairman synthesis and both-order final-answer preference. No A–E acceptance or research
effect claim follows from this partial engineering run.

## Artifacts

All are under `runs/mixed-family-pilot/live/` (ignored by git):

- `manifest.json`: frozen inputs, route/sample, settings, package versions and implementation hash.
- `requests.sqlite3` and `calls.json`: original requests, responses, statuses and cost accounting.
- `report.json`: original blocked outcome, not overwritten by the cleanup fix.
- `implementation-start.tar.gz`: source snapshot taken before the post-run cleanup fix.

Post-fix offline suite: 177 passed, 3 optional native-renderer checks skipped in the normal
environment. The three native checks passed separately in the isolated environment before
the live call, along with all 14 Databricks transport tests. These tests use fabricated responses.
