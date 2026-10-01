# Format-contract pilot continuations — 2026-09-25

**Final status: finished with measurement failures; no process is running.** All 337
scheduled generation/judgment calls completed. Two additional scoring calls failed the
unchanged consistency check; both are arms of one Inkling D-text event. D-choice is 27/27,
D-text is 16/17 complete pairs. All-attempt estimated cost is **US$1.557001068**.
See the [final engineering report](pivot-pilot-live-report.md). The record below is
chronological development history, not a list of currently running attempts.

Earlier checkpoint: v4 stopped after 72 calls (71 valid) at a D-text supplemental-score
consistency check. Question 24's three initial answers, all 51 formal replies across three
tones, and all 14 C readings are complete. One Inkling D-text argument/control pair is
complete. Another argument read lacks candidate E in its top-20; exact-prefix supplemental
scoring differs from the original overlapping tail logprobs by up to 0.999845 natural-log
units. The 0.01 tolerance is unchanged. Exact saved-prefix equality, token identity and
scoring-row alignment have been verified against raw records and Tinker documentation;
the reason for the numerical discrepancy is not established. No scores are mixed or
missing candidates filled with zero. B/C-text/E and questions 22/48 have not yet run.

V4 estimated cost: US$0.388979710; all three attempts: **US$0.440636345**.
An explicit new-directory continuation ran in
`runs/mixed-family-pilot/live-format-v4-continuation/`. It reuses all 72 checkpoint
records by exact request hash, including the invalid score record. Reused calls have
zero new charge; their original cost and source hashes are preserved in the continuation
manifest. Known supplemental-score disagreements remain invalid and their dependent
D metrics are null, not zero. All other errors remain fail-stop. The report will be
`completed_with_measurement_failures`, not a full pass, if any such failures remain.
This tests independent stages without repeating debate/C responses or changing prompts,
models, routes, sample, reasoning or the 0.01 consistency tolerance.

The continuation's **new-spend cap is US$29.55**. Adding all prior attempts' US$0.440636345
keeps the authorized maximum below US$30. Offline checks before launch: **195 passed,
3 optional native tests skipped**, lint and whitespace checks passed.

That continuation stopped after nine additional calls, at the first Gemini B response's
billing validation. The response itself was valid JSON. Databricks reported 603 prompt,
77 visible completion, 95 reasoning, and 775 total tokens: completion excludes reasoning
on this response. The adapter previously assumed completion already included reasoning.
The fixed adapter verifies total tokens and accepts either explicitly reconciled convention,
billing thinking exactly once; unexplained/missing totals still fail.

The saved response is reparsed **offline**, keeping the original checkpoint unchanged.
Its US$1.1354112 conservative hold becomes a US$0.001206975 token-cost estimate.
The correction, original-response hash and new accounting are recorded in the next manifest.
The revised cumulative cost before new calls is **US$0.455751208**, including all previous
attempts and the nine-call continuation. No Gemini judgment is regenerated.

The next directory was `runs/mixed-family-pilot/live-format-v4-usagefix/`,
continuing from `live-format-v4-continuation` with all **81 records** reused. Add
`--reconcile-databricks-usage` to the command below, change the two directory arguments
accordingly, and use a **US$29.54 new-spend cap**. This plus all prior costs stays below
US$30. Offline checks before this launch: **200 passed, 3 optional native tests skipped**;
lint and whitespace checks passed. This is still an incomplete engineering pilot, not
an accepted D measurement protocol or scientific result.

That next call exposed Gemini's optional reasoning-usage field: it reported 622 prompt,
78 completion and 700 total tokens without a separate reasoning count. The output total
is sufficient for billing; the reasoning partition remains explicitly unreported, not
evidence that thinking was disabled. The adapter now distinguishes that case from missing
or inconsistent totals. Direct-read verification is still strict. Its valid judgment was
also recovered offline, at US$0.0008349, without regeneration.

**Final continuation (finished):** `runs/mixed-family-pilot/live-format-v4-resumed/`, from
`live-format-v4-usagefix`, with **82 records** reused and both flags shown above.
Prior all-attempt cost: **US$0.456586108**; new-spend cap US$29.54 (aggregate < US$30).
Offline checks: **201 passed, 3 optional native tests skipped**. All previous directories
are retained as read-only development history; imported rows are not new observations.

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /tmp/committee-tinker-check.sqHb9X/venv/bin/python \
  -m llm_committee.pivot run \
  --questions runs/mixed-family-pilot/questions.json \
  --roster mixed_family --judge-model gemini-3.8-flash \
  --closed-provider databricks --databricks-profile un \
  --output runs/mixed-family-pilot/live-format-v4-continuation \
  --continue-from runs/mixed-family-pilot/live-format-v4 \
  --record-probability-failures --approve-spend-usd 29.55
```

Earlier checkpoint: v3 stopped before debate because Terra's initial C read returned
only an option letter, without the required full position text. All three initial answers
were valid; the fourth generation was rejected. V3 cost: US$0.00873408. No additional
reasoning or truncation was reported, and no missing paragraph was inferred or regenerated
under the same request. The v3 artifacts are preserved below.

V4 (`pivot-pilot-2026-09-25-v4-output-contracts`) additionally clarifies that C must return
two mandatory parts in one answer: an original-option letter first, then a full position
paragraph. All three models receive the same clarification; the choice remains the first
visible output token. D-text, reasoning settings, questions, routing and sampling are unchanged.
This corrects format compliance, not the definition of position change. It is a new measurement
prompt version, so earlier readings are not silently mixed into its results.

V4 was run at `runs/mixed-family-pilot/live-format-v4/`, with the same command
below except for `--output` and `--approve-spend-usd 29.94`. Previous attempts total
US$0.051656635; the new cap is rounded down so the aggregate remains below US$30.
The offline suite still passes: 182 passed, 3 optional native tests skipped.

## V3 attempt (historical record)

The user approved fixing output formatting and continuing the mixed-family pilot after
the first live attempt stopped on Inkling's non-JSON debate reply. This is an engineering
continuation, not another independent research replicate or a change in model selection.

## Fixed before starting

- Prompt version: `pivot-pilot-2026-09-25-v3-debate-json`.
- Only formal debate prompts change: append the same explicit JSON output contract after
  the full branch context for every model and tone. It names both required fields and
  includes their existing schema, with all four labels and null; no example selects a label.
- Initial-answer, C, D-text, judge and synthesis prompts are unchanged. Formal reasoning
  remains enabled; direct measurement settings are unchanged.
- JSON parsing now rejects duplicate fields rather than silently keeping the last value.
  Free text, code fences, absent fields and invalid labels still fail without repair,
  inferred labels or automatic resampling.
- Programmatic comparison with the first manifest confirmed identical questions, route
  trees, measurement sample, model configuration and generation settings.
- Offline test suite: 182 passed, 3 optional native-renderer tests skipped.

Because the debate prompt version changed, the three-question pilot is started in a new
directory under one version, without importing or silently continuing the earlier debate.
The first attempt is preserved as development history, not pooled into this run. Questions
may still enter the main 80 under the previously agreed final-protocol reuse rules.

## Budget and artifacts

Original authorization: US$30 total. First-attempt estimated cost: US$0.042922555.
This attempt's cap is **US$29.95**, rounding the remaining allowance down; combined caps
and prior charges are below US$30. Costs are token-based estimates, not account invoices.

Command launched:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /tmp/committee-tinker-check.sqHb9X/venv/bin/python \
  -m llm_committee.pivot run \
  --questions runs/mixed-family-pilot/questions.json \
  --roster mixed_family --judge-model gemini-3.8-flash \
  --closed-provider databricks --databricks-profile un \
  --output runs/mixed-family-pilot/live-format-v3 --approve-spend-usd 29.95
```

V3 is stopped, not running. The manifest,
journal and source snapshot are in `runs/mixed-family-pilot/live-format-v3/`. The prior
`runs/mixed-family-pilot/live/` is untouched. No deployment, main-study run, manuscript
edit, commit or push is included. Any new measurement or interface failure is preserved
and inspected; it does not authorize a retry loop or a relaxed measurement definition.
