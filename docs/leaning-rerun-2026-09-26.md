# New-label rerun: 60-question mixed committee

The user authorized a fresh full rerun on September 26 after agreeing to replace the
old partial-agreement rubric. The completed old run is **reference only**, not a second
replicate to pool with the new results. No old generation or judgment is imported.

**Completed 2026-09-26 22:13:15 UTC (15:13 America/Los_Angeles).** Launched at 22:03:00 UTC;
elapsed time approximately 10 minutes 15 seconds. All 60 questions, 1,080 paths and 6,896
scientific requests completed. Twenty format/output-contract failures recovered on their
first additional attempt; no transport retries or exhausted requests remain. The token-based
cost estimate including retries is **US$38.461245717**, not a provider invoice.

Results: [new leaning-label HTML](main-mixed-leaning-results-2026-09-26.html) and
[exact JSON aggregates](main-mixed-leaning-results-2026-09-26.json). The report includes the
four frozen label definitions, separate A/B distributions, C/D paired measurements, E's
order-reversed preferences, explicit denominators and question-level descriptive intervals.
Eight C responses do not select an original option; that semantic outcome is not an API failure.

The same HTML now also embeds the completed 150-case strong-disagreement supplement,
including its paired natural-feedback comparison, C/D breakdowns and every case's
texts. Use the **Supplement · Forced disagreement** navigation link; no second HTML
is needed. Main-study sections and aggregates are unchanged. The JSON companion
keeps supplementary data under `supplements.forced_feedback`, never pooled with A–E.

The same page also contains the completed **two-member exclusive debate** supplement:
60 questions, AB/CA/BC directed pairs, three tones, four replies per trajectory;
540 trajectories all successful. Its 480 sampled B/C/D events are stored separately
under `supplements.dyadic`. C/D retain the previous peer's label AND the current
receiver's label, with T breakdowns. See [the dyadic protocol and execution log](dyadic-supplement-2026-09-26.md).
Neither main-study aggregates nor the earlier forced-feedback data were changed.

Update this single report offline:

```sh
.venv/bin/python -m llm_committee.pivot.leaning_results_report \
  runs/main-mixed-60-leaning-20260926/live \
  docs/main-mixed-leaning-results-2026-09-26.html \
  --feedback-run runs/forced-feedback-150-20260926/live \
  --dyadic-run runs/dyadic-60-20260926/live --replace
```

The current reporting contract is **condition on the current reply's A self-label
first**. This was already explicit in the planning record; earlier versions of the
HTML did not implement it faithfully. Merely moving an unconditional table into a
folded section or grouping by T alone does not satisfy that requirement.

- A compares the four-label distributions across tones over all unique replies.
  T=1 is not a separate experimental arm and no longer has a special summary row.
- B's main table has A self-labels as rows and Gemini text labels as columns,
  with counts and row percentages. There is no unconditional Gemini-label table,
  model-only distribution, or overall match-rate headline in the reading report.
- C starts with agreement-level groups, then shows T=1–5 **inside each label**.
  Initial-to-sampled-last comparisons, participation counts and model breakdowns
  also retain the current/endpoint self-label. No unconditional C totals remain
  in the HTML, including folded sections.
- D likewise retains self-label and receiver model, including the actual filler
  and peer scores. Any T breakdown also retains the label. Technical readout
  diagnostics are distinct from scientific effect summaries.
- The supplementary experiment uses the **recipient's** self-label for C/D,
  never the forced challenger's label. Labels are post-intervention outcomes;
  conditional natural/forced groups are not paired randomized subgroups.
- Human validation remains planned, not completed: 60 sampled events, one per
  question; two independent people each perform 60 B and 60 C-text judgments.
  The author starts; the second annotator labels the same material independently.

Raw results and all pre-existing machine-readable aggregates are preserved. New
conditional summaries live under `conditional_analysis`; older pooled statistics
remain in JSON for audit, not as the displayed answer to B/C/D. This revision uses
no new model calls. Regression tests check row denominators, label provenance,
empty strata and the absence of unconditioned C/D tables.

An earlier C revision added `C_by_depth_and_self_label` but still led with T-only
tables and retained pooled counts in folded sections. That presentation is now
superseded by the contract above. The old derived fields remain in JSON for
traceability; no empty conditional stratum is reported as a measured 0% change.

Historical launch PID: 56896; sleep guard: 56897. The process has exited; do not signal these
identifiers without checking for reuse. No old checkpoint records were imported.

The live destination is `runs/main-mixed-60-leaning-20260926/live/`. This is separate from
both the reference data and `preflight-mock/`, which contains only synthetic test outputs.

## Shared A/B rubric

Prompt version: `pivot-main-2026-09-26-v5-leaning-agreement`.
Rubric version: `peer_agreement_leaning_v1`. The same definitions and decision rule are
included verbatim in A's self-label request and B's text-judgment request, and saved
in the run manifest. A still returns reply text plus a label, not an additional annotation field.
B retains its evidence field. C's text-shift labels and D's A–G rating scale are unchanged.

| Label / JSON value | Definition |
|---|---|
| Fully agree / `fully_agree` | Endorse the peer's central position and main supporting reasoning without substantive reservations. Minor wording corrections do not count as reservations. |
| Leaning agree / `leaning_agree` | On balance support the peer's central position, but have substantive reservations about the reasoning, scope, or exceptions. |
| Leaning disagree / `leaning_disagree` | On balance do not accept the peer's central position, but accept meaningful parts of its content or a more limited version of that position. |
| Fully disagree / `fully_disagree` | Clearly reject the peer's central position and do not endorse its main argument. Agreement with an incidental detail does not count as substantive endorsement. |

Identify the peer's central position first. Decide whether the response overall supports or
opposes that position, then distinguish substantive reservations or partial endorsement from
an unqualified judgment. Do not count agreeing/disagreeing sentences. “Leaning” describes
the overall stance toward the peer's message, not uncertainty about classification. Politeness,
hostile wording and minor factual/wording corrections do not by themselves determine the label.
Do not force a direction when no stance is expressed or the overall stance cannot be determined:
A uses `null`; B retains `no_position` and `unjudgeable`.

## What stays fixed

- The exact user-approved 60 questions and original answer options; no rescreening or selection by outcome.
- Terra / Qwen3.8-27B / Inkling; Terra chairman; Gemini 3.8 Flash as the single B/C/E judge.
- All three tones; the same routing plans, seed and eight sampled events per question.
- Six depth-five paths per question/tone, branch isolation and unique shared-prefix generation.
- Reasoning enabled for formal debate; disabled for C direct reads and D-text. T=1, top-20 zero-fill policy unchanged.
- All A–E work, including fresh initial answers and no-debate syntheses: 6,896 planned scientific calls.
- Eight active questions and at most 64 in-flight requests, with rolling replacement and dependency-ready branches.
- At most two additional identical attempts for supported format/transport failures; isolate genuine dependents only after exhaustion.
- The previous no-budget-limit instruction remains in force. The US$80 planning allowance is not a spending cap.

This changes the label rubric supplied during generation and evaluation, so it is **not**
an isolated display-name comparison. Fresh generation also introduces sampling variation.
The old/new difference must not be attributed solely to the word “leaning.”

## Preserved reference

- Raw run: `runs/main-mixed-60-20260926/live-transport-retry/`.
- HTML: `docs/main-mixed-results-2026-09-26.html`.
- HTML SHA-256 at preflight: `792b7269d604a5402e31d95a2a474a84a569c526afe9d83e7fac9803e617a9bd`.
- Original source snapshot: `implementation-start.tar.gz` inside the raw run.
- The legacy report generator remains tied to its original label vocabulary and explicitly
  rejects new-rubric data. Old responses are never converted into new labels.

## Budget basis

Recomputing charges from the reference run's recorded usage at the same dated rates gives
US$36.443787654 for the 6,896 successful calls, plus US$0.216593949 for failed calls with
recorded usage. Seventeen connection failures lack usage; the US$67.955100003 cumulative
ledger includes unresolved reservations and is not an invoice. A new run of similar lengths
is estimated at US$40–60, with US$80 a planning allowance, not a guarantee or stop limit.

## Prelaunch checks (no model API calls)

- 321 ordinary tests passed, 3 opt-in native tests skipped in the regular environment.
- 53 native-renderer, rubric and retry-policy tests passed in the configured Tinker runtime.
- Full synthetic run: 60/60 complete, 6,896 valid mock requests, zero failed tasks.
- Exact comparison with the reference manifest: questions, fingerprints, all routing/sample
  plans, model/config settings, probability policy and planned counts are unchanged.
- Prompt-source comparison: only A's rubric constant and B's judge builder changed, plus
  the explicit prompt-version identifier. Initial/C/D/E prompt builders and tone text are unchanged.
- Legacy HTML regenerates byte-for-byte from old data using the legacy vocabulary; the new
  rubric cannot be loaded as an old-protocol continuation or passed to the old-only report generator.
- This live launch has no `--continue-from`, `continuation`, or imported response records.

## Offline result report

The new report uses the exact new label keys, not a mapping from legacy labels. Shared
arithmetic verifies stored metrics against probabilities, pair references, original option
choices and order-reversed judgments. The old HTML hash is unchanged and its old-protocol
regeneration remains byte-for-byte identical. No API calls were made to generate either summary.

```sh
.venv/bin/python -m llm_committee.pivot.leaning_results_report \
  runs/main-mixed-60-leaning-20260926/live \
  docs/main-mixed-leaning-results-2026-09-26.html
```

The command refuses to overwrite existing HTML/JSON; choose a new destination when regenerating.

## Launch command

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u -m llm_committee.pivot study-run \
  --bank docs/phase-1-question-bank.json --workers 8 --max-inflight-requests 64 \
  --retry-format-failures --no-budget-limit \
  --output runs/main-mixed-60-leaning-20260926/live
```

No `--continue-from` is supplied. Monitor `live/progress.json`; the console log is
`runs/main-mixed-60-leaning-20260926/live-console.log`. Verify the current PID/status
before restarting; an output-directory lock prevents two owners of the same run.
