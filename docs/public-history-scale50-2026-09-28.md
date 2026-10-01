# Expand all six settings to fifty questions

The user approved thirty additional questions per setting, retaining the original
twenty (not sixty total). This is an extension, not another rerun or repetition.

## Frozen design

- Uniform seeded selection of 30 from the remaining 40 in the user-approved,
  country-context-screened 60-question bank. All six settings share these 30.
  No debate outputs, agreement labels, or measured effects enter selection.
- Preserve the original 20 question plans, generations, judgments and failures.
  New questions use the same generation prompts, reasoning, provider endpoints,
  configuration slots, measurement rules, and retry policy.
- Dyads: AB, CA, BC; four replies; two local-tone continuations sharing T1/T2.
  Eight base B/C/D measurement nodes per question, selected before generation.
- Triads: copy the archived per-question trees, two roots/six terminal paths,
  five replies, paired local tones sharing T1/T2. Collect A/B/E only.
- D: only the open-weight members in the mixed-family dyads. Reasoning-enabled
  formal debate; separate neutral reasoning-disabled D1/D2 reads.
- E: Terra chairman, Gemini 3.8 Flash, unchanged 190–210-word primary answers,
  both judge orders, twelve frozen diagnostic questions per added batch.
  E orders and diagnostic membership are frozen before new debate generation.
- All analyses preserve prior/current-label conditioning for local C/D. Report
  bootstrap intervals over questions, not turns as independent observations.
  Fifty questions improve coverage but do not guarantee statistical power.

## Execution

Frozen selection/plans and new journals:

`/Users/Chen/Documents/research/llm-committee/runs/public-history-extension30-20260928`

The first attempted engineering launch made no paid calls: its plan/output
directory layout triggered the source-protection guard, and the analysis-only
Python environment lacked Tinker. Its zero-cost records are retained at
`runs/public-history-extension30-20260928-preflight-unlaunched`. The plans now
live in a separate `plans/` subdirectory, and live work uses the same pinned
native environment as earlier experiments; no provider package was upgraded.

Both mixed-family one-question gates then completed, including recovered
connection timeouts. Exact audits reconstructed 89 dyadic and 68 triadic calls.
All 28 native probability reads passed tokenizer/read-index verification.
The triadic E primary answers contained 199–200 words, with both judge orders.

Launch only after those gates pass. Two 32-request pools (64 combined), each
with eight rolling questions. Each dyadic E job follows its own debate batch.
Successful gate outputs are resumed, not regenerated. No new spending cap.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m scripts.scale_public_history --live
```

Planned logical calls across the added batches: 14,834. The linear extrapolation
from the nine earlier journals is US$85.66, approximately US$111 with a 30%
allowance. This is not a provider invoice or a spending limit. Received-response
token estimates and unresolved transport reservations must be reported separately.

## Reporting

`scripts.report_scaled_public_history` creates analysis-only 20+30 views with
disjoint question checks, protocol/configuration checks, raw-record provenance,
and all original failure records. Views are explicitly forbidden from launching
or resuming model calls. The six reports are rebuilt from individual observations,
not averages of batch percentages. Original twenty-question reports remain intact;
the prior dashboard is retained as a reference before publishing the fifty-question
dashboard. No manuscript edits, commit or push are part of this task.

Status: all nine extension phases finished and all six 50-question reports are
published. The same-model triadic phase initially stopped after 14 completed
questions when an E verbosity diagnostic returned a
Databricks HTTP 400 output-limit error without billing usage. Formal debate and
probability measurement prompts were not implicated.

The user authorized a runtime fix and recovery. The original stopped journal is
unchanged. A separate `same_model/triadic-output-limit-recovery` journal imports
all 1,221 original attempts, including 1,220 successful requests and the failed
attempt with its unresolved reservation. Offline reconstruction verified every
saved success against the unchanged request generator. The recovered request
succeeded on its first additional attempt with identical prompts, reasoning and
output-token limits; no successful generation was repeated.

The new opt-in policy recognizes only the inspected output-limit HTTP error,
allows at most two additional identical attempts, and isolates an exhausted task
and its true dependents. Missing billing usage stays unknown; token summaries no
longer crash when a provider omits a count. Other unrecognized failures still
stop for inspection. The original frozen execution plan is not overwritten:
`recovery-overrides.json` records the replacement journal and manifest hash.
Original charges/reservations are carried into that replacement exactly once.

The recovered phase finished all 30 questions and all 60 primary E comparisons.
Its output-limit retry and two unrelated answer-length retries each succeeded on
the first additional attempt. All nine read-only audits passed against their
respective archived implementation snapshots. All 231 preservation-file hashes
and the recovery's imported rows remained unchanged. The finalizer rebuilt the
combined views and dashboard; publication still stops on an audit error.

Final 50-question coverage:

| Setting | Questions | Debate paths completed | E comparisons available |
| --- | ---: | ---: | ---: |
| Mixed-family, two members | 50 | 300/300 | 300/300 |
| Mixed-family, three members | 50 | 600/600 | 100/100 |
| Same-family, two members | 50 | 300/300 | 294/300 |
| Same-family, three members | 50 | 600/600 | 100/100 |
| Same-model, two members | 50 | 300/300 | 292/300 |
| Same-model, three members | 50 | 600/600 | 100/100 |

The 14 unavailable dyadic E comparisons reflect previously exhausted synthesis
requests and their dependent comparisons. They are excluded, never treated as
ties, losses or successful measurements. No extra retry allowance was added to
those exhausted requests.

For the **30-question extension only**, received-response token estimates total
**US$84.54**, including retries. Nine attempts have unresolved billing usage:
eight connection timeouts and the output-limit response, retaining **US$22.27**
in conservative reservations. These reservations are not known charges, and the
received-response estimate is not a provider invoice. The stopped source journal
is excluded from totals because its attempts are carried into the replacement.

Dashboard:
`/Users/Chen/Documents/research/llm-committee/docs/committee-experiment-dashboard.html`

The unchanged old dashboard is retained as
`docs/committee-experiment-dashboard-20-question-reference.html`. The expanded
reports retain separate settings, prior/current-label C/D conditioning, full
qualitative cases, and explicit missingness; no global C/D pooling was added.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python -u \
  -m scripts.finalize_scaled_public_history --wait \
  --audit-python /tmp/committee-tinker-check.sqHb9X/venv/bin/python
```

Verification before expansion: 515 regression tests passed, seven optional
tests skipped. The seven new selection/count tests and two merge-view tests
also verify disjointness, unchanged original schedules, preserved missingness,
non-dispatchable views and idempotent merging. Live gate audits additionally
verified the optional native tokenizer/probability path.

Recovery verification: 529 regression tests passed, seven optional tests skipped.
A subsequent 26-test focused run also passed, including the added snapshot-output
lifetime regression test. The final HTML checks verified 50 distinct questions
per setting, table dimensions, 24 triadic diagnostic cases across the two batches,
and preservation of source records. The reporting fixes made no model calls.
