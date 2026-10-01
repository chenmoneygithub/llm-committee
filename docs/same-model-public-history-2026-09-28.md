# Same-model: two- and three-member cohorts

The user authorized both same-model settings after the same-family run.
Scope: the same 20 approved questions, not the remaining 40 or additional
repetitions. A/B/C are three distinct member identities using GPT-5.6 Terra;
each receives its own initial-generation request and retains branch-local
participation history. Requests are not deduplicated merely because the
model name matches.

## Frozen design and execution

The dyadic and triadic plans and source hashes were checked to match their
same-family counterparts exactly. Initial answers, debates and measurements
are fresh; no model outputs from other rosters are reused. Formal calls use
medium reasoning and jointly generate reply/agreement/choice/position.
Only initial answers and public replies form subsequent debate history;
private updated position fields, historical tones and judgments are not
reinjected. Gemini 3.8 Flash remains the judge; Terra remains the chairman.

| Setting | Paths | Replies | Scope | Planned successful calls |
| --- | ---: | ---: | --- | ---: |
| Exclusive dyads AB / CA / BC | 120 | 360 | A/B/C and E1 member outcomes | 1,168 |
| Dyadic E2 follow-up | — | — | 120 primary pairs and 12 diagnostic cases | 540 |
| Three-member routing | 240 | 673 | A/B/E1/E2, with 40 primary pairs | 1,497 |

No D in either setting and no local C in the three-member setting. Dyadic C
retains previous/current agreement-label grouping. E1 remains initial-to-last
participation analysis, not grouped by final label/tone. E2 retains the exact
190–210-word primary synthesis constraint and both answer orders.

Two 32-request pools run concurrently (64 combined), with up to eight rolling
questions each. A one-question gate is audited, then the same journal resumes
to 20; no gate response is regenerated. Original request plus at most two
identical technical retries; first format-valid answer is retained. Only true
dependents are blocked on exhaustion. No new spending cap.

Live directories:

- `runs/public-history-same-model-dyadic-20-20260928/live`
- `runs/public-history-same-model-triadic-20-20260928/live`
- `runs/public-history-same-model-quality-20-20260928/live`

Only presentation adapters under `scripts/` are changed. Experiment code hashes
match the same-family runs. Reports explicitly label A/B/C, so repeated Terra
names cannot obscure member identity. Final output will populate the remaining
two tabs of `docs/committee-experiment-dashboard.html`.

## Earlier E failure: retry clarification

The same-family BC baseline for question `archived-global-63` already received
three attempts: 169, 150 and 167 whitespace-separated words. All failed the
190–210-word contract. Both continuations therefore lack their baseline
comparison (118/120 available pairs). The user asked whether it was retried;
this was verified directly from its read-only request journal. No fourth
attempt, changed prompt, relaxed length limit or retrospective replacement
was authorized or performed in this task. Previous results and the failure
record remain unchanged.

## Gate checks

Both first-question gates completed without retries or missing outputs.
The dyadic audit reconstructed all 57 requests and the complete question
record; the triadic audit reconstructed all 68 requests, both E2 comparisons
and the complete question record. The triadic primary answers were 198–202
words; the largest chairman input was 3,374 words. Both journals were resumed
to their 20-question plans without changing frozen code or regenerating the
gate. No manuscript edits, commit or push.

## Completed result

All formal debates are complete. No experiment remains running.

| Phase | Successful logical calls | Additional attempts | Complete outputs | Token-estimated USD |
| --- | ---: | ---: | --- | ---: |
| Dyadic A/B/C and E1 | 1,168 | 0 | 20/20 questions; 120/120 paths | 4.131723 |
| Dyadic E2 | 530 | 17 | 116/120 primary comparisons | 4.558268 |
| Triadic A/B/E | 1,497 | 2 | 20/20 questions; 240/240 paths; 40/40 comparisons | 9.743847 |
| Total | 3,195 | 19 | | 18.433838 |

Costs include gates, successful calls and invalid attempts, based on returned
token usage rather than provider invoices. The full phases took 129.65 seconds
(dyadic), 228.82 seconds (triadic), and 107.83 seconds (dyadic E2), excluding
their gates. These phases overlapped, so their durations are not additive wall
time.

Two E2 baseline requests exhausted their identical retries: question
`global-row-2402`, pairs CA and BC. The item asks whether people who speak a
different language would be unwanted neighbors. Each of the six raw responses
was exactly `{"answer":"B"}`; the model selected the listed option instead of
producing the requested 190–210-word synthesis. The original responses verify
that this was not parser truncation. No prompt or length requirement was changed.
Each missing baseline blocked four dependent order judgments across its two
continuations, leaving four unavailable primary comparisons in total. All other
planned E2 and diagnostic tasks completed. Missing comparisons are not ties or
losses. The earlier same-family E failure remains separate and unchanged.

The audits reconstructed all 3,195 successful request contexts and all 60
question records across the three journals. The E2 audit also verified all six
rejected outputs, two exhausted requests and eight blocked dependents. Both
answer orders were verified for all 116 available dyadic and 40 triadic primary
pairs. Accepted primary lengths are 191–210 words (dyadic) and 195–210 words
(triadic); the largest triadic chairman input is 4,190 words.

Outputs:

- `docs/turn-tone-dyadic-same-model-public-history-2026-09-28.html`
- `docs/turn-tone-triadic-same-model-public-history-2026-09-28.html`
- `docs/committee-experiment-dashboard.html`

The dashboard now has all six settings, with two dyadic tabs explicitly marked
as having missing E2 results (same-family 118/120; same-model 116/120). The two
new reports contain 142 tables with checked column counts and resolving local
links/anchors. A/B/C member identities are explicit in same-model endpoint tables
and qualitative transcripts. Previously saved reports and same-family journals
were checked against 77 pre-run file hashes; all are unchanged.

Verification: 494 regression tests passed, seven optional native tests skipped;
all 12 dashboard tests passed separately (506 total passes). Browser checks
covered all six tabs, both missingness banners, member identity labels, section
navigation, expandable transcripts and mobile width. No additional questions,
manuscript changes, commit or push.
