# GPT same-family: two- and three-member cohorts

The user authorized both 20-question same-family settings on September 27
(execution timestamps September 28 UTC). No additional questions or same-model
cohort are launched by this task.

## Frozen design

Slots A/B/C are GPT-5.6 Luna / Terra / Sol, using the configured Databricks
endpoints with medium reasoning. Gemini 3.8 Flash remains the external judge;
Terra remains the chairman. No provider, prompt, or generation-protocol change.
The model roster is the only change relative to each mixed-family setting.
Initial answers, replies and all judgments are fresh for this roster.

Both planners were checked to produce exactly the same plans and source hashes
as their corresponding mixed-family cohort. This includes questions/options,
routes, paired local-tone schedules, sample membership and available evaluation
orders. Positions are recorded with each formal reply but private updated fields
are not reinjected into later debate turns.

| Setting | Formal replies | Paths | Layers | Logical calls |
| --- | ---: | ---: | --- | ---: |
| Exclusive dyads AB / CA / BC | 360 | 120 | A/B/C, with E1 member outcomes | 1,168 |
| Dyadic final-answer follow-up | — | — | E2, 120 primary comparisons and diagnostics | 540 |
| Three-member routing | 673 | 240 | A/B/E1/E2, 40 primary comparisons and diagnostics | 1,497 |

No D calls in either cohort. No local C in the three-member cohort. C in the
dyadic report retains previous/current agreement-label conditioning. E1 remains
initial-to-last-participation outcome analysis, without local-label grouping.
E2 uses the same 190–210-word synthesis constraint and both answer orders.

## Execution and preservation

Two separate journals, 32 in-flight requests each (64 total) and up to eight
active questions per cohort. First audit one question, then resume its journal
to 20 without regenerating the gate. Existing bounded technical retries and
failure isolation remain unchanged. No new budget cap.

- Dyadic source: `runs/public-history-same-family-dyadic-20-20260928/live`
- Dyadic E2: `runs/public-history-same-family-quality-20-20260928/live`
- Triadic source: `runs/public-history-same-family-triadic-20-20260928/live`

Existing mixed-family outputs and frozen implementation files are not edited.
Report adapters under `scripts/` handle roster labels, omit unmeasured D
sections, and link the corresponding two-/three-member reports. Final results
will occupy the same-family tabs of `docs/committee-experiment-dashboard.html`.

## Later expansion to the other 40 questions

The approved bank contains 60 questions, including all current 20. Additional
questions need new generations, but existing results do not need regeneration
if the protocol remains unchanged. Current planners contain explicit 20-question
validation/counts; expansion is not merely changing a CLI number. Before a later
60-question rollout, generalize those counts and add a frozen extension plan
and separate append-only batch journal. Reuse the archived per-question triadic
routes, and freeze new dyadic/local-tone/sample plans for the 40 new questions
once, shared by all rosters. Do not rebalance or redraw the existing 20.

Combine the batches only for matching protocols and question-disjoint records,
with question-level uncertainty. No expansion has been launched here.

## Completed execution

Both 20-question cohorts have finished. No experiment remains running.

| Phase | Successful logical calls | Extra attempts | Completed outputs | Token-estimated USD |
| --- | ---: | ---: | --- | ---: |
| Dyadic A/B/C and E1 | 1,168 | 0 | 20/20 questions; 120/120 paths | 4.174499 |
| Dyadic E2 | 535 | 17 | 118/120 primary answer pairs | 4.721992 |
| Triadic A/B/E | 1,497 | 2 | 20/20 questions; 240/240 paths; 40/40 answer pairs | 9.485803 |
| Total | 3,200 | 19 | | 18.382295 |

Costs include gates, accepted calls and invalid attempts, using returned token
usage; they are not provider invoices. Full phases took 142.59 seconds (dyadic),
237.78 seconds (triadic), and 110.93 seconds (dyadic E2), excluding their initial
gates. The phases overlapped, so these durations should not be added as elapsed
wall time.

One E2 baseline, question `archived-global-63`, pair BC, failed the 190–210-word
constraint on the original attempt and both identical retries. The baseline
was therefore unavailable to both continuations; their four dependent judge
requests were blocked. No prompt/length bound was relaxed and no extra retry
beyond the frozen policy was dispatched. Both comparison rows use 19/20 pairs;
the two unavailable comparisons are not ties or losses. Other E2 and diagnostic
tasks completed, and all formal debates/B/C remain complete.

Read-only full audits reconstructed all 3,200 successful request contexts and
all 60 question records across the three journals. The E audit additionally
verified all three rejected attempts, the one exhausted request and its four
blocked dependents. All 118 available dyadic and 40 triadic primary pairs use
both answer orders. Accepted primary answers are 190–210 words (dyadic) and
193–210 words (triadic). The largest triadic chairman input is 4,078 words.

Final reports:

- `docs/turn-tone-dyadic-same-family-public-history-2026-09-28.html`
- `docs/turn-tone-triadic-same-family-public-history-2026-09-28.html`
- `docs/committee-experiment-dashboard.html`

The dashboard has four finished settings, with the dyadic same-family tab
explicitly marked as having missing E results, and two same-model placeholders.
Both original mixed-family reports remain byte-identical to the previous
dashboard snapshots. All 142 new report tables have consistent column counts;
local links and anchors resolve. Browser checks covered all four completed
tabs, the missingness banner/118-of-120 denominator, the same-roster cross-link,
and the two pending tabs. No manuscript changes, commit or push.
