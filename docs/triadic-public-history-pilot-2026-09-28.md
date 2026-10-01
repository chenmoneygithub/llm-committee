# Three-member public-history A/B/E pilot

## Scope

The user approved returning to three-member routing first after agreeing to omit
its local C/D analysis. Working roster: the existing mixed-family committee,
Terra / Qwen3.8-27B / Inkling; the GPT-family experiment remains separate and is
not started by this plan. The roster choice was surfaced to the user before any
paid dispatch. Use the same 20 approved questions as the current dyadic pilot.

Only A, B and E are collected. Formal reasoning-enabled turns still jointly
generate `reply`, `agreement`, `choice`, `position`; only public replies are
transmitted. Initial answers remain in their appropriate public branch context.
Updated private position/choice fields, historical tone instructions, agreement
labels and judgments are not injected into later debate inputs. Branches remain
isolated, with recorded positions indexed by branch, member and participation.

## Reused routing, fresh generations

The exact 20 question-specific trees are copied from the archived 60-question
leaning-label manifest. No routing is redrawn and no old model output is reused.
Each tree has two distinct starting exchanges, six terminal paths and five
formal replies per path; prefixes are shared. These trees contain 384 unique
nodes for one tone assignment in total. Natural random paths can involve two
or three identities; that coverage is reported without filtering by outcomes.

The frozen, roster-independent template is
`docs/turn-tone-triadic-shared-plan-2026-09-28.json`. It contains questions,
routes, tone schedules, sample membership, answer order and calibration selection,
plus hashes of its two archived sources. It contains no generated responses.

As in the paired dyadic experiment, two turn-local tone assignments share newly
generated T1/T2 and have different tones at every later node (T3–T5 here).
Assignments are prescheduled and approximately balanced by depth and receiver.
Combined: **673 unique replies and 240 terminal paths**, plus 60 initial answers.
This extends the setting; it does not isolate committee size from debate budget
against the shorter four-turn dyads.

## Measurements and report

- **A:** all unique replies, four strongly/leaning labels by current turn tone.
- **B:** eight distinct base nodes per question, sampled before generation at
  T2–T5. One per depth, then four additional nodes sampled without replacement
  with a mild terminal-depth preference. The two continuations use the same
  selected nodes; shared T2 requests are deduplicated. Total: 294 judgments by
  Gemini 3.8 Flash. Report self-label × text-judge label, not pooled agreement.
- **Local C/D:** deliberately absent. No position re-ask, local shift judgment,
  option-logprob read, argument/filler probe or surrogate global C/D table.
- **E1:** initial-to-last-participation choices for every participating member
  on every terminal path. Text judgments use two uniformly presampled leaves per
  question in both continuations, for every participating member. Identical
  initial/endpoint text pairs share a judgment: 210 unique calls. Initial-only
  members are retained in the records but not counted as observed unchanged
  endpoints. Text and option denominators are distinct. No last-turn label/tone
  grouping; results are reported by model.
- **E2:** the Terra chairman sees all three original answers without debate,
  versus those same answers plus one entire six-leaf discussion tree. Shared
  messages occur once, with parent links; only public replies are supplied.
  One baseline per question is shared across the two tone assignments:
  40 primary answer pairs. Primary answers use the same 190–210-word limit.
  Gemini compares both answer orders. “Debate wins” counts the percentage of
  judge decisions choosing debate, not percentage quality improvement.
- **Judge diagnostics:** the same three construction types as the dyadic E
  protocol, on 12 presampled questions: expanded wording, one contradictory
  reason, and off-topic content. Both-order comparisons and a separate automatic
  construction check are retained; they are not independent human validation.

The report includes all complete public trees, positions, endpoint assessments,
final answers, both-order explanations and all diagnostic cases. The dyadic
report remains separate and unchanged. Confidence intervals resample questions;
branches and tone continuations are not independent new questions.

## Execution

Protocol: `public-history-triadic-ABE-2026-09-28-v1`.

Planned successful logical calls:

| Purpose | Calls |
| --- | ---: |
| Initial answers | 60 |
| Formal debate replies | 673 |
| B text judgments | 294 |
| E1 endpoint text judgments | 210 |
| E2 syntheses and diagnostic variants | 96 |
| Both-order E judgments, including diagnostics | 152 |
| Diagnostic construction checks | 12 |
| Total | 1,497 |

Use the existing 64-request / eight-question rolling pool. Original request plus
at most two identical technical retries; first format-valid answer accepted.
Failures are logged and only true dependencies are blocked. Missing discussion
does not silently become a smaller-tree synthesis; missing judgments are not ties.

An estimate based on the previous pilot's successful per-model/per-purpose calls
is **US$9.19**, or approximately **US$11.94 with a 30% allowance**. The larger tree
and fifth turn can change token use; this is not an invoice, spending cap or
guarantee. No new budget cap is imposed.

First run a one-question engineering gate, audit every request and outcome,
inspect actual chairman input length, then resume the same journal to all 20.
Never regenerate the gate or change frozen code between phases.

```sh
python -m llm_committee.pivot.triadic_run \
  --plan docs/turn-tone-triadic-shared-plan-2026-09-28.json \
  --output runs/public-history-triadic-20-20260928/live \
  --roster mixed_family --live --questions 1
python -m scripts.audit_triadic \
  runs/public-history-triadic-20-20260928/live \
  --output runs/public-history-triadic-20-20260928/gate-audit.json
# Resume the same first command with --questions 20 after the gate passes.
python -m llm_committee.pivot.triadic_report \
  runs/public-history-triadic-20-20260928/live \
  docs/turn-tone-triadic-public-history-2026-09-28.html
```

OpenAI Docs' conversation-state guidance was consulted to verify explicit
history construction. Existing Databricks/Tinker adapters, model names, reasoning
settings and retry machinery are retained; no new provider API or model migration.

## Status

Implementation and frozen design prepared. The 12 new offline tests pass,
including all three rosters, exact 1,497-request reconstruction, no private-state
feedback, shared-prefix deduplication, E length/order checks, endpoint sampling,
no-call resume, and failure isolation. Full regression: **490 passed, 7 optional
native tests skipped**; provider/native-tokenizer code was not modified.

The one-question live gate completed at 2026-09-28 03:44:55 UTC: **68 successful
logical calls, 12/12 paths, 2/2 primary E2 comparisons**, zero extra attempts or
missing results. Token-estimated cost: **US$0.460789287**, already included in
the cohort total on resume. All 68 requests and the question record reconstructed
exactly. The largest chairman input had 3,758 whitespace-separated words;
primary answer lengths were 200–203 words. No private position feedback and no
local C/D requests were present. Gate audit and report are preserved alongside
the run. The same mixed-family journal was resumed to 20 questions; no gate
output was regenerated.

## Completed result

**All 20/20 questions, 240/240 branch paths and 40/40 primary E2 comparisons
completed successfully.** All 1,497 planned logical calls completed; nine extra
format/length attempts recovered nine requests, with no exhausted requests or
technical missingness. The resumed full phase took 292.33 seconds, excluding
the initial engineering gate. Total token-estimated cost, including that gate
and all extra attempts: **US$10.188771546** (not a provider invoice).

Full audit: `runs/public-history-triadic-20-20260928/full-audit.json`.
All 1,497 requests and 20 question records reconstructed exactly. All 20 archived
routes are unchanged; all 40 primary answer pairs use both presentation orders.
Primary answer lengths are 193–210 words. The largest full-tree chairman input
contains 4,667 whitespace-separated words. No old outputs were reused and no
local C/D calls occurred.

E1 contains 720 planned branch/member endpoints, of which 676 have a formal
participation. Of those, 664 have two comparable option letters. The 12 other
participating endpoints are Terra cases with a null initial or final option
(the permitted “none of the listed choices fits” output), not failed requests.
They remain in the full records and are excluded from option-change denominators.
The two-leaf text sample covers 228 participating endpoints using 210 distinct
initial/endpoint text judgments; shared endpoint pairs are judged once.

Final report (full absolute path):

`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-triadic-public-history-2026-09-28.html`

Its JSON sidecar retains exact counts and denominators. The HTML contains A/B,
E1 member outcomes, E2 quality comparisons, all 12 judge-diagnostic cases and
all 20 complete question/tree records. All 120 tables passed column-count checks;
all local/anchor links resolve, and browser screenshots of A/B/E1/E2 were inspected.
The 57 previously hashed source/reference files remain byte-for-byte unchanged.
The GPT-family cohort has not been launched. No manuscript edits, commit or push.
