# Fresh strongly/leaning dyadic rerun

User authorization: replace the four-level debate/judge labels with strongly
agree, leaning agree, leaning disagree, strongly disagree; rerun the 20-question
turn-level-tone experiment and regenerate the same HTML report. Old fully-label
outputs remain reference-only. No other model configuration is authorized here.

## Frozen scope

- Reuse the design, not outputs, in
  `turn-tone-dyadic-shared-plan-2026-09-26.json`. Design SHA-256:
  `b2f3563d1be00b7f3ce39a6b6f84c5c6a2e1f04d2a8c0785463b5533916070f3`.
- Exactly the same 20 questions, original options, AB/CA/BC routing, both
  private per-turn tone schedules and B/C/D sample/reference positions.
- Mixed roster only: GPT-5.6 Terra, Qwen3.8-27B, Inkling. Single Gemini 3.8 Flash
  judge for B/C. No same-model, same-family, three-member or Layer E run.
- Generate all 60 independent initial answers afresh. Each new path shares
  its new T1/T2 prefix, then generates both new T3/T4 continuations.
- Four labels and definitions are identical in A and B. Strong agreement or
  opposition concerns the central position; it does not require agreement or
  disagreement with every detail, nor does it describe forcefulness or certainty.
  Leaning denotes substantive qualifications, not uncertainty about the label.
- The seven-level D endorsement scale is unchanged. No old fully responses or
  judgments are converted to strongly responses or judgments.
- Debate reasoning is on. Neutral C/D direct readings disable reasoning.
  No earlier private tone instructions, explicit historical labels or measurement
  outputs enter debate history; earlier public text still does.

## Counts and accounting

Twenty questions × three pairs × (two shared replies + two original suffix
replies + two alternate suffix replies) = **360 unique formal replies**, with
120 assigned friendly, 120 neutral and 120 hostile. There is no global tone arm.

The original sample has 160 events, including 53 shared T2 events. The alternate
adds 107 T3/T4 events, giving **267 unique B/C events**. D uses eligible Qwen/Inkling
receivers. Identical pre-reply D-text inputs at T2/T3 are requested once; T3 is not
a tone contrast. The HTML's D-text table counts the shared T3 reading under the
original reply label and omits its alternate duplicate. Both branches' C/D-choice
measurements are retained after their new replies.

Planned logical calls:

| Purpose | Calls |
| --- | ---: |
| Fresh initial answers | 60 |
| Debate | 360 |
| Neutral C position reads (also D-choice where supported) | 387 |
| B text judge | 267 |
| C text comparisons | 481 |
| D-text peer/filler reads | 286 |
| Total | 1,841 |

One-question gate followed by an idempotent resume to 20 questions; eight active
questions and at most 64 in-flight requests. Existing bounded format/transport
retry and branch-isolation policies remain. No user-imposed spending cap.
Report received-response token estimates separately from unresolved reservations.

## Preservation and report

New run: `runs/strong-dyadic-20-20260926/live`.

New execution modules use separate versioned strong-label schemas and prompts.
Archived generation modules and original raw runs are unchanged. The optional
label-order argument added to the reporting-only table helper preserves its old
default behavior; it changes no generation or numerical calculations.

Canonical report:
`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-dyadic-pilot-2026-09-26.html`.

Before replacement, archive the previous HTML and JSON byte-for-byte as
`turn-tone-dyadic-fully-reference-2026-09-26.html` and `.json`.
Previous canonical checksums:

- HTML: `d417f7217a806accf8c15c457b1b6bd1bccad2bb3d7f39a10b95876666eb400f`
- JSON: `97ed150bfa891eb78b37c9eeb1133c2e8b7395138844e7117d61282f3f44c623`

The fixed-tone supplementary HTML/JSON are also preserved, with checksums:

- HTML: `5708f69605c780a40d72378bd15cf7b82dea037023485e9cc053f3fb89d5ab15`
- JSON: `1840c7b21e59387a75edf467f735f4981e6922bbf0b5ccd71c91216863c7bcfd`

B is a self-label × text-judge table, not a pooled match headline. Per the
2026-09-27 presentation revision, C's main table groups only by preceding peer
label × current receiver label, pooling turns, model pairs and tone assignments
within each label pair. Pool distinct reply counts, not percentages; shared T2
is counted once. The six columns are the two labels, replies, option changes,
reason/qualification adjustments and main-conclusion changes. Omit zero-valued
missing/unclassifiable columns and `/ valid` headings; show any exclusions in a
footnote if they occur. Retain T and assignment in raw data and qualitative
cases, without treating selected cases as evidence of a causal turn effect.
D follows the same two-label grouping, with separate Qwen and Inkling tables.
Pool T and assignment within each label/model group, but retain D's original
equal-question weighting: average matched readings within each question, then
average the question means. Do not average the former table-cell means.
D1 shows the fixed previous option's probability before → after, and the
difference in percentage points; D2 shows the fixed full text's expected 1–7
endorsement under filler → peer, and their difference in scale points. Explain
the reference, calculation, units and limits before the tables, with observed
D1 examples tied to C's option-switch counts on the exact same subset. Do not
interpret output probability as human-like confidence or a fraction switching.
Retain the existing D2 T3 deduplication under the original continuation label.
Include both previous-participation and initial-position references separately.
Show all 60 paired qualitative cases with shared prefixes once and suffixes side
by side. These remain 20 independent questions; sparse descriptive strata do
not support strong population conclusions. No manuscript edits or git actions.

## Execution log

- Full offline graph: all 1,841 requests and 120 branch paths completed.
- Offline read-only audit reconstructed every request and all 20 question
  records, verifying 60 paired T3 inputs and correct shared-request identity.
- Pre-run regression: 381 passed, 3 optional renderer tests skipped.
- Real gate: one question, 89 successful requests, six successful branches.
  Token-estimated received-response cost US$0.313446779. Every request and the
  report reconstructed exactly; all 32 direct reads had zero reasoning tokens.
- Full 20-question run resumed from the same journal at concurrency 64.
  Status: completed. The resumed process took 237.8 seconds. All 1,841 logical
  requests and 120 branches succeeded; eight format failures each recovered
  with one retry, no exhausted failures or transport retries. Observed peak
  concurrency was 64 requests across eight active questions.
- Received-response token estimate: **US$5.944951782**, including gate and
  retries. No unresolved-billing reservations remain. This is not an invoice.
- Final audit reconstructed all 1,841 request contexts and 20 reports exactly.
  All 673 direct reads had zero reasoning tokens; 543 probability reads used
  the original top-k data at read temperature 1. All 60 T3 request pairs
  differed only in the current private tone instruction and bookkeeping key.
  The 120 shared replies and 142 shared pre-reply D requests were counted once.
- Final regression: **383 passed, 3 optional native-renderer skips**.
- Archived original and fork runs were independently re-audited successfully:
  their frozen execution hashes, source records and exact request contexts
  remain unchanged.
- Canonical HTML and JSON now contain the new strong-label results only.
  Validation confirms 360 unique replies, 267 B/C events, 143 unique D-text
  peer/filler pairs, 60 paired cases, 120 side-by-side panels, valid links and
  table dimensions. The old fully HTML/JSON archives match the pre-replacement
  checksums above; both fixed-tone supplementary files are also byte-unchanged.
- Final report-focused suite: six tests passed after the explanatory notes were
  added. Browser inspection confirmed the real report's C table rendering and
  readable label/turn/assignment breakdown; automated link/table checks passed.
- 2026-09-27 C presentation revision: 37 main-table rows across four incoming-label
  sections became one 10-row, six-column table. Initial-to-final results use the
  same compact format in a separate collapsed view. All pre-existing JSON fields,
  A/B/D/E/paired/cases sections, raw run files and archives are unchanged; the JSON
  adds the two aggregated C summaries. Thirty-four focused tests passed, and a
  browser check confirmed the table fits without horizontal scrolling at 1440px.
- 2026-09-27 D presentation revision: D1 and D2's 106 main-table rows became
  31 rows across four five-column model-specific tables (D1: 8 + 8; D2: 8 + 7).
  The initial-reference view is separately collapsed. Only the D HTML section
  changed; all pre-existing JSON fields, other sections, source runs and archives
  are unchanged. Three label-pair/model summaries were added to the JSON and
  independently reconstructed from event-level readings using equal-question
  weighting. Thirty-eight focused tests passed. Browser inspection confirmed
  the tables fit without horizontal scrolling at 1440px.
