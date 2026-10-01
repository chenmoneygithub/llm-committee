# E follow-up: fixed output budget, current turn-level tone data

Authorization: add E using the current turn-based-tone experiment. Do not rerun
debate, reintroduce global-tone arms or change A–D. Source is the completed
strongly/leaning 20-question mixed-model dyadic run, not the older 60-question
fixed-tone study. Initial positions and four-turn paths are immutable inputs.

## Comparison unit

Default proposal communicated to the user: analyze each AB/CA/BC two-member
pair separately, rather than combine three independent dyads into a purported
three-member discussion. For each question and pair:

- Baseline: Terra synthesizes the pair's two actual original independent answers.
- Original continuation: those exact answers plus that pair's original four replies.
- Alternate continuation: those exact answers plus that pair's alternate four replies.

The source's T1/T2 prefix is common to its two continuations. Synthesis includes
all four public messages in correct order, not just terminal positions. No third
member's independent position, another pair's history, self-label, private tone
metadata, C/D measurement or judgment is included. The same baseline synthesis
is shared between the two continuations, not regenerated to favor either.

Both synthesis conditions use exactly the same chairman instructions, model,
reasoning setting and output budget. They differ only in supplied discussion.
The judge sees only the question/options and the two answer texts, in both
orders. Its existing Gemini 3.8 Flash rubric and forced-choice format remain
unchanged; 0.5 is order inconsistency, not an explicit tie.

## Output length and failure handling

Every primary answer must contain **190–210 whitespace-separated words**, checked
using `str.split()` before acceptance. No truncation, rewriting one condition
alone, or post-hoc selection by preference. Up to two identical additional
attempts after invalid JSON or length failure, symmetric across conditions.
All attempts remain in the journal with their cost and reasons. Exhaustion
marks the answer and its dependent comparisons unavailable, not a zero score.

This is the same narrow output budget, not exact word-for-word length matching.
Report actual paired word gaps. The previous unrestricted E is not pooled or
used as a causal pre/post comparison: its committees and tone design differ.

## Small diagnostic battery

Presample 12 distinct questions from IDs and seed 20260927, balanced over the
three pairs and two continuations, without using preferences or selecting
observed effects. The anchor is that case's newly generated debate synthesis.
Terra constructs three independent variants:

1. Same substantive content, expanded wording: 290–310 words; no additional
   claims, evidence, examples, reasons, qualifications or omissions permitted.
2. Moderate intended damage: retain the main conclusion and most reasoning,
   introduce one internally contradictory important reason, 190–210 words.
   A different legitimate stance does not count as damage.
3. Severe intended damage: answer an unrelated question about lunar phases,
   190–210 words.

The same blinded E judge compares the anchor to each variant in both orders.
A separate Gemini request checks whether the manipulations actually satisfy
their intended properties, with quotations and yes/no/uncertain assessments.
These checks are **not independent human validation**. Display all 12 selected
cases and all available preferences. Separately flag construction failures and
show a validity-checked diagnostic subset without treating it as a gold standard.
Never regenerate a variant because its preference result is inconvenient.

These diagnostics assess sensitivity and possible verbosity preference in this
small constructed set. They do not certify judge accuracy or establish the
absence of length bias. Primary answer-pair data are never selected by calibration.

## Planned scale and reporting

Twenty questions × three dyads × two continuations = 120 primary comparisons.
There are 60 baseline answers, 120 debate answers and 240 primary judge calls.
The battery adds 36 variants, 12 construction checks and 72 judge calls.
Total: 540 logical calls, before bounded retries. No open-weight generation or
new debate is required; existing Databricks Terra/Gemini endpoints are used.

Report each pair × continuation separately, with 20 independent questions per
row; do not label paths friendly/neutral/hostile from just their last tone.
Report stable preferences, order-inconsistent cases, score and descriptive
question-bootstrap interval; do not count the two judge orders as independent
questions. Include actual lengths, all 60 paired final-answer case views,
blind explanations, failures and calibration diagnostics in the same canonical
HTML. Existing A–D numerical data and tables must remain intact.

New execution files and a separate SQLite journal preserve the previous frozen
implementation and all source data. One-question engineering gate precedes the
full run. No spending cap is introduced, and estimates are not provider invoices.
No manuscript changes or git commit/push are authorized by this task.

## Execution and verification

Completed with the separate-dyad design described above. Original and alternate
paths were not combined into a three-member synthesis. Source A–D raw records,
all existing JSON summary fields, and the A/B/C/D/paired/cases HTML sections
are unchanged. E was appended to the same canonical report:

`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-dyadic-pilot-2026-09-26.html`

- Full offline simulation and exact-request reconstruction passed (540 calls).
- Real gate: 31 successful requests including all three diagnostic conditions;
  no retries. Primary outputs were 198–209 words. All request contexts and
  both-order comparisons were reconstructed exactly before expanding the run.
- Full execution: 20 questions, 120 complete primary pairs, all 540 logical
  calls successful. Eleven invalid-length attempts recovered with one retry
  each; no exhausted requests or unresolved billing reservations.
- Received-response token estimate, including gate and retries:
  **US$4.889548125**, not a provider invoice. No debate was regenerated.
- Final audit reconstructs all 540 requests and 20 question reports exactly;
  all source question hashes and source-manifest hash remain unchanged.
- Primary answers are 195–210 words. Group-level mean debate-minus-baseline
  length differences range from −1.10 to +0.80 words; the largest individual
  paired difference is 14 words. This is a narrow shared budget, not exact
  equality of every answer pair.
- Full regression: **393 passed, 3 optional native-renderer skips**. Focused
  execution/report suite: 16 passed. HTML contains 60 E cases and 12 complete
  diagnostic cases, with valid links/table dimensions. The new E section
  cannot silently disappear on regeneration: `--quality-source` is required.

### Descriptive results (not a universal quality claim)

| Pair | Original continuation score | Alternate continuation score |
| --- | ---: | ---: |
| AB: Terra–Qwen | 60.0% | 57.5% |
| CA: Inkling–Terra | 37.5% | 45.0% |
| BC: Qwen–Inkling | 70.0% | 67.5% |

Scores average the two opposite-order binary preferences; a split pair scores
0.5 and is not an explicit tie. Each cell has 20 questions. Individual
question-bootstrap intervals and all stable/split counts are in the HTML;
intervals are not simultaneous/multiplicity-adjusted. Rows must not be treated
as 120 independent questions. The BC tendency remains under this output budget,
but the direction is not uniform across dyads. This does not isolate what
caused any difference from the older 60-question global-tone study.

On the 12 constructed cases, the judge stably preferred the original shorter
answer to its expanded version in 9, the expansion in 1, and switched under
reversal in 2. It preferred the unmodified source to the contradiction variant
in both orders in 11/12 (one split), and to the off-topic version in 12/12.
All constructions passed the separate automatic property check; they remain
unvalidated by humans. This is not evidence that all length/style bias is absent
or that the judge can reliably distinguish subtle natural quality differences.

Pre-E HTML/JSON snapshots are preserved byte-for-byte inside the new E run as
`report-before-E.html` and `.json`; the hashes were respectively
`5460eb3ce5d732afb194e73eaef6380eecc121c851a2c014470c94c444ad20b3`
and `e4a187d04762cef69cd23090fa5e3cec33099ea944dcebd2786d2a351ae00170`.
The old fully-label and fixed-tone supplementary report files remain unchanged.

Regenerate the combined report without new model calls:

```sh
.venv/bin/python -m llm_committee.pivot.strong_report \
  runs/strong-dyadic-20-20260926/live \
  docs/turn-tone-dyadic-pilot-2026-09-26.html \
  --legacy docs/turn-tone-dyadic-fully-reference-2026-09-26.html \
  --supplementary docs/fixed-tone-supplementary-results-2026-09-26.html \
  --quality-source runs/turn-tone-quality-20-20260927/live
```
