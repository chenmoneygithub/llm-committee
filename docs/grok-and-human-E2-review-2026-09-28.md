# Independent E2 judge and blinded human review

Scope authorized by the user: **mixed-family, three-member final-answer quality only**.
No additional debate, chairman synthesis, B/C/D judging, or diagnostic generation.

## Grok check — completed

- Databricks endpoint: `databricks-grok-4-6`, verified ready before dispatch.
  All 200 provider responses identify `global.xai.grok-4.6`.
- Reused the exact existing Gemini rubric, question/options, answer texts,
  left/right presentation order, JSON schema, low reasoning effort and
  2,048-token output limit. Only judge model and local journal key changed.
- Fifty questions × two tone assignments × both answer orders = 200 calls.
  The two-call gate was reused, not repeated. All 200 completed without retries.
- Received-usage estimate: **US$0.7353**. No unknown usage reservations remain.
  Pricing uses the published Databricks rates at $0.07/DBU, a conservative 10%
  regional uplift, and no promotional discount. This is not a provider invoice.
  The local safety cap was $8; it was not reached.

| Tone assignment | Gemini debate wins | Grok debate wins |
| --- | ---: | ---: |
| Original | 78% [67%, 88%] | 79% [70%, 87%] |
| Alternate | 79% [69%, 89%] | 80% [70%, 89%] |

Wins are the fraction of decisions choosing debate, not percentage quality
improvement. Intervals bootstrap the 50 questions (10,000 resamples), without
multiplicity adjustment. Both judges evaluate the same pairs; their results are
kept separate, not pooled or majority-voted. The assignments share prefixes and
questions and are not independent replications.

The judges choose the same answer in 148/200 matched-order decisions (74%). Their
two-order verdicts match on 69/100 pairs. Grok's own preference changes with order
on 23/100 pairs; these are not explicit ties. Similar aggregate preferences do
not mean identical pair-level judgments or establish human accuracy. The existing
constructed diagnostics remain Gemini-only; Grok did not run that battery.

Run, frozen tasks, raw responses, usage, report and audit:
`/Users/Chen/Documents/research/llm-committee/runs/grok-mixed-triadic-E2-50-20260928`

Reproduce offline summaries with `scripts.grok_quality_check.summarize`; update
the original table/dashboard with `python -m scripts.report_grok_quality`.
The combined-report publisher preserves this extra column on future rebuilds.
Existing experiment journals are unchanged; pre-Grok copies of the affected
standalone report are retained in the judge run directory.

## Human tool — first 20-question review completed

Share only this standalone file:
`/Users/Chen/Documents/research/llm-committee/docs/human-answer-review-20.html`

- Uniform seeded selection of 20 distinct questions from all 50, one pair per
  question. Ten original and ten alternate assignments; A/B presentation balanced
  within each assignment. No model preferences or labels enter sampling.
- Full, unedited question/options and answer texts. No source labels, model names
  as metadata, experiment IDs, judge decisions, or answer-origin key in the HTML.
  Natural wording in an answer is not edited or guaranteed to be uninformative.
- Reviewer ID, A/B preference, substantive tie, cannot-judge, optional evidence.
  Ties and cannot-judge remain distinct; this is not the models' forced-choice task.
- Local autosave, resume after reload, JSON export/import, mobile layout and
  keyboard navigation. No network requests. Export a backup before closing or
  moving the HTML; storage availability depends on the browser.
- Export includes opaque item IDs, preferences and notes, not the answer origins.

The researcher-only mapping lives in
`runs/human-answer-review-20-20260928/researcher-key.json`. Do **not** distribute
that file with the HTML. Sampling and source hashes are frozen alongside it.
The first completed export has 20 binary preferences (no ties or exclusions):
14 debate and 6 baseline, decoded only after annotation. Sixteen ratings were
selected with Chinese displayed and four with English. This one-reviewer sample
does not establish human accuracy. Original-rubric agreement with the reviewer
is 13/20 for Gemini and 12/20 for Grok in the matching presentation order; the
completed rubric-sensitivity comparison below retains these original results.

Verification: 200 requests and raw payloads checked against frozen inputs, all
usage totals reconciled, source hashes unchanged. Unit tests cover exact request
reuse, two-order scoring and missingness, balanced source-blind sampling, and
idempotent report columns. Real Chrome tests passed for desktop/mobile rendering,
reviewer-specific save/reload, export/import, wrong-dataset rejection, export
without local storage, and absence of network calls or JavaScript errors. UI-test
labels existed only in an isolated test browser and are not research annotations.

## Bilingual reading aid — September 28

The same HTML now opens in Chinese on first use, with **EN / 中文** buttons at the
top right. It remembers the display language. Questions, survey options, both
answers, and the interface switch together. The English items, sample, opaque IDs,
answer sides, reviewer storage keys, and dataset ID (`2855f1a76da6c8635438da22`)
are unchanged. Switching language does not clear or alter a rating, its timestamp,
or a note. Existing English-only saved annotations can still be loaded/imported.

Gemini 3.8 Flash translated all 143 text segments (20 questions, their options,
and 40 answers) in **one successful batch request**, with low reasoning. It saw
only the already-blinded public text. Instructions required preserving arguments,
uncertainty, negation, numbers, and paragraph boundaries, without evaluating,
improving, or identifying the answers. All segments were returned; no paragraph
count or unusually-short-output warnings occurred. The two numeric-token warnings
were valid unit conversions: $1.3 billion → 13 亿美元 and $200–300 million →
2 亿至 3 亿美元, in both answers to one question.

The raw response and original translation remain frozen. A separate correction
file records 12 narrow source-fidelity edits, checked against the blinded English
text by the coding assistant—for example, correcting “blank-check support” from
“空头支票” to unrestricted support, and removing an added “completely” before
“agreement.” This is **not independent human validation**. The public HTML contains
the corrected text and its content hash, not the correction history or private
source mapping. English remains the reference when a translation is ambiguous.

Successful-response token-usage estimate: **US$0.091513125**, not a provider invoice.
An earlier request was rejected with HTTP 400: its flat schema had 143 properties,
exceeding Databricks' documented 64-key limit. It returned no usage counts, so its
journal conservatively retains a **US$1.1354112 unknown-billing reservation**; this
is not a measured charge. The successful request used a three-key array schema.
There were no retries of the successful request. No committee answers or model
preference judgments were regenerated.

Each exported response additionally records:

- `rating_language`: language displayed when the preference was selected;
- `rating_translation_id`: the Chinese text version, or null for English;
- `viewed_languages_at_rating`: languages viewed in this browser session before
  that selection, plus any previously saved rating-language history;
- `last_edit_language`: language displayed during the latest rating/note edit.

Merely switching language does not re-rate the answer. Editing a note in another
language does not change the rating's language. Legacy rated responses are marked
English, because the previous tool was English-only. The annotation export keeps
its original dataset/version identity and adds `annotation_format: bilingual-v1`.
These fields provide provenance; they do not remove translation-induced judgment
differences. Any future analysis should identify translation-assisted annotation,
not silently treat it as an English-only human evaluation.

Rebuild offline with `python -m scripts.human_answer_review`. The builder validates
the English source hash, item identities, options, answer sides, translation hash,
and each source-anchored correction before embedding it. Translation requests,
English snapshot and usage reports are under
`runs/human-answer-review-20-20260928/translation-zh-v2`; the rejected attempt stays
under `translation-zh-v1`. Original machine translations and fidelity corrections
are in `translation.zh.json` and `translation.zh-corrections.json`, respectively,
in the parent private directory. Do not distribute that directory with the HTML.

Verification: 35 relevant unit tests passed. Isolated Chrome checks passed for all
20 items in both languages, stable A/B text and ordering, switching without losing
ratings/notes or changing rating timestamps, language provenance, reload, old and
new JSON imports, old English-only browser sessions, export without local storage,
wrong-dataset rejection, and desktop/mobile layout. No JavaScript errors or network
requests occurred. Screenshots and the repeatable browser check are in the v2 run
directory. Browser-test labels were isolated fixtures, not research annotations.

## Process-reference-neutral judge revision — completed

Following the first human annotation and discussion of item 18, the user chose
to change only the judge instruction for now. Chairman prompts, all generated
answers, the annotation page, human responses, and old judge ratings remain
unchanged. A separately labelled old/new section is now attached to the original
report and dashboard. This is an exploratory rubric-sensitivity check motivated by an
observed disagreement, not a preregistered or held-out human validation.

Version `E2-process-reference-neutral-v2` appends exactly this instruction to the
existing E rubric, with no other rubric or forced-choice changes:

> Do not reward or penalize references to a committee, its members, or a discussion
> process merely because they appear. Judge the substantive reasoning rather than
> the presence or absence of these references. If an answer has a genuine
> evidential or logical gap, identify that specific gap; do not infer one solely
> from committee-related wording.

This does not ban considering substantive information about agreement among
contributors. Nor does it require overlooking unsupported claims or treating
genuine omissions as cosmetic. It removes a standalone bonus/penalty for references
to the answer's production process.

`scripts/prepare_process_neutral_judging.py` provides the versioned request
transformer and an **offline-only** preparation command. It does not contain a
live dispatch path. It prepares the same mixed-family triadic 100 answer pairs
for Gemini and Grok in both answer orders (400 logical judgments, now completed).
It includes all existing pairs, not just human/model disagreements. Exact answer
text, question/options, ordering, model, reasoning effort, output schema and token
limit are preserved. No annotations, source labels, or old preferences are placed
in the new model inputs. Only the developer instruction and versioned request key
change; old frozen experiment code and journals remain untouched.

```sh
python -m scripts.prepare_process_neutral_judging
```

Prepared inputs and manifest:
`/Users/Chen/Documents/research/llm-committee/runs/E2-process-reference-neutral-v2-20260928`

The preparation manifest retains its historical `prepared_not_dispatched` status;
`execution-report.json` records the subsequent completed live run. Both judges
completed 200/200 logical requests. No new debate, synthesis, translation, or
other model-configuration evaluation was run.

### Results: pooled assignments, questions as the resampling unit

| Judge | Old debate wins | Revised debate wins | Paired change, 95% CI | Changed decisions |
| --- | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash | 78.5% [70.5, 86.0] | 78.5% [70.5, 86.0] | 0.0 pp [−5.0, +4.5] | 16/200 |
| Grok 4.6 | 79.5% [72.5, 86.0] | 81.0% [75.0, 87.0] | +1.5 pp [−3.0, +6.0] | 33/200 |

These are preference frequencies, not quality-improvement magnitudes. Each
question first averages both assignments and both answer orders; intervals
bootstrap the 50 question means, not 200 independent decisions. Gemini switched
eight judgments in each direction. Grok switched 15 debate → baseline and
18 baseline → debate. Answer-order inconsistency remains: Gemini 13/100 →
15/100 pairs; Grok 23/100 → 26/100. An order-dependent split is not a substantive tie.

For the already-inspected human sample, matched-presentation-order agreement
is Gemini 13/20 → 12/20 and Grok 12/20 → 12/20. Across both presentation orders:

| Judge | Both agree with human, old → new | Both disagree, old → new | Order-dependent, old → new |
| --- | ---: | ---: | ---: |
| Gemini | 12 → 12 | 5 → 4 | 3 → 4 |
| Grok | 10 → 11 | 5 → 4 | 5 → 5 |

Item 18 remains unresolved, not a successful correction: the human selected
A (baseline), old Gemini and Grok selected B/B, new Gemini still selects B/B,
and new Grok selects B/A. These A/B identities match the annotation page, not
the survey option letters. Gemini's revised swapped-order explanation still
criticizes "awkward references" and an "uncleaned synthesis of prompt inputs";
the instruction has not demonstrably removed sensitivity to process wording.

The overall preference pattern is similar, but some individual judgments differ.
The new and old calls are separate stochastic samples, with no unchanged-prompt
rerun control; their differences cannot be attributed solely to the addendum.
The 20-question human sample motivated this revision and is not held out. No
new damage/sensitivity battery was run. The original diagnostic results apply
only to their original judge/rubric, not automatically to v2 or Grok.

### Execution, recovery and offline reproduction

The two-call gate (one call per judge) was reused. The original four-worker run
stopped after 282 successful judgments when a Grok timeout exposed a missing
model in the transport-retry allowlist. A separately frozen recovery entry point
extends only known Databricks network/timeouts for Grok `judge_e` requests.
Sequential recovery finished the remaining 118 without repeating any successful
judgment or changing any frozen request. The two-additional-attempt limit, first
valid response policy, original $8 cap, and unknown-usage reservations remain.
All four failed logical requests recovered; no pair is missing.

There were 404 physical attempts: 400 successful responses and four network
timeouts. Received-response token-usage estimate: **US$1.043981**. The four
timeouts retain **US$5.054615 in conservative unknown-billing reservations**, not
confirmed charges. Neither number is a provider invoice; unknown usage has not
been silently priced at zero. All provider payloads, endpoints, returned model
identities, HTTP statuses and parsed outputs were checked. Original answer and
old-judgment hashes, plus all pre-recovery attempt hashes, remain unchanged.

The standalone report includes two compact summary tables and expandable old/new
reasoning for all 20 human cases, with the original A/B identities and Chinese
reading aids. The six-tab dashboard links to the same comparison; the original
E2 table and original scores remain intact.

`/Users/Chen/Documents/research/llm-committee/docs/E2-judge-rubric-comparison-2026-09-28.html`

```sh
# Offline only; rebuild after any future base-report regeneration.
.venv/bin/python -m scripts.report_process_neutral_judging --attach
```

Execution/recovery manifests, raw journal, comparison JSON, annotation snapshot,
logs and browser checks are in the prepared run directory above. Recovery is
implemented separately in `scripts/recover_process_neutral_judging.py`; frozen
experiment and original runner code were not edited. All 28 relevant unit tests
pass, covering append-only requests, preserved inputs, bounded identical retries,
resumability without paid duplicates, accounting, answer identity, question-level
aggregation, and existing report/dashboard behavior. Expansion to other settings
is deferred, as requested.

An isolated Chrome check also passed: desktop/mobile layout without horizontal
overflow, both compact tables, all 20 expandable cases, item 18's Chinese/English
answers and both-order rationales, all six dashboard tabs, the preserved original
E2 table, and the new report link. No page errors or network requests occurred.
