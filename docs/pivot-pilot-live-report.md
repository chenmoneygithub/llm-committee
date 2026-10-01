# Mixed-family pilot — live engineering report

**2026-09-26 update:** the record below describes the original exact-score policy. The user
subsequently chose original top-20 probabilities with absent candidates approximated as zero,
followed by normalization, without supplemental scoring. A separate offline reanalysis now
calculates 27/27 D-choice reads and 17/17 D-text pairs under that policy, with no API calls.
The old scoring discrepancy is not claimed to be fixed. See [change record](pivot-topk-zero-fill-2026-09-26.md).

Status: **finished with two failed probability checks in one D-text pair**
(`completed_with_measurement_failures`, exit code 2). All nine debates and scheduled
generation/judging work finished; this is not full D acceptance. No process is running.
This is a three-question engineering pilot, not an effect estimate or a main-study result.
The main 80-question study has not started.

## Final coverage

| Measurement | Planned | Completed and usable at the format/interface level |
|---|---:|---:|
| Initial answers | 9 | 9 |
| A: formal replies and self-labels | 168 | 168 |
| B: reply-text judgments | 24 | 24 |
| C: option plus full-position reads | 40 | 40 |
| C: text-shift judgments | 32 | 32 |
| D-choice: complete native-option distributions | 27 | 27 |
| D-text: complete argument/control probability pairs | 17 | **16** |
| E: chairman syntheses | 12 | 12 |
| E: judgments across both answer orders | 18 | 18 |

The 339 journal records comprise 337 completed generation/judgment calls and two rejected
supplemental scoring calls. A completed letter-generation call alone does **not** establish
complete candidate probabilities. The two rejected scores are the argument and control
arms of the same Inkling event: question 24, hostile, node n14. Its D-text metric is null.
No events were resampled or replaced; there are still exactly 24 preselected B/C/D events.
Question 24 has 4/5 usable D-text pairs; questions 22 and 48 each have 6/6.

## What is being run

- Questions: archived GlobalOpinionQA 24 (gender and university education), 22 (greatest
  worldwide threat), and 48 (free expression in democracy), with original choices and the
  archived opinion-question screen.
- Committee: GPT-5.6 Terra, Qwen3.8-27B, full Inkling. Chairman: Terra. Judge: Gemini 3.8 Flash.
- Existing Databricks OAuth access for Terra/Gemini; Tinker for Qwen/Inkling.
- Three tones per question, one run each. Six terminal trajectories per debate, five turns
  per trajectory; shared prefixes generated once. The frozen route has 17, 20, and 19 unique
  reply nodes per tone for the three questions, respectively: 168 formal replies in total.
- One preselected sample of 24 events across the three questions is shared by B/C/D.
  Referencing each event's previous own position and initial position gives 40 C reads and
  32 distinct C-text comparisons. D covers only the two open models: 27 choice reads and
  17 argument/control pairs. E produces 12 syntheses and 18 order-swapped judgments.

Final artifacts: `runs/mixed-family-pilot/live-format-v4-resumed/`.
The manifest, SQLite journal and source snapshot identify the exact implementation,
requests and checkpoint lineage. Completed calls are reused by exact request hash, never
regenerated to obtain a valid format or a preferred result. Older prompt versions are
development records, not extra research observations.

## Issues found and handled

1. **Output format.** Inkling once returned free text instead of the formal JSON reply;
   a later Terra C read returned a choice without its opinion paragraph. These are preserved
   in the earlier attempts. V4 makes the common output contract explicit for every model.
2. **D supplemental scores: unresolved.** Inkling omitted some rating letters from its
   top-20 output probabilities. Re-scoring the exact saved prefix produced different
   overlapping log probabilities. The prefix, candidate-token identity and score position
   checks pass; the numerical cause is not established. The 0.01 logprob tolerance has not
   changed. Failed reads remain invalid; their dependent D metrics are null, never zero.
   Continuing independent work does not turn those failures into a passed D check.
3. **Gemini usage accounting: fixed.** Databricks can report completion and reasoning
   separately, or omit the reasoning partition while still giving a consistent total.
   Billing now reconciles these fields and counts reasoning once. Two already-received
   judgments were recovered offline without another model call; original responses and
   correction records are retained. Unexplained totals still stop the run.

## Checks completed during execution

All planned B/C-text evaluations and E comparisons completed. Actual saved payloads confirm:

- All nine no-debate/debate comparisons reuse identical initial answers. Every unique
  debate node appears once, not once per leaf. The largest chairman input was **6,010 tokens**.
- E's two presentations swap exactly the same answers.
- B receives only the question, options, peer text and reply. C-text receives only the
  question, options and before/after position text—not tone instructions, model names or
  A's self-reported label.
- C responses include both the choice and the full position. C and D direct reads have
  no reported new reasoning tokens. Formal debate requests keep reasoning enabled; an
  enabled model need not produce positive reasoning tokens on every easy question.
- All **61** native probability responses have exact agreement between their sampled-token
  logprob and the corresponding top-k entry. The observed discrepancy is in a separate
  supplemental scoring call, not within a single sampled response. The minimum candidate
  mass among complete native reads is 0.98886; this descriptive check does not excuse the
  two incomplete distributions or relax their consistency gate.
- A provider-disabled replay regenerated every expected request locally and reproduced
  `report.json` exactly from the journal. Any new generation would have raised an error.
  It made **zero paid calls**, changed no journal records and added no cost.

Automated checks: **201 passed, 3 optional tests skipped** in the project environment.
The three native-renderer tests also pass in the live SDK/tokenizer environment; the
combined native/provider/continuation suite there passed 36 tests using fabricated responses,
not additional paid calls. Lint and whitespace checks pass.

## Budget and interpretation

Authorization is **US$30 across all attempts**, not per output directory. **Final estimated
cost across all attempts: US$1.557001068 (about US$1.56).** This combines US$0.456586108
from preceding attempts/checkpoints with US$1.100414960 in new continuation calls. The
continuation's new-spend cap was US$29.54. These are token-based estimates, not provider
invoices. Imported calls have zero new charge; their original costs remain in the lineage
ledger, including the two documented offline accounting corrections.

This pilot does not validate judge accuracy, estimate statistical power, or justify pooling
the available D reads after failures. The D scoring interface needs resolution before the
main study. A's additional instruction-removal ablation and the human audit are not part
of this pilot. No paper changes, main-study launch, deployment, commit or push are included.
