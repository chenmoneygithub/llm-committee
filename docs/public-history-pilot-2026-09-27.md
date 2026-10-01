# Public-history-only pilot

Protocol: `public-history-dyadic-paired-2026-09-27-v1`.
Authorized scope: update the code and rerun the current 20-question mixed-family
pilot. Do not scale to more questions, rosters, or three-member routing yet.

## What changes

The reasoning-enabled formal turn still generates `reply`, `agreement`, `choice`,
and `position` together. The next turn sees the original question/options, the
dyad's original answer texts, and this branch's public reply history. It does
not receive updated private choice/position fields as `your_current_position`.
Only the public reply is transmitted; labels, tone metadata, private reasoning,
judgments, and measurement results are not part of history.

Initial positions are still ordinary historical answers, not deleted memories.
A model may mention its option or reasoning naturally in its public reply; this
text is not stripped. Structured positions remain in the ragged per-branch,
per-member record for analysis, but do not guide subsequent formal generations.

- C compares the choices/full positions produced in the formal generation.
- D1 replays the corresponding formal **input**, with reasoning disabled and a
  neutral one-letter option request. It sees neither the current formal output
  nor an extra stored-position state. The initial D1 read remains question-only.
- D2 adds the receiver's fixed prior full text as `position_to_evaluate`, without
  a separate old option field. Both arms see that same text and history; only
  the latest peer message changes to token-length-matched filler. This target is
  measurement-only and does not enter the real debate or D1.
- Original top-20 probabilities, zero-fill/renormalization, all seven D2 rating
  probabilities, and all three D2 metrics are retained. No entropy adjustment.

Positive D1 is increased probability support for the same old option. It need
not agree with the reasoning-enabled formal answer. Before/after replay histories
still differ, so removing explicit state does not turn D1 into an isolated causal
effect of the incoming argument, and does not guarantee reduced saturation.

## Why earlier data cannot replace this run

| Protocol | Position generation | Position feedback |
| --- | --- | --- |
| Earlier independent-position pilot | Separate neutral re-ask after a debate reply | No latest private state in debate |
| Explicit-position pilot | Same reasoning-enabled response as reply | Latest private choice and full position reinjected |
| This pilot | Same reasoning-enabled response as reply | Public initial answers/replies only |

The earlier independent-position pilot is close on feedback policy but not on
how positions were generated or how D1 was read. The explicit-position pilot
remains a valid different protocol, not corrupted data. Neither is relabeled as
this new cohort. All initial answers, formal replies, measurements, judgments,
and E outputs are regenerated; only the frozen experimental plan is reused.

## Frozen scope and execution

- Same approved 20 questions and original options.
- Terra / Qwen3.8-27B / Inkling; AB, CA, BC; four-turn dyads.
- Same paired turn-local tone assignments, newly shared T1/T2 and separate T3/T4.
- Same 267 unique sampled B/C nodes, one Gemini 3.8 Flash judge.
- Same 1,676 logical A–D calls; 360 unique formal replies; 120 branch paths.
- Same 540-call length-controlled E: Terra chairman, 190–210-word primary
  answers, both judge orders, 12 presampled calibration cases.
- One-question A–D engineering gate, native audit, then resume the same journal
  to 20 questions. E uses only the completed new source and a separate journal.
- Maximum 64 in-flight calls / eight rolling questions; bounded identical
  technical retries. No new spend cap. Prior A–E cost was approximately US$10.16
  by returned token usage; this is a reference estimate, not a spending promise.

Fresh A–D directory: `runs/public-history-dyadic-20-20260927/live`.
Fresh E directory: `runs/public-history-quality-20-20260927/live`.
Report: `docs/turn-tone-dyadic-public-history-2026-09-27.html`.
Existing runs/reports remain untouched. New protocol IDs prevent cross-protocol
resume, and each run snapshots the code before its first paid request.

The new report keeps local C/D grouped by both agreement labels, separates D
models, and includes individual D1 distributions and complete D1/D2 prompts.
Following the earlier reporting decision, initial-to-last member positions are
under E1 without last-turn label/tone grouping, and chairman quality is E2.
Endpoint choices use all branches; text summaries use only the previously
planned endpoint judgments, with explicit separate denominators. No extra
judging is added for presentation.

## Commands

```sh
python -m llm_committee.pivot.stateful_run \
  --plan docs/turn-tone-dyadic-shared-plan-2026-09-26.json \
  --output runs/public-history-dyadic-20-20260927/live \
  --public-history --live --questions 1
# After the read-only gate audit, resume with --questions 20.
python -m scripts.audit_stateful_dyadic \
  runs/public-history-dyadic-20-20260927/live \
  --output runs/public-history-dyadic-20-20260927/full-audit.json
python -m llm_committee.pivot.quality_run \
  --source runs/public-history-dyadic-20-20260927/live \
  --output runs/public-history-quality-20-20260927/live --live
python -m scripts.report_stateful_dyadic \
  runs/public-history-dyadic-20-20260927/live \
  docs/turn-tone-dyadic-public-history-2026-09-27.html \
  --quality-source runs/public-history-quality-20-20260927/live
```

Use the configured Tinker environment for live execution/native auditing.
Tests and mock runs do not call endpoints. No manuscript edits or git actions.

## Completed execution and verification

The complete **20-question A–E pilot finished successfully**. No scale-up or
additional measurement calls were launched after completion.

| Phase | Completed outputs | Successful logical calls | Extra attempts | Estimated cost (USD) |
| --- | --- | ---: | ---: | ---: |
| A–D | 60 initial answers, 360 replies, 120/120 paths, 267 sampled B/C events | 1,676 | 13 | 5.287488 |
| E | 120/120 primary quality comparisons, plus 12 planned calibration cases | 540 | 12 | 4.815330 |
| Total | No missing questions, paths, or primary comparisons | 2,216 | 25 | 10.102818 |

Costs use returned token usage and include retries and the engineering gates;
they are estimates, not provider invoices. Twenty-four logical requests needed
retries; all recovered, with zero exhausted requests. The first A–D question's
81 successful logical calls (US$0.277534795) are already included above: the
full run resumed the same journal without regenerating the gate question.

Read-only audits saved in `runs/public-history-dyadic-20-20260927/full-audit.json`
and `runs/public-history-quality-20-20260927/full-audit.json` verified:

- Exact reconstruction of all 1,676 A–D requests and all 20 question records.
- All 508 native probability-read positions and reasoning-off settings
  (222 D1 reads, 286 D2 reads).
- All 143 D2 argument/control pairs differ only in the incoming message.
- Exact reconstruction of all 540 E requests and all 20 E question records;
  both judge orders verified for all 120 primary comparisons.
- Primary E answers contain 191–210 words, within the 190–210-word constraint.
  E did not regenerate or modify its source debates.

Verification: **478 tests passed, 7 optional native tests skipped**; the seven
cached native-tokenizer/renderer tests passed separately offline. A final
focused public-history/report regression run passed all **26 tests**. The
archived explicit-state cohort's **1,676 requests and all 20 question records**
were also reconstructed exactly with the retained old-version behavior.
All **62** previously hashed files under the two archived run roots and four
reference report files remain byte-for-byte unchanged.

The final HTML includes fresh A–E results, 143 individual D1 change dots,
10 complete prompt/probability examples, and all 120 paired E comparisons.
E1 records all 240 branch/member choice endpoints and the 214 presampled text
endpoint judgments; no extra judgments were purchased for reporting. Automated
checks confirm all 68 tables have consistent column counts and all local/anchor
links resolve. No new formal-debate or D1 prompt includes `your_current_position`.

Inkling still has a modal probability above 99% in 83 of 92 discussion D1 reads;
Qwen has 53 of 90. This is a distribution diagnostic, not a pooled C/D conclusion
or evidence of a token-position error. Comparisons with the older explicit-state
pilot are descriptive: both initial answers and debate continuations are freshly
sampled here, so differences do not isolate a feedback-policy effect by themselves.
Inspect the conditional results, distributions, and full prompts before scaling.
