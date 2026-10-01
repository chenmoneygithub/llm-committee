# Forced disagreement on current public-history dyads

Status: completed. All 300 cases have all planned measurements, with 600 new
formal replies and no exhausted requests. The full offline/native-token audit
and desktop/mobile report checks passed. Results are attached to the existing
mixed-family two-member report and six-setting dashboard.
This is not the completed September 26 supplement. Its old independent-position
reads, labels, and measurements must not be reused as the current protocol.

## Agreed scope and stopping rule

- Start with the mixed-family two-member paths: AB, CA, BC over the existing
  Terra / Qwen3.8-27B / Inkling roster. Other settings are out of scope for now.
- Initial independent opinions are T0 (initial), not the first debate turn.
- T1 is the first peer response to an initial opinion.
- A sample is one intervention location: one forced opposing reply followed by
  one normal reply from the member being challenged. It is not a bundle of both
  intervention times, nor an independent question.
- Early sample: force opposition at T1, observe the recipient at T2, then STOP.
  Do not generate T3/T4, and do not append archived T3/T4 after the changed prefix.
- Late sample: reuse the original T1/T2, force opposition at T3, observe the
  recipient at T4, then STOP.
- Early and late interventions are separate branches. Do not force opposition
  at both T1 and T3 in the same branch.
- Tone remains turn-local. Preserve each case's scheduled tone at each turn;
  do not introduce a global tone condition. The force instruction is private to
  the challenger and is not supplied to the recipient or external judges.

The user proposed 300 samples total. Under the definition above, that means
600 new formal debate replies, not 300 cases each expanded into both early and
late interventions. B/C/D judgments, probability reads, and any E synthesis are
additional calls, not included in that count. The completed received-response
estimate is $18.787089, including retries, with zero unresolved reservations.

## Frozen sample

Use only the current completed 50-question mixed-family dyadic data. Sample from
question/pair/turn metadata, never by agreement labels or observed movements.
The frozen allocation is 150 early / 150 late: each of the 50 questions contributes
one early and one late case for each of AB, CA and BC. For late cases, source
continuations are selected with seed 20260928 and balanced 25 original / 25
alternate per dyad. These are per-turn tone sequences, not global tone conditions.
The original/alternate tone sequences share exact T1/T2 generations: deduplicate
shared early events rather than counting them twice as separate evidence.
Preserve question, ordered pair, source prefix, per-turn tones, intervention
time and source request IDs. Uncertainty is clustered by question; 300 events
are not 300 independent questions.

## Natural comparison and measurements

Reuse the corresponding archived natural continuation at the SAME endpoint:
through T2 for early cases, through T4 for late cases. Never compare an early
two-turn intervention with an untruncated four-turn natural outcome.

- Formal debate still emits reply, agreement, choice and full position in one
  reasoning-enabled generation. Only public history enters later debate turns;
  do not reinject private updated positions or labels.
- Request substantive opposition to the incoming message's central conclusion,
  not merely harsher wording. Do not force the self-report field to contain a
  predetermined label or regenerate valid outputs to obtain that label.
- A records the challenger's and recipient's actual labels separately.
- B checks label/text correspondence for both roles, blind to the intervention.
- C measures recipient initial position -> T2 for early cases, and recipient
  T2 position -> T4 for late cases, using recorded formal choices/full texts.
- D follows those same recipients and references, only for the supported open
  models: neutral, no-reasoning D1 option readouts and D2 argument/filler
  endorsement of the recipient's fixed pre-feedback full position. Reads never
  enter the debate. They are separate calls, not probabilities from reasoning-
  enabled formal outputs.
- C/D tables retain [challenger actual label, recipient actual label], with
  early and late results separated. Label-conditioned comparisons are
  descriptive: the intervention can change membership in those label groups.

## E boundary

The stopping rule takes precedence over the earlier proposal to extend every
case to four turns for E. No extra debate turns should be generated just for E.
E is included without additional debate turns. Compare natural and forced
syntheses at matching endpoints within each timing
group, with the same chairman, answer-length contract and blind two-order
judging. An early natural synthesis must use the T2-truncated record, not the
archived four-turn synthesis. Do not pool early/late preferences as if the
chairman had received equal amounts of discussion across those groups.

The old 4+2 counting assumption is superseded: early samples now need two new
formal replies, just like late samples. Existing main-study records and reports
remain untouched.

## Agreed reporting destination

The user confirmed that these results belong in the existing HTML as a
supplementary experiment, not as a replacement main-study result or a seventh
model/routing setting.

- Attach a clearly labelled "Supplementary experiment: forced disagreement"
  section to `docs/turn-tone-dyadic-mixed-family-50-public-history.html` and
  refresh its embedded view in the existing six-tab
  `docs/committee-experiment-dashboard.html`, under Mixed family / 2 members.
- Keep the current main-study tables, observations and denominators unchanged.
  Store supplement data separately in the report JSON and in a new run journal.
- Explain the intervention, natural comparison, sample unit, actual coverage
  and stopping points before showing scores. Report T1 -> T2 and T3 -> T4
  separately; never present them as equal-length full debates.
- Show A/B separately for the instructed challenger and the normal recipient.
  Preserve the challenger/recipient label breakdowns for C/D; display counts
  alongside rates or probability shifts and explain their direction/reference.
- Include expandable paired examples with the shared context, natural/forced
  feedback, recipient replies and before/after recorded positions. Private
  intervention instructions may appear as labelled inspection metadata, but
  are never injected into the recipient or judge inputs.
- If E is included, label it as a same-endpoint natural-versus-intervention
  comparison, separate from the main study's debate-versus-no-debate results.
- Do not publish mock outputs or incomplete runs as completed experimental
  findings. New experimental results are published only after the completed
  run and its source/probability audits pass.

## Frozen execution

New journal and records:
`/Users/Chen/Documents/research/llm-committee/runs/forced-feedback-public-history-300-20260928`

Implementation lives in separate scripts, leaving all archived pivot code and
source journals unchanged. Every reused measurement must have identical model,
prompt, schema and generation settings (only the local request key differs).
The first source success is selected deterministically, not by its outcome.
Failed old measurements remain in their original journals; any new supplemental
measurement has its own request/attempt provenance.

The full graph contains 4,900 logical measurement/generation tasks including
archived reuse. Only 600 are new debate replies; actual new-call totals and reused
counts will be reported after execution. Gate: six cases, 68 new calls, 30 exact
archived reuses, 49.55 seconds, $0.356589 received-response estimate, no failures
or unknown usage. All 12 new probability reads passed native-token alignment
verification. The gate is resumed, not repeated.

Full run: up to 64 in-flight requests and eight rolling question groups. The
existing two-additional-identical-attempt policy and local failure isolation
apply, including the narrowly recognized provider output-limit response. No
output is retried or excluded to obtain stronger disagreement or larger movement.
Unknown charges remain reserved; no new spending cap was imposed. Gate-based
linear cost estimate is about $18, not an invoice or a guaranteed upper bound.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m scripts.run_forced_feedback_public_history --live --questions 50 \
  --max-inflight 64 --workers 8
```

Before the live gate, 109 relevant regression tests passed. Gate audit reconstructed
every new request and reused measurement, checked both public-history protocols
and actual native probability tokens, verified equal-endpoint syntheses, and
confirmed the original source files remained unchanged.

## Completed execution and verification

- 50/50 questions; 300/300 intervention cases; 600/600 new formal debate replies.
- 3,433 successful new calls; 1,467 exact archived measurement reuses. Their sum
  is the frozen 4,900-task graph. All reused requests match model, settings,
  messages and schema, excluding only the request key.
- 27 additional format attempts recovered 25 logical requests; none exhausted.
  No transport or output-limit retries were needed. These attempts are retained,
  not discarded. Semantic non-choices remain unavailable in C, not technical
  failures and not unchanged choices.
- Received-response estimate: $18.787088984; unresolved billing reservations: $0.
  This is token-based accounting, not a provider invoice or a budget cap.
- Live gate: 49.55 seconds; resumed full phase: 354.81 seconds. Completed gate
  successes were replayed from the journal, never regenerated.
- Full audit reconstructed all 300 cases, verified all 3,433 new successes and
  all 1,467 reused records, and independently checked all 650 new probability
  reads against their native token positions. All source hashes are unchanged.
  Audit hashes bind the manifest, final report and exact question records.
- The published main HTML is byte-for-byte unchanged outside the four inserted
  supplement blocks; main JSON observations are unchanged. The same six tabs
  remain. All 300 examples, combined timing/model filters, probability displays,
  standalone links and mobile width were checked in isolated headless Chrome,
  with no external network requests or browser errors.

Publication commands (offline; no paid model calls):

```sh
.venv/bin/python -m scripts.report_forced_feedback_public_history
uv run --offline --no-project --with playwright python -m scripts.check_forced_feedback_report
```

## Descriptive findings

These are supplementary observations, not a replacement for the main tables.
Label groups are post-intervention descriptions, not randomized subgroups.

- A: instructed challengers self-report strongly disagree in 54/150 early cases
  (36.0%) and 26/150 late cases (17.3%); all remaining challenger labels are
  leaning disagree. No outputs were filtered or retried to obtain a label.
- B: among challenger replies self-labeled leaning disagree, Gemini rates the
  text as strongly disagree for 48/96 early and 78/124 late. This is a self/text
  mismatch under one judge, not human-validated evidence that either is wrong.
- C: the most frequent forced label pair is leaning disagree / leaning disagree.
  Early: 0/78 available choices change, 9/79 texts adjust reasons/qualifications,
  and 0/79 change the main conclusion. Late: 0/91 choices change, 6/92 texts adjust,
  and 0/92 change the main conclusion. Other label pairs are retained separately;
  these numbers must not be generalized to all recipient responses.
- D: conditional model-specific tables report D1 prior-option probability and
  all three D2 signals (modal-category change, original-mode probability change,
  expected-rating change). D1 and D2 do not measure the same object, and early D1
  also changes from no public discussion to public-discussion input. Its full
  pre/post change is not the effect of forced feedback alone. Per-case plots and
  same-endpoint natural controls are provided, rather than unconditional averages.
- E: forced-answer wins over natural-discussion answers are 55.3% [49.0, 62.0]
  early and 60.3% [54.3, 66.3] late. Both have 150 pairs over 50 question clusters;
  two-order inconsistent preferences occur in 34/150 and 25/150 respectively.
  Mean natural/forced lengths are 201.38/201.80 words early and 201.79/202.25 late.
  All syntheses meet 190–210 words. These are one judge's preferences, not a
  direct measure of quality gains. No new human review or damage battery ran.

Full report path:
`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-dyadic-mixed-family-50-public-history.html`

Dashboard path:
`/Users/Chen/Documents/research/llm-committee/docs/committee-experiment-dashboard.html`

Machine-readable results and verification are in `results.json`, `audit.json`
and `browser-audit.json` in the run directory. The original gate record remains
in `gate-report.json`, `gate-audit.json` and `gate.log`.
