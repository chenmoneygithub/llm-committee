# Strong-disagreement feedback supplement

This is a new supplementary experiment, not a replacement or relabeling of the
completed main study. Human-authored feedback is out of scope for this run.

## Question and paired comparison

Does a newly generated, explicitly opposing peer response produce more change in
the recipient's position than the naturally occurring feedback at the same point?

For each case, freeze a prefix of an existing discussion. If A just spoke and B
would respond next, construct two separate continuations:

1. **Natural feedback:** reuse B's archived reply, then generate a fresh reply from A.
2. **Forced opposition:** regenerate B's reply with a private instruction to oppose
   A's central conclusion, then generate a fresh reply from A.

Both continuations have the identical pre-feedback history. A does not receive
B's private instruction. The later messages from the original discussion are
discarded for this experiment, and the new reply always goes back to A. No
existing A–B–A routing pattern is required.

The natural feedback is archived whereas the forced feedback is newly generated;
this is a paired comparison of these two feedback constructions, not repeated
sampling of all possible replies. It does not isolate argumentative strength,
style, length, and wording from one another.

## Frozen sample

- Source: `runs/main-mixed-60-leaning-20260926/live` (read-only).
- 150 cases across all 60 questions: 30 questions contribute two cases and 30
  contribute three. Exactly 50 cases per recipient model and 50 per delivery tone.
- Models: GPT-5.6 Terra, Qwen3.8-27B, and Inkling. D is measured for the 100 cases
  with an open-model recipient.
- Choose a source leaf, then a cut among its eligible feedback positions T=2–4.
  The recipient's fresh response is at T=3–5. Sampling uses route metadata only,
  never agreement labels or measured outcomes. Duplicate cut nodes and source
  leaves within a question/tone are excluded.
- Seed: 20260926. Realized feedback depths: 34 at T=2, 63 at T=3, 53 at T=4.
  These are not equal-sized depth strata.
- The first six presampled cases cover all six ordered model pairs and serve as
  an engineering check. They remain in the 150-case dataset.

## Private instruction to the feedback generator

> For this reply, take the opposing side to the central conclusion of the
> immediately preceding contribution. This assigned stance overrides
> agreement-seeking instructions, while retaining the assigned delivery tone.
> Clearly identify the conclusion you reject and defend a substantively opposing
> position with concrete reasons. Do not merely criticize wording or add minor
> qualifications. Do not invent facts or citations. Use the usual response format
> and length limit, and report the agreement label that best describes your actual
> reply.

We retain every technically valid response, including one that does not express
full disagreement. There are no retries or exclusions to obtain a desired label.

## Measurements

Use the existing four-label definitions unchanged: fully agree, leaning agree,
leaning disagree, fully disagree. Record the feedback generator's label separately
from the recipient's label. The main breakdown is by the **recipient's** label for
its new response to the feedback, with model and trajectory position also shown.

- **C options:** one fresh pre-feedback own-position read shared across the two
  arms, and one read after each new recipient reply. Record selected-option
  changes, including unjudgeable/missing readings separately.
- **C text:** Gemini 3.8 Flash compares the full pre/post position texts, blind to
  the arm, self-label and forced instruction. Report unchanged, adjusted,
  conclusion changed, and unjudgeable counts. No B re-rating or E synthesis.
- **D options:** capture the option probabilities from the same C reads for Qwen
  and Inkling; no extra generation for this measurement.
- **D own-text endorsement:** hold the full pre-feedback position fixed. Before
  generating the recipient's formal reply, separately read endorsement after
  the feedback and after length-matched procedural filler. Repeat in both arms.
  Use the same seven-category scale and original-top-20, missing-as-zero
  normalization as the main experiment, at read temperature 1.

All position/probability reads are side measurements: their outputs never enter
formal debate history. Debate keeps reasoning enabled; direct probability and
position reads use the established no-reasoning settings and validation.

Report probabilities/scale values with reference values, denominators, and clear
sign definitions. Less endorsement of one's prior text is not automatically
agreement with the peer. Labels are post-intervention outcomes: label-conditioned
comparisons are descriptive, not randomized causal subgroups. Treat questions,
not individual turns, as the independent unit for uncertainty; retain matched
natural/forced pairs and average within question before pooling.

Current reading-report policy: every C/D table must retain the recipient's
self-label, including T, participation-count and model breakdowns. Do not lead
with a pooled change rate or a T-only table. The earlier unconditional paired
contrasts remain unchanged in the JSON audit, but are no longer rendered in the
HTML. Do not replace them with purported paired causal effects within labels:
the two arms may assign the same case different labels. Show empty label strata
explicitly as unavailable, not a measured zero effect.

## Execution and preservation

There are 1,600 planned new logical requests: 150 forced feedback generations,
300 recipient replies, 450 C position reads, 300 C text judgments, and 400 D-text
reads. The 150 natural feedback messages are reused without new calls.

Reuse the main study's durable request journal and identical technical retries
(at most two additional attempts). An exhausted recognized local failure blocks
only its dependents; independent arms/cases continue. Preserve all attempts and
report partial cases. Unknown billing/provenance failures still stop dispatch.

Run the six-case gate at up to 16 in-flight requests, then resume the same output
directory for all 150 cases at up to 64 requests across eight active question
groups. All graph mutations belong to the coordinator, not provider threads.
Successful calls are replayed from the journal on resume and not regenerated.
No spending cap, per the user's standing instruction.

Separate output directory:

`/Users/Chen/Documents/research/llm-committee/runs/forced-feedback-150-20260926/live`

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m llm_committee.pivot.forced_feedback_run \
  --source runs/main-mixed-60-leaning-20260926/live \
  --output runs/forced-feedback-150-20260926/live \
  --live --cases 6 --max-inflight 16
```

After verifying the gate, use the identical command with `--cases 150
--max-inflight 64`. Frozen manifests, source hashes, code snapshot, raw requests,
and per-case records are kept in the separate output directory. Main-study
artifacts and manuscript files are not modified.

## Execution log

- Offline full run: 150/150 complete, 1,600 simulated calls. Offline resume
  issued zero new calls.
- Dedicated supplement/report tests plus scheduling, retries and probability
  regression tests: 64 passed. Tests cover private-instruction isolation,
  exclusion of original later/sibling messages, C/D reference contexts,
  local retry exhaustion, balanced sampling, paired aggregation and reuse.
- Real six-case gate: 6/6 complete, 64 calls, no failures, 50.42 seconds;
  token-based estimated charge $0.2883. All six ordered model pairs covered.
- Full run resumed at 2026-09-27 00:08 UTC (September 26 local), PID 67083,
  up to 64 in-flight calls. These launch identifiers are historical; always
  verify `live/progress.json` and the current process before restarting.
- Logs: `preflight-console.log` and `full-console.log` in the supplementary
  run's parent directory.
- Completed: 150/150 cases, 1,600 successful logical requests plus six failed
  format attempts, all recovered on identical retries. No exhausted retries,
  transport retries, blocked descendants or final failed cases.
- Full phase: 221.06 seconds; combined with the 50.42-second gate, approximately
  4.5 minutes of execution. Peak concurrency 62 (configured ceiling 64).
  Final token-based estimated charge: $7.548825317, including retries.
- Offline final verification reconstructed all 150 per-case records and all
  1,600 expected requests from the journal without a new API call or journal
  changes. All 850 successful direct position/probability reads had zero
  reasoning tokens. All 60 original main-study question records and its
  manifest retained their original hashes.
- Current report: `/Users/Chen/Documents/research/llm-committee/docs/main-mixed-leaning-results-2026-09-26.html#forced-feedback`.
  Per user request, all supplement tables and case texts are embedded in the same
  main-study HTML. The earlier standalone `forced-feedback-results-2026-09-26.html`
  is retained as an archived rendering, not the primary reading entry point.
  The main report's JSON companion stores this dataset separately under
  `supplements.forced_feedback`; main-study figures are unchanged.
  C option comparisons retain unjudgeable choices as such; complete execution
  does not imply every scientific measurement is judgeable.

Update the combined HTML offline:

```sh
.venv/bin/python -m llm_committee.pivot.leaning_results_report \
  runs/main-mixed-60-leaning-20260926/live \
  docs/main-mixed-leaning-results-2026-09-26.html \
  --feedback-run runs/forced-feedback-150-20260926/live --replace
```
