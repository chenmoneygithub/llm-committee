# Matched tone continuations after T2

Authorized scope: keep the completed 20-question dyadic pilot and add one
alternate continuation for each AB/CA/BC path. Do not run three-member debates.

## Design frozen before new outputs

- Same 20 questions, models, original options, independent initial answers,
  and four-reply routing as the completed turn-level-tone pilot.
- Reuse the exact T1/T2 generated text, private tone assignments, neutral
  position readings and other identical requests. Do not regenerate the prefix.
- Assign a different private tone to each alternate T3 and T4, before new
  outputs. Seed 20260928; randomized constrained matching uses only original
  tone assignments. Preserve original tone counts within each pair × T block:
  40 friendly, 40 neutral, 40 hostile among the 120 new replies. Neither tone
  switches nor their directions are selected using observed labels or effects.
- T3 compares the same incoming text/history under different current private
  tone instructions. T4 compares downstream continuations whose T3 messages
  and current tone instructions differ; it does not isolate T4's immediate effect.
- Generated public text remains in its own branch. Historical private tones,
  self-label metadata and measurement outputs are never fed into the discussion.
- Debate reasoning remains enabled; C/D probes remain neutral with reasoning
  disabled. Prompt text, model parameters and the four-label rubric are unchanged.
- No new repetitions or outcome-based sample selection. Single continuation
  pairs contain generation variability; a case difference is not a certainty
  about that question's causal effect.

## Measurement and accounting

Use the original B/C/D event sample. The 53 sampled T2 events are shared, not
new observations. Add 107 sampled T3/T4 events with their original counterparts.
Both preceding peer and current receiver labels, T, assignment, and D receiver
model remain visible. Label strata are descriptive, post-generation groups;
subtracting different strata across assignments is not a paired treatment test.

D-text at T3 precedes the changed receiver reply and has exactly the same neutral
input under both assignments. Reuse it: equality is structural, not evidence of
no tone effect. At T4, the incoming T3 message changes, so both argument and
length-matched filler probes are rebuilt. D-choice after T3/T4 follows the new C
position reading. Preserve initial and previous-own-participation references.

- 60 paired paths: 120 shared replies + 120 original suffix replies + 120 new
  suffix replies = **360 unique formal replies**, not 480.
- New calls: 120 debate + 107 position + 107 B judgments + 214 C judgments +
  72 D-text = **620** logical requests. Reuse **541** original successful requests.
- Shared prefixes and identical neutral probes are not counted twice. There
  are still 20 independent questions, not 120 independent trajectories.
- Bounded retries and the global 64-request limit are unchanged. A recoverable
  exhausted failure affects its dependency chain, not all questions. No imposed
  spending cap. Token estimates and unknown-billing reservations stay separate.

## Preservation and reporting

Original raw run: `runs/turn-tone-dyadic-20-20260926/live` (read-only).
New run: `runs/turn-tone-fork-20-20260926/live`.
New execution modules are separate; the archived generation code stays unchanged.

Append the paired-continuation results and full side-by-side transcripts to:
`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-dyadic-pilot-2026-09-26.html`

Keep the original-assignment tables intact and identify the new comparison
separately. Do not mix old global-tone studies into this report; those remain in
the fixed-tone supplementary HTML. No manuscript edits or commits in this task.

Status: completed. All 60 alternate suffixes succeeded, with 120 new formal
replies and 107 paired sampled events. The main pilot HTML now includes the
paired continuation and all 60 side-by-side cases. No three-member run started.

## Confirmed reuse across the remaining two model configurations

The user confirmed retaining this trajectory design for the same-model and
same-family configurations. The machine-readable frozen template is
`turn-tone-dyadic-shared-plan-2026-09-26.json`. It contains the actual 20 questions
and original options, AB/CA/BC logical routes, both per-question tone schedules,
and the exact original B/C/D sampling and position-reference locations. Do not
draw new routes, tone assignments or sample locations for each configuration.

The existing project mappings remain Terra/Terra/Terra for same-model, and
Luna/Terra/Sol for same-family, with distinct logical A/B/C member identities.
This preserves previous model choices; it does not claim fresh endpoint
availability checks. Neither remaining configuration was launched in this turn.

Reuse **design**, not outputs, across configurations. Each new configuration
generates its own initial answers and shared T1/T2 prefix, then forks its own
two T3/T4 continuations. No mixed-family text, position readings, judgments or
probabilities are imported into those configurations, including the Terra
seat. Within one configuration only, identical shared-prefix and neutral
pre-reply measurements are reused. Each full new configuration therefore has
360 unique replies, not merely the 120 added in the mixed-family continuation.

B/C retain the common sample. D remains limited to eligible open-weight members
in the mixed-family configuration; it is not silently expanded to the other
two configurations. This decision does not start a three-member study, enlarge
the 20-question subset or add final-answer synthesis. Keep model configurations
separate in reporting and preserve both labels, T, assignment and D model.

## Execution log

- Full offline simulation: 20 questions, 60 alternate suffixes, all 620 new
  requests completed; no shared-prefix request was dispatched again.
- Five focused tests passed: deterministic balanced alternate assignments,
  same-prefix/different-T3-tone validation, exact reuse, resume without duplicate
  dispatch, tampered-prefix rejection and failure isolation.
- Real gate: 30 successful new requests (6 debate), 26 source requests reused;
  three successful alternate paths, no retries/failures. Process time 36.62 s;
  token-estimated cost US$0.091255867.
- Read-only gate audit reconstructed all 30 requests and the question record,
  verified all 26 reused requests against their original journal, and checked
  current-tone-only prompts, neutral probes and zero reasoning in nine direct
  reads. All 20 source question hashes and frozen execution hashes match.
- Before report extension, main HTML checksum:
  `b77ffe83788dc0b66616c6e2c28ab5f9897584d97eac24a9e41944e35b3b6f93`;
  main JSON `662ea88a0b3e7ff81fb262865d82e49be831dc902c852395ca1c25d799ec86e2`.
  Fixed-tone supplementary HTML checksum:
  `5708f69605c780a40d72378bd15cf7b82dea037023485e9cc053f3fb89d5ab15`.
- Full execution resumed the same journal without repeating gate requests.
  Resumed-process duration: 71.27 seconds. All 620 logical requests succeeded;
  one invalid-format attempt was recovered by one retry. No failed trajectories,
  missing measurements, unresolved billing attempts or exhausted retries.
- Incremental token-estimated cost: **US$1.937376827**, including the invalid
  attempt. No retained reservations in this new run. This is not an invoice;
  it excludes the original pilot's cost and its unresolved-billing reservation.
- Final read-only audit exactly reconstructed 620 new requests and all 20
  question reports, checked all 541 reused requests against the original
  journal, and verified that each of the 60 T3 request pairs differs only in
  its current private tone instruction (plus its bookkeeping key). All 179
  direct readings had zero reasoning tokens; all 143 probability readings
  used the original top-k data at read temperature 1. All source hashes match.
- Full regression: 373 passed, 3 optional native-renderer tests skipped.
  Focused execution/report/recovery suite: 76 passed. The audit initially
  compared in-memory tuples to JSON lists; canonicalizing both sides fixed
  this audit-only representation mismatch. No prompts, data or execution code
  changed, and no experiment requests were rerun for that fix.
- HTML verification: exactly one document and paired section, 60 matched cases,
  120 two-column turn panels, 107 sampled event pairs, valid tables and links.
  All pre-existing original-assignment tables and JSON fields are unchanged.
  The fixed-tone supplementary and old canonical HTML/JSON are byte-unchanged.
  Original raw records and frozen generation modules remain untouched.
