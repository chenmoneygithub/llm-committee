# Integrated-position rerun · September 27

User authorization: preserve the old report as reference and redo the latest
main setup using the newly approved integrated debate/position protocol.
This means the **20-question turn-tone dyadic mixed-family experiment**, not a
new 60-question/global-tone run, same-family roster, or three-member routing.

## Frozen setup

- Plan: `turn-tone-dyadic-shared-plan-2026-09-26.json`.
- Design SHA-256: `b2f3563d1be00b7f3ce39a6b6f84c5c6a2e1f04d2a8c0785463b5533916070f3`.
- A: GPT-5.6 Terra; B: Qwen3.8-27B; C: Inkling.
- AB, CA, BC; four replies per path; original/alternate local-tone assignments
  share newly generated T1/T2 and diverge in prescribed tones at T3/T4.
- 60 fresh initial answers, 360 fresh unique replies, 120 branch paths.
- Same presampled 267 B/C events; Gemini 3.8 Flash is the single judge.
- Four formal output fields: reply, agreement, choice, position. Reasoning medium.
- C reads the formal positions. D1/D2 are separate neutral no-reasoning reads,
  only for Qwen/Inkling. No entropy adjustment or supplemental scoring.
- 1,676 logical A–D calls: 60 initial, 360 debate, 267 B, 481 C,
  222 D1 and 286 D2 arm requests.
- At most 64 in-flight requests, eight rolling questions, two identical retries
  for eligible technical failures. No spending stopping limit, per prior request.

Run directory: `runs/stateful-dyadic-20-20260927/live`.
The full prompt implementation is snapshotted there before the first paid call.
No old model output is imported; the first question is reused within this same
new frozen run, without being regenerated during expansion to 20.

## Gate and current status

The first question (`archived-global-14`) completed all 81 requests without a
failure, including 18 formal replies and 24 D1/D2 requests. Recorded token-price
estimate: $0.260831512, not an invoice. Read-only reconstruction verified every
request and the complete question record. Independent native decoding verified
the probability token, native prompt, raw top-20 row, and disabled reasoning for
both open-weight models. All seven argument/control pairs differ only in the
incoming message. Gate result: `runs/stateful-dyadic-20-20260927/gate-audit.json`.

The full run is complete: 20/20 questions, 120/120 paths, and all 1,676 logical
requests succeeded. Nine additional technical attempts recovered nine requests;
none were exhausted. Full-run process elapsed time was 207.62 seconds, excluding
the separate gate and audit. Recorded token-cost estimate including retries:
**$5.383296584**, not a provider invoice. Peak concurrency reached 64 requests
across eight active questions. Logs: `gate.log`, `full.log`.

The full read-only audit reconstructed all 1,676 requests and all 20 question
records exactly. All 508 D requests passed independent native-prompt/output-token/
top-20 checks, with reasoning disabled in the accepted readings. All 143 D2 pairs
differ only in the incoming argument/filler message. Full audit:
`runs/stateful-dyadic-20-20260927/full-audit.json`.

## E follow-up

The existing length-controlled E protocol was rerun using these
new initial positions and public replies. The only source-adapter change is that
the original answer text now comes from `initial_positions[member].position`.
No private updated position, self-label or tone metadata is added to synthesis.
Terra chairman, Gemini judge, 190–210 whitespace-separated words for both arms,
both presentation orders, 120 comparisons, and 12 presampled calibration cases
with verbosity/moderate/severe damage probes remain unchanged (540 logical calls).
The old E results were not appended to new A–D.

E completed all 540 logical requests, all 120 primary comparisons and all 12
calibration cases. Fourteen additional length/format attempts recovered 11
requests; none were exhausted. Runtime: 90.14 seconds. Token-cost estimate:
**$4.77760657**. Combined A–E estimate: **$10.160903154**. These are calculated
from returned token usage/pricing, not billed charges or pending reservations.

E directory: `runs/stateful-dyadic-quality-20-20260927/live`. Its
`../full-audit.json` reconstructs all 540 requests, verifies both orders for all
120 primary comparisons, and confirms every accepted primary synthesis has
190–210 words. All source question files and the source manifest stayed unchanged.

## Preserved reference report

Copied byte-for-byte, including the old E section:

- `turn-tone-dyadic-independent-position-reference-2026-09-27.html`
  SHA-256 `2181c3567925c9c0b27e426915350b0b386bd76d960104c3555b31695f033f8e`.
- `turn-tone-dyadic-independent-position-reference-2026-09-27.json`
  SHA-256 `a366674cb37d2c7b7342622eedb2e4fc037120b0317fcba5f055f1a0d734ea9b`.

The original old report paths and run journals also remain untouched. Reference
copies are in the same directory to preserve existing relative links.

## New report and auditing

New report target: `docs/turn-tone-dyadic-stateful-2026-09-27.html`, with JSON
sidecar. Both are now generated, including fresh E in the same HTML.
B conditions on the receiver's self-label; C/D retain both the peer's
preceding label and the receiver's current label. No globally pooled C/D headline.
C uses formal choices/full text, not separate position generations. D2 reports
modal-category change, fixed control-modal probability change, and expected
rating difference, with definitions and full prompt examples. Shared neutral
T3 reads count once under the original continuation's observed labels.

Read-only audit: `python -m scripts.audit_stateful_dyadic RUN --output AUDIT.json`.
Offline renderer: `python -m scripts.report_stateful_dyadic RUN OUTPUT.html`;
append `--quality-source E_RUN` only for the matching fresh E cohort.

Verification: full regression suite **454 passed, 5 skipped**; the five native
tokenizer/renderer tests separately passed offline. After adding one extra
stateful-to-E embedding test, all three report tests passed. Ruff and whitespace
checks passed. The preserved reference HTML/JSON hashes still match. No manuscript
edits, archived-output conversion, commit or push.

## Offline D1 distribution and qualitative diagnostic

The subsequent review added D1 distributions to the same A–E HTML and a linked
[ten-case diagnostic](stateful-d1-distribution-cases-2026-09-27.html). No model
was called and no experiment data, prompt protocol, or paid journal was changed.
The plots retain both agreement labels and separate models. Each point is one
unique sampled comparison; the outline diamond retains the table's question-equal
mean. The ±1pp band is only a display convention, not a test of equivalence.

The offline diagnostic independently recomputes all **141 usable D1 comparisons**
from their saved candidate log probabilities and validates **28 verbatim quotations**
across ten deliberately selected cases. The two events with no formal reference
choice remain unavailable rather than being assigned zero. Shared T3 neutral reads
are still counted once under original-continuation labels, as in the main report.

What the close reading supports:

- Large initial increases can compare a bare-question, no-reasoning distribution
  that differs from the formal reasoning-enabled initial choice against a later
  prompt explicitly supplied with that choice and position. These do not isolate
  the effect of the peer's argument.
- Later-turn D1 increases also coexist with concessions and an unchanged formal
  option. One Qwen event switches formal choice B→C even while neutral D1 for old
  B rises 6.64pp; a positive D1 difference is not synonymous with strengthening
  the actual formal choice.
- An incoming leaning-disagree reply can explicitly share the same survey option
  and instead dispute its justification. Labels therefore do not guarantee opposed
  option choices.
- A saved paired continuation has the same two labels but D1 changes +36.95pp and
  −2.90pp. The more qualified formal position is visible in the latter. This is
  illustrative, not an estimate of a single tone's causal effect.

Cases were selected after seeing the outcomes for diagnostic contrasts, not to
estimate how often these patterns occur. Interpretations are assistant-authored;
existing Gemini judgments are shown separately, and quoted model assertions are
not fact-checked claims about the world. D1's input-comparability problem is
documented, **not fixed or supplemented with new probability reads**.

Rebuild the diagnostic with:

```sh
.venv/bin/python -m scripts.stateful_probability_diagnostics \
  runs/stateful-dyadic-20-20260927/live \
  docs/stateful-d1-case-notes-2026-09-27.json \
  docs/stateful-d1-distribution-cases-2026-09-27.html
.venv/bin/python -m scripts.report_stateful_dyadic \
  runs/stateful-dyadic-20-20260927/live \
  docs/turn-tone-dyadic-stateful-2026-09-27.html \
  --quality-source runs/stateful-dyadic-quality-20-20260927/live \
  --diagnostic-source docs/stateful-d1-distribution-cases-2026-09-27.html
```

Focused verification: all **7 diagnostic/main-report tests passed**. Old reference
reports, all paid request records and both original protocol snapshots are preserved.
