# Two-member exclusive debate supplement

User-approved design: 60 existing screened questions, three tones, directed AB/CA/BC
trajectories, four formal replies each. A = GPT-5.6 Terra, B = Qwen3.8-27B,
C = Inkling. AB means A's independent initial answer is sent to B, then
B/A/B/A generate the four replies. CA and BC rotate the roles. Each model
starts one trajectory and receives first in another; each speaks twice per dyad.

This produces **540 trajectories and 2,160 new debate replies**, not including
measurement calls. The source main study remains immutable and separate.
The initial independent answers are reused from the completed leaning-label run;
each trajectory sees only its two members. No third-member answer, other tone,
other dyad, measurement output or private self-label enters its discussion context.
Prompts, model versions, tone wording and leaning-agreement rubric stay unchanged.
No Layer E synthesis or quality comparison is added by this supplement.

## Measurements and sampling

Layer A retains all four turns. For B/C/D, preselect eight events per question
(480 total) without reading outputs. There are nine directed-pair × tone strata;
omit one per question, balanced over the question bank, and sample one position
from each remaining stratum. Eligible positions are T=2/3/4 with weights 1/1/1.5.
T=1 receives an initial answer, which has no peer-agreement label; do not invent
one or treat it as agreement. Tone is retained as metadata, not a primary B/C/D split.

Both C/D conditioning variables are retained: the immediately preceding peer
reply's self-label, which in this alternating dyad refers to the receiver's latest
public contribution; and the receiver's current self-label referring to that peer
reply. These are descriptive response-label strata, not randomized treatments.
Agreement with a public contribution is not proof of agreement with every part
of an independently read full position. Null labels remain explicitly not reported.

C compares the receiver's same-trajectory previous own participation (or initial
position) with the current one, via original-option choice and Gemini 3.8 Flash's
full-text judgment. Sampled final own participations at T=3/4 also get a separate
initial-to-endpoint text comparison; its endpoint labels do not explain the whole
history. B uses that same single judge, blind to self-labels/tone. D is limited to
Qwen/Inkling: C's option distributions plus fixed prior full-position endorsement
after incoming peer text versus length-matched procedural filler. D-text excludes
the receiver's new reply; C readouts do not enter debate. Debate reasoning stays
on; C/D direct reads use none, original top-20, absent candidates approximated
as zero, normalization at T=1. No top-two margin or entropy adjustment.

## Execution and reporting

Independent question/tone/dyad chains run concurrently; each chain respects
dependencies. Rolling eight-question pool, global 64-request limit. Existing
format/transport recovery policy: at most two additional authorized attempts,
immutable attempt logging, first valid response retained, no outcome-based retry.
Exhausted failures isolate dependent tasks; report trajectory and measurement
failures separately. No newly imposed spending cap, consistent with authorization.

Run offline tests, then one real question as an engineering gate (not selected by
observed effects). If successful, resume the same frozen manifest/journal for all
60 questions. Keep a source-code snapshot, source hashes, sampling plan and all
raw outputs. Report separately inside the canonical combined HTML; never pool
two- and three-member results. C/D tables retain both labels and T, plus model for D.
Counts are sampled-event counts, not 480 independent questions. No new human
validation or committee-size causal claim is implied.

Canonical HTML:
`/Users/Chen/Documents/research/llm-committee/docs/main-mixed-leaning-results-2026-09-26.html`

Source:
`runs/main-mixed-60-leaning-20260926/live`

Planned output:
`runs/dyadic-60-20260926/live`

## Execution log

- Full offline run: 60 questions, 540 successful trajectories, 5,130 simulated
  requests (2,160 debate, 1,016 position, 480 B judgments, 838 C judgments,
  636 D-text reads). These simulated outputs are not research data.
- Dedicated execution, scheduling, recovery, probability and agreement tests:
  88 passed. Reporting tests: 53 passed including the combined-page guards.
  Additional core/provider/probability regressions: 194 passed, three optional
  native-renderer tests skipped in the project environment.
- Real one-question engineering gate: completed 9/9 trajectories and 89/89
  requests, no retries or final failures, 75.41 seconds, estimated charge
  US$0.392215049. All 30 direct reads generated zero reasoning tokens; 25 have
  open-model probability readouts. The gate is included in the full run.

Execution command (use `--questions 1 --max-inflight 16` for the initial gate):

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m llm_committee.pivot.dyadic_run \
  --source runs/main-mixed-60-leaning-20260926/live \
  --output runs/dyadic-60-20260926/live \
  --live --questions 60 --max-inflight 64
```

**Completed 2026-09-27 01:20:32 UTC (September 26, 18:20 Pacific).** All 60
questions, 540 trajectories and 5,130 logical requests succeeded. Twenty-two
requests recovered from formatting failures using 23 additional attempts;
no exhausted failures or transport retries. Peak concurrency 64. Full phase
441.58 seconds plus the 75.41-second gate (about 8 minutes 37 seconds total,
excluding offline engineering work). Token-based total estimate, including
the gate and retries: **US$22.513522776**, not a provider invoice.

All 60 question reports and 5,130 exact request contexts were reconstructed
offline. All 1,652 successful direct reads had zero reasoning tokens;
1,307 had original top-20 probability readouts. The 60 original main-study
question files and its manifest were verified unchanged. Audit results:
`runs/dyadic-60-20260926/audit.json`.

Final combined regression run: **335 passed, 3 optional native-renderer tests
skipped**. The canonical HTML now contains 11 dyadic tables and all 480 sampled
event texts, separately from both earlier cohorts. Every C/D table retains both
label dimensions; T and receiver model are retained as applicable. The prior
main-study JSON data and forced-feedback supplementary JSON data have identical
hashes before and after this addition.

Regenerate the single combined HTML offline:

```sh
.venv/bin/python -m llm_committee.pivot.leaning_results_report \
  runs/main-mixed-60-leaning-20260926/live \
  docs/main-mixed-leaning-results-2026-09-26.html \
  --feedback-run runs/forced-feedback-150-20260926/live \
  --dyadic-run runs/dyadic-60-20260926/live --replace
```

Repeat the read-only native-tokenizer audit with:

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python \
  -m scripts.audit_dyadic_run runs/dyadic-60-20260926/live
```
