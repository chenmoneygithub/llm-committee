# Integrated positions in formal debate

Protocol: `stateful-dyadic-paired-2026-09-27-v1`.
Implementation and offline verification are complete. The user subsequently
authorized the paid mixed-family rerun; see
[execution and archive record](stateful-dyadic-rerun-2026-09-27.md).

## What changed

Each initial generation returns `choice` and `position`. Each formal debate turn
uses reasoning and returns exactly four fields in the same generation:

| Field | Meaning |
| --- | --- |
| `reply` | Argument replying specifically to the immediate incoming peer message |
| `agreement` | Strongly/leaning agree/disagree with that message, not with the survey question |
| `choice` | Current selection from the original question's options; null if none fits |
| `position` | Complete current view on the original question, with reasons; unchanged views are allowed |

The receiver sees the original question/options, the public history of this
branch, its own latest recorded choice/full position on this branch, and the
immediate incoming message. The incoming message occurs once, separately from
its ancestors. The receiver's initial position is not duplicated as a second
historical contribution when it is already the supplied current position.
Full positions remain paragraphs, not separate claim/reason fields.

Only `reply` is broadcast as a peer's turn in later histories. The receiver's
own structured position is supplied as its state; other members' structured
positions/choices, historical agreement labels, tone instructions and private
reasoning are not broadcast. Only the current formal turn's tone instruction is
applied. Current-tone/output instructions are a trailing task message; the common
identity/context prefix is stable. This is a new prompt protocol, not a relabeling
of archived generations.

State is resolved from the nearest same-member ancestor, **not** from a global
latest response or model-name dictionary. Equal-model members still have distinct
member IDs. Original/alternate paths share T1/T2 once and then inherit independent
state. An unsampled formal turn still produces and propagates its position.

## C: analysis, not another member generation

C compares choices deterministically and asks the external judge to compare the
two formal position texts. The judge sees question/options plus before/after
texts, not the transcript or self-labels. Adjacent comparisons use the same
member's previous participation; initial-to-endpoint comparisons remain separate.
There are no standalone `position` requests. State records carry the originating
formal request key so the measurements can be traced to the exact output.

## D: independent, sampled side reads

D runs **only in mixed_family**, only for its Qwen/Inkling members. It does not run
in same_model or same_family. B/C/D retain the same presampled event plan.

- **D1** reuses the discussion input that preceded the corresponding formal
  generation and asks directly for one original option letter with reasoning
  disabled. It does not see that generation's newly emitted choice, position or
  reply. An initial D1 read sees only the original question/member identity, not
  the generated initial answer. Readouts for earlier participation/initial
  references are reused by ID rather than sampled again for each comparison.
- **D2** uses the same pre-turn history and formal prior position in both arms.
  Only the incoming argument is replaced by approximately token-matched filler
  in the control. Both arms ask for one A–G endorsement rating of that **fixed
  prior text**, not of the newly updated position.

Both probes retain the earlier neutral, no-reasoning measurement policy. Thus
"same input" means the same question, discussion material and pre-turn state;
the formal current-tone/reply task is replaced by a neutral direct-rating task.
D1 is **not** the probability attached to the reasoning-enabled formal answer,
and neither probe measures an internal belief or updates any debate state.
Original top-20 provenance and the existing zero-fill/renormalization policy are
retained; no new scoring calls or entropy adjustment are introduced.

D2 records all seven probabilities plus three complementary measures:

1. Modal rating transition and signed category difference (A=1 through G=7).
2. Argument-minus-control probability change, in percentage points, of the
   **fixed control-modal category**.
3. Expected rating in each arm and their signed difference.

The modal category is not the stochastically emitted letter. All tied modes are
saved; an ambiguous modal transition or control reference is reported as null,
not broken arbitrarily. These are event records, not a new global aggregation.
Keep the peer/current-label conditioning for later reports. Neutral pre-turn D1
and D2 reads at T3 are shared across the paired continuations (T3's formal tone
has not yet acted on any public history). Shared request IDs and flags must be
used to avoid counting those reads twice.

## Cache-friendly ordering without scientific dependencies

Within a sampled event, wait for D1 to finish, then run D2 argument, then D2
filler. Other samples/questions stay parallel. These are **soft scheduling
dependencies**: terminal failure also releases the next read. A probability
failure cannot stop formal debate, poison a position, or prevent an independently
valid control request. Actual argument/control data dependencies are unchanged.

The D1 and D2-argument prompts share the entire discussion prefix; the two D2 arms
share everything before the incoming message. No extra warm-up requests are sent.
Native reasoning templates are not modified to force a cache hit. Tinker controls
cache placement and retention; improved hits are not guaranteed. The new run
report summarizes actual provider-reported input/cached/output tokens by model
and purpose, including received retry attempts.

## Execution and provenance

The new entry point supports the existing three rosters over the frozen 20-question
paired dyadic plan: AB, CA and BC, four turns, turn-local tones, original/alternate
continuations. It does not silently start a new three-member routing experiment.
The prompt/context helpers also support branched ancestry, tested independently.

Use a **fresh output directory**. Existing journals, completed HTML reports,
quality follow-ups and archived protocols are not overwritten or converted.
The existing strong-label HTML loader intentionally rejects this new protocol;
its old C/D descriptions must not be used for new data without an explicit report
update. E is not launched by this entry point.

Offline engineering check (no endpoints or credentials):

```sh
.venv/bin/python -m llm_committee.pivot.stateful_run \
  --plan docs/turn-tone-dyadic-shared-plan-2026-09-26.json \
  --output /tmp/llm-committee-stateful-offline-20260927 \
  --mock --questions 2 --roster mixed_family
```

`--live` is required explicitly for paid calls. The authorized one-question live
gate has now verified four-field compliance and native D1/D2 probability alignment;
the separately recorded full rerun uses the same frozen implementation and plan.
Offline tests alone cannot establish model behavior or cache hits.

Reports include `positions[assignment/pair][member_id]`, a chronological list of
initial and participating-turn records, each with `T`, `participation_index`,
`node_id` and `source_request`. Lists are branch/member-specific; failures may
leave missing suffixes, so use the explicit indices rather than assuming dense
global-turn arrays. Shared initial/prefix records retain the same source IDs.

## Verification on September 27

- Full repository suite: **451 passed**; the 5 opt-in native-renderer tests are
  skipped in the ordinary environment and were separately run with cached native
  tokenizers: **5 passed**, with no real sampler/session.
- Full 20-question mixed-family **offline mock**: 1,676 completed calls, 120
  successful branch paths, no failures. All generations are synthetic fixtures;
  its token-cost figures are not actual spending or a live budget estimate.
- Same-model and same-family execution, failed D1/D2 reads, isolated formal-turn
  failures, exact checkpoint resume, modal-rating ties, and missing/extra JSON
  fields are covered by tests.
- Ruff and whitespace checks pass. No paid experiment, manuscript edit, old-data
  conversion, commit or push was performed.
