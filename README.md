# LLM Committee

## Repository contents and local artifacts

This repository includes the experiment code, tests, protocol documents, frozen
question/routing plans, and lightweight summaries. Raw run archives (`runs/`)
and large generated HTML/JSON reports are excluded from this code repository;
their private paper-repository archive is linked below. Caches and credentials
are never included. Links to local reports below work only after the artifacts
are restored. Report-generation scripts are included; some artifact-dependent
integration tests also require the original local archives.

## OpenRouter

Set `OPENROUTER_API_KEY` on your own machine, then run the GPT experiment with:

```sh
python -m scripts.run_openrouter --members 2 --roster same_family --questions 10 --live --output runs/my-openrouter-run
```

Use `--members 3` for three-member routing or `--roster same_model` for three
Terra slots. This entry point uses one frozen turn-level tone assignment and
OpenRouter for every member/judge request. Omit `--live` to inspect the plan
without calling any endpoint. There are no account-balance checks or separate
billing queries; ordinary responses and retry/resume records are still saved.
Historical entry points retain their original provider defaults, so use this
OpenRouter entry point for new GPT runs. Mixed-family native probability reads
are not supported through this adapter; there is no automatic Tinker fallback.

Research data and the complete HTML reports are preserved in the private
[paper repository](https://github.com/chenmoneygithub/llm-committee-paper/tree/main/artifacts/2026-09-redesign).

## September 2026 redesign

**Current public-history study (September 28):** all six model/routing settings
now cover 50 questions each: the unchanged original 20 plus the same 30 additional
screened questions. Existing generations and failures are preserved. Formal turns produce
`reply`, `agreement`, `choice`, and `position`, but later turns see only the
initial answers and public replies—not updated private position fields. D remains
exclusive to mixed-family dyads; three-member runs collect A/B/E. See the
[extension plan and execution record](docs/public-history-scale50-2026-09-28.md)
and the [six-setting dashboard](docs/committee-experiment-dashboard.html).
The dashboard contains all six 50-question reports. All formal debate paths completed;
dyadic E has 300/300, 294/300 and 292/300 available comparisons for mixed-family,
same-family and same-model respectively. Missing comparisons remain missing, not ties.
The [20-question dashboard](docs/committee-experiment-dashboard-20-question-reference.html)
is retained for reference. A same-model triadic output-limit failure was recovered
without regenerating any of its 1,220 successful requests; all nine extension audits passed.

**Forced-disagreement supplement (September 28):** all 300 current-protocol
mixed-family dyadic interventions completed: 150 at T1→T2 and 150 at T3→T4,
600 new formal replies, no exhausted requests. Natural comparisons use the same
endpoint; C/D retain both speakers' actual label combinations. All 650 new
probability reads passed native-token checks. Received-response cost estimate:
$18.79, including retries; no unresolved reservations. The supplement and all
paired examples are in the existing Mixed family / 2 members dashboard tab,
without changing main-study scores. See the
[protocol, counts and findings](docs/forced-feedback-public-history-plan-2026-09-28.md)
or the [updated report](docs/turn-tone-dyadic-mixed-family-50-public-history.html#forced-disagreement-v1).

**Independent E2 check:** Grok 4.6 additionally judged the mixed-family three-member
answers only; its original-rubric 79%/80% debate preference rates appear beside
Gemini's 78%/79% in the existing table. A completed judge-only rubric revision
reuses those same 100 pairs and both answer orders: pooling assignments within
50 questions, Gemini's debate wins remain 78.5%, and Grok's change from 79.5% to
81.0%. Agreement with the first human review is 13/20 → 12/20 for Gemini and
12/20 → 12/20 for Grok, comparing the same presentation order. This is exploratory,
not held-out validation or evidence that wording sensitivity is fixed. No other
configuration was rejudged. See the
[old/new HTML comparison](docs/E2-judge-rubric-comparison-2026-09-28.html) and
[run record](docs/grok-and-human-E2-review-2026-09-28.md).
The [blind human-review tool](docs/human-answer-review-20.html) samples 20 distinct
questions, hides answer origins, saves locally, and exports annotations as JSON.
It now includes an EN / 中文 switch for the questions, options, answers, and UI.
Chinese is a machine-translated reading aid; both languages share the same
annotations, and exports record the language used when each rating was selected.

**Earlier integrated-position implementation (September 27):** the new
[integrated-position protocol](docs/stateful-debate-protocol-2026-09-27.md) generates
`reply`, `agreement`, `choice`, and `position` in each formal reasoning-enabled
debate turn. C analyzes those recorded positions without re-asking the member;
D uses independent no-reasoning probes only for the mixed-family setup. Use
`python -m llm_committee.pivot.stateful_run` for this version. The paid 20-question
mixed-family rerun is complete: all 120 paths and 120 E comparisons succeeded,
with all 508 probability reads independently checked at the native token level.
[Open the new A–E report](docs/turn-tone-dyadic-stateful-2026-09-27.html) or see the
[run record](docs/stateful-dyadic-rerun-2026-09-27.md). Estimated token cost: $10.16.
The [previous report](docs/turn-tone-dyadic-independent-position-reference-2026-09-27.html)
is preserved byte-for-byte as reference only. The runs and reports below retain their original protocols.

**New-label rerun (September 26):** the user authorized a fresh 60-question mixed-committee
run using **fully agree / leaning agree / leaning disagree / fully disagree**, with a shared,
explicit A/B rubric. The earlier partial-agreement run and its HTML are reference only and
remain unchanged. See the [rerun protocol and current status](docs/leaning-rerun-2026-09-26.md).
No old results are relabeled or pooled into the new run.
The rerun completed in **10 minutes 15 seconds**: all 60 questions and 1,080 paths succeeded,
with estimated token-based cost **US$38.46** including retries.
[Open the new leaning-label results report](docs/main-mixed-leaning-results-2026-09-26.html)
for A–E tables, explanations and the four label definitions. The earlier report below is reference only.

The new experiment uses a preplanned, branch-isolated debate tree and separate A–E
measurements. On September 26 the **60-question mixed-family main run** (Terra / Qwen3.8-27B /
Inkling; three tones) was launched with three question workers and no budget stopping limit,
as requested. Costs and failures remain recorded. See the [run guide and progress paths](docs/main-mixed-run-2026-09-26.md),
[pilot guide](docs/pivot-pilot.md), and [research protocol](docs/pivot-protocol-2026-09-25.md).
After a redundant JSON field stopped the run, the saved response was recovered offline
with no regeneration; identical duplicate fields are now tolerated, while conflicting values still fail.
The user-approved upgrade, `--workers 8 --max-inflight-requests 64`, schedules independent
branches, tones and measurements through one global request pool, refilling question slots.
`--retry-format-failures` permits at most two additional, identical requests for malformed
responses and retains the first format-valid response. Exhaustion marks the task missing and
blocks only actual dependents; it does not stop unrelated branches or questions. Measurement
failures do not delete discussion. Every attempt, raw response, charge and failure is preserved.
Known Databricks connection/timeouts now receive the same maximum of two additional identical
attempts, with short backoff; they are marked missing only after exhaustion. Original failure
records and unresolved cost reservations are retained. Unknown provider/billing/integrity errors
still stop for inspection. The earlier direct-skip behavior was an overly conservative policy,
corrected at the user's request; it is not the current behavior.
The final continuation is `runs/main-mixed-60-20260926/live-transport-retry/`;
it completed at **21:24:01 UTC on September 26: 60/60 questions fully complete,
1,080/1,080 debate paths successful, zero remaining technical failures**.
The [offline HTML results report](docs/main-mixed-results-2026-09-26.html) includes A–E
tables, narrative interpretation, explicit denominators and question-level descriptive intervals.
Its [JSON companion](docs/main-mixed-results-2026-09-26.json) retains unrounded aggregates and source hashes.
Databricks workers share serialized OAuth refresh
while keeping model requests parallel; this avoids independent concurrent token initialization.
The preceding checkpoint finished at 21:03:54 UTC on September 26: **60 questions processed, 53 fully complete,
7 with recorded missing data; 1,077/1,080 debate trajectories successful, 3 failed**.
Its status was `completed_with_failures`. The final continuation recovered its 17 connection-failed
requests on the first additional attempt and completed 15 blocked dependents, reusing all 6,864
valid scientific outputs. No successful response was regenerated. Seven C outputs explicitly
decline to select an original option; these are semantic outcomes, not unresolved API failures,
and option-change tables exclude them only from comparisons requiring a selected option.
The user also authorized identical, bounded retries when Inkling unexpectedly reasons in an
effort-`none` C/D read. Noncompliant outputs remain invalid; successful readings are not rerun.
The [Terra diagnostic](docs/terra-position-read-investigation-2026-09-26.md) succeeded and its
saved retry can now be imported without another generation. The original records stay untouched.
See the run guide for live status and `trajectory-outcomes.json`: planned/success/failed/blocked/
pending paths, with separate measurement support and retry counts. Shared-prefix failures may
affect multiple trajectories; technical attempts are not additional research observations.

`python -m llm_committee.pivot --help` provides offline question import and mixed-roster
planning. Pilot questions are retained in the approved [60-question main-study bank](docs/phase-1-question-bank.md); only real records collected under the
final protocol can be reused. Synthetic outputs are never study data.

Mixed and same-family synthetic dry runs are implemented, including D-choice/D-text for
the mixed roster. Native Tinker adapters and exact-prefix candidate scoring have offline
tests. The three-question live pilot finished all nine debates. On September 26 the user
changed D to use original top-20 probabilities, assign absent candidates zero, and normalize
without supplemental scoring. Offline reanalysis calculates all 27 D-choice reads and 17
D-text pairs under this explicit truncation approximation; the old failed scores remain
archived, not repaired. This made no API calls and left A/B/C/E unchanged
([policy and verification](docs/pivot-topk-zero-fill-2026-09-26.md)). Total estimated cost
including pilot debugging was $1.56 before the main launch. `study-plan` prices the
actual main-study calls. Live execution requires approved inputs, optional provider
dependencies and an explicit spending policy (a cap or `--no-budget-limit`). The additional A instruction ablation is not
yet implemented. The new engine does not call the legacy multi-thread committee.

Closed models use the repository's existing Databricks OAuth profile `un` by default;
Qwen/Inkling use Tinker. Direct OpenAI/Gemini API keys are only needed if explicitly
selecting `--closed-provider direct`. See the pilot guide for the approved command and progress.

## Published-study toolkit (legacy)

The documentation below describes the earlier engine and layer definitions, retained for
compatibility and historical reproduction. In the new protocol, C measures position change;
instruction removal belongs to A, not C.

Multi-agent LLM debate with **measurable disagreement**. A committee of models debates a question
under a configurable tone, the full debate trajectory is captured, and four evaluator layers let
you check whether the disagreement is *real* — instead of trusting any single signal.

Built for the paper *"What Does Multi-Agent LLM Debate Actually Change? A Layered Analysis of
Disagreement and Answer Quality"* (arXiv: https://arxiv.org/abs/2609.08016), where these four
layers measurably come apart: debate readily changes what agents say, but the evidence that it
changes what they persistently endorse — or improves the final answer — is much weaker.

## The flow

```
question + config ──> committee debate ──> trajectory ──> evaluators (pick A/B/C/D)
```

```python
import dspy
from llm_committee import LLMCommitteeSync
from llm_committee.evaluators import layer_a

committee = LLMCommitteeSync(
    num_members=3,
    chairman_lm=dspy.LM("openai/gpt-4o"),
    member_lms=[dspy.LM("openai/gpt-5-mini"),
                dspy.LM("openrouter/qwen/qwen3-32b"),
                dspy.LM("openrouter/google/gemini-2.5-flash-lite")],
    max_iterations=2,
    debate_stance="hostile",   # predefined tones: "neutral" | "friendly" | "hostile"
)

result = committee(
    committee_task="Debate the statement and reach a considered position.",
    agent_task="",
    task_input={"question": "Remote work makes teams more productive. Agree or disagree?"},
    task_context={},
)

print(result.final_judgement)
print(layer_a.summarize(result.trajectory_graph))
```

See `examples/debate_and_evaluate.py` for the full walkthrough.

## Debate tones

Every debate runs under a standing *stance instruction*, set by `debate_stance`:

| Tone | Instruction |
|---|---|
| `neutral` | none — the committee's default dynamics |
| `friendly` | seek common ground |
| `hostile` | stress-test every position |

In our study, moving friendly → hostile collapses self-reported full agreement by tens of points —
which is exactly why the label alone proves nothing, and why the evaluator layers exist.

## The evaluator layers

Each layer answers a *different* question; don't read one as another.

| Layer | Question | Entry point | Cost |
|---|---|---|---|
| **A** | Does the model *say* it disagrees? | `evaluators.layer_a.summarize(trajectory)` | free |
| **B** | Does the reply *text* actually push back? | `evaluators.layer_b.judge_pushback(...)` — a condition-blind judge that sees only the parent turn and reply; use a model family outside the committee | 1 judge call / turn |
| **C** | Does the position survive *deleting* the tone instruction that elicited it? | `evaluators.layer_c` — re-issues the turn without the instruction, paired with a matched retained re-ask as the resampling noise floor | re-queries the members |
| **D** | Does the debater's *token-level* stance distribution move? | `evaluators.layer_d` — reads next-token logprobs over a Likert probe | open-weight endpoints with logprobs |

Layer A is emitted by the same call that received the tone instruction — it is the *confounded*
signal the other layers exist to check. Layer D needs OpenAI-compatible endpoints exposing
logprobs (`LLM_COMMITTEE_ENDPOINTS`; see `llm_committee/evaluators/endpoints.example.json`).

## Installation

```bash
pip install -e .
```

```bash
python -m pytest tests/
```

## Repository map

| Path | What it is |
|---|---|
| `llm_committee/` | The committee engine: debate orchestration (`committee_sync.py`), trajectory graph, LM client. |
| `llm_committee/evaluators/` | The four disagreement layers plus trajectory/statistics helpers. |
| `examples/` | Runnable walkthroughs. |
| `tests/` | Network-free unit tests. |

The September experiment harnesses and analysis pipelines are included under
`llm_committee/pivot/` and `scripts/`, with protocols and frozen inputs under
`docs/`. Raw experiment archives and large reports remain local, as described above.

## License

MIT — see `LICENSE`.
