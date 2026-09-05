# LLM Committee

Multi-agent LLM debate with **measurable disagreement**. A committee of models debates a question
under a configurable tone, the full debate trajectory is captured, and four evaluator layers let
you check whether the disagreement is *real* — instead of trusting any single signal.

Built for the paper *"LLM debate readily changes what agents say — the evidence that it changes
what they persistently endorse, or improves the final answer, is much weaker"* (link forthcoming),
where these four layers measurably come apart.

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

The study's experiment harnesses, frozen artifacts, and analysis pipelines live in the research
archive, not here; every number in the paper regenerates from that archive.

## License

MIT — see `LICENSE`.
