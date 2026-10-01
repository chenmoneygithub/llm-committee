"""Opt-in cached-tokenizer checks, with fabricated sampler responses and no endpoint."""

import math
import os
from types import SimpleNamespace

import pytest

from llm_committee.pivot import stateful_prompts
from llm_committee.pivot.dyadic import make_route
from llm_committee.pivot.models import Question, Request, Screen
from llm_committee.pivot.probabilities import candidate_ids, distribution, locate_visible_start
from llm_committee.pivot.tinker_provider import TinkerProvider, make_renderer

pytestmark = pytest.mark.skipif(
    os.environ.get("PIVOT_NATIVE_RENDERERS") != "1", reason="Opt-in native tokenizer check, no real sampling"
)


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "thinkingmachines/Inkling"])
@pytest.mark.parametrize("explicit_position", [True, False])
def test_native_D1_alignment_and_shared_D_prefixes(model, explicit_position):
    import tinker

    q = Question(
        "fixture",
        "Should public libraries be free?",
        ("Yes", "No"),
        Screen(("gpt-5.5", "opus-4.8", "gemini-3.5-flash"), 3, "fixture", "0" * 64),
    )
    initial = {m: {"choice": "A", "position": f"Member {m}: libraries should be free."} for m in range(3)}
    route = make_route()
    replies = {
        "BC-1": {
            "reply": "Free access helps people on low incomes; public funding can cover operating costs.",
            "agreement": "leaning_agree",
            "choice": "A",
            "position": "I favor free library access.",
        }
    }
    d1 = stateful_prompts.choice_messages(q, 1, initial, route, replies, "BC-2", explicit_position=explicit_position)
    argument = stateful_prompts.text_messages(
        q, initial, route, replies, "BC-2", replies["BC-1"]["reply"], explicit_position=explicit_position
    )
    control = stateful_prompts.text_messages(
        q, initial, route, replies, "BC-2", "The moderator noted the time.", explicit_position=explicit_position
    )
    tokenizer, renderer = make_renderer(model, "none")
    kwargs = {"effort": 0.0} if model == "thinkingmachines/Inkling" else {}

    def rendered(messages):
        native = [{"role": "system" if m.role == "developer" else m.role, "content": m.text} for m in messages]
        return native, renderer.build_generation_prompt(native, **kwargs).to_ints()

    messages, prompt = rendered(d1)
    argument_tokens, control_tokens = rendered(argument)[1], rendered(control)[1]

    def lcp(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))

    if explicit_position:
        assert lcp(prompt, argument_tokens) > lcp(argument_tokens, control_tokens) > 0
    else:
        assert lcp(argument_tokens, control_tokens) > lcp(prompt, argument_tokens) > 0
    if model == "thinkingmachines/Inkling":
        full, _ = renderer.build_supervised_example(messages + [{"role": "assistant", "content": "A"}], **kwargs)
        assert full.to_ints()[: len(prompt)] == prompt
        tokens = full.to_ints()[len(prompt) :]
    else:
        tokens = tokenizer.encode("A") + renderer.get_stop_sequences()
    index = locate_visible_start(tokenizer, tokens, "A")
    ids = candidate_ids(tokenizer, q.labels)
    topk, logs = [None] * len(tokens), [-1.0] * len(tokens)
    topk[index] = [(ids["A"], math.log(0.6)), (ids["B"], math.log(0.3))]
    logs[index] = math.log(0.6)
    sequence = SimpleNamespace(
        tokens=tokens, logprobs=logs, topk_logprobs=topk, stop_reason="stop", sequence_id="fixture"
    )
    seen = []

    def sample(**args):
        seen.append(args)
        return SimpleNamespace(result=lambda **_: SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0))

    provider = TinkerProvider(service=object(), sdk=tinker)
    provider.client_for = lambda _: SimpleNamespace(sample=sample)
    result = provider.generate(Request("fixture", "d_choice", model, d1, "none", 128, candidate_labels=q.labels))
    assert result.status == "completed" and result.reasoning_tokens == 0
    assert stateful_prompts.parse_choice(result.text, q) == {"choice": "A"}
    assert result.readout["sampled_output_index"] == index
    assert seen[0]["prompt"].to_ints() == prompt
    assert distribution(result.readout, q.labels)["probabilities"]["A"] == pytest.approx(2 / 3)
