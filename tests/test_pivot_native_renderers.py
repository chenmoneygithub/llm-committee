"""Opt-in local tokenizer/SDK tests; never create a real Tinker service/session.

PIVOT_NATIVE_RENDERERS=1 enables public tokenizer loading in an environment with
the pilot-tinker extra installed. Normal CI skips these; transport responses are fixtures.
"""

import math
import os
from types import SimpleNamespace

import pytest

from llm_committee.pivot.models import Message, Request
from llm_committee.pivot.probabilities import candidate_ids, distribution, locate_visible_start
from llm_committee.pivot.tinker_provider import TinkerProvider, make_renderer

pytestmark = pytest.mark.skipif(
    os.environ.get("PIVOT_NATIVE_RENDERERS") != "1", reason="Opt-in native renderer check, no live model calls"
)


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "thinkingmachines/Inkling"])
def test_native_direct_render_and_choice_alignment(model):
    import tinker

    tokenizer, renderer = make_renderer(model, "none")
    text = "A\nMy complete position."
    messages = [{"role": "user", "content": "Choose A or B, then explain."}]
    if model == "thinkingmachines/Inkling":
        prompt = renderer.build_generation_prompt(messages, effort=0.0).to_ints()
        full, _ = renderer.build_supervised_example(messages + [{"role": "assistant", "content": text}], effort=0.0)
        assert full.to_ints()[: len(prompt)] == prompt
        tokens = full.to_ints()[len(prompt) :]
    else:
        tokens = tokenizer.encode(text) + renderer.get_stop_sequences()
    index = locate_visible_start(tokenizer, tokens, text)
    ids = candidate_ids(tokenizer, ("A", "B"))
    assert tokens[index] == ids["A"]
    topk = [None] * len(tokens)
    topk[index] = [(ids["A"], math.log(0.6)), (ids["B"], math.log(0.3))]
    logs = [-1.0] * len(tokens)
    logs[index] = math.log(0.6)
    sequence = SimpleNamespace(
        tokens=tokens, logprobs=logs, topk_logprobs=topk, stop_reason="stop", sequence_id="fixture"
    )
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(result=lambda **_: SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0))

    provider = TinkerProvider(service=object(), sdk=tinker)
    provider.client_for = lambda model: SimpleNamespace(sample=sample)
    request = Request(
        "test", "position", model, (Message("user", messages[0]["content"]),), "none", 1200, candidate_labels=("A", "B")
    )
    result = provider.generate(request)
    assert result.status == "completed" and result.reasoning_tokens == 0
    assert result.text == text
    assert result.readout["sampled_output_index"] == index
    assert distribution(result.readout, ("A", "B"))["probabilities"]["A"] == pytest.approx(2 / 3)
    assert calls[0]["sampling_params"].temperature == 1
    assert calls[0]["sampling_params"].top_p == 1
    assert calls[0]["sampling_params"].top_k == -1


def test_native_inkling_reasoning_is_detected_despite_effort_zero():
    import tinker

    tokenizer, renderer = make_renderer("thinkingmachines/Inkling", "none")
    messages = [{"role": "user", "content": "Output A."}]
    prompt = renderer.build_generation_prompt(messages, effort=0.0).to_ints()
    full, _ = renderer.build_supervised_example(
        messages
        + [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "I will think first."}, {"type": "text", "text": "A"}],
            }
        ],
        effort=0.0,
    )
    tokens = full.to_ints()[len(prompt) :]
    sequence = SimpleNamespace(
        tokens=tokens, logprobs=None, topk_logprobs=None, stop_reason="stop", sequence_id="fixture"
    )
    provider = TinkerProvider(service=object(), sdk=tinker)
    provider.client_for = lambda model: SimpleNamespace(
        sample=lambda **_: SimpleNamespace(
            result=lambda **_: SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)
        )
    )
    result = provider.generate(
        Request(
            "test",
            "d_text",
            "thinkingmachines/Inkling",
            (Message("user", "Output A."),),
            "none",
            128,
            candidate_labels=("A", "B"),
        )
    )
    assert result.status == "unexpected_reasoning" and result.reasoning_tokens > 0
    assert result.readout is None
    assert "I will think first." in str(result.raw["parsed_message"])
