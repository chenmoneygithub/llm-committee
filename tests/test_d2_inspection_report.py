"""Offline diagnostic checks must detect shifted indices, not trust the saved index."""

import copy
import math
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.d2_inspection_report import (
    LABELS,
    MODEL_NAMES,
    arm_html,
    inspect_request,
    publish,
    select_cases,
)
from llm_committee.pivot.models import digest


class Tokenizer:
    texts = {**{i + 32: k for i, k in enumerate(LABELS)}, 100: "<message>", 101: "<text>", 102: "<end>", 999: "PROMPT"}

    def decode(self, ids):
        return "".join(self.texts[i] for i in ids)

    def encode(self, text, **kwargs):
        return [next(i for i, value in self.texts.items() if value == text)]


class Renderer:
    def parse_response(self, tokens):
        return {"role": "assistant", "content": "G"}, SimpleNamespace(is_clean=True)

    def build_generation_prompt(self, messages, **kwargs):
        return SimpleNamespace(to_ints=lambda: [999])


def entry(framed=True):
    index = 2 if framed else 0
    tokens = [100, 101, 38, 102] if framed else [38, 102]
    ids = dict(zip(LABELS, range(32, 39), strict=True))
    scores = {k: math.log(p) for k, p in zip(LABELS, [0.01] * 5 + [0.15, 0.70], strict=True)}
    topk = [[[t, 0.0]] for t in tokens]
    topk[index] = [[ids[k], score] for k, score in scores.items()]
    logs = [0.0] * len(tokens)
    logs[index] = scores["G"]
    prefix = [999] + tokens[:index]
    saved = {
        "candidate_ids": ids,
        "candidate_logprobs": scores,
        "sampled_output_index": index,
        "sampled_token_id": 38,
        "sampled_token_logprob": scores["G"],
        "prefix_token_ids": prefix,
        "prefix_sha256": digest(prefix),
        "sampling_temperature": 1.0,
        "read_temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    return {
        "request": {
            "model": "thinkingmachines/Inkling" if framed else "Qwen/Qwen3.8-27B",
            "effort": "none",
            "messages": [{"role": "user", "text": "<script>not executable</script>\nFULL PROMPT"}],
        },
        "parsed": {"rating": "G", "_readout": copy.deepcopy(saved)},
        "response": {
            "text": "G",
            "reasoning_tokens": 0,
            "readout": copy.deepcopy(saved),
            "raw": {
                "renderer": "fixture",
                "prompt_token_ids": [999],
                "sampled_token_ids": tokens,
                "sampled_logprobs": logs,
                "topk_logprobs": topk,
                "stop_reason": "stop",
                "probability_readout": copy.deepcopy(saved),
                "sampling_temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        },
    }


@pytest.mark.parametrize("framed,index", [(True, 2), (False, 0)])
def test_native_rating_position_and_probability_reconstruction(framed, index):
    original = entry(framed)
    result = inspect_request(original, Tokenizer(), Renderer())
    assert result["passed"]
    assert result["read_index"] == result["independent_rating_index"] == index
    assert result["candidate_mass"] == pytest.approx(0.9)
    assert result["normalized_probabilities"]["G"] == pytest.approx(0.7 / 0.9)
    assert sum(result["normalized_probabilities"].values()) == pytest.approx(1)
    assert [r["text"] for r in result["tokens"] if r["used_for_D2"]] == ["G"]
    assert result["decoded_scoring_prefix"] == ("PROMPT<message><text>" if framed else "PROMPT")


def test_wrong_saved_index_is_flagged_even_with_consistent_saved_prefix():
    original = entry()
    for saved in (
        original["parsed"]["_readout"],
        original["response"]["readout"],
        original["response"]["raw"]["probability_readout"],
    ):
        saved.update(
            sampled_output_index=0,
            sampled_token_id=100,
            prefix_token_ids=[999],
            prefix_sha256=digest([999]),
            sampled_token_logprob=0.0,
        )
    result = inspect_request(original, Tokenizer(), Renderer())
    assert not result["passed"]
    assert result["independent_rating_index"] == 2 and result["read_index"] == 0
    assert not result["checks"]["rating_token_position"]
    assert not result["checks"]["candidate_scores_from_saved_row"]
    assert result["recomputed_rating"] is None


def test_inconsistent_sample_logprob_is_not_hidden_by_normalization():
    original = entry()
    original["response"]["raw"]["sampled_logprobs"][2] -= 0.1
    result = inspect_request(original, Tokenizer(), Renderer())
    assert not result["passed"]
    assert not result["checks"]["sampled_vs_topk_logprob"]


def test_render_keeps_full_messages_raw_arrays_and_escapes_content():
    result = inspect_request(entry(), Tokenizer(), Renderer())
    document = arm_html(result, "Control")
    soup = BeautifulSoup(document, "html.parser")
    assert not soup.select("script")
    assert soup.select_one(".full-prompt pre").get_text() == "<script>not executable</script>\nFULL PROMPT"
    assert len(soup.select(".token")) == 4 and len(soup.select(".token.read")) == 1
    assert "<text>" in soup.select_one(".token.read").find_previous_sibling().get_text()
    assert "sampled_output_index" in soup.get_text()
    assert "Original top-20 logprobs at every generated position" in soup.get_text()
    for node in soup.select("table"):
        width = len(node.select("thead th"))
        assert all(len(row.select("td")) == width for row in node.select("tbody tr"))


def test_selection_is_deterministic_and_five_distinct_pairs_per_model():
    cases = []
    for model in MODEL_NAMES:
        for i, delta in enumerate((-1, -0.8, -0.4, -0.01, 0, 0.1, 0.2)):
            cases.append(
                {
                    "model": model,
                    "id": model + str(i),
                    "reported_delta": delta,
                    "peer_label": "leaning_disagree" if i in (1, 2) else "strongly_agree",
                    "receiver_label": "strongly_agree",
                }
            )
    selected = select_cases(cases)
    assert selected == select_cases(list(reversed(cases)))
    assert len({c["id"] for c in selected}) == len(selected) == 10
    for model in MODEL_NAMES:
        group = [c for c in selected if c["model"] == model]
        assert [c["reported_delta"] for c in group] == [-1, -0.8, 0, -0.01, 0.2]
        assert not any(c["selection_fallback"] for c in group)


def test_existing_unrelated_report_is_never_overwritten(tmp_path):
    output = tmp_path / "main.html"
    output.write_text("PRESERVE")
    with pytest.raises(ValueError, match="unrelated"):
        publish(tmp_path / "run", output)
    assert output.read_text() == "PRESERVE"
