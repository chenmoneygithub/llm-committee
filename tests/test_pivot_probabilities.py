"""No-network tests of probability semantics, native transport boundaries and D controls."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.budget import estimate_budget
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, PilotConfig, Question, Request, Screen, digest
from llm_committee.pivot.planning import plan_question, route_from_plan
from llm_committee.pivot.probabilities import (
    RATING_LABELS,
    capture_readout,
    choice_change,
    distribution,
    locate_visible_start,
    merge_scored_readout,
    own_position_difference,
)
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.questions import load_questions, validate_input_review
from llm_committee.pivot.runner import expected_counts, manifest_for, run_pilot
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.tinker_provider import TinkerProvider, message_content


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


@pytest.mark.parametrize("status", ["success", "errored", "interrupted"])
def test_tinker_close_supplies_required_session_status(status):
    calls = []
    service = SimpleNamespace(
        close=lambda value: (calls.append(value) or SimpleNamespace(result=lambda **kwargs: None))
    )
    provider = TinkerProvider(service=service, sdk=object())
    provider.close(status)
    assert calls == [status]


def q():
    return Question(
        "test", "Should public parks be free?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "test fixture", "0" * 64)
    )


def sample_readout(labels=("A", "B")):
    scores = [(ord(label), math.log(0.9 / len(labels))) for label in labels]
    return capture_readout(
        Tokenizer(),
        [1000],
        Tokenizer().encode("<>A\nView!"),
        "A\nView!",
        labels,
        [None, None, scores],
        [None, None, scores[0][1]],
    )


def test_readout_keeps_generated_framing_and_exact_output_position():
    r = sample_readout()
    assert r["prefix_token_ids"] == [1000, ord("<"), ord(">")]
    assert r["sampled_output_index"] == 2
    assert r["prefix_sha256"] == digest(r["prefix_token_ids"])
    d = distribution(r, ("A", "B"))
    assert d["probabilities"] == {"A": 0.5, "B": 0.5}
    assert d["candidate_mass"] == pytest.approx(0.9)


@pytest.mark.parametrize("invalid", ["missing", "nan", "positive", "temperature", "truncated"])
def test_invalid_distribution_cannot_be_normalized_into_success(invalid):
    r = sample_readout()
    if invalid == "missing":
        del r["candidate_logprobs"]["B"]
    elif invalid == "nan":
        r["candidate_logprobs"]["B"] = math.nan
    elif invalid == "positive":
        r["candidate_logprobs"]["B"] = 0.1
    elif invalid == "temperature":
        r["read_temperature"] = 2
    else:
        r["top_p"] = 0.9
    with pytest.raises(ValueError):
        distribution(r, ("A", "B"))


def test_ambiguous_or_misaligned_output_is_not_rescored_at_another_letter():
    with pytest.raises(ValueError, match="uniquely"):
        locate_visible_start(Tokenizer(), Tokenizer().encode("A\nView! A\nView!"), "A\nView!")
    with pytest.raises(ValueError, match="without whitespace"):
        locate_visible_start(Tokenizer(), Tokenizer().encode(" A"), " A")


def test_supplemental_scores_check_identity_and_existing_logprobs():
    original = sample_readout()
    complete = {**original, "source": "target_prompt_logprobs"}
    original = {**original, "candidate_logprobs": {"A": original["candidate_logprobs"]["A"]}}
    assert merge_scored_readout(original, complete, ("A", "B"))["source"] == "target_prompt_logprobs"
    with pytest.raises(ValueError, match="prefix_sha256"):
        merge_scored_readout(original, {**complete, "prefix_sha256": "other"}, ("A", "B"))
    with pytest.raises(ValueError, match="disagree"):
        merge_scored_readout(original, {**complete, "candidate_logprobs": {"A": -10, "B": -1}}, ("A", "B"))


def test_choice_and_text_metrics_are_not_mislabeled_as_opponent_movement():
    a = {"probabilities": {"A": 0.7, "B": 0.3}}
    b = {"probabilities": {"A": 0.6, "B": 0.4}}
    assert choice_change(a, b, "A") == pytest.approx(-10)
    assert choice_change(a, b, None) is None
    control = {"probabilities": dict.fromkeys(RATING_LABELS, 0.0)}
    argument = {"probabilities": dict.fromkeys(RATING_LABELS, 0.0)}
    control["probabilities"]["G"] = 1
    argument["probabilities"]["F"] = 1
    assert own_position_difference(argument, control) == -1


def test_d_context_omits_peer_then_inserts_each_arm_once_without_future_or_siblings():
    question = q()
    plan = plan_question(question, PilotConfig())
    route = route_from_plan(plan)
    initial = {m: f"INITIAL-UNIQUE-{m}" for m in range(3)}
    replies = {n.id: {"reply": f"REPLY-UNIQUE-{n.id}"} for n in route.nodes}
    for event in plan["events"]:
        node = route.get(event["node_id"])
        peer = replies[node.parent]["reply"] if node.parent else initial[node.sender]
        argument = prompts.d_text_messages(question, event, initial, route, replies, "FIXED FULL OPINION", peer)
        control = prompts.d_text_messages(question, event, initial, route, replies, "FIXED FULL OPINION", "FILLER")
        assert argument[:-2] == control[:-2] and argument[-1] == control[-1]
        assert argument[-2] != control[-2]
        assert "FIXED FULL OPINION" in argument[-3].text
        text = "\n".join(m.text for m in control)
        ancestors_before_peer = {n.id for n in route.path(node.id)[:-2]}
        for other in route.nodes:
            assert (replies[other.id]["reply"] in text) == (other.id in ancestors_before_peer)
        if node.parent is None:
            assert peer not in text
        assert peer in argument[-2].text and "FILLER" in control[-2].text


def test_thinking_detection_handles_both_renderer_formats():
    assert message_content({"role": "assistant", "content": "A", "reasoning_content": "thinking"}) == ("A", "thinking")
    assert message_content(
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking"}, {"type": "text", "text": "A"}]}
    ) == ("A", "thinking")
    with pytest.raises(ValueError, match="without tools"):
        message_content({"role": "assistant", "content": "A", "tool_calls": ["unexpected"]})


def test_native_score_request_targets_correct_row_without_new_history():
    captured = []
    saved = sample_readout()
    scores = list(saved["candidate_logprobs"].values())

    def sample(**kwargs):
        captured.append(kwargs)
        rows = len(saved["prefix_token_ids"])
        matrix = [[0, 0] for _ in range(rows - 1)] + [scores]
        result = SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[999], sequence_id="score-sequence")],
            prompt_cache_hit_tokens=0,
            target_prompt_logprobs=SimpleNamespace(to_numpy=lambda: SimpleNamespace(tolist=lambda: matrix)),
            prompt_logprobs=[None] * rows + [saved["sampled_token_logprob"]],
        )
        return SimpleNamespace(result=lambda **_: result)

    sdk = SimpleNamespace(
        TensorData=lambda **kwargs: SimpleNamespace(**kwargs),
        ModelInput=SimpleNamespace(from_ints=lambda tokens: tokens),
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    provider = TinkerProvider(service=object(), sdk=sdk)
    provider.client_for = lambda model: SimpleNamespace(sample=sample)
    request = Request(
        "score", "candidate_scores", "Qwen/Qwen3.8-27B", (), "none", 1, candidate_labels=("A", "B"), scoring=saved
    )
    result = provider.generate(request)
    assert result.status == "completed"
    call = captured[0]
    assert call["prompt"] == [*saved["prefix_token_ids"], saved["sampled_token_id"]]
    assert call["target_prompt_logprobs"].shape == [3, 2]
    assert call["target_prompt_logprobs"].sparse_crow_indices == [0, 0, 0, 2]
    assert call["sampling_params"].max_tokens == 1
    assert result.input_tokens == 4 and result.output_tokens == 1
    assert merge_scored_readout(saved, json.loads(result.text), ("A", "B"))["source"] == "target_prompt_logprobs"


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_topk_uses_zero_fill_or_explicit_legacy_scoring(tmp_path, legacy):
    class PartialProvider(MockProvider):
        def generate(self, request):
            if request.scoring is not None:
                self.calls.append(request)
                full = {
                    **request.scoring,
                    "candidate_logprobs": request.scoring["test_full_scores"],
                    "source": "target_prompt_logprobs",
                }
                return Completion(json.dumps(full), 100, 1)
            result = super().generate(request)
            if result.readout:
                full = result.readout["candidate_logprobs"]
                result = replace(
                    result, readout={**result.readout, "test_full_scores": full, "candidate_logprobs": {"A": full["A"]}}
                )
            return result

    question, config = q(), PilotConfig()
    manifest = manifest_for((question,), config, mock=True)
    if legacy:
        manifest.pop("probability_readout")
    provider = PartialProvider()
    journal = Journal(tmp_path / "partial.sqlite3", manifest, 1000)
    try:
        result = run_pilot((question,), config, manifest, journal, provider)
        counts = expected_counts(manifest)
        assert len(provider.calls) == counts["logical_calls_with_scoring_allowance"]
        assert any(call.purpose == "candidate_scores" for call in provider.calls) == legacy
        if not legacy:
            for d in result["questions"][0]["D"]["choice_readings"].values():
                assert d["probabilities"] == {"A": 1.0, "B": 0.0}
                assert d["missing_candidate_labels"] == ["B"]
        count = len(provider.calls)
        assert run_pilot((question,), config, manifest, journal, provider) == result
        assert len(provider.calls) == count
    finally:
        journal.close()


def test_reasoning_in_direct_read_is_saved_and_not_retried(tmp_path):
    class ThinkingProvider(MockProvider):
        def generate(self, request):
            self.calls.append(request)
            return Completion("A", 100, 20, reasoning_tokens=10, raw={"thinking": "actual raw output"})

    provider = ThinkingProvider()
    request = Request("bad-read", "d_text", "thinkingmachines/Inkling", (), "none", 128, candidate_labels=RATING_LABELS)
    journal = Journal(tmp_path / "thinking.sqlite3", {}, 10)
    try:
        with pytest.raises(RunBlocked, match="Invalid response"):
            journal.call(request, provider, prompts.parse_rating)
        record = journal.audit()["calls"][0]
        assert record["response"]["raw"]["thinking"] == "actual raw output"
        with pytest.raises(RunBlocked, match="invalid"):
            journal.call(request, provider, prompts.parse_rating)
        assert len(provider.calls) == 1
    finally:
        journal.close()


def test_input_review_is_fingerprint_bound(tmp_path):
    question = q()
    document = {
        "purpose": "engineering_pilot",
        "questions": [asdict(question)],
        "question_fingerprints": [question.fingerprint],
    }
    source = tmp_path / "questions.json"
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="input-context review"):
        validate_input_review(source, load_questions(source))
    document["input_review"] = {
        "status": "complete",
        "items": [{"fingerprint": question.fingerprint, "status": "usable", "reason": "standalone question"}],
    }
    source.write_text(json.dumps(document))
    validate_input_review(source, load_questions(source))
    document["input_review"]["items"][0]["fingerprint"] = "other"
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exact question"):
        validate_input_review(source, load_questions(source))


@pytest.mark.parametrize("legacy", [False, True])
def test_budget_counts_scoring_only_for_legacy_policy_without_cache_assumptions(legacy):
    config = PilotConfig(judge_model="gemini-3.8-flash")
    manifest = manifest_for((q(),), config, mock=True)
    if legacy:
        manifest.pop("probability_readout")
    budget = estimate_budget(manifest)
    for scenario in budget["scenarios"].values():
        assert (
            scenario["logical_calls_including_all_scoring_allowances"]
            == expected_counts(manifest)["logical_calls_with_scoring_allowance"]
        )
        assert scenario["usd_with_30_percent_reserve"] == pytest.approx(1.3 * scenario["usd"])
        assert (
            scenario["usd_with_30_percent_reserve_and_inkling_regular_price"] > scenario["usd_with_30_percent_reserve"]
        )
        for row in scenario["rows"]:
            if "scoring" in row["stage"]:
                assert legacy
                assert row["output_tokens"] == row["calls"]
    assert budget["scenarios"]["high_tokens"]["usd"] > budget["scenarios"]["central_tokens"]["usd"]


def test_live_question_review_blocks_before_any_client(tmp_path, monkeypatch, capsys):
    import llm_committee.pivot.__main__ as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Unreviewed input reached a live provider")

    monkeypatch.setattr(cli, "LiveProviders", forbidden)
    question = q()
    source = tmp_path / "unreviewed.json"
    source.write_text(
        json.dumps(
            {
                "purpose": "engineering_pilot",
                "questions": [asdict(question)],
                "question_fingerprints": [question.fingerprint],
            }
        )
    )
    assert (
        cli.main(
            ["run", "--questions", str(source), "--output", str(tmp_path / "not_created"), "--approve-spend-usd", "30"]
        )
        == 2
    )
    assert "input-context review" in capsys.readouterr().out
    assert not (tmp_path / "not_created").exists()
