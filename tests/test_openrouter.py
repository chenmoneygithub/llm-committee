"""Offline OpenRouter transport and single-assignment routing tests."""

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_committee.pivot.models import Message, PilotConfig, Request, canonical
from llm_committee.pivot.openrouter_provider import BASE_URL, OpenRouterProvider, openrouter_payload, parse_openrouter_response
from llm_committee.pivot.providers import LiveProviders, cost_usd
from llm_committee.pivot.stateful_run import run as run_dyadic
from llm_committee.pivot.stateful_study import StatefulGraph
from llm_committee.pivot.triadic_run import run as run_triadic
from llm_committee.pivot.triadic_study import TriadicGraph
from scripts.run_openrouter import OpenRouterMock, prepare


def request(**kwargs):
    return replace(Request("q/initial", "initial", "gpt-5.6-terra", (Message("developer", "instruction"), Message("user", "question")),
                           "medium", 4096, schema={"type": "object"}), **kwargs)


def body():
    return {
        "id": "gen-fixture", "model": "openai/gpt-5.6-terra", "provider": "OpenAI",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"answer":"ok"}'}}],
        "usage": {"cost": 0.012345, "prompt_tokens": 1000, "completion_tokens": 150,
                  "prompt_tokens_details": {"cached_tokens": 300},
                  "completion_tokens_details": {"reasoning_tokens": 100}},
    }


def test_settings_and_raw_response_are_preserved():
    req = request()
    payload = openrouter_payload(req)
    assert payload["model"] == "openai/gpt-5.6-terra"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["max_tokens"] == 4096 and payload["stream"] is False
    assert payload["provider"]["require_parameters"] is True
    assert payload["messages"] == [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}]
    assert payload["response_format"]["json_schema"]["strict"] is True
    result = parse_openrouter_response(req, body(), payload)
    assert result.status == "completed" and result.raw["response"] == body()
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens, result.cached_tokens) == (1000, 150, 100, 300)
    assert cost_usd(req.model, result, closed_provider="openrouter") == 0.012345


def test_missing_optional_cost_does_not_block_generation():
    value = body()
    del value["usage"]["cost"]
    result = parse_openrouter_response(request(), value, {})
    assert result.status == "completed"
    assert cost_usd("gpt-5.6-terra", result, closed_provider="openrouter") == cost_usd("gpt-5.6-terra", result)


def test_incomplete_and_substituted_responses_are_not_accepted():
    value = body()
    value["choices"][0]["finish_reason"] = "length"
    assert parse_openrouter_response(request(), value, {}).status == "incomplete"
    value = body()
    value["model"] = "openai/gpt-5.6-luna"
    assert parse_openrouter_response(request(), value, {}).status == "unexpected_model"


@pytest.mark.parametrize("changes", [{"model": "unknown"}, {"cache": True}, {"candidate_labels": ("A", "B")}, {"scoring": {"x": 1}}])
def test_unsupported_settings_have_no_fallback(changes):
    with pytest.raises(ValueError):
        openrouter_payload(request(**changes))


def test_only_openrouter_credential_is_used_and_no_accounting_calls(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key-must-not-be-used")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()
    with pytest.raises(ValueError, match="no Tinker fallback"):
        LiveProviders(judge=True, mixed=True, closed_provider="openrouter")
    sent = []
    # No GET method: a completion must not call /key, /generation, or /models.
    client = SimpleNamespace(post=lambda url, **kw: sent.append((url, kw)) or SimpleNamespace(status_code=200, json=body))
    result = OpenRouterProvider(client=client, api_key="fixture-private-key").generate(request())
    assert len(sent) == 1 and sent[0][0] == BASE_URL + "/chat/completions"
    assert sent[0][1]["headers"]["Authorization"] == "Bearer fixture-private-key"
    assert "fixture-private-key" not in canonical(result.raw)


@pytest.mark.parametrize("members", [2, 3])
def test_single_assignment_is_a_closed_subset_of_original_graph(members):
    m, contexts = prepare(members=members, mock=True)
    graph_type = StatefulGraph if members == 2 else TriadicGraph
    cfg = PilotConfig(**m["config"])
    cfg.validate()
    assert cfg.members == ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
    assert cfg.closed_provider == "openrouter" and cfg.debate_effort == "medium"
    assert m["planned_counts"]["questions"] == 10
    assert m["planned_counts"]["branch_paths"] == 10 * (3 if members == 2 else 6)
    assert m["planned_counts"]["initial_answers"] == 30
    for p in m["plans"]:
        g = graph_type(contexts[p["question_id"]], p, m)
        assert all(t.dependencies <= g.tasks.keys() for t in g.tasks.values())
        assert all("/alternate/" not in key for key in g.tasks)
        assert not any(t.purpose.startswith("d_") for t in g.tasks.values())
    if members == 2:
        assert m["planned_counts"]["formal_replies"] == 120


def test_same_model_and_question_limit():
    manifest, _ = prepare(questions=1, roster="same_model", mock=True)
    assert PilotConfig(**manifest["config"]).members == ("gpt-5.6-terra",) * 3
    with pytest.raises(ValueError, match="exceeds"):
        prepare(questions=21)


@pytest.mark.parametrize("members", [2, 3])
def test_mock_run_has_no_phantom_alternate_paths_and_resumes_without_calls(tmp_path, members):
    m, contexts = prepare(members=members, questions=1, mock=True)
    runner = run_dyadic if members == 2 else run_triadic
    options = {"provider_factory": lambda: OpenRouterMock(members)}
    if members == 2:
        options["token_count"] = lambda model, text: len(text.split())
    result = runner(m, contexts, tmp_path, question_limit=1, request_limit=4, **options)
    assert result["status"] == "completed"
    assert result["trajectory_statuses"] == {"success": 3 if members == 2 else 6}
    record = json.loads(next((tmp_path / "questions").glob("*.json")).read_text())
    assert set(record["branches"]) == {"original"}
    assert all(key.startswith("original/") for key in record["positions"])
    runner(m, contexts, tmp_path, question_limit=1, request_limit=4, **options)
    with sqlite3.connect(tmp_path / "requests.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == m["planned_counts"]["logical_calls"]
