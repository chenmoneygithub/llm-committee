"""Offline Databricks transport tests: no account access or real generations."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.databricks_provider import (
    DatabricksProvider,
    databricks_payload,
    parse_databricks_response,
)
from llm_committee.pivot.models import Completion, Message, PilotConfig, Request, canonical
from llm_committee.pivot.providers import LiveProviders, cost_usd, reservation_usd
from llm_committee.pivot.storage import Journal, RunBlocked


def request(**changes):
    return replace(
        Request(
            "test/C",
            "position",
            "gpt-5.6-terra",
            (Message("developer", "rule"), Message("user", "question")),
            "none",
            1200,
        ),
        **changes,
    )


def response():
    return {
        "choices": [{"message": {"role": "assistant", "content": "A\nMy opinion"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0},
    }


def test_payload_preserves_messages_effort_and_schema():
    r = request(effort="medium", schema=prompts.INITIAL_SCHEMA)
    payload = databricks_payload(r)
    assert payload["messages"] == [{"role": "system", "content": "rule"}, {"role": "user", "content": "question"}]
    assert payload["reasoning_effort"] == "medium"
    assert payload["max_tokens"] == 1200 and payload["stream"] is False
    assert payload["service_tier"] == "default"
    assert payload["response_format"]["json_schema"]["schema"] == r.schema
    assert not {"store", "tools", "prompt_cache_options", "temperature"}.intersection(payload)
    assert "response_format" not in databricks_payload(request())


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "gpt-5.5"},
        {"cache": True},
        {"candidate_labels": ("A", "B")},
        {"model": "gemini-3.8-flash", "effort": "none", "schema": prompts.C_SCHEMA},
        {"messages": (Message("user", "question"), Message("developer", "late instruction"))},
    ],
)
def test_no_silent_model_setting_or_role_fallback(changes):
    with pytest.raises(ValueError):
        databricks_payload(request(**changes))


def test_structured_reasoning_is_not_public_text():
    raw = response()
    raw["choices"][0]["message"]["content"] = [
        {"type": "reasoning", "summary": [{"text": "private"}]},
        {"type": "text", "text": '{"position":"answer"}'},
    ]
    raw["usage"]["reasoning_tokens"] = 10
    c = parse_databricks_response(request(effort="medium"), raw, {})
    assert c.text == '{"position":"answer"}' and c.reasoning_tokens == 10
    assert c.output_tokens == 20  # Already includes reasoning; do not add it twice.
    assert c.raw["response"] == raw
    assert parse_databricks_response(request(), raw, {}).status == "unexpected_reasoning"


@pytest.mark.parametrize("visible,total,expected", [(77, 775, 172), (172, 775, 172), (77, 774, None), (77, None, None)])
def test_gemini_billing_reconciles_reasoning_against_reported_total(visible, total, expected):
    raw = response()
    raw["usage"] = {"prompt_tokens": 603, "completion_tokens": visible, "reasoning_tokens": 95, "total_tokens": total}
    c = parse_databricks_response(request(model="gemini-3.8-flash", effort="low", schema=prompts.B_SCHEMA), raw, {})
    assert c.output_tokens == expected and c.reasoning_tokens == 95
    assert c.raw["response"] == raw
    if expected is None:
        with pytest.raises(ValueError, match="billing token"):
            cost_usd("gemini-3.8-flash", c)
    else:
        assert cost_usd("gemini-3.8-flash", c, closed_provider="databricks") == pytest.approx(0.001206975)


def test_gemini_total_can_verify_bill_without_reporting_reasoning_partition():
    raw = response()
    raw["usage"] = {"prompt_tokens": 622, "completion_tokens": 78, "total_tokens": 700}
    r = request(model="gemini-3.8-flash", effort="low", schema=prompts.B_SCHEMA)
    c = parse_databricks_response(r, raw, {})
    assert c.output_tokens == 78
    assert c.raw["reasoning_usage_reported"] is False
    assert cost_usd(r.model, c, closed_provider="databricks") == pytest.approx(0.0008349)
    raw["usage"]["total_tokens"] = 701
    assert parse_databricks_response(r, raw, {}).output_tokens is None


def test_missing_reasoning_usage_does_not_validate_direct_read():
    raw = response()
    del raw["usage"]["reasoning_tokens"]
    c = parse_databricks_response(request(), raw, {})
    assert c.status == "unverified_direct_read" and c.raw["response"] == raw
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
    raw["usage"]["prompt_tokens_details"] = {"cached_tokens": 25, "cache_write_tokens": 5}
    c = parse_databricks_response(request(), raw, {})
    assert c.status == "completed" and c.cached_tokens == 25 and c.cache_write_tokens == 5


def test_direct_read_rejects_visible_reasoning_even_if_reported_zero():
    raw = response()
    raw["choices"][0]["message"]["reasoning_content"] = "visible reasoning"
    assert parse_databricks_response(request(), raw, {}).status == "unexpected_reasoning"


def test_no_reasoning_only_fallback_or_tool_output():
    raw = response()
    raw["choices"][0]["message"]["content"] = [{"type": "reasoning", "summary": [{"text": "private"}]}]
    raw["choices"][0]["finish_reason"] = "length"
    c = parse_databricks_response(request(effort="medium"), raw, {})
    assert c.text == "" and c.status == "incomplete"
    raw["choices"][0]["message"]["tool_calls"] = [{"id": "unexpected"}]
    assert parse_databricks_response(request(effort="medium"), raw, {}).status == "unexpected_message"


def test_oauth_workspace_header_refresh_without_archiving_credentials():
    auth_calls, sent = [], []

    def authenticate():
        auth_calls.append(1)
        return {"Authorization": "Bearer fixture-private"}

    config = SimpleNamespace(host="https://example.databricks.com", workspace_id="42", authenticate=authenticate)

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return SimpleNamespace(
            status_code=200, headers={"served-model-name": "databricks-gpt-5-6-terra"}, json=response
        )

    provider = DatabricksProvider(config=config, client=SimpleNamespace(post=post))
    for _ in range(2):
        c = provider.generate(request())
        assert "fixture-private" not in canonical(c.raw)
        assert c.raw["endpoint"] == "databricks-gpt-5-6-terra"
    assert len(auth_calls) == 2
    assert sent[0][0].endswith("/serving-endpoints/databricks-gpt-5-6-terra/invocations")
    assert sent[0][1]["headers"]["X-Databricks-Workspace-Id"] == "42"


def test_parallel_workers_share_serialized_oauth_but_not_model_requests(monkeypatch):
    from llm_committee.pivot import databricks_provider as module

    lock = threading.Lock()
    configurations, auth_active, auth_peak, auth_calls = 0, 0, 0, 0
    gate = threading.Barrier(16, timeout=10)

    def authenticate():
        nonlocal auth_active, auth_peak, auth_calls
        with lock:
            auth_active += 1
            auth_calls += 1
            auth_peak = max(auth_peak, auth_active)
        time.sleep(0.001)
        with lock:
            auth_active -= 1
        return {"Authorization": "Bearer fixture-private"}

    def factory(profile):
        nonlocal configurations
        configurations += 1
        assert profile == "un"
        return SimpleNamespace(host="https://example.databricks.com", workspace_id="42", authenticate=authenticate)

    def post(*args, **kwargs):
        gate.wait()  # The model requests must overlap; auth lock cannot span HTTP.
        return SimpleNamespace(status_code=200, headers={}, json=response)

    monkeypatch.setattr(module, "_AUTH_CONFIGS", {})
    monkeypatch.setattr(module, "_new_workspace_config", factory)

    def invoke(_):
        return DatabricksProvider(client=SimpleNamespace(post=post)).generate(request())

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(invoke, range(16)))
    assert configurations == 1 and auth_peak == 1 and auth_calls == 17
    assert all(r.status == "completed" and "fixture-private" not in canonical(r.raw) for r in results)


def test_failed_oauth_initialization_is_not_cached(monkeypatch):
    from llm_committee.pivot import databricks_provider as module

    calls = []

    def fail(profile):
        calls.append(profile)
        raise ValueError("authentication unavailable")

    monkeypatch.setattr(module, "_AUTH_CONFIGS", {})
    monkeypatch.setattr(module, "_new_workspace_config", fail)
    with pytest.raises(ValueError, match="authentication unavailable"):
        DatabricksProvider(client=SimpleNamespace())
    assert calls == ["un"] and not module._AUTH_CONFIGS


def test_http_error_preserved_and_full_hold_retained(tmp_path):
    cfg = SimpleNamespace(host="https://example.databricks.com", workspace_id="42", authenticate=lambda: {})
    http = SimpleNamespace(
        post=lambda *a, **kw: SimpleNamespace(
            status_code=400, headers={}, json=lambda: {"error_code": "INVALID_PARAMETER_VALUE", "message": "bad format"}
        )
    )
    provider = DatabricksProvider(config=cfg, client=http)
    journal = Journal(tmp_path / "journal.sqlite3", {"config": {"closed_provider": "databricks"}}, 30)
    try:
        with pytest.raises(RunBlocked, match="reconcile"):
            journal.call(request(), provider, lambda t: {})
        audit = journal.audit()
        call = audit["calls"][0]
        assert call["status"] == "billing_unknown"
        assert call["response"]["raw"]["response"]["error_code"] == "INVALID_PARAMETER_VALUE"
        assert call["charge_usd"] == reservation_usd(request(), closed_provider="databricks")
        with pytest.raises(RunBlocked, match="inspect the journal"):
            journal.call(request(), provider, lambda t: {})
    finally:
        journal.close()


def test_databricks_dispatch_does_not_require_direct_keys(monkeypatch):
    from llm_committee.pivot import databricks_provider, providers

    sent = []
    fake = SimpleNamespace(generate=lambda r: sent.append(r.model), close=lambda: None)
    monkeypatch.setattr(databricks_provider, "DatabricksProvider", lambda **kw: fake)

    def forbidden():
        raise AssertionError("Do not initialize the direct-key adapters")

    monkeypatch.setattr(providers, "OpenAIProvider", forbidden)
    monkeypatch.setattr(providers, "GeminiProvider", forbidden)
    p = LiveProviders(judge=True)
    p.generate(request())
    p.generate(request(model="gemini-3.8-flash", effort="low", schema=prompts.C_SCHEMA))
    p.close()
    assert sent == ["gpt-5.6-terra", "gemini-3.8-flash"]


def test_explicit_transport_is_validated_and_budgeted():
    PilotConfig().validate()
    with pytest.raises(ValueError, match="cache"):
        PilotConfig(cache=True).validate()
    PilotConfig(closed_provider="direct", cache=True).validate()
    with pytest.raises(ValueError, match="explicitly"):
        PilotConfig(closed_provider="other").validate()
    c = Completion("text", 100, 20)
    assert cost_usd("gpt-5.6-terra", c, closed_provider="databricks") == pytest.approx(
        cost_usd("gpt-5.6-terra", c) * 1.1
    )
    assert cost_usd("thinkingmachines/Inkling", c, closed_provider="databricks") == cost_usd(
        "thinkingmachines/Inkling", c
    )
