"""The authorized diagnostic preserves failures and stops at the first valid retry."""

import json
import sqlite3
from dataclasses import asdict

import pytest

from llm_committee.pivot.databricks_provider import databricks_payload
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, Message, Question, Request, Screen
from llm_committee.pivot.prompts import parse_position
from llm_committee.pivot.storage import Journal, RunBlocked
from scripts.investigate_position_read import investigate


class Responses:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.requests = []
        self.closed = False

    def generate(self, request):
        self.requests.append(request)
        text = next(self.texts)
        return Completion(
            text,
            100,
            10,
            raw={
                "provider": "databricks",
                "http_status": 200,
                "request_payload": databricks_payload(request),
                "response": {
                    "id": f"response-{len(self.requests)}",
                    "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                },
            },
        )

    def close(self):
        self.closed = True


def source_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    q = Question("fixture", "Should parks be free?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
    manifest = {
        "config": {"closed_provider": "databricks", "databricks_profile": "un"},
        "questions": [asdict(q)],
        "execution": {"budget_policy": "no_limit_user_requested"},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    request = Request(
        "fixture/mixed_family/C/initial/0",
        "position",
        "gpt-5.6-terra",
        (Message("user", "Choice on first line; full position below."),),
        "none",
        1200,
    )
    journal = Journal(source / "requests.sqlite3", manifest, None)
    try:
        with pytest.raises(RunBlocked):
            journal.call(request, Responses(["A"]), lambda text: parse_position(text, q))
        audit = journal.audit()
    finally:
        journal.close()
    return source, request, manifest, audit


@pytest.mark.parametrize(
    "texts,expected_calls,expected_status",
    [
        (["A\nComplete position", "B\nMust never select a later response"], 1, "first_valid_retry_recorded"),
        (["A", "B\nComplete position"], 2, "first_valid_retry_recorded"),
        (["A", "A"], 2, "same_failure_on_both_retries"),
    ],
)
def test_bounded_identical_payload_retries_preserve_original(tmp_path, texts, expected_calls, expected_status):
    source, request, manifest, before = source_fixture(tmp_path)
    provider = Responses(texts)
    result = investigate(source, request.key, tmp_path / "diagnostic", retries=2, provider_factory=lambda: provider)
    assert len(provider.requests) == expected_calls and provider.closed
    assert all(r.document() == request.document() for r in provider.requests)
    assert result["status"] == expected_status and result["source_unchanged"]
    assert result["main_study_resumed"] is False
    assert result["new_charged_or_reserved_usd"] == sum(a["charge_usd"] for a in result["attempts"])
    journal = Journal(source / "requests.sqlite3", manifest, None)
    try:
        assert journal.audit() == before
    finally:
        journal.close()


def test_changed_payload_is_rejected_without_opening_provider(tmp_path):
    source, request, _, _ = source_fixture(tmp_path)
    with sqlite3.connect(source / "requests.sqlite3") as db:
        response = json.loads(db.execute("SELECT response FROM calls").fetchone()[0])
        response["raw"]["request_payload"]["max_tokens"] = 1
        db.execute("UPDATE calls SET response=?", (json.dumps(response),))

    def forbidden():
        raise AssertionError("Do not open a provider for a mismatched payload")

    with pytest.raises(ValueError, match="payload changed"):
        investigate(source, request.key, tmp_path / "diagnostic", retries=2, provider_factory=forbidden)
    assert not (tmp_path / "diagnostic").exists()


def test_uncertain_transport_is_logged_and_not_automatically_retried(tmp_path):
    source, request, _, _ = source_fixture(tmp_path)

    class Timeout(Responses):
        def generate(self, request):
            self.requests.append(request)
            raise TimeoutError("Uncertain provider response")

    provider = Timeout([])
    result = investigate(source, request.key, tmp_path / "diagnostic", retries=2, provider_factory=lambda: provider)
    assert len(provider.requests) == 1 and provider.closed
    assert result["status"] == "different_failure_requires_inspection"
    assert result["attempts"][0]["status"] == "uncertain"
    assert result["new_charged_or_reserved_usd"] > 0
