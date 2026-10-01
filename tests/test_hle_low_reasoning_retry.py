"""Synthetic-only tests for the isolated effort-toggle diagnostic."""

import json

from llm_committee.pivot.models import Completion, Message, Request, canonical
from scripts.hle_low_reasoning_retry import MODEL, TimedProvider, execute_case, low_request, render, saved_results


def example():
    request = Request(
        "synthetic/initial/1",
        "initial",
        MODEL,
        (Message("developer", "Synthetic instruction"), Message("user", "Which number is even? A. 2 B. 3")),
        "medium",
        16384,
        {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["A", "B"]}, "position": {"type": "string"}},
            "required": ["choice", "position"],
            "additionalProperties": False,
        },
    )
    return {"source_key": request.key, "request": request.document(), "reference_answer": "A"}


def test_only_effort_and_storage_key_change():
    source = example()["request"]
    result = low_request(source).document()
    assert {k for k in source if canonical(source[k]) != canonical(result[k])} == {"key", "effort"}
    assert result["effort"] == "low" and result["max_output_tokens"] == 16384


def test_single_sample_resume_and_local_reference(tmp_path):
    class Fake:
        calls = 0

        def generate(self, request):
            self.calls += 1
            assert "reference_answer" not in canonical(request.document())
            return Completion(json.dumps({"choice": "A", "position": "Two is even."}), 20, 20, reasoning_tokens=5)

    source, provider = example(), Fake()
    manifest = {"execution": {"budget_policy": "no_limit_user_requested"}}
    first = execute_case(source, TimedProvider(provider), manifest, tmp_path)
    second = execute_case(source, TimedProvider(provider), manifest, tmp_path)
    assert provider.calls == 1 and first == second
    assert saved_results([source], tmp_path) == [first]
    assert first["status"] == "completed" and first["correct"] is True
    assert first["response"]["raw"]["diagnostic_elapsed_seconds"] >= 0
    page = render(
        {
            "cases": [first],
            "completed": 1,
            "correct": 1,
            "received_response_estimate_usd": 0.01,
            "unresolved_reservations_usd": 0,
        }
    )
    assert "不能据此估计整体准确率" in page and "Two is even." in page


def test_invalid_attempt_not_silently_repeated(tmp_path):
    class Broken:
        calls = 0

        def generate(self, request):
            self.calls += 1
            return Completion("unfinished", 20, 20, status="incomplete")

    provider = Broken()
    manifest = {"execution": {"budget_policy": "no_limit_user_requested"}}
    for _ in range(2):
        result = execute_case(example(), TimedProvider(provider), manifest, tmp_path)
        assert result["status"] == "invalid" and result["correct"] is None
    assert provider.calls == 1
    assert saved_results([example()], tmp_path)[0]["status"] == "invalid"
