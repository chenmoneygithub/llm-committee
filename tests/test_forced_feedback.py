"""Supplementary forks must preserve frozen prefixes and isolate private instructions."""

import json
from collections import Counter
from dataclasses import asdict

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.forced_feedback import FORCE_INSTRUCTION, VERSION, FeedbackGraph, case_route, select_cases
from llm_committee.pivot.forced_feedback_run import run
from llm_committee.pivot.models import SCREEN_JUDGES, TONES, Completion, PilotConfig, Question, Screen, digest
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.runner import manifest_for


@pytest.fixture(scope="module")
def fixture():
    questions = tuple(
        Question(f"q{i}", f"Policy topic {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(60)
    )
    config = PilotConfig(judge_model="gemini-3.8-flash")
    original = manifest_for(questions, config, mock=True)
    cases = select_cases(original)
    contexts = {
        q.id: {
            "question": asdict(q),
            "plan": plan,
            "initial_answers": {str(i): f"INITIAL-{i}" for i in range(3)},
            "formal_replies": {
                tone: {
                    node["id"]: {"reply": f"ARCHIVED::{tone}::{node['id']}::END", "agreement": "fully_agree"}
                    for node in plan["route"]
                }
                for tone in TONES
            },
        }
        for q, plan in zip(questions, original["plans"], strict=True)
    }
    manifest = {
        **original,
        "protocol_version": VERSION,
        "source": {"directory": "/nonexistent/frozen-test-source", "contexts_sha256": digest(contexts)},
        "cases": cases,
        "planned_counts": {"logical_calls": 1600},
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
    }
    # Production snapshots are loaded from JSON, including options and screen judges.
    return manifest, json.loads(json.dumps(contexts))


def test_frozen_balanced_sampling_and_call_counts(fixture):
    manifest, contexts = fixture
    cases = manifest["cases"]
    assert cases == select_cases(manifest)
    assert Counter(c["recipient"] for c in cases) == {0: 50, 1: 50, 2: 50}
    assert Counter(c["tone"] for c in cases) == dict.fromkeys(TONES, 50)
    assert Counter(Counter(c["question_id"] for c in cases).values()) == {2: 30, 3: 30}
    assert len({(c["recipient"], c["challenger"]) for c in cases[:6]}) == 6
    assert len({(c["question_id"], c["tone"], c["cut_node"]) for c in cases}) == 150
    counts = Counter()
    for case in cases:
        graph = FeedbackGraph(contexts[case["question_id"]], [case], manifest)
        counts.update(task.purpose for task in graph.tasks.values())
        route = case_route(graph.context, case)
        last = route.get("feedback_return")
        assert last.parent == case["cut_node"]
        assert (last.sender, last.receiver) == (case["challenger"], case["recipient"])
        assert last.depth == case["receiver_T"]
        assert len(route.nodes) == case["feedback_T"] + 1
    assert counts == {"debate": 450, "position": 450, "judge_c": 300, "d_text": 400}


def test_contexts_exclude_old_future_other_arm_and_private_instructions(fixture):
    manifest, contexts = fixture
    count = MockProvider().token_count
    for case in manifest["cases"][:6]:
        graph = FeedbackGraph(contexts[case["question_id"]], [case], manifest)
        route = case_route(graph.context, case)
        values = {
            graph.key(case, "before"): {"choice": "A", "position": "MEASUREMENT-ONLY-BEFORE"},
            graph.key(case, "forced/feedback"): {"reply": "NEW-FORCED-FEEDBACK", "agreement": "fully_disagree"},
        }
        for arm in ("natural", "forced"):
            values[graph.key(case, f"{arm}/reply")] = {"reply": f"RETURN-{arm}", "agreement": "leaning_agree"}
            values[graph.key(case, f"{arm}/position")] = {"choice": "B", "position": f"AFTER-{arm}"}
        prefix_nodes = {n.id for n in route.path(case["cut_node"])[:-1]}
        archived = graph.context["formal_replies"][case["tone"]]
        for task in graph.tasks.values():
            request = task.build(values, count)
            text = "\n".join(m.text for m in request.messages)
            forced_generation = task.key.endswith("forced/feedback")
            assert (FORCE_INSTRUCTION in text) == forced_generation
            if forced_generation:
                assert FORCE_INSTRUCTION in request.messages[0].text
                assert "NEW-FORCED-FEEDBACK" not in text
                assert archived[case["cut_node"]]["reply"] not in text
            for node_id, reply in archived.items():
                if node_id not in prefix_nodes | {case["cut_node"]}:
                    assert reply["reply"] not in text
            if "/forced/" in task.key:
                assert archived[case["cut_node"]]["reply"] not in text
                assert "RETURN-natural" not in text
            if "/natural/" in task.key:
                assert "NEW-FORCED-FEEDBACK" not in text
                assert "RETURN-forced" not in text
            if request.purpose == "debate":
                assert "MEASUREMENT-ONLY-BEFORE" not in text
            if request.purpose == "d_text":
                assert "RETURN-" not in text
                assert "MEASUREMENT-ONLY-BEFORE" in text
            assert request.effort == (
                "medium" if request.purpose == "debate" else "low" if request.purpose == "judge_c" else "none"
            )
        if case["recipient"] != 0:
            for arm in ("natural", "forced"):
                argument = graph.tasks[graph.key(case, f"{arm}/d_text/argument")].build(values, count)
                control = graph.tasks[graph.key(case, f"{arm}/d_text/control")].build(values, count)
                different = [(a, b) for a, b in zip(argument.messages, control.messages, strict=True) if a != b]
                assert len(different) == 1
                assert prompts.FILLER_SENTENCE in different[0][1].text


def test_gate_resume_never_regenerates_and_labels_belong_to_recipient(fixture, tmp_path):
    manifest, contexts = fixture
    output = tmp_path / "run"
    report = run(manifest, contexts, output, case_limit=6, request_limit=16)
    assert report["status"] == "preflight_completed"
    assert report["successful_cases"] == 6
    assert report["call_status_counts"] == {"completed": 64}
    for case in manifest["cases"][:6]:
        result = json.loads((output / "cases" / f"{case['id']}.json").read_text())
        assert result["status"] == "completed"
        for arm in ("natural", "forced"):
            data = result["arms"][arm]
            assert data["receiver_self_label"] == data["receiver_reply"]["agreement"]
            assert (data["D_text"] is not None) == (case["recipient"] != 0)
            assert data["C_text"]["label"] == "unchanged"

    def forbidden():
        raise AssertionError("Resume must not create a provider for completed requests")

    resumed = run(manifest, contexts, output, case_limit=6, provider_factory=forbidden)
    assert resumed["call_status_counts"] == report["call_status_counts"]
    assert resumed["charged_or_reserved_usd"] == report["charged_or_reserved_usd"]


def test_exhausted_feedback_failure_leaves_natural_arm_and_other_cases_running(fixture, tmp_path):
    manifest, contexts = fixture
    first = manifest["cases"][0]
    failure_key = f"{first['question_id']}/{VERSION}/{first['id']}/forced/feedback"
    attempted = []

    class BrokenFeedback(MockProvider):
        def generate(self, request):
            if request.key.startswith(failure_key):
                attempted.append(request.key)
                return Completion("INVALID JSON", 100, 2)
            return super().generate(request)

    output = tmp_path / "failed"
    report = run(manifest, contexts, output, case_limit=6, request_limit=16, provider_factory=BrokenFeedback)
    assert report["status"] == "completed_with_failures"
    assert len(attempted) == 3
    assert report["successful_cases"] == 5 and not report["fatal_errors"]
    result = json.loads((output / "cases" / f"{first['id']}.json").read_text())
    assert result["arms"]["natural"]["receiver_reply"]
    assert result["arms"]["natural"]["C_text"]
    assert result["arms"]["forced"]["receiver_reply"] is None
