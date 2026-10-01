"""Closed-book question adaptation without leakage or changes to archived opinion protocols."""

import copy
import json
from collections import Counter
from dataclasses import asdict

import pytest

from llm_committee.pivot.dyadic import make_route
from llm_committee.pivot.models import Completion, canonical
from llm_committee.pivot.strong_run import run
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import (
    BenchmarkQuestion,
    SuperGPQAGraph,
    SuperGPQAMockProvider,
    context,
    correctness_transition,
    d1_metrics,
    debate_messages,
    initial_messages,
    prepare,
)
from scripts.supergpqa_pilot import audit, publish, summaries


@pytest.fixture
def fixture_bank(tmp_path):
    path = tmp_path / "plan/bank.json"
    row = {
        "uuid": "unit-test",
        "question": "Which number is prime?",
        "options": ["4", "7", "9"],
        "answer_letter": "B",
        "answer": "7",
        "difficulty": "middle",
    }
    atomic_json(
        path,
        {
            "source": {"dataset": "SYNTHETIC"},
            "scope": "SYNTHETIC TEST ONLY",
            "selected": [{"domain": "math", "screen_note": "EVALUATION_ONLY_SECRET", "source_row": row}],
        },
    )
    return path


@pytest.fixture
def prepared(fixture_bank):
    return prepare(fixture_bank, mock=True)


def test_counts_and_mixed_roster(prepared):
    manifest, contexts = prepared
    assert manifest["planned_counts"]["by_purpose"] == {"initial": 3, "debate": 12, "d_choice": 10, "synthesis": 6}
    assert manifest["planned_counts"]["logical_calls"] == 31
    assert manifest["config"]["judge_model"] is None
    assert manifest["config"]["initial_tokens"] == 8192
    assert manifest["design"]["tone_assignments"] == 1
    assert len(contexts) == 1


def test_objective_schema_and_no_fake_opinion_screen():
    q = BenchmarkQuestion("test", "What is 2+3?", ("4", "5", "6"))
    messages = initial_messages(q, 0)
    text = canonical([asdict(m) for m in messages])
    assert "academic multiple-choice" in text
    assert "survey question" not in text
    assert "none fits" not in text
    assert not hasattr(q, "screen") and not hasattr(q, "answer_letter")


@pytest.mark.parametrize("node_id", [n.id for n in make_route().nodes])
def test_public_history_only_no_third_member_or_tone_metadata(node_id):
    q = BenchmarkQuestion("test", "What is 2+3?", ("4", "5", "6"))
    route = make_route()
    initial = {m: {"choice": "A", "position": f"INITIAL_{m}"} for m in range(3)}
    replies = {
        n.id: {
            "reply": f"PUBLIC_{n.id}",
            "position": f"PRIVATE_{n.id}",
            "choice": "C",
            "agreement": "strongly_disagree",
        }
        for n in route.nodes
    }
    n = route.get(node_id)
    messages = debate_messages(q, initial, route, replies, node_id)
    text = canonical([asdict(m) for m in messages[:-1]])
    assert "PRIVATE_" not in text and "strongly_disagree" not in text
    assert "your_current_position" not in text
    # Only the root sender's initial answer was publicly transmitted. A member
    # also retains its own initial answer; it does not see an unshared peer initial.
    pair = {route.path(node_id)[0].sender, n.receiver}
    for m in range(3):
        assert text.count(f"INITIAL_{m}") == int(m in pair)
    for ancestor in route.nodes:
        assert text.count(f"PUBLIC_{ancestor.id}") == int(ancestor in route.path(node_id)[:-1])
    assert tuple(messages[:-1]) == context(q, n.receiver, initial, route, replies, node_id)
    changed = copy.deepcopy(replies)
    for v in changed.values():
        v.update(position="DIFFERENT PRIVATE TEXT", choice="B", agreement="strongly_agree")
    assert debate_messages(q, initial, route, changed, node_id) == messages


def test_answer_key_never_changes_any_model_request(prepared):
    manifest, contexts = prepared
    plan = manifest["plans"][0]
    stored = contexts[plan["question_id"]]
    alternate = {**stored, "answer_letter": "C", "screen_note": "ANOTHER_SECRET"}
    one, two = (SuperGPQAGraph(c, plan, manifest) for c in (stored, alternate))
    provider = SuperGPQAMockProvider()
    while not one.complete:
        task = one.ready()
        request = task.build(one.values, lambda *_: 0)
        assert request == two.tasks[task.key].build(two.values, lambda *_: 0)
        assert "EVALUATION_ONLY_SECRET" not in canonical(request.document())
        assert "ANOTHER_SECRET" not in canonical(request.document())
        response = provider.generate(request)
        value = task.parse(response.text)
        if request.candidate_labels:
            value["_readout"] = response.readout
        one.accept(task, request, value)
        two.accept(two.tasks[task.key], request, copy.deepcopy(value))


def test_tv_and_correct_answer_probability_are_different_signals():
    p = {"probabilities": {"A": 0.7, "B": 0.2, "C": 0.1}}
    q = {"probabilities": {"A": 0.2, "B": 0.6, "C": 0.2}}
    result = d1_metrics(p, q, "B")
    assert result["total_variation"] == pytest.approx(0.5)
    assert result["correct_probability_change_pp"] == pytest.approx(40)
    assert d1_metrics(p, q, "A")["correct_probability_change_pp"] == pytest.approx(-50)
    assert d1_metrics(None, q, "A") is None
    assert correctness_transition("A", "B", "B") == "wrong_to_right"
    assert correctness_transition("B", "A", "B") == "right_to_wrong"
    assert correctness_transition(None, "A", "B") is None


def test_complete_resume_audit_and_html(prepared, tmp_path):
    from bs4 import BeautifulSoup

    manifest, contexts = prepared
    output = tmp_path / "outputs/mock"
    provider = SuperGPQAMockProvider()
    result = run(
        manifest,
        contexts,
        output,
        question_limit=1,
        request_limit=8,
        workers=1,
        graph_type=SuperGPQAGraph,
        mock_provider_type=SuperGPQAMockProvider,
        provider_factory=lambda: provider,
        token_count=lambda *_: 0,
    )
    assert result["status"] == "completed"
    assert len(provider.calls) == 31
    checked = audit(output)
    assert checked["requests_reconstructed"] == 31
    run(
        manifest,
        contexts,
        output,
        question_limit=1,
        graph_type=SuperGPQAGraph,
        mock_provider_type=SuperGPQAMockProvider,
        provider_factory=lambda: provider,
        token_count=lambda *_: 0,
    )
    assert len(provider.calls) == 31
    record = json.loads(next((output / "questions").glob("*.json")).read_text())
    assert len(record["events"]) == 12 and len(record["endpoints"]) == 6
    assert Counter(e["T"] for e in record["events"]) == {1: 3, 2: 3, 3: 3, 4: 3}
    assert all(e["previous_label"] is None for e in record["events"] if e["T"] == 1)
    summary = summaries([record])
    assert sum(r["n"] for r in summary["C_conditional"]) == 9
    assert sum(r["n"] for r in summary["D1_conditional"]) == 6
    assert sum(r["n"] for r in summary["agreement_by_model"]) == 12
    assert summary["coverage"] == {
        "initial_answers": 3,
        "replies": 12,
        "complete_trajectories": 3,
        "failed_tasks": 0,
        "blocked_tasks": 0,
    }
    phases = Counter()
    for row in summary["D1_measurement_types"]:
        phases[row["measurement_type"]] += row["n"]
    assert phases == {"first_participation": 2, "later_participation": 4}
    page = tmp_path / "report.html"
    publish(output, page)
    soup = BeautifulSoup(page.read_text(), "html.parser")
    assert "620" in soup.get_text()  # Protocol plan size, not pretending this fixture is the full study.
    assert "TV" in soup.get_text() and "ΔP(正确答案)" in soup.get_text()
    assert "较大的首次跳变不能只归因于同伴说服" in soup.get_text()
    assert "首次参与输入" in soup.get_text()
    for anchor in soup.select('nav a[href^="#"]'):
        assert soup.find(id=anchor["href"][1:])
    for table in soup.select("table"):
        width = len(table.select("thead th"))
        assert all(len(tr.select("td")) == width for tr in table.select("tbody tr"))


def test_exhausted_task_is_missing_not_an_incorrect_answer(prepared, tmp_path):
    manifest, contexts = prepared

    class BrokenInitial(SuperGPQAMockProvider):
        def generate(self, request):
            if request.key.endswith("initial/2") and request.purpose == "initial":
                return Completion("not JSON", 100, 10, raw={"mock": True})
            return super().generate(request)

    output = tmp_path / "failed-output"
    result = run(
        manifest,
        contexts,
        output,
        question_limit=1,
        request_limit=8,
        workers=1,
        graph_type=SuperGPQAGraph,
        mock_provider_type=BrokenInitial,
        token_count=lambda *_: 0,
    )
    assert result["status"] == "completed_with_failures"
    checked = audit(output)
    assert checked["questions_reconstructed"] == 1
    assert checked["requests_reconstructed"] == checked["successful_requests"]
    summary = publish(output, tmp_path / "failed.html")
    missing = next(m for m in summary["models"] if m["model"] == "thinkingmachines/Inkling")
    assert missing["initial_n"] == missing["endpoint_n"] == 0
    assert summary["initial_disagreement_denominator"] == 0
    assert summary["observed_initial_disagreement_denominator"] == 1
    assert summary["coverage"]["complete_trajectories"] == 1
    assert summary["coverage"]["failed_tasks"] == 1
    assert all(e["n"] == 0 for e in summary["chairman"] if e["pair"] != "AB")
