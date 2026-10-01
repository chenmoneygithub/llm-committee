"""Frozen three-member routes, public histories, and A/B/E-only scope."""

import json
import sqlite3
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot import stateful_prompts
from llm_committee.pivot.models import Question, canonical
from llm_committee.pivot.storage import RunBlocked
from llm_committee.pivot.triadic_report import publish
from llm_committee.pivot.triadic_run import run
from llm_committee.pivot.triadic_study import (
    TriadicGraph,
    TriadicMockProvider,
    make_design,
    prepare,
    route_for,
)
from scripts.audit_triadic import audit

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/turn-tone-triadic-shared-plan-2026-09-28.json"


@pytest.fixture(scope="module")
def prepared():
    return prepare(PLAN, mock=True)


@pytest.fixture(scope="module")
def completed(prepared, tmp_path_factory):
    m, contexts = prepared
    output = tmp_path_factory.mktemp("triadic")
    report = run(m, contexts, output)
    assert report["status"] == "completed"
    return output


def test_plan_reuses_exact_questions_and_archived_routes():
    plan = json.loads(PLAN.read_text())
    rebuilt = make_design(plan["sources"]["question_plan"], plan["sources"]["route_manifest"])
    assert rebuilt == plan
    dyadic = json.loads(Path(plan["sources"]["question_plan"]).read_text())
    assert [p["question"] for p in plan["plans"]] == [p["question"] for p in dyadic["plans"]]
    assert len(plan["plans"]) == 20
    for p in plan["plans"]:
        route = route_for(p)
        assert len(route.leaves) == 6
        assert len(set(p["sampled_b_nodes"])) == 8
        assert len(set(p["sampled_endpoint_leaves"])) == 2
        assert set(p["sampled_endpoint_leaves"]) <= {n.id for n in route.leaves}
        for n in route.nodes:
            assert (p["tones"]["original"][n.id] == p["tones"]["alternate"][n.id]) == (n.depth <= 2)


@pytest.mark.parametrize("roster", ["mixed_family", "same_family", "same_model"])
def test_scope_and_shared_prefixes(roster):
    m, contexts = prepare(PLAN, roster=roster, mock=True)
    assert m["planned_counts"]["logical_calls"] == 1497
    assert m["planned_counts"]["formal_replies"] == 673
    assert m["planned_counts"]["local_C_calls"] == m["planned_counts"]["D_calls"] == 0
    p = m["plans"][0]
    g = TriadicGraph(contexts[p["question_id"]], p, m)
    counts = Counter(t.purpose for t in g.tasks.values())
    assert "judge_c" not in counts and "d_choice" not in counts and "d_text" not in counts
    for n in g.route.nodes:
        assert (g.key("original", f"debate/{n.id}") == g.key("alternate", f"debate/{n.id}")) == (n.depth <= 2)
    assert g.key("baseline", "E2/baseline") == g.key("alternate", "E2/baseline")


def test_public_history_is_ancestor_only_without_private_state(prepared):
    m, contexts = prepared
    p = m["plans"][0]
    q, route = Question.from_dict(contexts[p["question_id"]]["question"]), route_for(p)
    initial = {i: {"choice": "A", "position": f"INITIAL_{i}"} for i in range(3)}
    replies = {
        n.id: {
            "reply": f"PUBLIC_{n.id}",
            "agreement": "strongly_disagree",
            "choice": "B",
            "position": f"PRIVATE_{n.id}",
        }
        for n in route.nodes
    }
    for n in route.nodes:
        text = canonical(
            [vars(x) for x in stateful_prompts.turn_context(q, initial, route, replies, n.id, explicit_position=False)]
        )
        path = route.path(n.id)
        assert "PRIVATE_" not in text and "your_current_position" not in text
        assert "agreement" not in text and "choice" not in text
        for other in route.nodes:
            assert (f"PUBLIC_{other.id}" in text) == (other in path[:-1])
        assert text.count("incoming_peer_message") == 1


def test_complete_counts_and_frozen_request_audit(completed):
    report = json.loads((completed / "report.json").read_text())
    assert report["completed_questions"] == 20
    assert report["trajectory_statuses"] == {"success": 240}
    assert report["quality_comparison_statuses"] == {"success": 40}
    a = audit(completed)
    assert a["requests_reconstructed_exactly"] == a["completed_requests"] == 1497
    assert a["questions_reconstructed_exactly"] == 20
    assert a["primary_pairs_both_orders_verified"] == 40
    assert a["primary_answer_word_range"] == [200, 200]


def test_recorded_positions_and_endpoint_sampling(prepared, completed):
    m, _ = prepared
    for p in m["plans"]:
        r = json.loads((completed / "questions" / f"{p['question_id']}.json").read_text())
        route = route_for(p)
        assert len(r["E1"]) == 36
        for e in r["E1"]:
            state = r["positions"][f"{e['assignment']}/{e['leaf']}"][str(e["member"])]
            expected = [n for n in route.path(e["leaf"]) if n.receiver == e["member"]]
            assert len(state) == len(expected) + 1
            assert state[0]["T"] == 0
            assert [x["node_id"] for x in state[1:]] == [n.id for n in expected]
            assert e["participated"] == bool(expected)
            assert e["text_sampled"] == (e["leaf"] in p["sampled_endpoint_leaves"] and bool(expected))
            assert (e["text_judgment"] is not None) == e["text_sampled"]
            if not expected:
                assert e["choice_changed"] is None
        # Shared prefix outputs remain exactly identical across the paired schedules.
        for n in route.nodes:
            if n.depth <= 2:
                assert r["branches"]["original"]["replies"][n.id] == r["branches"]["alternate"]["replies"][n.id]


def test_report_has_only_ABE_and_readable_wins(completed, tmp_path):
    output = publish(completed, tmp_path / "triadic.html")
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert len(soup.select(".question-case")) == 20
    assert len(soup.select(".e-calibration-case")) == 12
    assert {s["id"] for s in soup.select("section[id]")} == {"a", "b", "e1", "e2", "cases"}
    assert "Local C/D are intentionally not collected" in soup.get_text()
    assert "Debate wins (%) [95% interval]" in soup.get_text()
    assert "OFFLINE MOCK" in soup.get_text()
    for t in soup.select("table"):
        assert all(len(r.select("td")) == len(t.select("thead th")) for r in t.select("tbody tr"))
    summary = json.loads(output.with_suffix(".json").read_text())
    assert len(summary["formal"]) == 673
    assert len(summary["B"]) == 294
    assert len(summary["E1_endpoints"]) == 720
    assert len(summary["E2_rows"]) == 40


def test_resume_generates_nothing_and_frozen_change_is_rejected(prepared, completed):
    m, contexts = prepared

    class NoCalls:
        def generate(self, request):
            raise AssertionError("Completed calls must not regenerate")

    result = run(m, contexts, completed, provider_factory=NoCalls)
    assert result["status"] == "completed"
    changed = json.loads(canonical(m))
    changed["plans"][0]["tones"]["original"]["n00"] = "different"
    with pytest.raises(RunBlocked, match="manifest"):
        run(changed, contexts, completed)


def test_failed_measurement_does_not_delete_debates(prepared, tmp_path):
    m, contexts = prepared
    p = m["plans"][0]
    g = TriadicGraph(contexts[p["question_id"]], p, m)
    target = next(t.key for t in g.tasks.values() if t.purpose == "judge_e1")
    seen = Counter()

    class BrokenEndpoint(TriadicMockProvider):
        def generate(self, request):
            seen[request.key] += 1
            answer = super().generate(request)
            return replace(answer, text="not JSON") if request.key == target else answer

    report = run(m, contexts, tmp_path / "measurement-failure", question_limit=1, provider_factory=BrokenEndpoint)
    assert report["status"] == "completed_with_failures"
    assert seen[target] == 3
    assert report["trajectory_statuses"] == {"success": 12}
    assert report["quality_comparison_statuses"] == {"success": 2}
    assert not report["fatal_errors"]


def test_shared_debate_failure_blocks_only_dependents(prepared, tmp_path):
    m, contexts = prepared
    p = m["plans"][0]
    g = TriadicGraph(contexts[p["question_id"]], p, m)
    target = g.key("original", "debate/n00")
    seen = Counter()

    class BrokenDebate(TriadicMockProvider):
        def generate(self, request):
            seen[request.key] += 1
            answer = super().generate(request)
            return replace(answer, text="not JSON") if request.key == target else answer

    output = tmp_path / "debate-failure"
    report = run(m, contexts, output, question_limit=1, provider_factory=BrokenDebate)
    assert report["status"] == "completed_with_failures"
    assert seen[target] == 3
    assert report["trajectory_statuses"]["success"] > 0
    assert report["trajectory_statuses"]["failed"] > 0
    assert report["quality_comparison_statuses"] == {"failed": 2}
    with sqlite3.connect(output / "requests.sqlite3") as db:
        requests = [json.loads(r) for (r,) in db.execute("SELECT request FROM calls")]
    assert not any(r["purpose"] == "synthesis" and r["key"].endswith(("E2/original", "E2/alternate")) for r in requests)


def test_frozen_tone_mismatch_rejected(prepared):
    m, contexts = prepared
    p = json.loads(canonical(m["plans"][0]))
    n = next(n for n in route_for(p).nodes if n.depth == 3)
    p["tones"]["alternate"][n.id] = p["tones"]["original"][n.id]
    with pytest.raises(ValueError, match="tones"):
        TriadicGraph(contexts[p["question_id"]], p, m)
