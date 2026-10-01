"""Recorded positions are observations, never private state injected into debate/D1."""

import json
import sqlite3
from pathlib import Path

import pytest

from llm_committee.pivot import prompts, stateful_prompts
from llm_committee.pivot.dyadic import make_route
from llm_committee.pivot.models import Question, canonical
from llm_committee.pivot.quality_study import prepare as prepare_quality
from llm_committee.pivot.stateful_run import run
from llm_committee.pivot.stateful_study import PUBLIC_HISTORY_VERSION, StatefulGraph, StatefulMockProvider, prepare
from llm_committee.pivot.storage import RunBlocked
from scripts.audit_stateful_dyadic import audit

PLAN = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"


@pytest.fixture(scope="module")
def prepared():
    return prepare(PLAN, mock=True, public_history=True)


@pytest.fixture(scope="module")
def question(prepared):
    manifest, contexts = prepared
    return Question.from_dict(contexts[manifest["plans"][0]["question_id"]]["question"])


def material(route):
    initial = {m: {"choice": "A", "position": f"[INITIAL POSITION {m}]"} for m in range(3)}
    replies = {
        n.id: {
            "reply": f"[PUBLIC REPLY {n.id}]",
            "agreement": "strongly_disagree",
            "choice": "B",
            "position": f"[PRIVATE POSITION {n.id}]",
        }
        for n in route.nodes
    }
    return initial, replies


@pytest.mark.parametrize("nid", [n.id for n in make_route().nodes])
def test_formal_and_D1_have_only_original_answers_and_public_history(question, nid):
    route = make_route()
    initial, replies = material(route)
    node = route.get(nid)
    debate = stateful_prompts.debate_messages(
        question, "hostile", initial, route, replies, nid, explicit_position=False
    )
    d1 = stateful_prompts.choice_messages(
        question, node.receiver, initial, route, replies, nid, explicit_position=False
    )
    assert debate[:-1] == d1[:-1]
    context = canonical([m.text for m in debate[:-1]])
    assert "your_current_position" not in context and "PRIVATE POSITION" not in context
    assert "strongly_disagree" not in context and prompts.TONE_TEXT["hostile"] not in context
    assert context.count(initial[node.receiver]["position"]) == 1
    assert context.count(initial[route.path(nid)[0].sender]["position"]) == 1
    ancestors = {n.id for n in route.path(nid)[:-1]}
    for n in route.nodes:
        assert context.count(replies[n.id]["reply"]) == int(n.id in ancestors)
    assert "public branch history" in debate[-1].text
    assert "ONLY to incoming_peer_message" in debate[-1].text
    assert "may remain unchanged" in debate[-1].text
    assert "your_current_position" not in debate[-1].text
    assert "reply, agreement" not in context


@pytest.mark.parametrize("nid", ["AB-2", "BC-3", "CA-4"])
def test_D2_adds_fixed_prior_text_only_as_measurement_target(question, nid):
    route = make_route()
    initial, replies = material(route)
    node = route.get(nid)
    incoming = replies[node.parent]["reply"]
    a = stateful_prompts.text_messages(question, initial, route, replies, nid, incoming, explicit_position=False)
    b = stateful_prompts.text_messages(question, initial, route, replies, nid, "FILLER", explicit_position=False)
    d1 = stateful_prompts.choice_messages(
        question, node.receiver, initial, route, replies, nid, explicit_position=False
    )
    assert [i for i in range(len(a)) if a[i] != b[i]] == [len(a) - 2]
    assert d1[:-1] == (*a[:-3], a[-2])
    target = json.loads(a[-3].text)["position_to_evaluate"]
    assert set(target) == {"source", "text"}
    old = route.previous_own(nid)
    assert target["text"] == (replies[old.id] if old else initial[node.receiver])["position"]
    assert replies[nid]["position"] not in canonical([m.text for m in a])
    assert "position_to_evaluate.text" in a[-1].text


@pytest.fixture(scope="module")
def completed(tmp_path_factory, prepared):
    manifest, contexts = prepared
    output = tmp_path_factory.mktemp("public-history-offline")
    result = run(manifest, contexts, output, question_limit=20)
    assert result["status"] == "completed"
    return output


def test_full_run_layout_and_audit(prepared, completed):
    manifest, contexts = prepared
    old, old_contexts = prepare(PLAN, mock=True)
    assert contexts == old_contexts
    assert manifest["plans"] == old["plans"]
    assert manifest["planned_counts"] == old["planned_counts"]
    assert manifest["protocol_version"] == PUBLIC_HISTORY_VERSION
    assert not manifest["design"]["explicit_position_feedback"]
    result = audit(completed)
    assert result["public_history_only_debate_and_D1"]
    assert result["completed_requests"] == result["requests_reconstructed_exactly"] == 1676
    assert result["argument_control_pairs_differ_only_in_incoming"] == 143
    assert result["questions_reconstructed_exactly"] == 20
    with pytest.raises(RunBlocked, match="Frozen manifest"):
        run(old, contexts, completed, question_limit=1)


def test_changing_recorded_positions_never_changes_later_debate_or_D1(prepared, completed):
    manifest, contexts = prepared
    plan = manifest["plans"][0]
    graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
    with sqlite3.connect(f"file:{completed}/requests.sqlite3?mode=ro", uri=True) as db:
        entries = [(json.loads(r), json.loads(v)) for r, v in db.execute("SELECT request,parsed FROM calls")]
    calls = {r["key"]: r for r, _ in entries if r["key"] in graph.tasks}
    graph.values = {r["key"]: v for r, v in entries if r["key"] in graph.tasks}
    for key, value in graph.values.items():
        if graph.tasks[key].purpose == "debate":
            value["position"] = "DO NOT REINJECT PRIVATE POSITION"
            value["choice"] = None
    count = StatefulMockProvider().token_count
    for key, task in graph.tasks.items():
        if task.purpose in ("debate", "d_choice"):
            assert canonical(task.build(graph.values, count).document()) == canonical(calls[key])


def test_quality_uses_matching_new_initial_answers_and_public_replies(completed):
    manifest, contexts = prepare_quality(completed, mock=True)
    assert manifest["planned_counts"]["logical_calls"] == 540
    for qid, context in contexts.items():
        source = json.loads((completed / "questions" / f"{qid}.json").read_text())
        assert context["initial_answers"] == {m: v["position"] for m, v in source["initial_positions"].items()}


def test_new_report_explains_protocol_and_separates_endpoints(completed, tmp_path):
    from bs4 import BeautifulSoup

    from scripts.report_stateful_dyadic import publish

    output = publish(completed, tmp_path / "public-history.html")
    summary = json.loads(output.with_suffix(".json").read_text())
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert summary["protocol_version"] == PUBLIC_HISTORY_VERSION
    assert len(soup.select("circle.event-dot")) == 143
    assert "not reinjected" in soup.get_text()
    assert "Initial position → last participation" not in soup.select_one("#c").get_text()
    assert sum(g["paths"] for g in summary["E_member_endpoints"]) == 240
    assert sum(g["sampled_text_pairs"] for g in summary["E_member_endpoints"]) == 214
    probes = soup.select_one("#probes").get_text()
    assert "D1 previous input" in probes and "D1 current input" in probes
    assert '"position_to_evaluate"' in probes and '"your_current_position"' not in probes
    for t in soup.select("table"):
        assert all(len(r.select("td")) == len(t.select("thead th")) for r in t.select("tbody tr"))
