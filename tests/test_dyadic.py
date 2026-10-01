"""Dyadic routing, label targets, context isolation and durable parallel execution."""

import json
from collections import Counter
from dataclasses import asdict

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.dyadic import PAIRS, VERSION, DyadicGraph, make_route, plan_questions
from llm_committee.pivot.dyadic_run import run
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.models import SCREEN_JUDGES, TONES, Completion, PilotConfig, Question, Screen, digest
from llm_committee.pivot.providers import MockProvider


@pytest.fixture(scope="module")
def fixture():
    questions = [
        Question(f"q{i:02}", f"Policy {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(60)
    ]
    contexts = {
        q.id: {"question": asdict(q), "initial_answers": {str(m): f"INITIAL-MEMBER-{m}" for m in range(3)}}
        for q in questions
    }
    contexts = json.loads(json.dumps(contexts))
    manifest = {
        "kind": "offline_mock",
        "protocol_version": VERSION,
        "config": asdict(PilotConfig(judge_model="gemini-3.8-flash")),
        "source": {"directory": "/nonexistent/immutable-source", "contexts_sha256": digest(contexts)},
        "plans": plan_questions(questions, 20260926),
        "planned_counts": {},
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
    }
    return manifest, contexts


def test_directed_routes_and_each_previous_label_targets_receiver():
    route = make_route()
    assert len(route.nodes) == 12 and len(route.leaves) == 3
    for pair, (sender, receiver) in PAIRS.items():
        nodes = route.path(f"{pair}-4")
        assert [n.receiver for n in nodes] == [receiver, sender, receiver, sender]
        assert nodes[0].sender == sender
        for n in nodes[1:]:
            peer = route.get(n.parent)
            assert peer.sender == n.receiver
            previous_own = route.previous_own(n.id)
            assert peer.parent == (previous_own.id if previous_own else None)


def test_presampled_without_labels_and_budget(fixture):
    manifest, contexts = fixture
    plans = manifest["plans"]
    questions = [Question.from_dict(c["question"]) for c in contexts.values()]
    assert plans == plan_questions(questions, 20260926)
    counts = Counter()
    for plan in plans:
        assert len(plan["events"]) == 8
        assert len({(e["pair"], e["tone"]) for e in plan["events"]}) == 8
        assert all(e["T"] in (2, 3, 4) for e in plan["events"])
        assert all("self_label" not in str(e) for e in plan["events"])
        graph = DyadicGraph(contexts[plan["question_id"]], plan, manifest)
        counts.update(t.purpose for t in graph.tasks.values())
    assert counts["debate"] == 2160 and counts["judge_b"] == 480
    assert counts["initial"] == counts["synthesis"] == counts["judge_e"] == 0
    assert set(Counter(tuple(p["omitted_measurement_stratum"]) for p in plans).values()) == {6, 7}


def test_no_third_member_other_trajectory_or_measurement_leakage(fixture):
    manifest, contexts = fixture
    plan = manifest["plans"][0]
    graph = DyadicGraph(contexts[plan["question_id"]], plan, manifest)
    values = {}
    for tone in TONES:
        for n in graph.route.nodes:
            values[graph.key(f"{tone}/debate/{n.id}")] = {
                "reply": f"PUBLIC::{tone}::{n.id}::END",
                "agreement": "fully_disagree",
            }
    for reading in plan["readings"]:
        values[graph.key(f"C/{reading['id']}")] = {"choice": "A", "position": f"MEASUREMENT::{reading['id']}"}
    for task in graph.tasks.values():
        request = task.build(values, MockProvider().token_count)
        text = "\n".join(m.text for m in request.messages)
        assert request.effort == (
            "medium" if request.purpose == "debate" else "low" if request.purpose.startswith("judge_") else "none"
        )
        if request.purpose == "debate":
            tone, _, nid = task.key.split("/")[-3:]
            node = graph.route.get(nid)
            pair = nid.split("-")[0]
            assert f"INITIAL-MEMBER-{next(m for m in range(3) if m not in PAIRS[pair])}" not in text
            allowed = {n.id for n in graph.route.path(nid)[:-1]}
            for t in TONES:
                for n in graph.route.nodes:
                    if n.id not in allowed or t != tone:
                        assert f"PUBLIC::{t}::{n.id}::END" not in text
            assert "MEASUREMENT::" not in text
            assert '"agreement":"fully_disagree"' not in text
            assert len(task.dependencies) == (1 if node.parent else 0)
    for e in plan["events"]:
        if e["member"] == 0:
            continue
        a = graph.tasks[graph.key(f"Dtext/{e['id']}/argument")].build(values, MockProvider().token_count)
        b = graph.tasks[graph.key(f"Dtext/{e['id']}/control")].build(values, MockProvider().token_count)
        differences = [(x, y) for x, y in zip(a.messages, b.messages, strict=True) if x != y]
        assert len(differences) == 1 and prompts.FILLER_SENTENCE in differences[0][1].text
        text = "\n".join(m.text for m in a.messages)
        assert f"PUBLIC::{e['tone']}::{e['node_id']}::END" not in text


def test_gate_and_resume_have_no_duplicate_requests(fixture, tmp_path):
    manifest, contexts = fixture
    report = run(manifest, contexts, tmp_path / "run", question_limit=1, request_limit=16)
    assert report["status"] == "preflight_completed"
    assert report["trajectory_statuses"] == {"success": 9}
    qid = manifest["plans"][0]["question_id"]
    record = json.loads((tmp_path / "run" / "questions" / f"{qid}.json").read_text())
    assert len(record["events"]) == 8
    for e in record["events"]:
        history = record["formal_replies"][e["tone"]]
        node = make_route().get(e["node_id"])
        assert e["previous_peer_self_label"] == history[node.parent]["agreement"]
        assert e["current_self_label"] == history[node.id]["agreement"]
        assert (e["D_text"] is not None) == (e["member"] != 0)
        assert e["C_text"]["label"] == "unchanged"

    def forbidden():
        raise AssertionError("No completed request may be regenerated")

    resumed = run(manifest, contexts, tmp_path / "run", question_limit=1, provider_factory=forbidden)
    assert resumed["call_status_counts"] == report["call_status_counts"]
    assert resumed["charged_or_reserved_usd"] == report["charged_or_reserved_usd"]


def test_failure_does_not_stop_other_trajectories(fixture, tmp_path):
    manifest, contexts = fixture
    attempts = []
    qid = manifest["plans"][0]["question_id"]
    target = f"{qid}/{VERSION}/friendly/debate/AB-1"

    class Broken(MockProvider):
        def generate(self, request):
            if request.key == target:
                attempts.append(request.key)
                return Completion("INVALID", 100, 2)
            return super().generate(request)

    report = run(manifest, contexts, tmp_path / "failed", question_limit=1, provider_factory=Broken)
    assert len(attempts) == 3
    assert report["status"] == "completed_with_failures"
    assert report["trajectory_statuses"] == {"success": 8, "failed": 1}
    assert not report["fatal_errors"]
