"""Turn-specific private instructions must never contaminate neutral measurements/history."""

import json
from collections import Counter
from dataclasses import asdict

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.dyadic import PAIRS
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.models import SCREEN_JUDGES, TONES, Completion, PilotConfig, Question, Screen, digest
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.turn_tone import STREAM, VERSION, TurnToneGraph, plan_questions
from llm_committee.pivot.turn_tone_run import run


@pytest.fixture(scope="module")
def fixture():
    questions = [
        Question(f"q{i:02}", f"Policy topic {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(20)
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
        "source": {"directory": "/nonexistent/frozen-main-study", "contexts_sha256": digest(contexts)},
        "plans": plan_questions(list(contexts)),
        "planned_counts": {},
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
    }
    return manifest, contexts


def test_frozen_balanced_tones_and_small_pilot(fixture):
    manifest, contexts = fixture
    assert manifest["plans"] == plan_questions(list(reversed(contexts)))
    assignments, purposes = Counter(), Counter()
    for plan in manifest["plans"]:
        graph = TurnToneGraph(contexts[plan["question_id"]], plan, manifest)
        purposes.update(t.purpose for t in graph.tasks.values())
        assignments.update(plan["tone_schedule"].values())
        assert len(plan["events"]) == 8
        assert len({e["node_id"] for e in plan["events"]}) == 8
        for event in plan["events"]:
            node = graph.route.get(event["node_id"])
            assert event["T"] in (2, 3, 4)
            assert event["current_tone"] == plan["tone_schedule"][node.id]
            assert event["previous_tone"] == plan["tone_schedule"][node.parent]
    assert assignments == dict.fromkeys(TONES, 80)
    assert purposes["debate"] == 240 and purposes["judge_b"] == 160
    assert purposes["initial"] == purposes["synthesis"] == purposes["judge_e"] == 0


def test_only_current_private_tone_and_always_neutral_probes(fixture):
    manifest, contexts = fixture
    plan = manifest["plans"][0]
    graph = TurnToneGraph(contexts[plan["question_id"]], plan, manifest)
    values = {
        graph.key(f"{STREAM}/debate/{n.id}"): {"reply": f"PUBLIC-REPLY::{n.id}::END", "agreement": "fully_disagree"}
        for n in graph.route.nodes
    }
    values.update(
        {graph.key(f"C/{r['id']}"): {"choice": "A", "position": f"MEASUREMENT::{r['id']}"} for r in plan["readings"]}
    )
    for task in graph.tasks.values():
        request = task.build(values, MockProvider().token_count)
        text = "\n".join(m.text for m in request.messages)
        if request.purpose == "debate":
            node_id = task.key.split("/")[-1]
            tone = plan["tone_schedule"][node_id]
            pair = node_id.split("-")[0]
            third = next(m for m in range(3) if m not in PAIRS[pair])
            assert f"INITIAL-MEMBER-{third}" not in text
            assert request.messages[0].text == prompts.BASE + "\n" + prompts.TONE_TEXT[tone] + "\n" + prompts.REPLY_RULE
            history = "\n".join(m.text for m in request.messages[1:])
            for nonneutral in ("friendly", "hostile"):
                assert prompts.TONE_TEXT[nonneutral] not in history
            assert '"agreement":"fully_disagree"' not in history
            assert "MEASUREMENT::" not in text
            allowed = {n.id for n in graph.route.path(node_id)[:-1]}
            for n in graph.route.nodes:
                if n.id not in allowed:
                    assert f"PUBLIC-REPLY::{n.id}::END" not in text
        if request.purpose in ("position", "d_text"):
            assert request.effort == "none"
            assert request.messages[0].text == prompts.BASE + "\n"
            assert all(prompts.TONE_TEXT[t] not in text for t in ("friendly", "hostile"))
    for e in plan["events"]:
        if e["member"] == 0:
            continue
        a = graph.tasks[graph.key(f"Dtext/{e['id']}/argument")].build(values, MockProvider().token_count)
        b = graph.tasks[graph.key(f"Dtext/{e['id']}/control")].build(values, MockProvider().token_count)
        differences = [(x, y) for x, y in zip(a.messages, b.messages, strict=True) if x != y]
        assert len(differences) == 1 and prompts.FILLER_SENTENCE in differences[0][1].text
        assert f"PUBLIC-REPLY::{e['node_id']}::END" not in "\n".join(m.text for m in a.messages)


def test_one_question_gate_and_replay_preserve_labels_and_schedule(fixture, tmp_path):
    manifest, contexts = fixture
    report = run(manifest, contexts, tmp_path / "live", question_limit=1, request_limit=16)
    assert report["status"] == "preflight_completed" and report["trajectory_statuses"] == {"success": 3}
    plan = manifest["plans"][0]
    result = json.loads((tmp_path / "live" / "questions" / f"{plan['question_id']}.json").read_text())
    assert result["tone_schedule"] == plan["tone_schedule"]
    assert set(result["formal_replies"]) == {STREAM}
    assert len(result["formal_replies"][STREAM]) == 12
    for e in result["events"]:
        peer = result["formal_replies"][STREAM][f"{e['pair']}-{e['T'] - 1}"]
        current = result["formal_replies"][STREAM][e["node_id"]]
        assert e["previous_peer_self_label"] == peer["agreement"]
        assert e["current_self_label"] == current["agreement"]
        assert e["probe_tone"] == "neutral"
        assert (e["D_text"] is not None) == (e["member"] != 0)

    def forbidden():
        raise AssertionError("Completed requests cannot be regenerated")

    second = run(manifest, contexts, tmp_path / "live", question_limit=1, provider_factory=forbidden)
    assert second["call_status_counts"] == report["call_status_counts"]


def test_failure_isolated_to_one_pair_not_whole_question(fixture, tmp_path):
    manifest, contexts = fixture
    qid = manifest["plans"][0]["question_id"]
    target = f"{qid}/{VERSION}/{STREAM}/debate/AB-1"
    attempts = []

    class Broken(MockProvider):
        def generate(self, request):
            if request.key == target:
                attempts.append(request.key)
                return Completion("INVALID", 100, 2)
            return super().generate(request)

    report = run(manifest, contexts, tmp_path / "fail", question_limit=1, provider_factory=Broken)
    assert len(attempts) == 3 and not report["fatal_errors"]
    assert report["trajectory_statuses"] == {"success": 2, "failed": 1}


def test_tone_schedule_cannot_change_on_resume(fixture, tmp_path):
    manifest, contexts = fixture
    run(manifest, contexts, tmp_path / "freeze", question_limit=1)
    changed = json.loads(json.dumps(manifest))
    changed["plans"][0]["tone_schedule"]["AB-1"] = (
        "hostile" if changed["plans"][0]["tone_schedule"]["AB-1"] != "hostile" else "friendly"
    )
    with pytest.raises(RuntimeError, match="Frozen manifest"):
        run(changed, contexts, tmp_path / "freeze", question_limit=1)
