"""Current-protocol intervention boundaries, reusable controls and A–E references."""

import copy
import json
import sqlite3
from collections import Counter

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.quality_study import synthesis_messages
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.storage import Journal
from scripts.forced_feedback_public_history import (
    FORCE,
    QUALITY,
    FeedbackGraph,
    FeedbackMockProvider,
    prepare,
    request_identity,
    select_cases,
)
from scripts.run_forced_feedback_public_history import run
from scripts.scale_public_history import read


@pytest.fixture(scope="module")
def prepared():
    return prepare(mock=True)


def graph(prepared):
    manifest, contexts = prepared
    plan = manifest["plans"][0]
    return FeedbackGraph(contexts[plan["question_id"]], plan, manifest)


def fill(g):
    provider = FeedbackMockProvider()
    requests = {}
    while task := g.ready():
        req = task.build(g.values, provider.token_count)
        response = provider.generate(req)
        parsed = task.parse(response.text)
        if req.candidate_labels:
            parsed["_readout"] = response.readout
        g.accept(task, req, parsed)
        requests[req.key] = req
    assert g.successful
    return requests


def text(request):
    def strings(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return []

    result = []
    for message in request.messages:
        result.append(message.text)
        try:
            result.extend(strings(json.loads(message.text)))
        except ValueError:
            pass
    return "\n".join(result)


def test_three_hundred_events_not_six_hundred_interventions(prepared):
    manifest, _ = prepared
    cases = [c for p in manifest["plans"] for c in p["cases"]]
    assert len(cases) == 300 and len({c["question_id"] for c in cases}) == 50
    assert Counter(c["cut_T"] for c in cases) == {1: 150, 3: 150}
    assert Counter(c["pair"] for c in cases) == {"AB": 100, "CA": 100, "BC": 100}
    for pair in ("AB", "CA", "BC"):
        assert Counter(c["source_assignment"] for c in cases if c["pair"] == pair and c["cut_T"] == 3) == {
            "original": 25,
            "alternate": 25,
        }
    assert manifest["planned_counts"]["formal_replies"] == 600
    assert manifest["planned_counts"]["by_purpose_including_reuse"]["debate"] == 600
    assert select_cases(c["question_id"] for c in manifest["plans"]) == manifest["plans"]


def test_private_assignment_and_future_do_not_leak(prepared):
    g = graph(prepared)
    requests = fill(g)
    for case in g.plan["cases"]:
        cid = case["id"]
        challenger = requests[g.key(f"{cid}/forced/feedback")]
        recipient = requests[g.key(f"{cid}/forced/return")]
        assert FORCE in challenger.messages[-1].text and FORCE not in text(recipient)
        assert challenger.effort == recipient.effort == "medium"
        assert set(challenger.schema["properties"]) == {"reply", "agreement", "choice", "position"}
        assert "your_current_position" not in text(challenger)
        assert "your_current_position" not in text(recipient)
        assert g.values[g.key(f"{cid}/forced/feedback")]["reply"] in text(recipient)
        for nid, old in g.archived(case).items():
            if nid.startswith(case["pair"] + "-") and int(nid[-1]) >= case["cut_T"]:
                assert old["reply"] not in text(challenger)
                assert old["reply"] not in text(recipient)
        for arm in ("natural", "forced"):
            req = requests[g.measurements[cid]["arms"][arm]["E_answer"]]
            public = json.loads(req.messages[-1].text)
            assert len(public["discussion"]) == case["return_T"]
            assert {r["member"] for r in public["initial_answers"]} == {case["recipient"], case["challenger"]}
            assert all(set(r) == {"id", "member", "reply_to", "text"} for r in public["discussion"])
        for k, req in requests.items():
            if cid in k and req.purpose in ("judge_b", "judge_c", "judge_e", "d_choice", "d_text", "synthesis"):
                assert FORCE not in text(req)


def test_readouts_use_recipient_pre_turn_inputs_not_new_position(prepared):
    g = graph(prepared)
    requests = fill(g)
    for case in g.plan["cases"]:
        cid = case["id"]
        meta = g.measurements[cid]
        if case["recipient"] == 0:
            assert "D1_before" not in meta and "D1" not in meta["arms"]["forced"]
            continue
        returned = g.values[g.key(f"{cid}/forced/return")]
        for arm in ("natural", "forced"):
            keys = meta["arms"][arm]
            d1, argument, control = (requests[keys[k]] for k in ("D1", "D2_argument", "D2_control"))
            assert all(r.effort == "none" for r in (d1, argument, control))
            assert returned["position"] not in text(d1)
            assert returned["reply"] not in text(d1)
            assert "position_to_evaluate" not in text(d1)
            assert argument.messages[:-2] == control.messages[:-2]
            assert argument.messages[-1] == control.messages[-1]
            target = json.loads(argument.messages[-3].text)["position_to_evaluate"]["text"]
            assert target == g.before(case)["position"]
            assert prompts.FILLER_SENTENCE in control.messages[-2].text
    record = g.report(FeedbackMockProvider().token_count)
    assert len(record["cases"]) == 6
    assert all(c["status"] == "completed" for c in record["cases"])


def test_natural_late_synthesis_is_exact_existing_chairman_input(prepared):
    g = graph(prepared)
    old_context = read(QUALITY / "source-contexts.json")[g.question.id]
    for case in g.plan["cases"]:
        if case["cut_T"] == 3:
            assert g.synthesis(case, {}, "natural") == synthesis_messages(
                old_context, case["pair"], case["source_assignment"]
            )


def test_forced_wins_maps_target_side_in_both_orders(prepared):
    g = graph(prepared)
    fill(g)
    for target_votes in ((True, True), (True, False), (False, False)):
        for meta in g.measurements.values():
            for order, forced_wins in zip(meta["E_orders"], target_votes, strict=True):
                target = order["target_side"]
                preference = target if forced_wins else ("right" if target == "left" else "left")
                g.values[order["key"]] = {"preference": preference, "evidence": "Test mapping"}
        cases = g.report(FeedbackMockProvider().token_count)["cases"]
        assert all(c["E_forced_wins"] == sum(target_votes) / 2 for c in cases)
        assert all(c["E_order_inconsistent"] == (target_votes[0] != target_votes[1]) for c in cases)


def test_static_measurement_reuse_requires_exact_prompt_and_model(prepared, tmp_path):
    g = graph(prepared)
    key = next(k for k in g.eligible_reuse if k.endswith("/C"))
    request = g.tasks[key].build({}, FeedbackMockProvider().token_count)
    context = copy.deepcopy(g.context)
    context["archive"] = {
        request_identity(request.document()): {
            "value": {"label": "unchanged", "evidence": "existing"},
            "provenance": {"directory": "fixture", "storage_key": "original", "row_sha256": "fixture"},
        }
    }
    reused = FeedbackGraph(context, g.plan, g.manifest)
    journal = Journal(tmp_path / "requests.sqlite3", g.manifest, None, should_stop=lambda: True)
    reused.restore(journal, OfflineProvider(FeedbackMockProvider().token_count))
    assert key in reused.reused and reused.values[key]["evidence"] == "existing"
    assert journal.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 0
    assert request_identity({**request.document(), "model": "different"}) not in context["archive"]
    journal.close()


def test_mock_gate_resume_never_redispatches_success(tmp_path):
    calls = []

    class Provider(FeedbackMockProvider):
        def generate(self, request):
            calls.append(request.key)
            return super().generate(request)

    output = tmp_path / "mock"
    kwargs = {
        "mock": True,
        "questions": 1,
        "max_inflight": 12,
        "provider_factory": Provider,
        "token_count": FeedbackMockProvider().token_count,
    }
    report = run(output, **kwargs)
    assert report["status"] == "preflight_completed" and report["completed_cases"] == 6
    assert report["completed_new_calls_by_purpose"]["debate"] == 12
    first = list(calls)
    assert run(output, **kwargs)["completed_cases"] == 6
    assert calls == first
    with sqlite3.connect(output / "requests.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM calls").fetchone()[0] == len(first)
