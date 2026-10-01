"""Four-field formal generations, isolated state and independent probability reads."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from llm_committee.pivot import prompts, stateful_prompts, strong_prompts
from llm_committee.pivot.dyadic import make_route
from llm_committee.pivot.failures import retryable_format_record
from llm_committee.pivot.models import Question, canonical
from llm_committee.pivot.planning import Node, Route
from llm_committee.pivot.stateful_run import run
from llm_committee.pivot.stateful_study import VERSION, StatefulGraph, StatefulMockProvider, prepare, rating_metrics
from llm_committee.pivot.storage import RunBlocked
from llm_committee.pivot.strong_agreement import AGREEMENT

PLAN = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"


@pytest.fixture(scope="module")
def prepared():
    return prepare(PLAN, mock=True)


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
            "position": f"[FULL OWN POSITION {n.id}]",
        }
        for n in route.nodes
    }
    return initial, replies


def test_schema_and_archived_protocol_are_distinct(question):
    schema = stateful_prompts.reply_schema(question)
    assert schema["required"] == ["reply", "agreement", "choice", "position"]
    assert strong_prompts.REPLY_SCHEMA["required"] == ["reply", "agreement"]
    assert prompts.INITIAL_SCHEMA["required"] == ["position"]
    for agreement in (*AGREEMENT, None):
        for choice in (*question.labels, None):
            value = {"reply": "My response", "agreement": agreement, "choice": choice, "position": "My full position"}
            assert prompts.parse_json(canonical(value), schema) == value
    for patch in ({"position": ""}, {"choice": "INVALID"}, {"agreement": "fully_agree"}, {"extra": "field"}):
        with pytest.raises(ValueError):
            prompts.parse_json(canonical({**value, **patch}), schema)
    with pytest.raises(ValueError):
        prompts.parse_json(canonical({k: v for k, v in value.items() if k != "position"}), schema)
    duplicate = '{"reply":"r","agreement":"leaning_agree","choice":"A","position":"x","position":"y"}'
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prompts.parse_json(duplicate, schema)


@pytest.mark.parametrize("node_id", [n.id for n in make_route().nodes])
def test_prompt_exact_ancestry_immediate_peer_and_own_state(question, node_id):
    route = make_route()
    initial, replies = material(route)
    node = route.get(node_id)
    messages = stateful_prompts.debate_messages(question, "hostile", initial, route, replies, node_id)
    context = canonical([m.text for m in messages[:-1]])
    state = json.loads(messages[-3].text)["your_current_position"]
    previous = route.previous_own(node_id)
    assert state["position"] == (replies[previous.id] if previous else initial[node.receiver])["position"]
    assert state["source"] == (previous.id if previous else f"initial/{node.receiver}")
    peer = json.loads(messages[-2].text)["incoming_peer_message"]
    assert peer["member"] == node.sender
    assert peer["source"] == (node.parent or f"initial/{node.sender}")
    incoming = replies[node.parent]["reply"] if node.parent else initial[node.sender]["position"]
    assert peer["contribution"] == incoming
    assert context.count(incoming) == 1
    ancestor_ids = {n.id for n in route.path(node_id)[:-1]}
    for n in route.nodes:
        assert context.count(replies[n.id]["reply"]) == int(n.id in ancestor_ids)
        assert context.count(replies[n.id]["position"]) == int(previous is not None and n.id == previous.id)
    assert context.count(initial[node.receiver]["position"]) == 1
    assert "strongly_disagree" not in context  # Self-labels are private, not public transcript.
    assert prompts.TONE_TEXT["hostile"] not in context
    assert "may remain unchanged" in messages[-1].text
    assert "ONLY to incoming_peer_message" in messages[-1].text
    assert prompts.TONE_TEXT["hostile"] in messages[-1].text


def test_branch_specific_state_never_uses_latest_sibling(question):
    route = Route(
        (
            Node("root", None, 0, 1, 1),
            Node("trunk", "root", 1, 0, 2),
            Node("left", "trunk", 0, 1, 3),
            Node("right", "trunk", 0, 2, 3),
            Node("left-return", "left", 1, 0, 4),
            Node("right-return", "right", 2, 0, 4),
        )
    )
    initial, replies = material(route)
    for side in ("left", "right"):
        state = stateful_prompts.own_state(0, initial, route, replies, f"{side}-return")
        assert state["source"] == "trunk"
        text = canonical(
            [m.text for m in stateful_prompts.turn_context(question, initial, route, replies, f"{side}-return")]
        )
        other = "right" if side == "left" else "left"
        assert replies[other]["reply"] not in text
        assert replies[f"{other}-return"]["position"] not in text


@pytest.mark.parametrize("roster", ["mixed_family", "same_model", "same_family"])
def test_frozen_layout_all_rosters_and_D_scope(roster):
    manifest, contexts = prepare(PLAN, mock=True, roster=roster)
    assert manifest["protocol_version"] == VERSION
    counts = manifest["planned_counts"]["by_purpose"]
    assert counts["initial"] == 60 and counts["debate"] == 360
    assert counts["judge_b"] == 267 and counts["judge_c"] == 481
    assert counts.get("position", 0) == 0
    assert manifest["planned_counts"]["formal_position_records"] == 420
    assert manifest["design"]["D_enabled"] == (roster == "mixed_family")
    if roster == "mixed_family":
        assert counts["d_choice"] == 222 and counts["d_text"] == 286
    else:
        assert "d_choice" not in counts and "d_text" not in counts
    frozen = json.loads(PLAN.read_text())
    for item, plan in zip(frozen["plans"], manifest["plans"], strict=True):
        assert contexts[plan["question_id"]] == {"question": item["question"]}
        graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
        for arm in ("original", "alternate"):
            assert plan["arms"][arm]["tone_schedule"] == item[f"{arm}_tones"]
            assert [e["node_id"] for e in plan["arms"][arm]["events"]] == item["sampled_node_ids"]
        for n in graph.route.nodes:
            assert (
                graph.key("original", f"turn_level/debate/{n.id}")
                == graph.key("alternate", f"turn_level/debate/{n.id}")
            ) == (n.depth <= 2)
            assert (
                graph.key("original", f"Dchoice/turn_level/{n.id}")
                == graph.key("alternate", f"Dchoice/turn_level/{n.id}")
            ) == (n.depth <= 3)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, prepared):
    manifest, contexts = prepared
    output = tmp_path_factory.mktemp("stateful-offline")
    result = run(manifest, contexts, output, question_limit=1, workers=2, request_limit=8)
    assert result["status"] == "preflight_completed"
    assert result["successful_questions"] == 1
    assert result["trajectory_statuses"] == {"success": 6}
    return output, manifest, contexts


def records(completed):
    output, manifest, contexts = completed
    plan = manifest["plans"][0]
    graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
    with sqlite3.connect(output / "requests.sqlite3") as db:
        rows = db.execute("SELECT rowid, request, parsed FROM calls ORDER BY rowid").fetchall()
    calls = {json.loads(req)["key"]: json.loads(req) for _, req, _ in rows}
    graph.values = {json.loads(req)["key"]: json.loads(value) for _, req, value in rows}
    graph.submitted = set(graph.values)
    record = json.loads((output / "questions" / f"{graph.question.id}.json").read_text())
    return graph, calls, record, {json.loads(req)["key"]: rowid for rowid, req, _ in rows}


def test_formal_four_fields_and_C_reads_actual_outputs(completed):
    graph, calls, record, _ = records(completed)
    assert "position" not in {r["purpose"] for r in calls.values()}
    for key, req in calls.items():
        value = graph.values[key]
        if req["purpose"] == "debate":
            assert set(value) == {"reply", "agreement", "choice", "position"}
            assert req["effort"] == "medium" and not req["candidate_labels"]
        if req["purpose"] == "initial":
            assert set(value) == {"choice", "position"}
            assert req["effort"] == "medium"
    for arm, branch in record["branches"].items():
        for e in branch["events"]:
            for field in ("position_before", "position_initial", "position_after"):
                p = e[field]
                assert p["position"] == graph.values[p["source_request"]]["position"]
                assert p["choice"] == graph.values[p["source_request"]]["choice"]
            judge = calls[graph.key(arm, f"Cjudge/{e['adjacent_pair']}")]
            texts = json.loads(judge["messages"][1]["text"])
            assert texts["before"] == e["position_before"]["position"]
            assert texts["after"] == e["position_after"]["position"]
            assert set(texts) == {"question", "options", "before", "after"}
            assert e["C_choice_changed"] == (e["position_before"]["choice"] != e["position_after"]["choice"])
    assert len(record["positions"]) == 6
    for states in record["positions"].values():
        assert len(states) == 2
        for entries in states.values():
            assert [e["participation_index"] for e in entries] == [0, 1, 2]
            assert len({e["source_request"] for e in entries}) == 3


def test_D_replays_pre_turn_inputs_only_and_cache_order(completed):
    graph, calls, record, rowids = records(completed)
    for arm, plan in graph.plan["arms"].items():
        for r in plan["readings"]:
            key = graph.key(arm, f"Dchoice/{r['id']}")
            if key not in calls:
                continue
            req = calls[key]
            formal = calls[graph.source_key(arm, r["id"])]
            assert req["effort"] == "none" and req["candidate_labels"] == list(graph.question.labels)
            assert req["messages"][:-1] == formal["messages"][:-1]
            assert graph.values[formal["key"]]["position"] not in canonical(req["messages"])
        for e in record["branches"][arm]["events"]:
            if not e["D_text"]:
                continue
            d1, a, c = [
                graph.key(arm, suffix)
                for suffix in (
                    f"Dchoice/{e['current_reading']}",
                    f"Dtext/{e['id']}/argument",
                    f"Dtext/{e['id']}/control",
                )
            ]
            assert calls[a]["effort"] == calls[c]["effort"] == "none"
            assert calls[d1]["messages"][:-1] == calls[a]["messages"][:-1]
            am, cm = calls[a]["messages"], calls[c]["messages"]
            assert [i for i, (x, y) in enumerate(zip(am, cm, strict=True)) if x != y] == [len(am) - 2]
            state = json.loads(am[-3]["text"])["your_current_position"]
            assert state["position"] == e["position_before"]["position"] == e["D_text"]["fixed_full_position"]
            assert e["position_after"]["position"] not in canonical(am)
            assert rowids[d1] < rowids[a] < rowids[c]
            assert a not in graph.tasks[c].dependencies
            assert d1 not in graph.tasks[a].dependencies
            assert graph.cache_after[c] == {a}


def test_measurement_outputs_do_not_affect_formal_or_probe_inputs(completed):
    graph, calls, _, _ = records(completed)
    provider = StatefulMockProvider()
    for key in graph.values:
        if graph.tasks[key].purpose not in ("initial", "debate"):
            graph.values[key] = {"_readout": "FORBIDDEN_MEASUREMENT", "position": "FORBIDDEN_MEASUREMENT"}
    for key, task in graph.tasks.items():
        if task.purpose in ("initial", "debate", "d_choice", "d_text"):
            rebuilt = task.build(graph.values, provider.token_count)
            assert canonical(rebuilt.document()) == canonical(calls[key])


@pytest.mark.parametrize("purpose", ["d_choice", "d_text", "debate"])
def test_failures_retry_without_poisoning_independent_work(tmp_path, prepared, purpose):
    manifest, contexts = prepared
    plan = manifest["plans"][0]
    graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
    node_id = next(e["node_id"] for e in plan["arms"]["original"]["events"] if e["member"] in (1, 2))
    suffix = {
        "d_choice": f"Dchoice/turn_level/{node_id}",
        "d_text": f"Dtext/turn_level/{node_id}/argument",
        "debate": "turn_level/debate/AB-3",
    }[purpose]
    arm = "alternate" if purpose == "debate" else "original"
    target = graph.key(arm, suffix)
    seen = []

    class Broken(StatefulMockProvider):
        def generate(self, req):
            seen.append(req.key)
            result = super().generate(req)
            return replace(result, text="INVALID") if req.key == target else result

    result = run(manifest, contexts, tmp_path / purpose, question_limit=1, request_limit=8, provider_factory=Broken)
    assert result["status"] == "completed_with_failures"
    assert seen.count(target) == 3
    assert not result["fatal_errors"]
    if purpose == "debate":
        assert result["trajectory_statuses"] == {"success": 5, "failed": 1}
        assert graph.key("original", "turn_level/debate/AB-4") in seen
        assert graph.key("alternate", "turn_level/debate/AB-4") not in seen
    else:
        assert result["trajectory_statuses"] == {"success": 6}
        assert graph.key("original", f"Dtext/turn_level/{node_id}/control") in seen
    old = list(seen)
    run(manifest, contexts, tmp_path / purpose, question_limit=1, provider_factory=Broken)
    assert seen == old  # Resume reuses successes and preserves exhausted failures.


def test_successful_resume_and_protocol_mismatch(completed):
    output, manifest, contexts = completed

    def forbidden_provider():
        raise AssertionError("A complete replay must not initialize any endpoint")

    result = run(manifest, contexts, output, question_limit=1, provider_factory=forbidden_provider)
    assert result["status"] == "preflight_completed"
    assert result["cache_usage"]
    assert sum(g["responses"] for g in result["cache_usage"]) == result["call_status_counts"]["completed"]
    assert json.loads((output / "progress.json").read_text())["cache_usage"] == result["cache_usage"]
    with pytest.raises(ValueError, match="archived"):
        run({**manifest, "protocol_version": "old"}, contexts, output)
    changed = {**manifest, "prompt_version": "changed"}
    with pytest.raises(RunBlocked, match="Frozen manifest"):
        run(changed, contexts, output)


def test_all_three_rating_metrics_and_ties():
    control = {"probabilities": dict(zip("ABCDEFG", [0, 0, 0, 0, 0, 0.1, 0.9], strict=True))}
    argument = {"probabilities": dict(zip("ABCDEFG", [0, 0, 0, 0, 0.2, 0.74, 0.06], strict=True))}
    metrics = rating_metrics(argument, control)
    assert metrics["modal_rating_control"] == 7 and metrics["modal_rating_argument"] == 6
    assert metrics["modal_rating_change"] == -1
    assert metrics["reference_category"] == "G"
    assert metrics["reference_probability_change_pp"] == pytest.approx(-84)
    assert metrics["mean_own_agreement_argument_minus_control"] == pytest.approx(5.86 - 6.9)
    tie = {"probabilities": dict(zip("ABCDEFG", [0, 0, 0, 0, 0, 0.5, 0.5], strict=True))}
    tied = rating_metrics(argument, tie)
    assert tied["modal_labels_control"] == ["F", "G"]
    assert tied["modal_rating_change"] is None and tied["reference_probability_change_pp"] is None


def test_choice_parse_is_a_single_original_option(question):
    assert stateful_prompts.parse_choice("A", question) == {"choice": "A"}
    for text in ("A\nNew opinion", "UNJUDGEABLE", "<think>reason</think>A", "AA", " A", "A "):
        with pytest.raises(ValueError, match="D-choice"):
            stateful_prompts.parse_choice(text, question)


def test_D1_format_and_reasoning_retries_do_not_hide_alignment_errors():
    req = {"model": "thinkingmachines/Inkling", "purpose": "d_choice", "effort": "none", "candidate_labels": ["A", "B"]}
    response = {
        "status": "invalid_native_output",
        "text": " A",
        "reasoning_tokens": 0,
        "raw": {
            "provider": "tinker",
            "model": req["model"],
            "effort": "none",
            "stop_reason": "stop",
            "parsed_message": {"role": "assistant"},
            "validation_error": "Direct output must begin with the requested letter, without whitespace",
        },
    }
    row = {"status": "invalid", "response": canonical(response), "error": "Response status: invalid_native_output"}
    assert retryable_format_record(req, row)
    # A valid one-letter response with broken alignment remains an integrity error.
    response["text"] = "A"
    response["raw"]["validation_error"] = "Cannot uniquely align parsed content with raw sampled tokens"
    assert not retryable_format_record(req, {**row, "response": canonical(response)})
    response["status"] = "unexpected_reasoning"
    response["reasoning_tokens"] = 12
    assert retryable_format_record(
        req, {**row, "response": canonical(response), "error": "Response status: unexpected_reasoning"}
    )


@pytest.mark.parametrize("roster", ["same_model", "same_family"])
def test_closed_rosters_execute_without_D_or_cross_member_state(tmp_path, roster):
    manifest, contexts = prepare(PLAN, mock=True, roster=roster)
    output = tmp_path / roster
    result = run(manifest, contexts, output, question_limit=1, request_limit=8)
    assert result["status"] == "preflight_completed"
    assert all(not row["purpose"].startswith("d_") for row in result["cache_usage"])
    graph, calls, _, _ = records((output, manifest, contexts))
    for arm in graph.plan["arms"]:
        for node in graph.route.nodes:
            req = calls[graph.key(arm, f"turn_level/debate/{node.id}")]
            state = json.loads(req["messages"][-3]["text"])["your_current_position"]
            previous = graph.route.previous_own(node.id)
            source = f"turn_level/{previous.id}" if previous else f"initial/{node.receiver}"
            assert state["position"] == graph.values[graph.source_key(arm, source)]["position"]
