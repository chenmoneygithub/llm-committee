"""No-network tests for the new protocol, including restart/budget/branch invariants."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from llm_committee.pivot import PilotConfig, Question, Screen, prompts
from llm_committee.pivot.__main__ import main
from llm_committee.pivot.models import AGREEMENT, SCREEN_JUDGES, TONES, Completion, Message, Request, canonical
from llm_committee.pivot.planning import make_route, plan_question, route_from_plan
from llm_committee.pivot.providers import GeminiProvider, MockProvider, OpenAIProvider, cost_usd, openai_payload
from llm_committee.pivot.questions import import_screened_archive, load_questions
from llm_committee.pivot.runner import expected_counts, manifest_for, run_pilot
from llm_committee.pivot.storage import BudgetExceeded, Journal, RunBlocked


@pytest.fixture
def question():
    return Question(
        "fixture",
        "Should public parks remain free to enter?",
        ("Yes", "No"),
        Screen(SCREEN_JUDGES, 2, "synthetic-test-only", "0" * 64),
    )


@pytest.fixture
def config():
    return PilotConfig(roster="same_family", judge_model="gemini-3.8-flash")


def test_screening_does_not_accept_a_single_judge(question):
    with pytest.raises(ValueError, match="at least two"):
        replace(question, screen=replace(question.screen, opinion_votes=1)).validate()


def test_debate_cannot_disable_reasoning():
    with pytest.raises(ValueError, match="reasoning enabled"):
        PilotConfig(debate_effort="none").validate()


@pytest.mark.parametrize("seed", range(100))
def test_route_and_sampling_invariants(question, config, seed):
    plan = plan_question(question, replace(config, seed=seed))
    route = route_from_plan(plan)
    assert len(route.leaves) == 6
    assert 15 <= len(route.nodes) <= 24
    assert all(n.depth == 5 for n in route.leaves)
    assert len(plan["events"]) == 8
    assert len({e["id"] for e in plan["events"]}) == 8
    assert {e["T"] for e in plan["events"]} == {1, 2, 3, 4, 5}
    assert len({r["id"] for r in plan["readings"]}) == len(plan["readings"])
    assert len({p["id"] for p in plan["text_pairs"]}) == len(plan["text_pairs"])
    for event in plan["events"]:
        node = route.get(event["node_id"])
        own_previous = next((n for n in reversed(route.path(node.id)[:-1]) if n.receiver == node.receiver), None)
        assert event["previous_reading"] == (
            f"{event['tone']}/{own_previous.id}" if own_previous else f"initial/{node.receiver}"
        )
        assert event["participation_index"] == route.participation(node.id)
        n_pool = sum(n.depth == node.depth for n in route.nodes) * 3
        w = 1.5 if node.depth == 5 else 1
        assert event["inclusion_probability"] == pytest.approx((1 + 3 * w / 5.5) / n_pool)
        assert bool(event["final_pair"]) == bool(event["final_trajectories"])


def test_route_is_reused_across_rosters_but_not_forced_equal_across_questions(question, config):
    a = plan_question(question, config)
    b = plan_question(question, replace(config, roster="same_model"))
    assert a["route"] == b["route"]
    assert a == plan_question(question, config)
    shapes = {
        canonical(make_route(replace(question, text=f"Question {i}?"), config.seed).nodes[0].__dict__)
        for i in range(10)
    }
    assert len(shapes) > 1


def test_branch_context_has_no_siblings_or_global_latest_position(question, config):
    route = make_route(question, config.seed)
    initial = {m: f"INITIAL-{m}" for m in range(3)}
    replies = {n.id: {"reply": f"UNIQUE-{n.id}", "agreement": "fully_disagree"} for n in route.nodes}
    for node in route.nodes:
        messages = prompts.debate_messages(question, "hostile", initial, route, replies, node.id)
        text = canonical([m.text for m in messages])
        ancestors = {n.id for n in route.path(node.id)[:-1]}
        for other in route.nodes:
            assert (f"UNIQUE-{other.id}" in text) == (other.id in ancestors)
        assert messages[-1].text == prompts.DEBATE_OUTPUT_CONTRACT
        tail = json.loads(messages[-2].text)
        previous = route.previous_own(node.id)
        assert tail["your_latest_own_contribution_on_this_branch"] == (
            replies[previous.id]["reply"] if previous else initial[node.receiver]
        )


def test_debate_output_contract_is_identical_for_all_members_and_tones(question, config):
    route = make_route(question, config.seed)
    initial = {m: f"INITIAL-{m}" for m in range(3)}
    replies = {n.id: {"reply": f"REPLY-{n.id}", "agreement": "fully_agree"} for n in route.nodes}
    for tone in TONES:
        for node in route.nodes:
            messages = prompts.debate_messages(question, tone, initial, route, replies, node.id)
            assert messages[-1] == Message("user", prompts.DEBATE_OUTPUT_CONTRACT)
            assert canonical(prompts.REPLY_SCHEMA) in messages[-1].text
    plan = plan_question(question, config)
    for reading in plan["readings"]:
        assert prompts.DEBATE_OUTPUT_CONTRACT not in canonical(
            [m.text for m in prompts.position_messages(question, reading, initial, route, replies)]
        )


@pytest.mark.parametrize("text", [
    "Fully agreed. A substantive ordinary-text reply.",
    '```json\n{"reply":"x","agreement":"fully_agree"}\n```',
    '{"reply":"x","agreement":"fully_agree","agreement":"fully_disagree"}',
    '{"reply":"x","agreement":"Fully agreed"}',
    '{"reply":"x","reply":"y","agreement":"fully_agree"}',
    '{"reply":"x","agreement":"fully_agree","extra":1}',
    '{"reply":"x","agreement":"unknown","agreement":"unknown"}',
    '{"reply":"x","agreement":true,"agreement":1}',
    '{"reply":null,"agreement":"fully_agree","reply":null}',
    '{"agreement":"fully_agree","agreement":"fully_agree"}',
])
def test_invalid_reply_formats_are_rejected_without_label_inference(text):
    with pytest.raises(ValueError):
        prompts.parse_json(text, prompts.REPLY_SCHEMA)


def test_identical_duplicate_fields_are_tolerated_without_changing_values():
    assert prompts.parse_json(
        '{"agreement":"leaning_agree","reply":"Original reply",'
        '"agreement":"leaning_agree","reply":"Original reply"}',
        prompts.REPLY_SCHEMA,
    ) == {"agreement": "leaning_agree", "reply": "Original reply"}


def test_measurement_includes_current_reply_and_preserves_tone(question, config):
    plan = plan_question(question, config)
    route = route_from_plan(plan)
    initial = {m: f"INITIAL-{m}" for m in range(3)}
    replies = {n.id: {"reply": f"UNIQUE-{n.id}", "agreement": "fully_agree"} for n in route.nodes}
    for reading in plan["readings"]:
        messages = prompts.position_messages(question, reading, initial, route, replies)
        if reading["node_id"]:
            assert f"UNIQUE-{reading['node_id']}" in canonical([m.text for m in messages])
        else:
            assert "UNIQUE-" not in canonical([m.text for m in messages])
        if reading["tone"] == "hostile":
            assert prompts.TONE_TEXT["hostile"] in messages[0].text


def test_missing_labels_are_not_defaulted_and_position_is_choice_first(question):
    with pytest.raises(ValueError):
        prompts.parse_json('{"reply":"hello"}', prompts.REPLY_SCHEMA)
    with pytest.raises(ValueError):
        prompts.parse_position("Let me think.\nA\nMy position", question)
    with pytest.raises(ValueError):
        prompts.parse_position("A\n<think>secret</think>\nMy position", question)
    with pytest.raises(ValueError, match="full position"):
        prompts.parse_position("A", question)
    assert prompts.parse_position("B\nMy full position", question)["choice"] == "B"
    assert prompts.parse_position("UNJUDGEABLE\nNo applicable position", question)["choice"] is None


def test_end_to_end_and_resume_without_paid_or_mock_replay(tmp_path, question, config):
    manifest = manifest_for((question,), config, mock=True)
    journal = Journal(tmp_path / "run.sqlite3", manifest, 1000)
    provider = MockProvider()
    try:
        result = run_pilot((question,), config, manifest, journal, provider)
        assert len(provider.calls) == expected_counts(manifest)["logical_calls"]
        q = result["questions"][0]
        assert q["D"]["status"] == "not_applicable"
        assert len(q["sampled_events"]) == 8
        for request in provider.calls:
            if request.purpose in ("debate", "initial"):
                assert request.effort != "none"
            if request.purpose == "position":
                assert request.effort == "none" and request.schema is None
            if request.purpose in ("debate", "synthesis"):
                assert "SYNTHETIC position " not in canonical(request.document())
        for tone in TONES:
            assert set(q["A"][tone]["counts"]) == set(AGREEMENT)
            e = q["E"]["preferences"][tone]
            assert e["orders"][0]["debate_side"] != e["orders"][1]["debate_side"]
            assert e["order_inconsistent"] == (e["debate_score"] == 0.5)
            assert len(q["positions"][tone]) == 6
            for members in q["positions"][tone].values():
                for sequence in members.values():
                    assert [s["time_index"] for s in sequence] == sorted({s["time_index"] for s in sequence})
                    assert all(s["reading_id"] in q["C_readings"] for s in sequence)
        count = len(provider.calls)
        repeated = run_pilot((question,), config, manifest, journal, provider)
        assert repeated == result
        assert len(provider.calls) == count
        assert all(c["status"] == "completed" for c in journal.audit()["calls"])
    finally:
        journal.close()


def small_request(**changes):
    request = Request(
        "test", "initial", "gpt-5.6-terra", (Message("user", "hello"),), "medium", 100, prompts.INITIAL_SCHEMA
    )
    return replace(request, **changes)


def test_budget_reserves_before_dispatch(tmp_path):
    journal = Journal(tmp_path / "run.sqlite3", {}, 0.001)
    provider = MockProvider()
    try:
        with pytest.raises(BudgetExceeded):
            journal.call(small_request(), provider, json.loads)
        assert provider.calls == [] and journal.charged_usd == 0
    finally:
        journal.close()


def test_uncertain_request_never_retries(tmp_path):
    class TimeoutProvider:
        count = 0

        def generate(self, request):
            self.count += 1
            raise TimeoutError("Possibly already billed")

    provider = TimeoutProvider()
    journal = Journal(tmp_path / "run.sqlite3", {}, 100)
    try:
        for _ in range(2):
            with pytest.raises(RunBlocked):
                journal.call(small_request(), provider, json.loads)
        assert provider.count == 1
        assert journal.charged_usd > 0
        assert journal.audit()["calls"][0]["status"] == "uncertain"
    finally:
        journal.close()


def test_direct_read_with_reasoning_is_saved_as_invalid(tmp_path):
    class ThinkingProvider:
        def generate(self, request):
            return Completion("A\nPosition", 100, 20, reasoning_tokens=10)

    journal = Journal(tmp_path / "run.sqlite3", {}, 100)
    try:
        with pytest.raises(RunBlocked, match="Invalid response"):
            journal.call(
                small_request(purpose="position", effort="none", schema=None), ThinkingProvider(), lambda t: {}
            )
        assert journal.audit()["calls"][0]["response"]["text"] == "A\nPosition"
        assert journal.audit()["calls"][0]["status"] == "invalid"
    finally:
        journal.close()


def test_resume_rejects_configuration_changes(tmp_path):
    journal = Journal(tmp_path / "run.sqlite3", {"config": 1}, 100)
    journal.close()
    with pytest.raises(ValueError, match="different manifest"):
        Journal(tmp_path / "run.sqlite3", {"config": 2}, 100)
    with pytest.raises(ValueError, match="different manifest"):
        Journal(tmp_path / "run.sqlite3", {"config": 1}, 101)


def test_duplicate_pending_dispatch_is_blocked_across_connections(tmp_path):
    one = Journal(tmp_path / "run.sqlite3", {}, 100)
    two = Journal(tmp_path / "run.sqlite3", {}, 100)
    request = small_request()
    mocked = MockProvider()

    class InterleavedProvider:
        def generate(self, incoming):
            with pytest.raises(RunBlocked, match="pending"):
                two.call(incoming, mocked, json.loads)
            return mocked.generate(incoming)

    try:
        one.call(request, InterleavedProvider(), json.loads)
        assert len(mocked.calls) == 1
    finally:
        one.close()
        two.close()


def test_cache_write_charges_and_reasoning_are_counted():
    c = Completion("x", 1000, 500, cached_tokens=200, cache_write_tokens=300, reasoning_tokens=450)
    assert cost_usd("gpt-5.6-terra", c) == pytest.approx((500 * 2 + 200 * 0.2 + 300 * 2 * 1.25 + 500 * 12) / 1e6)
    with pytest.raises(ValueError):
        cost_usd("gpt-5.6-terra", replace(c, cached_tokens=1001))


def test_openai_payload_has_no_global_state_no_tools_and_explicit_reasoning():
    request = small_request(messages=tuple(Message("user", str(i)) for i in range(8)), cache=True)
    payload = openai_payload(request)
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False and payload["tools"] == []
    assert "previous_response_id" not in payload and "temperature" not in payload
    assert payload["truncation"] == "disabled"
    assert sum("prompt_cache_breakpoint" in m["content"][0] for m in payload["input"]) == 4
    assert "prompt_cache_breakpoint" not in payload["input"][-1]["content"][0]


def test_judge_builders_do_not_add_condition_identity_or_self_labels(question):
    from llm_committee.pivot.agreement import AGREEMENT_RUBRIC_TEXT

    # A generic reminder that hostile wording is not disagreement is part of the
    # same rubric for every item, not disclosure of this item's assigned tone.
    for messages in (
        prompts.b_messages(question, "peer", "reply"),
        prompts.c_messages(question, "before", "after"),
        prompts.e_messages(question, "left answer", "right answer"),
    ):
        text = canonical([m.text.replace(AGREEMENT_RUBRIC_TEXT, "") for m in messages])
        assert "gpt-5.6" not in text and "hostile" not in text and "friendly" not in text
        item = json.loads(messages[-1].text)
        assert not {"tone", "agreement", "self_label", "model", "member"} & item.keys()
    assert "choice" not in json.loads(prompts.c_messages(question, "before", "after")[-1].text)


def test_chairman_sees_shared_nodes_once_and_no_measurements(question, config):
    route = make_route(question, config.seed)
    initial = {m: f"original-{m}" for m in range(3)}
    replies = {n.id: {"reply": n.id, "agreement": "fully_agree"} for n in route.nodes}
    baseline = prompts.synthesis_messages(question, initial, route, None)
    debate = prompts.synthesis_messages(question, initial, route, replies)
    b, d = json.loads(baseline[-1].text), json.loads(debate[-1].text)
    assert baseline[0] == debate[0]
    assert b["initial_answers"] == d["initial_answers"]
    assert b["discussion"] == []
    assert len(d["discussion"]) == len(route.nodes)
    assert len({n["id"] for n in d["discussion"]}) == len(route.nodes)


def test_cli_live_requires_explicit_budget():
    with pytest.raises(SystemExit):
        main(["run", "--questions", "missing.json", "--output", "unused"])


def test_cli_dry_run_is_resumable_and_does_not_construct_live_clients(tmp_path, question, monkeypatch, capsys):
    import llm_committee.pivot.__main__ as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline command attempted to instantiate a paid provider")

    monkeypatch.setattr(cli, "LiveProviders", forbidden)
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            {
                "purpose": "engineering_pilot",
                "questions": [asdict(question)],
                "reserved_pilot_fingerprints": [question.fingerprint],
            }
        )
    )
    assert main(["plan", "--questions", str(source)]) == 0
    argv = [
        "dry-run",
        "--roster",
        "same_family",
        "--questions",
        str(source),
        "--output",
        str(tmp_path / "dry"),
        "--judge-model",
        "gemini-3.8-flash",
    ]
    assert main(argv) == 0
    first = (tmp_path / "dry" / "report.json").read_text()
    assert main(argv) == 0
    assert (tmp_path / "dry" / "report.json").read_text() == first
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["api_spend_usd"] == 0


def test_archive_import_retains_options_and_filters_by_votes_not_confidence(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    rows = []
    for i in range(4):
        rows.append(
            {
                "index": i,
                "keep": i != 3,
                "question_class": "debatable_opinion",
                "keep_votes": 2 if i != 3 else 1,
                "n_judges": 3,
                "question": f"Question {i}",
                "mean_confidence": 0,
            }
        )
        (records / f"{i}.json").write_text(
            json.dumps(
                {
                    "benchmark": "global_opinions",
                    "index": i,
                    "question": f"Question {i} full text?",
                    "options": ["Yes", "No"],
                    "grade": "DO NOT USE",
                }
            )
        )
    screen = tmp_path / "screen.json"
    screen.write_text(json.dumps({"jury": ["gpt-5.5", "opus-4.8", "gemini"], "per_question": rows}))
    document = import_screened_archive(screen, records, count=3, seed=1)
    assert {q["id"] for q in document["questions"]} == {"archived-global-0", "archived-global-1", "archived-global-2"}
    assert len(document["question_fingerprints"]) == 3
    assert document["main_study_reuse"]["pilot_questions_may_be_in_main_80"] is True
    assert document["main_study_reuse"]["reuse_results_only_under_final_protocol"] is True
    assert "reserved_pilot_fingerprints" not in document
    assert "main_study_rule" not in document
    assert all(q["options"] == ("Yes", "No") for q in document["questions"])
    assert "DO NOT USE" not in canonical(document)


@pytest.mark.parametrize("field", ["question_fingerprints", "reserved_pilot_fingerprints"])
def test_current_and_legacy_question_manifests_remain_readable(tmp_path, question, field):
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            {
                "purpose": "engineering_pilot",
                "questions": [asdict(question)],
                field: [question.fingerprint],
            }
        )
    )
    assert load_questions(source) == (question,)


def test_conflicting_fingerprint_fields_are_rejected(tmp_path, question):
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            {
                "purpose": "engineering_pilot",
                "questions": [asdict(question)],
                "question_fingerprints": [question.fingerprint],
                "reserved_pilot_fingerprints": ["wrong"],
            }
        )
    )
    with pytest.raises(ValueError, match="Conflicting"):
        load_questions(source)


def test_mixed_plan_includes_both_d_readouts_and_keeps_shared_sample(question):
    config = PilotConfig(judge_model="gemini-3.8-flash")
    assert config.roster == "mixed_family"
    manifest = manifest_for((question,), config, mock=True)
    plan = manifest["plans"][0]
    original_plan = plan_question(question, config)
    assert plan["events"] == original_plan["events"]
    assert plan["D"]["status"] == "planned"
    assert manifest["execution"]["implemented"] is True
    assert manifest["execution"]["live_verified"] is False
    assert manifest["main_study_reuse"]["pilot_questions_may_be_in_main_80"] is True
    assert all(not x.startswith("D:") for x in manifest["excluded"])
    open_events = [e for e in plan["events"] if e["member"] in (1, 2)]
    open_readings = [r for r in plan["readings"] if r["member"] in (1, 2)]
    assert [p["event_id"] for p in plan["D"]["text_pairs"]] == [e["id"] for e in open_events]
    assert plan["D"]["choice_reading_ids"] == [r["id"] for r in open_readings]
    for pair, event in zip(plan["D"]["text_pairs"], open_events, strict=True):
        assert pair["fixed_full_position_reading"] == event["previous_reading"]
        assert pair["arms"] == ["argument", "control"]
    counts = expected_counts(manifest)
    assert counts["D_calls"] == len(open_events) * 2
    assert counts["D_choice_readings"] == len(open_readings)
    assert counts["logical_calls_with_scoring_allowance"] == counts["logical_calls"]
    assert counts["D_choice_supplemental_scoring_allowance"] == 0
    assert counts["D_text_supplemental_scoring_allowance"] == 0


@pytest.mark.parametrize("command", ["dry-run", "run"])
def test_unsupported_roster_stops_before_provider_or_outputs(tmp_path, question, monkeypatch, capsys, command):
    import llm_committee.pivot.__main__ as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Unimplemented pilot tried to construct a provider")

    monkeypatch.setattr(cli, "LiveProviders", forbidden)
    monkeypatch.setattr(cli, "MockProvider", forbidden)
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            {
                "purpose": "engineering_pilot",
                "questions": [asdict(question)],
                "question_fingerprints": [question.fingerprint],
            }
        )
    )
    argv = [command, "--roster", "same_model", "--questions", str(source), "--output", str(tmp_path / "not_created")]
    if command == "run":
        argv += ["--approve-spend-usd", "30"]
    assert main(argv) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["api_spend_usd"] == 0 and "pilot entry point" in status["reason"]
    assert not (tmp_path / "not_created").exists()


def test_mixed_runner_includes_real_d_requests_and_resumes(tmp_path, question):
    config = PilotConfig()
    manifest = manifest_for((question,), config, mock=True)
    provider = MockProvider()
    journal = Journal(tmp_path / "mixed.sqlite3", manifest, 1000)
    try:
        result = run_pilot((question,), config, manifest, journal, provider)
        data = result["questions"][0]
        assert data["D"]["status"] == "synthetic"
        assert data["D"]["choice_readings"] and data["D"]["text_events"]
        assert len(provider.calls) == expected_counts(manifest)["logical_calls"]
        assert sum(r.purpose == "d_text" for r in provider.calls) == expected_counts(manifest)["D_calls"]
        for event in data["sampled_events"]:
            assert (event["D_text"] is not None) == (event["member"] in (1, 2))
            assert (event["D_choice_adjacent_pp"] is not None) == (event["member"] in (1, 2))
        first_calls = len(provider.calls)
        assert run_pilot((question,), config, manifest, journal, provider) == result
        assert len(provider.calls) == first_calls
    finally:
        journal.close()


def test_provider_adapters_parse_usage_without_network():
    from types import SimpleNamespace

    raw = {
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 100},
        },
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
    }
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(model_dump=lambda: raw)

    parsed = OpenAIProvider(SimpleNamespace(responses=SimpleNamespace(create=create))).generate(small_request())
    assert parsed.reasoning_tokens == 100 and parsed.output_tokens == 200
    assert parsed.cache_write_tokens == 20
    assert sent[0]["extra_body"]["prompt_cache_options"]["mode"] == "explicit"

    class HTTP:
        def post(self, url, **kwargs):
            assert kwargs["json"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "LOW"
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 80},
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "not final"},
                                    {"text": '{"label":"unchanged","evidence":"x"}'},
                                ]
                            },
                        }
                    ],
                },
            )

    parsed = GeminiProvider(HTTP()).generate(small_request(model="gemini-3.8-flash", schema=prompts.C_SCHEMA))
    assert parsed.output_tokens == 100 and parsed.reasoning_tokens == 80
    assert "not final" not in parsed.text
