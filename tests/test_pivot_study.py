"""Main-study scheduling, explicit uncapped accounting and exact pilot reuse; no network."""

import hashlib
import json
import threading
import time
from copy import deepcopy

import pytest

from llm_committee.pivot.__main__ import main
from llm_committee.pivot.continuation import load_checkpoint
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, PilotConfig, Question, Screen
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.questions import load_approved_bank
from llm_committee.pivot.runner import expected_counts, manifest_for, run_pilot
from llm_committee.pivot.storage import Journal, RunBlocked, RunStopped
from llm_committee.pivot.study import prepare_study, run_study


def setup_study(tmp_path, count=4):
    questions = tuple(
        Question(f"q{i}", f"Opinion topic {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(count)
    )
    config = PilotConfig(judge_model="gemini-3.8-flash")
    manifest = manifest_for(questions, config, mock=True)
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    manifest["study"] = {"scope": "approved_bank_mixed_family", "workers": 3, "question_bank": {"path": str(bank)}}
    manifest["execution"].update(budget_policy="no_limit_user_requested", spend_cap_usd=None)
    return questions, config, manifest


def test_uncapped_requires_explicit_policy(tmp_path):
    with pytest.raises(ValueError, match="no-limit"):
        Journal(tmp_path / "bad.sqlite3", {}, None)
    manifest = {"execution": {"budget_policy": "no_limit_user_requested"}}
    journal = Journal(tmp_path / "uncapped.sqlite3", manifest, None)
    assert journal.cap_usd is None
    journal.close()


def test_stop_prevents_new_reservations_but_allows_cached_reads(tmp_path):
    from llm_committee.pivot.models import Message, Request

    stop = threading.Event()
    journal = Journal(tmp_path / "stop.sqlite3", {}, 10, should_stop=stop.is_set)
    provider = MockProvider()
    request = Request("one", "initial", "gpt-5.6-terra", (Message("user", "test"),), "medium", 100)
    try:
        first = journal.call(request, provider, json.loads)
        stop.set()
        assert journal.call(request, provider, json.loads) == first
        with pytest.raises(RunStopped):
            journal.call(
                Request("two", "initial", request.model, request.messages, "medium", 100), provider, json.loads
            )
        assert len(provider.calls) == 1 and len(journal.audit()["calls"]) == 1
    finally:
        journal.close()


@pytest.mark.parametrize("workers", [3, 8])
def test_parallel_study_is_bounded_isolated_accounted_and_replayable(tmp_path, capsys, workers):
    questions, config, manifest = setup_study(tmp_path, count=workers + 1)
    manifest["study"]["workers"] = workers
    output = tmp_path / "parallel"
    lock = threading.Lock()
    active = peak = 0
    keys = []

    class Tracking(MockProvider):
        def generate(self, request):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
                keys.append(request.key)
            time.sleep(0.001)
            try:
                return super().generate(request)
            finally:
                with lock:
                    active -= 1

    result = run_study(questions, config, manifest, None, output, provider_factory=Tracking)
    assert result["status"] == "completed" and result["completed_questions"] == len(questions)
    assert 1 < peak <= workers
    assert len(keys) == len(set(keys)) == expected_counts(manifest)["logical_calls"]
    assert result["spend_cap_usd"] is None and result["new_charged_or_reserved_usd"] > 0
    baseline_manifest = manifest_for(questions, config, mock=True)
    journal = Journal(tmp_path / "serial.sqlite3", baseline_manifest, 1000)
    try:
        baseline = run_pilot(questions, config, baseline_manifest, journal, MockProvider())
    finally:
        journal.close()
    for q, expected in zip(questions, baseline["questions"], strict=True):
        actual = json.loads((output / "questions" / f"{q.id}.json").read_text())
        assert actual["questions"] == json.loads(json.dumps([expected]))

    def forbidden():
        raise AssertionError("Completed questions must replay without constructing providers")

    replay = run_study(questions, config, manifest, None, output, provider_factory=forbidden)
    assert replay["status"] == "completed"
    assert replay["new_charged_or_reserved_usd"] == result["new_charged_or_reserved_usd"]
    assert len(keys) == expected_counts(manifest)["logical_calls"]
    changed = deepcopy(manifest)
    changed["study"]["workers"] = 2
    with pytest.raises(RunBlocked, match="manifest changed"):
        run_study(questions, config, changed, None, output, provider_factory=forbidden)


def test_eight_worker_pool_refills_without_waiting_for_the_slowest_question(tmp_path):
    questions, config, manifest = setup_study(tmp_path, count=9)
    manifest["study"]["workers"] = 8
    replacement_started = threading.Event()

    class SlowFirstQuestion(MockProvider):
        def generate(self, request):
            if request.key.startswith("q8/"):
                replacement_started.set()
            if request.key == "q0/mixed_family/initial/0":
                # A batch scheduler deadlocks here until the timeout: q8 must enter
                # the pool while q0 still occupies one of the initial eight slots.
                assert replacement_started.wait(timeout=10), "Pool did not refill after another question finished"
            return super().generate(request)

    result = run_study(
        questions, config, manifest, None, tmp_path / "rolling", provider_factory=SlowFirstQuestion
    )
    assert replacement_started.is_set()
    assert result["status"] == "completed" and result["completed_questions"] == 9
    assert result["new_call_status_counts"] == {"completed": expected_counts(manifest)["logical_calls"]}


def test_error_stops_queued_work_and_no_automatic_retry(tmp_path):
    questions, config, manifest = setup_study(tmp_path)
    manifest["study"]["workers"] = 1
    calls = []

    class Broken(MockProvider):
        def generate(self, request):
            calls.append(request.key)
            return Completion("INVALID JSON", 100, 1)

    result = run_study(questions, config, manifest, None, tmp_path / "failed", provider_factory=Broken)
    assert result["status"] == "blocked" and len(calls) == 1
    with pytest.raises(RunBlocked, match="unresolved requests"):
        run_study(questions, config, manifest, None, tmp_path / "failed", provider_factory=Broken)
    assert len(calls) == 1


def test_existing_pilot_is_verified_offline_before_main_calls(tmp_path):
    questions, config, manifest = setup_study(tmp_path, count=3)
    pilot_manifest = manifest_for(questions[:1], config, mock=True)
    source = tmp_path / "pilot"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(pilot_manifest))
    journal = Journal(source / "requests.sqlite3", pilot_manifest, 100)
    try:
        run_pilot(questions[:1], config, pilot_manifest, journal, MockProvider())
        old_keys = {r["key"] for r in journal.audit()["calls"]}
    finally:
        journal.close()
    inherited = load_checkpoint(source, manifest, allow_question_expansion=True)
    manifest["continuation"] = inherited[0]
    keys = []

    class Tracking(MockProvider):
        def generate(self, request):
            keys.append(request.key)
            return super().generate(request)

    result = run_study(questions, config, manifest, inherited, tmp_path / "study", provider_factory=Tracking)
    assert result["completed_questions"] == 3 and result["reused_checkpoint_records"] == len(old_keys)
    assert not old_keys.intersection(keys)
    changed = deepcopy(manifest)
    changed["plans"][0]["events"][0]["T"] += 1
    with pytest.raises(ValueError, match="existing plans"):
        load_checkpoint(source, changed, allow_question_expansion=True)
    changed = deepcopy(manifest)
    changed["kind"] = "paid_main_study"
    with pytest.raises(ValueError, match="synthetic"):
        load_checkpoint(source, changed, allow_question_expansion=True)


def fixture_bank(tmp_path):
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "per_question": [
                    {
                        "index": i,
                        "question": f"Topic {i}?",
                        "keep": True,
                        "keep_votes": 3,
                        "n_judges": 3,
                        "question_class": "debatable_opinion",
                    }
                    for i in range(60)
                ]
            }
        )
    )
    document = {
        "purpose": "main_study_question_bank",
        "status": "user_approved_question_set",
        "question_count": 60,
        "candidate_count": 60,
        "original_opinion_screen": {
            "screening_source": str(screen),
            "screening_sha256": hashlib.sha256(screen.read_bytes()).hexdigest(),
        },
        "final_selection": {"approval": {"source": "user"}},
        "questions": [
            {
                "review_number": i + 1,
                "archive_index": i,
                "question": f"Topic {i}?",
                "options": ["Yes", "No"],
                "opinion_votes": 3,
                "user_decision": "保留",
                "input_review_note": "fixture",
                "pilot_question": False,
            }
            for i in range(60)
        ],
    }
    bank = tmp_path / "approved.json"
    bank.write_text(json.dumps(document))
    return bank, document


def test_partial_main_continuation_reuses_calls_without_claiming_all_questions_complete(tmp_path):
    questions, config, manifest = setup_study(tmp_path, count=3)
    manifest["study"]["workers"] = 1
    stop = threading.Event()
    source = tmp_path / "partial-main"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(source / "requests.sqlite3", manifest, None, should_stop=stop.is_set)

    class StopDuringSecondQuestion(MockProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.key.startswith("q1/"):
                stop.set()
            return result

    try:
        with pytest.raises(RunStopped):
            run_pilot(questions, config, manifest, journal, StopDuringSecondQuestion())
        old = journal.audit()
        old_keys = {r["key"] for r in old["calls"]}
    finally:
        journal.close()
    inherited = load_checkpoint(source, manifest, allow_question_expansion=True)
    assert inherited[0]["study_extension"]["reused_question_ids"] == ["q0", "q1"]
    target = deepcopy(manifest)
    target["study"]["workers"] = 8
    target["continuation"] = inherited[0]
    new_keys = []

    class Tracking(MockProvider):
        def generate(self, request):
            new_keys.append(request.key)
            return super().generate(request)

    result = run_study(questions, config, target, inherited, tmp_path / "continued", provider_factory=Tracking)
    assert result["status"] == "completed" and result["completed_questions"] == 3
    assert result["question_reports"]["q0"]["replayed"] is True
    assert result["question_reports"]["q1"]["replayed"] is False
    assert not old_keys.intersection(new_keys)
    assert len(old_keys) + len(new_keys) == expected_counts(manifest)["logical_calls"]
    assert result["prior_charged_or_reserved_usd"] == old["charged_or_reserved_usd"]
    assert result["spend_cap_usd"] is None
    journal = Journal(source / "requests.sqlite3", manifest, None)
    try:
        assert journal.audit() == old
    finally:
        journal.close()


def test_main_bank_requires_approved_exact_60_with_unchanged_evidence(tmp_path):
    bank, document = fixture_bank(tmp_path)
    questions, provenance = load_approved_bank(bank)
    assert len(questions) == 60 and provenance["items"][0]["review_number"] == 1
    assert questions[0].id == "archived-global-0" and questions[0].options == ("Yes", "No")
    prepared = prepare_study(bank, no_budget_limit=True)
    assert prepared[2]["kind"] == "paid_main_study"
    assert prepared[2]["execution"]["spend_cap_usd"] is None
    with pytest.raises(ValueError, match="spending cap"):
        prepare_study(bank)
    document["questions"][0]["user_decision"] = "待审核"
    bank.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="approved"):
        load_approved_bank(bank)
    document["questions"][0]["user_decision"] = "保留"
    document["questions"].pop()
    bank.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="60-question"):
        load_approved_bank(bank)


def test_study_cli_requires_spending_policy_before_live_execution():
    with pytest.raises(SystemExit):
        main(["study-run", "--bank", "unused.json", "--output", "unused"])
