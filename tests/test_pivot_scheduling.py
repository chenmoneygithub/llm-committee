"""Branch-ready scheduling changes order, never contexts, samples, outputs or call count."""

import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_committee.pivot.continuation import load_checkpoint
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, PilotConfig, Question, Screen, canonical
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.runner import expected_counts, manifest_for, run_pilot
from llm_committee.pivot.scheduling import run_ready_pool
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.study import run_study
from llm_committee.pivot.taskgraph import QuestionGraph, Task


def setup(tmp_path, count=4, *, requests=64, workers=8):
    questions = tuple(
        Question(f"q{i}", f"Policy topic {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(count)
    )
    config = PilotConfig(judge_model="gemini-3.8-flash")
    manifest = manifest_for(questions, config, mock=True)
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    manifest["study"] = {
        "scope": "approved_bank_mixed_family",
        "workers": workers,
        "max_inflight_requests": requests,
        "scheduler": "ready_nodes_v1",
        "question_bank": {"path": str(bank)},
    }
    manifest["execution"].update(budget_policy="no_limit_user_requested", spend_cap_usd=None)
    return questions, config, manifest


def test_parallel_graph_matches_every_serial_request_and_result(tmp_path):
    questions, config, manifest = setup(tmp_path, count=9)
    baseline = Journal(tmp_path / "serial.sqlite3", manifest, None)
    try:
        serial_provider = MockProvider()
        serial = run_pilot(questions, config, manifest, baseline, serial_provider)
        serial_requests = {r.key: canonical(r.document()) for r in serial_provider.calls}
    finally:
        baseline.close()
    requests = {}
    lock = threading.Lock()
    active = peak = 0

    class Tracking(MockProvider):
        def generate(self, request):
            nonlocal active, peak
            with lock:
                assert request.key not in requests, "Shared prefix was generated twice"
                requests[request.key] = canonical(request.document())
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.004)
                return super().generate(request)
            finally:
                with lock:
                    active -= 1

    output = tmp_path / "parallel"
    result = run_study(questions, config, manifest, None, output, provider_factory=Tracking)
    assert result["status"] == "completed" and result["completed_questions"] == len(questions)
    assert result["peak_active_questions"] == 8
    assert 3 < peak <= 64 and result["peak_in_flight_requests"] <= 64
    assert requests == serial_requests
    assert len(requests) == expected_counts(manifest)["logical_calls"]
    for question, expected in zip(questions, serial["questions"], strict=True):
        actual = json.loads((output / "questions" / f"{question.id}.json").read_text())
        assert actual["questions"] == json.loads(json.dumps([expected]))

    def forbidden():
        raise AssertionError("Completed run must replay without opening any provider")

    replay = run_study(questions, config, manifest, None, output, provider_factory=forbidden)
    assert replay["completed_questions"] == len(questions)
    assert replay["new_charged_or_reserved_usd"] == result["new_charged_or_reserved_usd"]


def test_global_pool_reaches_64_but_never_multiplies_cap_by_question():
    graphs = []
    for i in range(8):
        graph = object.__new__(QuestionGraph)
        graph.question = SimpleNamespace(id=f"q{i}")
        graph.values, graph.submitted = {}, set()
        graph.tasks = {}
        for j in range(12):
            key = f"q{i}/{j}"
            request = SimpleNamespace(key=key, candidate_labels=())
            graph.tasks[key] = Task(key, frozenset(), lambda v, c, r=request: r, json.loads)
        graphs.append(graph)
    gate = threading.Barrier(64, timeout=10)
    lock = threading.Lock()
    active = peak = calls = 0
    keys = set()

    def execute(request, parse):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            index = calls
            peak = max(peak, active)
            assert request.key not in keys
            keys.add(request.key)
        try:
            if index <= 64:
                gate.wait()
            return {"value": request.key}
        finally:
            with lock:
                active -= 1

    errors, done = [], []
    stats = run_ready_pool(
        graphs,
        question_limit=8,
        request_limit=64,
        token_count=None,
        execute=execute,
        stop=threading.Event(),
        on_start=lambda g: None,
        on_done=lambda g: done.append(g.question.id),
        on_error=lambda g, e: errors.append(e),
        on_progress=lambda s: None,
    )
    assert not errors and len(done) == 8 and calls == 96
    assert peak == stats["peak_in_flight_requests"] == 64


def test_slow_branch_does_not_hold_other_tones_or_new_questions(tmp_path):
    questions, config, manifest = setup(tmp_path, count=2, requests=16, workers=1)
    route = manifest["plans"][0]["route"]
    first_root = next(n["id"] for n in route if n["parent"] is None)
    held_key = f"q0/mixed_family/friendly/debate/{first_root}"
    independent_tone = threading.Event()
    calls = []
    lock = threading.Lock()

    class CheckDependencies(MockProvider):
        def generate(self, request):
            if request.key == held_key:
                assert independent_tone.wait(10), "Unrelated tone was forced to wait"
            if request.key.startswith("q0/mixed_family/hostile/debate/"):
                independent_tone.set()
            with lock:
                calls.append(request.key)
            return super().generate(request)

    result = run_study(questions, config, manifest, None, tmp_path / "tones", provider_factory=CheckDependencies)
    assert result["status"] == "completed" and result["completed_questions"] == 2
    assert independent_tone.is_set()
    assert len(calls) == len(set(calls)) == expected_counts(manifest)["logical_calls"]


def test_nonprefix_parallel_checkpoint_is_fully_verified_and_reused(tmp_path):
    questions, config, manifest = setup(tmp_path, count=3)
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(source / "requests.sqlite3", manifest, None)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    provider = MockProvider()
    try:
        # Reverse ready order creates a saved checkpoint that is NOT a serial-run prefix.
        for _ in range(18):
            task = next(
                t
                for t in reversed(list(graph.tasks.values()))
                if t.key not in graph.submitted and t.dependencies <= graph.values.keys()
            )
            request = task.build(graph.values, provider.token_count)
            graph.accept(task, request, journal.call(request, provider, task.parse))
        before = journal.audit()
    finally:
        journal.close()
    inherited = load_checkpoint(source, manifest, allow_question_expansion=True)
    target = deepcopy(manifest)
    target["continuation"] = inherited[0]
    old_keys = {r["key"] for r in before["calls"]}
    called = []

    class Tracking(MockProvider):
        def generate(self, request):
            assert request.key not in old_keys
            called.append(request.key)
            return super().generate(request)

    result = run_study(questions, config, target, inherited, tmp_path / "continued", provider_factory=Tracking)
    assert result["status"] == "completed"
    assert len(called) + len(old_keys) == expected_counts(manifest)["logical_calls"]
    assert result["prior_charged_or_reserved_usd"] == before["charged_or_reserved_usd"]
    assert result["spend_cap_usd"] is None
    journal = Journal(source / "requests.sqlite3", manifest, None)
    try:
        assert journal.audit() == before
    finally:
        journal.close()


def test_dag_question_pool_replenishes_while_an_earlier_question_is_still_running(tmp_path):
    questions, config, manifest = setup(tmp_path, count=9)
    ninth_started = threading.Event()

    class HoldFirstQuestion(MockProvider):
        def generate(self, request):
            if request.key.startswith("q8/"):
                ninth_started.set()
            if request.key == "q0/mixed_family/initial/0":
                assert ninth_started.wait(10), "Completed question slot was not refilled"
            return super().generate(request)

    result = run_study(questions, config, manifest, None, tmp_path / "refill", provider_factory=HoldFirstQuestion)
    assert ninth_started.is_set() and result["status"] == "completed"
    assert result["completed_questions"] == 9 and result["peak_active_questions"] == 8


def test_failure_does_not_release_descendants_or_retry(tmp_path):
    questions, config, manifest = setup(tmp_path, count=2)
    called = []

    class Broken(MockProvider):
        def generate(self, request):
            called.append(request.key)
            return Completion("INVALID JSON", 100, 2)

    output = tmp_path / "broken"
    result = run_study(questions, config, manifest, None, output, provider_factory=Broken)
    assert result["status"] == "blocked"
    assert len(called) == len(set(called)) <= 6
    assert all("/initial/" in key for key in called)
    assert "pending" not in result["new_call_status_counts"]
    with pytest.raises(RunBlocked, match="unresolved requests"):
        run_study(questions, config, manifest, None, output, provider_factory=Broken)


def test_graph_restoration_refuses_changed_request_before_provider_construction(tmp_path):
    questions, config, manifest = setup(tmp_path, count=1)
    journal = Journal(tmp_path / "saved.sqlite3", manifest, None)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    provider = MockProvider()
    try:
        task = graph.ready()
        request = task.build({}, provider.token_count)
        journal.call(request, provider, task.parse)
        journal.db.execute("UPDATE calls SET request_hash='changed'")
        with pytest.raises(RunBlocked, match="Request changed"):
            graph.restore(journal, OfflineProvider(provider.token_count))
        assert len(provider.calls) == 1
    finally:
        journal.close()
