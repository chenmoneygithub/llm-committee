"""Bounded format retries, exact-request recovery, branch isolation and missingness accounting."""

import json
import sqlite3
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from llm_committee.pivot.continuation import load_checkpoint, reuse_diagnostic_retry, validate_attempts
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY, LocalTaskFailure, retry_key, retryable_format_record
from llm_committee.pivot.models import Completion, Message, canonical, digest
from llm_committee.pivot.planning import route_from_plan
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.runner import expected_counts, run_pilot
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.study import run_study
from llm_committee.pivot.taskgraph import QuestionGraph
from tests.test_pivot_scheduling import setup


def configured(tmp_path, count=1):
    questions, config, manifest = setup(tmp_path, count=count)
    manifest["execution"]["failure_policy"] = FORMAT_RETRY_POLICY
    return questions, config, manifest


def rows_at(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY rowid")]


@pytest.mark.parametrize("malformed_count", [1, 2, 3])
def test_retry_journal_is_bounded_immutable_identical_and_first_valid(tmp_path, malformed_count):
    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task = graph.ready()
    recorded = []

    class Formats(MockProvider):
        def generate(self, request):
            recorded.append(canonical(request.document()))
            completion = super().generate(request)
            return replace(completion, text="invalid JSON") if len(recorded) <= malformed_count else completion

    provider = Formats()
    request = task.build({}, provider.token_count)
    path = tmp_path / "attempts.sqlite3"
    journal = Journal(path, manifest, None)
    try:
        if malformed_count == 3:
            with pytest.raises(LocalTaskFailure):
                journal.call(request, provider, task.parse)
        else:
            assert journal.call(request, provider, task.parse)["position"]
        before = rows_at(path)
        assert len(recorded) == min(3, malformed_count + 1)
        assert len(set(recorded)) == 1
        assert [r["key"] for r in before] == [retry_key(request.key, n) for n in range(len(before))]
        assert len({r["request_hash"] for r in before}) == 1
        assert before[0]["status"] == "invalid" and before[0]["parsed"] is None
        assert journal.charged_usd == sum(r["charge"] for r in before) > 0
        if malformed_count == 3:
            with pytest.raises(LocalTaskFailure):
                journal.call(request, OfflineProvider(provider.token_count), task.parse)
        else:
            journal.call(request, OfflineProvider(provider.token_count), task.parse)
        assert rows_at(path) == before
        assert len(recorded) == len(before)
    finally:
        journal.close()


def test_invalid_record_does_not_allow_changed_request_retry(tmp_path):
    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task = graph.ready()
    request = task.build({}, MockProvider().token_count)
    journal = Journal(tmp_path / "changed.sqlite3", manifest, None)
    try:

        class Broken(MockProvider):
            def generate(self, request):
                return Completion("bad", 10, 10)

        with pytest.raises(RunBlocked):
            journal._call_once(request, Broken(), task.parse)
        before = journal.audit()
        changed = replace(request, messages=(Message("user", "A different context"),))
        with pytest.raises(RunBlocked, match="Request changed"):
            journal.call(changed, OfflineProvider(None), task.parse)
        assert journal.audit() == before
    finally:
        journal.close()


@pytest.mark.parametrize("failure", ["transport", "billing", "reasoning", "probability"])
def test_nonformat_failures_do_not_retry(tmp_path, failure):
    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task = graph.ready()
    calls = []

    class Broken(MockProvider):
        def generate(self, request):
            calls.append(request.key)
            if failure == "transport":
                raise TimeoutError("uncertain")
            good = super().generate(request)
            if failure == "billing":
                return replace(good, input_tokens=-1)
            return replace(good, reasoning_tokens=1 if failure == "reasoning" else 0, readout=None)

    request = task.build({}, MockProvider().token_count)
    if failure in ("reasoning", "probability"):
        request = replace(request, effort="none", candidate_labels=("A", "B") if failure == "probability" else ())
    journal = Journal(tmp_path / "nonformat.sqlite3", manifest, None)
    try:
        with pytest.raises(RunBlocked):
            journal.call(request, Broken(), task.parse)
        assert len(calls) == 1
    finally:
        journal.close()


@pytest.mark.parametrize("target", ["debate", "C", "initial", "B", "Dtext", "E"])
def test_exhaustion_isolates_true_dependents_and_reports_all_trajectories(tmp_path, target):
    questions, config, manifest = configured(tmp_path, count=2)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    route = route_from_plan(graph.plan)
    if target == "debate":
        suffix = f"friendly/debate/{route.nodes[0].id}"
    elif target == "initial":
        suffix = "initial/0"
    elif target == "C":
        suffix = "C/initial/0"
    elif target == "B":
        suffix = next(k.split("/mixed_family/")[1] for k in graph.tasks if "/B/" in k)
    elif target == "Dtext":
        suffix = next(k.split("/mixed_family/")[1] for k in graph.tasks if "/Dtext/" in k)
    else:
        suffix = "E/friendly/order0"
    failed_key = f"q0/mixed_family/{suffix}"
    calls = Counter()
    lock = threading.Lock()

    class OneFailure(MockProvider):
        def generate(self, request):
            with lock:
                calls[request.key] += 1
            valid = super().generate(request)
            if request.key != failed_key:
                return valid
            return replace(valid, text="A" if target == "C" else "invalid")

    output = tmp_path / target
    result = run_study(questions, config, manifest, None, output, provider_factory=OneFailure)
    assert result["status"] == "completed_with_failures"
    assert result["completed_questions"] == 2
    assert result["fully_successful_questions"] == result["questions_with_missing_data"] == 1
    assert calls[failed_key] == 3 and all(n == 1 for k, n in calls.items() if k != failed_key)
    assert result["format_retries"] == {
        "additional_attempts": 2,
        "exhausted_logical_requests": 1,
        "recovered_logical_requests": 0,
    }
    assert result["trajectories"]["planned"] == 36 and result["trajectories"]["pending"] == 0
    assert sum(result["trajectories"][s] for s in ("success", "failed", "blocked")) == 36
    q0 = json.loads((output / "questions/q0.json").read_text())
    assert list(q0["task_failures"]) == [failed_key]
    assert not (q0["blocked_tasks"].keys() & calls.keys()), "Blocked tasks were dispatched"
    q1 = json.loads((output / "questions/q1.json").read_text())
    assert q1["status"] == "completed"
    if target in ("C", "B", "Dtext", "E"):
        assert result["trajectories"]["success"] == 36
        assert len(q0["questions"][0]["formal_replies"]["friendly"]) == len(route.nodes)
        if target != "E":
            assert all(p["debate_score"] is not None for p in q0["questions"][0]["E"]["preferences"].values())
    if target == "debate":
        affected = sum(route.nodes[0].id in [n.id for n in route.path(leaf.id)] for leaf in route.leaves)
        assert result["trajectories"]["failed"] == affected
        assert q0["questions"][0]["E"]["preferences"]["friendly"]["debate_score"] is None
        assert result["trajectories"]["blocked"] == 0
    if target == "initial":
        assert result["trajectories"]["blocked"] > 0 and result["trajectories"]["failed"] == 0
    if target == "C":
        assert any(e["C_choice_initial_changed"] is None for e in q0["questions"][0]["sampled_events"])
        assert q0["questions"][0]["D"]["status"] == "completed", "Terra C failure does not invalidate open-model D"
    if target == "E":
        assert q0["questions"][0]["E"]["preferences"]["friendly"]["debate_score"] is None
    before = rows_at(output / "requests.sqlite3")

    def forbidden():
        raise AssertionError("Terminal partial run must replay without a provider")

    replay = run_study(questions, config, manifest, None, output, provider_factory=forbidden)
    assert replay["trajectories"] == result["trajectories"]
    assert rows_at(output / "requests.sqlite3") == before
    inherited = load_checkpoint(output, manifest, allow_question_expansion=True)
    new_manifest = {**manifest, "continuation": inherited[0]}
    continued = run_study(
        questions, config, new_manifest, inherited, tmp_path / "continued", provider_factory=forbidden
    )
    assert continued["trajectories"] == result["trajectories"]
    assert continued["new_charged_or_reserved_usd"] == 0


def test_isolation_policy_preserves_no_failure_requests_and_reports(tmp_path):
    questions, config, manifest = configured(tmp_path, count=3)
    journal = Journal(tmp_path / "serial.sqlite3", manifest, None)
    try:
        provider = MockProvider()
        serial = run_pilot(questions, config, manifest, journal, provider)
        requests = {r.key: digest(r.document()) for r in provider.calls}
    finally:
        journal.close()
    output = tmp_path / "parallel"
    result = run_study(questions, config, manifest, None, output, provider_factory=MockProvider)
    assert result["status"] == "completed" and result["trajectories"]["success"] == 54
    assert {r["key"]: r["request_hash"] for r in rows_at(output / "requests.sqlite3")} == requests
    assert len(requests) == expected_counts(manifest)["logical_calls"]
    for q, expected in zip(questions, serial["questions"], strict=True):
        actual = json.loads((output / "questions" / f"{q.id}.json").read_text())["questions"]
        assert actual == json.loads(json.dumps([expected]))


def test_partly_attempted_checkpoint_resumes_remaining_retry_budget(tmp_path):
    questions, config, manifest = configured(tmp_path)
    output = tmp_path / "partial"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(output / "requests.sqlite3", manifest, None)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task = graph.ready()
    provider = MockProvider()
    request = task.build({}, provider.token_count)

    class Bad(MockProvider):
        def generate(self, request):
            return replace(super().generate(request), text="bad")

    try:
        for n in range(2):
            with pytest.raises(RunBlocked):
                journal._call_once(request, Bad(), task.parse, storage_key=retry_key(request.key, n))
    finally:
        journal.close()
    before = rows_at(output / "requests.sqlite3")
    result = run_study(questions, config, manifest, None, output, provider_factory=MockProvider)
    assert result["status"] == "completed"
    assert result["format_retries"]["additional_attempts"] == 2
    assert result["format_retries"]["recovered_logical_requests"] == 1
    assert rows_at(output / "requests.sqlite3")[:2] == before


def test_diagnostic_reuse_is_verified_readonly_and_charged_once(tmp_path):
    from scripts.investigate_position_read import investigate
    from tests.test_position_diagnostic import Responses, source_fixture

    source, request, manifest, _ = source_fixture(tmp_path)
    diagnostic = tmp_path / "diagnostic"
    result = investigate(
        source, request.key, diagnostic, retries=2, provider_factory=lambda: Responses(["A\nComplete position"])
    )
    rows = rows_at(source / "requests.sqlite3")
    descriptor = {
        "source_directory": str(source),
        "source_manifest_sha256": digest(manifest),
        "import_rows_sha256": digest(rows),
        "prior_charged_or_reserved_usd": sum(r["charge"] for r in rows),
    }
    inherited = reuse_diagnostic_retry(diagnostic, (descriptor, rows))
    assert rows_at(source / "requests.sqlite3") == rows
    assert (
        inherited[0]["prior_charged_or_reserved_usd"]
        == descriptor["prior_charged_or_reserved_usd"] + result["new_charged_or_reserved_usd"]
    )
    target_manifest = deepcopy(manifest)
    target_manifest["execution"]["failure_policy"] = FORMAT_RETRY_POLICY
    journal = Journal(tmp_path / "import.sqlite3", target_manifest, None)
    try:
        journal.import_checkpoint(*inherited)
        journal.import_checkpoint(*inherited)
        assert len(journal.audit()["calls"]) == 2 and journal.charged_usd == 0
        got = journal.call(request, OfflineProvider(None), lambda _: pytest.fail("Should reuse valid parsing"))
        assert got["position"] == "Complete position"
    finally:
        journal.close()
    with pytest.raises(ValueError, match="already imported"):
        reuse_diagnostic_retry(diagnostic, inherited)
    bad_rows = deepcopy(rows)
    bad_rows[0]["response"] = "tampered"
    with pytest.raises(ValueError, match="provenance"):
        reuse_diagnostic_retry(diagnostic, (descriptor, bad_rows))


def test_retry_sequence_rejects_success_resampling_and_gaps(tmp_path):
    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task, provider = graph.ready(), MockProvider()
    journal = Journal(tmp_path / "records.sqlite3", manifest, None)
    try:
        request = task.build({}, provider.token_count)
        journal.call(request, provider, task.parse)
        rows = rows_at(tmp_path / "records.sqlite3")
    finally:
        journal.close()
    extra = {**rows[0], "key": retry_key(request.key, 1)}
    with pytest.raises(ValueError, match="after success"):
        validate_attempts([*rows, extra], allow_retries=True)
    extra["key"] = retry_key(request.key, 2)
    with pytest.raises(ValueError, match="missing attempts"):
        validate_attempts([*rows, extra], allow_retries=True)


def test_exhausted_judge_transport_failure_is_missing_not_tie_or_global_stop(tmp_path):
    import httpx

    questions, config, manifest = configured(tmp_path, count=2)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    failed_key = next(k for k in graph.tasks if "/B/" in k)
    calls = Counter()

    class ConnectionFailure(MockProvider):
        def generate(self, request):
            calls[request.key] += 1
            if request.key == failed_key:
                raise httpx.ConnectError("test connection not established")
            return super().generate(request)

    output = tmp_path / "network"
    result = run_study(questions, config, manifest, None, output, provider_factory=ConnectionFailure)
    assert result["status"] == "completed_with_failures" and result["completed_questions"] == 2
    assert result["trajectories"]["success"] == 36
    assert result["transport_missing_requests"] == 1 and not result["format_retries"]["additional_attempts"]
    assert not result["format_retries"]["exhausted_logical_requests"]
    assert result["task_failures"][failed_key]["status"] == "failed_after_transport_retries"
    assert calls[failed_key] == 3, "Original plus exactly two additional attempts before skipping"
    assert result["transport_retries"] == {
        "additional_attempts": 2,
        "recovered_logical_requests": 0,
        "exhausted_logical_requests": 1,
    }
    rows = rows_at(output / "requests.sqlite3")
    failed = next(r for r in rows if r["key"] == failed_key)
    assert failed["status"] == "uncertain" and failed["charge"] > 0 and failed["response"] is None

    def forbidden():
        raise AssertionError("A recorded unavailable task must not be regenerated")

    inherited = load_checkpoint(output, manifest, allow_question_expansion=True)
    target = {**manifest, "continuation": inherited[0]}
    replay = run_study(questions, config, target, inherited, tmp_path / "network-continued", provider_factory=forbidden)
    assert replay["transport_missing_requests"] == 1 and replay["new_charged_or_reserved_usd"] == 0
    assert replay["prior_charged_or_reserved_usd"] == result["new_charged_or_reserved_usd"]
    assert rows_at(output / "requests.sqlite3") == rows


@pytest.mark.parametrize("error_name", ["ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"])
def test_transport_retry_first_success_exact_request_and_preserved_reservation(tmp_path, error_name):
    import httpx

    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    task = graph.ready()
    calls = []

    class FailOnce(MockProvider):
        def generate(self, request):
            calls.append(request.document())
            if len(calls) == 1:
                raise getattr(httpx, error_name)("fixture transient connection error")
            return super().generate(request)

    provider = FailOnce()
    request = task.build({}, provider.token_count)
    path = tmp_path / "network-attempts.sqlite3"
    journal = Journal(path, manifest, None)
    try:
        got = journal.call(request, provider, task.parse)
        assert got["position"] and calls == [request.document()] * 2
        rows = rows_at(path)
        assert [r["status"] for r in rows] == ["uncertain", "completed"]
        assert rows[0]["error"] == error_name and rows[0]["charge"] > 0
        assert rows[1]["charge"] > 0 and journal.charged_usd == sum(r["charge"] for r in rows)
        assert rows[0]["response"] is None and rows[0]["request_hash"] == rows[1]["request_hash"]
        assert journal.call(request, OfflineProvider(provider.token_count), task.parse) == got
        assert rows_at(path) == rows and len(calls) == 2
    finally:
        journal.close()


@pytest.mark.parametrize("purpose", ["debate", "judge_b", "position"])
def test_continue_skipped_connection_repairs_only_missing_tasks_and_descendants(tmp_path, purpose):
    import httpx

    questions, config, manifest = configured(tmp_path)
    source = tmp_path / "legacy-skipped"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    failed_key = (
        next(k for k, t in graph.tasks.items() if t.purpose == purpose and t.model == "gpt-5.6-terra")
        if purpose != "judge_b"
        else next(k for k, t in graph.tasks.items() if t.purpose == purpose)
    )

    class OneUnavailable(MockProvider):
        def generate(self, request):
            if request.key == failed_key:
                raise httpx.ConnectError("old skipped request")
            return super().generate(request)

    provider = OneUnavailable()
    journal = Journal(source / "requests.sqlite3", manifest, None)
    try:
        while task := graph.ready():
            request = task.build(graph.values, provider.token_count)
            try:
                # Reproduce the old policy: one failed attempt, then mark its dependents missing.
                got = journal._call_once(request, provider, task.parse)
            except RunBlocked:
                assert task.key == failed_key
                graph.reject(
                    task, LocalTaskFailure(task.key, [task.key], "ConnectError", kind="transport_unavailable_no_retry")
                )
            else:
                graph.accept(task, request, got)
        assert graph.complete and not graph.successful
        expected_new = {failed_key, *graph.blocked}
    finally:
        journal.close()
    before = rows_at(source / "requests.sqlite3")
    inherited = load_checkpoint(source, manifest, allow_question_expansion=True)
    sent = []

    class Tracking(MockProvider):
        def generate(self, request):
            sent.append(request.key)
            assert request.key in expected_new, "A previously successful task was regenerated"
            return super().generate(request)

    output = tmp_path / "repaired"
    result = run_study(
        questions, config, {**manifest, "continuation": inherited[0]}, inherited, output, provider_factory=Tracking
    )
    assert result["status"] == "completed" and result["trajectories"]["success"] == 18
    assert set(sent) == expected_new and len(sent) == len(expected_new)
    assert result["transport_retries"]["recovered_logical_requests"] == 1
    assert result["transport_missing_requests"] == 0
    assert rows_at(source / "requests.sqlite3") == before


@pytest.mark.parametrize(
    "text,should_retry", [("G", False), ("GAdditional unsolicited prose", True), ("G\nExtra text", True)]
)
def test_native_alignment_error_retries_only_if_output_contract_also_fails(text, should_retry):
    request = {"purpose": "d_text", "effort": "none", "candidate_labels": list("ABCDEFG")}
    response = {
        "status": "invalid_native_output",
        "text": text,
        "reasoning_tokens": 0,
        "raw": {
            "provider": "tinker",
            "stop_reason": "stop",
            "parsed_message": {"role": "assistant"},
            "validation_error": "Cannot uniquely align parsed content with raw sampled tokens",
        },
    }
    row = {"status": "invalid", "response": canonical(response), "error": "Response status: invalid_native_output"}
    assert retryable_format_record(request, row) == should_retry
    response["reasoning_tokens"] = 1
    row["response"] = canonical(response)
    assert not retryable_format_record(request, row)


def test_native_malformed_rating_uses_identical_bounded_retry(tmp_path):
    questions, config, manifest = configured(tmp_path)
    graph = QuestionGraph(questions[0], config, manifest["plans"][0], manifest)
    called = Counter()
    failed_key = next(k for k in graph.tasks if "/Dtext/" in k)

    class MalformedNative(MockProvider):
        def generate(self, request):
            called[request.key] += 1
            good = super().generate(request)
            if request.key == failed_key and called[request.key] == 1:
                return replace(
                    good,
                    text="GExtra text",
                    status="invalid_native_output",
                    readout=None,
                    raw={
                        "provider": "tinker",
                        "stop_reason": "stop",
                        "parsed_message": {"role": "assistant"},
                        "validation_error": "Cannot uniquely align parsed content with raw sampled tokens",
                    },
                )
            return good

    result = run_study(questions, config, manifest, None, tmp_path / "native-format", provider_factory=MalformedNative)
    assert result["status"] == "completed" and called[failed_key] == 2
    assert result["format_retries"]["recovered_logical_requests"] == 1


@pytest.mark.parametrize("bad_attempts", [1, 3])
def test_inkling_unexpected_reasoning_retry_keeps_settings_and_isolates_exhaustion(tmp_path, bad_attempts):
    questions, config, manifest = configured(tmp_path)
    key = "q0/mixed_family/C/initial/2"
    calls = []

    class ReasoningLeak(MockProvider):
        def generate(self, request):
            good = super().generate(request)
            if request.key == key:
                calls.append(canonical(request.document()))
                if len(calls) <= bad_attempts:
                    return replace(
                        good,
                        status="unexpected_reasoning",
                        reasoning_tokens=2,
                        readout=None,
                        raw={"provider": "tinker", "model": request.model, "effort": request.effort},
                    )
            return good

    output = tmp_path / "inkling-contract"
    result = run_study(questions, config, manifest, None, output, provider_factory=ReasoningLeak)
    assert len(calls) == min(bad_attempts + 1, 3) and len(set(calls)) == 1
    assert json.loads(calls[0])["effort"] == "none"
    assert result["trajectories"]["success"] == 18
    assert result["status"] == ("completed" if bad_attempts == 1 else "completed_with_failures")
    attempts = [r for r in rows_at(output / "requests.sqlite3") if json.loads(r["request"])["key"] == key]
    assert attempts[0]["status"] == "invalid" and attempts[0]["parsed"] is None
    assert json.loads(attempts[0]["response"])["reasoning_tokens"] == 2
    if bad_attempts == 3:
        assert result["format_retries"]["exhausted_logical_requests"] == 1
    else:
        assert result["format_retries"]["recovered_logical_requests"] == 1
        assert json.loads(attempts[-1]["response"])["reasoning_tokens"] == 0
    inherited = load_checkpoint(output, manifest, allow_question_expansion=True)

    def forbidden():
        raise AssertionError("Saved successes/exhaustions must not be regenerated")

    replay = run_study(
        questions,
        config,
        {**manifest, "continuation": inherited[0]},
        inherited,
        tmp_path / "continued-inkling",
        provider_factory=forbidden,
    )
    assert replay["status"] == result["status"] and replay["new_charged_or_reserved_usd"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"model": "Qwen/Qwen3.8-27B"},
        {"purpose": "debate"},
        {"effort": "medium"},
    ],
)
def test_reasoning_exception_is_scoped_to_inkling_direct_measurements(change):
    request = {"model": "thinkingmachines/Inkling", "purpose": "position", "effort": "none", **change}
    response = {
        "status": "unexpected_reasoning",
        "reasoning_tokens": 2,
        "raw": {"provider": "tinker", "model": request["model"], "effort": request["effort"]},
    }
    row = {"status": "invalid", "response": canonical(response), "error": "Response status: unexpected_reasoning"}
    assert not retryable_format_record(request, row)
