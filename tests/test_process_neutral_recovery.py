"""Recovery retains old attempts, skips successes and scopes network retries."""

import json
import sqlite3

import httpx
import pytest

from llm_committee.pivot import storage
from llm_committee.pivot.models import Completion, canonical
from scripts.prepare_process_neutral_judging import prepare
from scripts.recover_process_neutral_judging import grok_transport_recovery, prepare_recovery, recover
from scripts.run_process_neutral_judging import CAP_USD, prepare_execution


class Judge:
    calls = []

    def generate(self, request):
        self.calls.append(request.document())
        return Completion(canonical({"preference": "left", "evidence": "Fixture only"}), 100, 30)

    def close(self):
        pass


class Timeout(Judge):
    def generate(self, request):
        self.calls.append(request.document())
        raise httpx.ReadTimeout("Fixture timeout")


def test_extension_is_narrow_and_restored():
    original = storage.local_transport_record
    request = {"model": "grok-4-6", "purpose": "judge_e"}
    row = {"status": "uncertain", "response": None, "error": "ReadTimeout"}
    with grok_transport_recovery():
        fn = storage.local_transport_record
        assert fn(request, row, "databricks")
        assert not fn(request, row, "direct")
        assert not fn({**request, "purpose": "debate"}, row, "databricks")
        assert not fn(request, {**row, "error": "ValueError"}, "databricks")
        assert not fn(request, {**row, "response": "{}"}, "databricks")
    assert storage.local_transport_record is original


def fixture_run(output):
    from scripts.grok_quality_check import request_from

    prepare(output=output)
    manifest, tasks = prepare_execution(output)
    journal = storage.Journal(output / "requests.sqlite3", manifest, CAP_USD)
    # The old runner saves a successful Gemini call and one uncertain Grok call.
    journal.call(request_from(tasks[0]["request"]), Judge(), json.loads)
    with pytest.raises(storage.RunBlocked):
        journal.call(request_from(tasks[1]["request"]), Timeout(), json.loads)
    journal.close()
    return tasks


def test_recovery_reuses_success_and_retains_timeout_and_cost(tmp_path):
    output = tmp_path / "revision"
    tasks = fixture_run(output)
    Judge.calls = []
    result = recover(output, 2, Judge)
    assert canonical(Judge.calls) == canonical([tasks[1]["request"]])
    assert result["completed_calls"] == 2 and result["attempts"] == 3
    assert result["cost_accounting"]["unresolved_reservations_usd"] > 0
    assert recover(output, 2, Judge) == result
    assert len(Judge.calls) == 1
    prepare_recovery(output)
    with sqlite3.connect(output / "requests.sqlite3") as db:
        db.execute("UPDATE calls SET charge=0 WHERE status='uncertain'")
    with pytest.raises(ValueError, match="Pre-recovery attempt"):
        prepare_recovery(output)


def test_recovery_never_exceeds_two_additional_attempts(tmp_path):
    output = tmp_path / "revision"
    fixture_run(output)
    Timeout.calls = []
    result = recover(output, 2, Timeout)
    assert len(Timeout.calls) == 2
    assert result["completed_calls"] == 1 and result["attempts"] == 4
    assert len(result["noncompleted_attempts"]) == 3
    assert recover(output, 2, Timeout) == result
    assert len(Timeout.calls) == 2
