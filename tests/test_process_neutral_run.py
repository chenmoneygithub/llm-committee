"""No-call resume and exact, narrowly scoped dispatch of the judge revision."""

import sqlite3

import pytest

from llm_committee.pivot.models import Completion, canonical
from scripts.prepare_process_neutral_judging import prepare
from scripts.run_process_neutral_judging import execution_report, run


class FakeJudge:
    calls = []

    def generate(self, request):
        self.calls.append(request.document())
        return Completion(canonical({"preference": "left", "evidence": "Fixture only"}), 100, 30)

    def close(self):
        pass


def test_two_model_gate_resumes_without_rejudging_and_accounts_for_cost(tmp_path):
    output = tmp_path / "revision"
    _, tasks = prepare(output=output)
    FakeJudge.calls = []
    first = run(output, 2, FakeJudge)
    assert first["status"] == "incomplete"
    assert first["completed_calls"] == 2 and first["attempts"] == 2
    assert first["by_model"] == {"gemini-3.8-flash": 1, "grok-4-6": 1}
    assert first["cost_accounting"]["received_response_estimate_usd"] > 0
    assert first["cost_accounting"]["unresolved_reservations_usd"] == 0
    assert run(output, 2, FakeJudge) == first
    assert len(FakeJudge.calls) == 2
    next_report = run(output, 4, FakeJudge)
    assert next_report["completed_calls"] == 4 and next_report["attempts"] == 4
    assert {canonical(r) for r in FakeJudge.calls} == {canonical(t["request"]) for t in tasks[:4]}
    assert next_report["source_unchanged"] and next_report["old_judgments_unchanged"]


def test_report_refuses_a_changed_journal_request(tmp_path):
    output = tmp_path / "revision"
    prepare(output=output)
    run(output, 2, FakeJudge)
    with sqlite3.connect(output / "requests.sqlite3") as db:
        db.execute("UPDATE calls SET request=json_set(request,'$.effort','high')")
    with pytest.raises(ValueError, match="unplanned or changed"):
        execution_report(output)
