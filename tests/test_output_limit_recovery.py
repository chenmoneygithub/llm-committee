"""Narrow output-limit retries preserve request settings, billing uncertainty and checkpoints."""

import json
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from llm_committee.pivot.databricks_provider import ENDPOINTS, databricks_payload
from llm_committee.pivot.failures import (
    FORMAT_RETRY_POLICY,
    OUTPUT_LIMIT_RETRY_POLICY,
    LocalTaskFailure,
    local_output_limit_record,
)
from llm_committee.pivot.models import Completion, Message, Request, canonical
from llm_committee.pivot.providers import reservation_usd
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.stateful_run import cache_usage
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.triadic_run import run
from llm_committee.pivot.triadic_study import TriadicMockProvider, prepare
from scripts.audit_triadic import audit
from scripts.recover_triadic_output_limit import prepare as prepare_recovery
from scripts.recover_triadic_output_limit import verify_preservation
from scripts.scale_public_history import execution_specs

PLAN = Path(__file__).resolve().parents[1] / "docs/turn-tone-triadic-shared-plan-2026-09-28.json"
ERROR = "Could not finish the message because max_tokens or model output limit was reached. Please try again with higher max_tokens."


def output_limit(request):
    return Completion(
        "",
        None,
        None,
        status="http_error",
        raw={
            "provider": "databricks",
            "http_status": 400,
            "endpoint": ENDPOINTS[request.model],
            "request_payload": databricks_payload(request),
            "response": {
                "error_code": "BAD_REQUEST",
                "message": json.dumps(
                    {
                        "error": {
                            "message": ERROR,
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                ),
            },
        },
    )


def request_fixture():
    return Request("test", "synthesis", "gpt-5.6-terra", (Message("user", "Answer this question"),), "medium", 4096)


@pytest.mark.parametrize("change", [None, "401", "403", "429", "overflow", "payload", "content", "endpoint"])
def test_output_error_recognition_is_exact(change):
    request = request_fixture()
    value = asdict(output_limit(request))
    if change in ("401", "403", "429"):
        value["raw"]["http_status"] = int(change)
    elif change == "overflow":
        value["raw"]["response"]["message"] = json.dumps(
            {"error": {"message": "Context length exceeded", "type": "invalid_request_error"}}
        )
    elif change == "payload":
        value["raw"]["request_payload"]["max_tokens"] += 1
    elif change == "content":
        value["text"] = "Actual valid answer with missing usage"
    elif change == "endpoint":
        value["raw"]["endpoint"] = "different"
    row = {"status": "billing_unknown", "response": canonical(value)}
    assert local_output_limit_record(request.document(), row, "databricks") == (change is None)
    assert not local_output_limit_record(request.document(), row, "direct")


@pytest.mark.parametrize("bad_count", [1, 2, 3])
def test_bounded_identical_retry_keeps_unknown_usage(tmp_path, bad_count):
    manifest = {
        "config": {"closed_provider": "databricks"},
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": OUTPUT_LIMIT_RETRY_POLICY,
        },
    }
    request, seen = request_fixture(), []

    class Provider:
        def generate(self, request):
            seen.append(canonical(request.document()))
            return output_limit(request) if len(seen) <= bad_count else Completion('{"answer":"yes"}', 100, 10)

    journal = Journal(tmp_path / "requests.sqlite3", manifest, None)
    try:
        if bad_count == 3:
            with pytest.raises(LocalTaskFailure, match="failed_after_output_limit_retries"):
                journal.call(request, Provider(), json.loads)
        else:
            assert journal.call(request, Provider(), json.loads) == {"answer": "yes"}
        rows = journal.db.execute("SELECT * FROM calls ORDER BY rowid").fetchall()
        assert len(seen) == min(3, bad_count + 1) and len(set(seen)) == 1
        assert all(
            row[3] == "billing_unknown" and row[4] == reservation_usd(request, closed_provider="databricks")
            for row in rows[:bad_count]
        )
        if bad_count == 3:
            with pytest.raises(LocalTaskFailure):
                journal.call(request, OfflineProvider(None), json.loads)
        else:
            journal.call(request, OfflineProvider(None), json.loads)
        assert rows == journal.db.execute("SELECT * FROM calls ORDER BY rowid").fetchall()
        usage = cache_usage(tmp_path)[0]
        assert usage["responses_with_unknown_usage"] == bad_count
        assert usage["cached_input_percent"] is None
    finally:
        journal.close()


def test_no_opt_in_still_blocks(tmp_path):
    manifest = {
        "config": {"closed_provider": "databricks"},
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
        },
    }

    class Provider:
        def generate(self, request):
            return output_limit(request)

    journal = Journal(tmp_path / "requests.sqlite3", manifest, None)
    try:
        with pytest.raises(RunBlocked):
            journal.call(request_fixture(), Provider(), json.loads)
        assert journal.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1
    finally:
        journal.close()


def test_triadic_diagnostic_failure_is_local_and_snapshot_auditable(tmp_path):
    manifest, contexts = prepare(PLAN, roster="same_model", mock=True)
    calls = Counter()

    class FailedDiagnostic(TriadicMockProvider):
        def generate(self, request):
            calls[request.key] += 1
            if "/calibration/verbosity/variant" in request.key:
                return output_limit(request)
            return super().generate(request)

    output = tmp_path / "local-failure"
    report = run(manifest, contexts, output, question_limit=3, provider_factory=FailedDiagnostic)
    assert report["status"] == "completed_with_failures"
    assert all(n == 3 if "/calibration/verbosity/variant" in k else n == 1 for k, n in calls.items())
    assert report["trajectory_statuses"] == {"success": 36}
    assert report["quality_comparison_statuses"] == {"success": 6}
    result = audit(output)
    assert result["failed_tasks_verified"] > 0 and result["missing_primary_pairs"] == 0
    subprocess.run(
        [sys.executable, "-m", "scripts.audit_frozen_run", str(output), "--kind", "triadic"],
        check=True,
        capture_output=True,
    )


def test_recovery_imports_without_regenerating_successes(tmp_path):
    manifest, contexts = prepare(PLAN, roster="same_model", mock=True)
    manifest["execution"].pop("output_limit_retry_policy")

    class Broken(TriadicMockProvider):
        def generate(self, request):
            if "/calibration/verbosity/variant" in request.key:
                return output_limit(request)
            return super().generate(request)

    source, output = tmp_path / "same_model/triadic", tmp_path / "same_model/recovery"
    report = run(manifest, contexts, source, question_limit=3, request_limit=1, workers=1, provider_factory=Broken)
    assert report["status"] == "blocked"
    with sqlite3.connect(source / "requests.sqlite3") as db:
        successful = {json.loads(r)["key"] for (r,) in db.execute("SELECT request FROM calls WHERE status='completed'")}
    assert successful
    recovered, loaded = prepare_recovery(source, output)
    assert loaded == contexts and recovered["recovery"]["scientific_protocol_unchanged"]
    assert verify_preservation(output)["preserved_successful_requests"] == len(successful)
    assert prepare_recovery(source, output) == (recovered, loaded)
    specs = [{"roster": "same_model", "setting": "triadic", "output_30": str(source)}]
    atomic_json(tmp_path / "execution-plan.json", {"settings": specs})
    assert execution_specs(tmp_path)[0]["output_30"] == str(output)
    assert json.loads((tmp_path / "execution-plan.json").read_text())["settings"] == specs
    seen = []

    class Resume(TriadicMockProvider):
        def generate(self, request):
            assert request.key not in successful
            seen.append(request.key)
            return super().generate(request)

    result = run(recovered, loaded, output, question_limit=3, provider_factory=Resume)
    assert result["status"] == "preflight_completed" and result["completed_questions"] == 3 and seen
    assert verify_preservation(output)["successful_requests_regenerated"] == 0
